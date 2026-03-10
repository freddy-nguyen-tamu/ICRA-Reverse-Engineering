from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .clustering.clusterer import DCAClusterer, ICRAClusterer, WCAClusterer
from .clustering.utility import compute_factors
from .config import ProtocolName, ScenarioConfig, SimConfig
from .metrics import RunMetrics, avg_role_changes, count_isolation_clusters, first_dead_time
from .mobility.gauss_markov import GaussMarkovMobility
from .node import Node, Role
from .radio import build_neighbor_tables
from .rl.qlearning import QLearningStrategy, generate_action_space, network_state, reward_transform
from .routing.router import Router
from .utils import clamp


def _init_positions_random(n: int, width_m: float, height_m: float) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    for _ in range(n):
        positions.append((random.uniform(0.0, width_m), random.uniform(0.0, height_m)))
    return positions


@dataclass
class SimulationResult:
    metrics: RunMetrics
    weight_history: List[Tuple[float, float, float, float]]


def _apply_control_overhead(
    nodes: Dict[int, Node],
    cfg: SimConfig,
    protocol: ProtocolName,
) -> Tuple[float, int]:
    """
    Approximate control-plane cost per reclustering interval.
    Returns (protocol_cost_s, control_messages_count)
    """
    alive = [n for n in nodes.values() if n.e_j > 0]
    if not alive:
        return 0.0, 0

    msg_count = 0
    protocol_cost_s = 0.0

    # hello + utility/status advertisement to neighbors
    for n in alive:
        deg = len([j for j in n.neighbors if nodes[j].e_j > 0])
        tx_msgs = 2
        rx_msgs = 2 * deg

        n.e_j -= tx_msgs * cfg.e_ctrl_tx_j
        n.e_j -= rx_msgs * cfg.e_ctrl_rx_j
        protocol_cost_s += tx_msgs * cfg.control_msg_proc_delay_s
        protocol_cost_s += rx_msgs * cfg.control_msg_proc_delay_s
        msg_count += tx_msgs + rx_msgs

    # RL feedback / GS strategy exchange only for ICRA
    if protocol == "icra":
        for n in alive:
            n.e_j -= cfg.e_rl_feedback_j
            protocol_cost_s += 2.0 * cfg.control_msg_proc_delay_s
            msg_count += 2

    # CH declaration / join / join-ack
    ch_ids = {n.node_id for n in alive if n.role == Role.CH}
    for n in alive:
        if n.role == Role.CH:
            deg = len([j for j in n.neighbors if nodes[j].e_j > 0])
            n.e_j -= cfg.e_ctrl_tx_j
            for _ in range(deg):
                n.e_j -= cfg.e_ctrl_rx_j
            protocol_cost_s += (1 + deg) * cfg.control_msg_proc_delay_s
            msg_count += 1 + deg
        elif n.cluster_head is not None and n.cluster_head in nodes:
            n.e_j -= cfg.e_ctrl_tx_j
            n.e_j -= cfg.e_ctrl_rx_j
            protocol_cost_s += 2.0 * cfg.control_msg_proc_delay_s
            msg_count += 2

    return protocol_cost_s, msg_count


def run_simulation(
    protocol: ProtocolName,
    scenario: ScenarioConfig,
    n_nodes: int,
    cfg: SimConfig,
) -> SimulationResult:
    random.seed(cfg.seed + hash((protocol, scenario.scenario, n_nodes)) % 10_000)

    width_m = cfg.area_km[0] * 1000.0
    height_m = cfg.area_km[1] * 1000.0
    comm_radius_m = cfg.comm_radius_km * 1000.0

    positions = _init_positions_random(n_nodes, width_m, height_m)
    nodes: Dict[int, Node] = {}
    for i in range(n_nodes):
        x, y = positions[i]
        heading = random.uniform(-math.pi, math.pi)

        if scenario.constant_speed:
            speed = scenario.speed_low_m_s
        else:
            speed = random.uniform(scenario.speed_low_m_s, scenario.speed_high_m_s)

        if abs(scenario.init_energy_low_j - scenario.init_energy_high_j) < 1e-9:
            e0 = scenario.init_energy_low_j
        else:
            e0 = random.uniform(scenario.init_energy_low_j, scenario.init_energy_high_j)

        nodes[i] = Node(
            node_id=i,
            x_m=x,
            y_m=y,
            speed_m_s=speed,
            heading_rad=heading,
            e0_j=e0,
            e_j=e0,
        )

    mobility = GaussMarkovMobility(
        alpha=cfg.gauss_markov_alpha,
        speed_range=(scenario.speed_low_m_s, scenario.speed_high_m_s),
        area_m=(width_m, height_m),
        speed_noise_std=cfg.speed_noise_std,
        heading_noise_std=cfg.heading_noise_std,
    )

    router = Router(
        comm_radius_m=comm_radius_m,
        data_rate_kbps=cfg.data_rate_kbps,
        packet_size_bytes=cfg.packet_size_bytes,
        per_hop_processing_delay_s=cfg.per_hop_processing_delay_s,
        mac_contention_delay_s=cfg.mac_contention_delay_s,
        queueing_delay_s=cfg.queueing_delay_s,
        max_hops=cfg.max_hops,
    )

    icra_clusterer = ICRAClusterer(
        comm_radius_m=comm_radius_m,
        lht_threshold_s=cfg.lht_threshold_s,
        lht_cap_s=cfg.lht_cap_s,
        v_max=scenario.speed_high_m_s,
        join_hysteresis_margin=cfg.join_hysteresis_margin,
    )
    wca_clusterer = WCAClusterer(comm_radius_m=comm_radius_m)
    dca_clusterer = DCAClusterer(comm_radius_m=comm_radius_m, lht_cap_s=cfg.lht_cap_s)

    weight_history: List[Tuple[float, float, float, float]] = []
    q_strategy: Optional[QLearningStrategy] = None
    if protocol == "icra":
        actions = generate_action_space(step=0.05)
        q_strategy = QLearningStrategy(
            actions=actions,
            alpha=cfg.q_alpha,
            gamma=cfg.q_gamma,
            epsilon=cfg.q_epsilon,
            epsilon_min=cfg.q_epsilon_min,
            epsilon_decay=cfg.q_epsilon_decay,
        )

    cluster_cost_sum_s = 0.0
    cluster_cost_samples = 0

    isolation_cluster_sum = 0.0
    isolation_cluster_samples = 0

    packets_generated = 0
    packets_delivered = 0
    delay_sum_s = 0.0

    dead_time: Dict[int, float] = {}
    dead_flag = {i: False for i in nodes.keys()}

    last_roles: Dict[int, bool] = {}
    interval_role_changes_by_node: Dict[int, int] = {}
    interval_energy_start: Dict[int, float] = {}

    prev_state = None
    prev_action = None

    build_neighbor_tables(nodes, comm_radius_m=comm_radius_m)

    t = 0.0
    while t < cfg.sim_time_s:
        build_neighbor_tables(nodes, comm_radius_m=comm_radius_m)

        if protocol == "icra":
            assert q_strategy is not None
            for node in nodes.values():
                if node.e_j <= 0:
                    continue
                factors = compute_factors(
                    node=node,
                    nodes=nodes,
                    comm_radius_m=comm_radius_m,
                    n_total=len(nodes),
                    lht_cap_s=cfg.lht_cap_s,
                    v_max=scenario.speed_high_m_s,
                )
                node.s1 = factors.s1_energy
                node.s2 = factors.s2_degree
                node.s3 = factors.s3_vel_sim
                node.s4 = factors.s4_lht

            s = network_state(nodes)
            a = q_strategy.select_action(s)
            weights = a
            prev_state, prev_action = s, a
            weight_history.append(weights)

            cluster_res = icra_clusterer.cluster(nodes, weights=weights, factors_already_set=True)
        elif protocol == "wca":
            cluster_res = wca_clusterer.cluster(nodes)
        elif protocol == "dca":
            cluster_res = dca_clusterer.cluster(nodes)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        # protocol cost, not Python CPU time
        protocol_cost_s, _ = _apply_control_overhead(nodes, cfg, protocol)
        cluster_cost_sum_s += protocol_cost_s
        cluster_cost_samples += 1

        iso = count_isolation_clusters(cluster_res.clusters, threshold=1)
        isolation_cluster_sum += iso
        isolation_cluster_samples += 1

        interval_role_changes_by_node = {i: 0 for i, n in nodes.items() if n.e_j > 0}

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            current_is_ch = node.role == Role.CH
            prev_is_ch = last_roles.get(node.node_id, None)

            if prev_is_ch is not None and prev_is_ch != current_is_ch:
                node.role_change_count += 1
                interval_role_changes_by_node[node.node_id] += 1

            last_roles[node.node_id] = current_is_ch

        interval_energy_start = {i: n.e_j for i, n in nodes.items()}

        steps = int(cfg.clustering_interval_s / cfg.dt_s)
        for _ in range(steps):
            if t >= cfg.sim_time_s:
                break

            for node in nodes.values():
                if node.e_j <= 0:
                    continue
                mobility.step(node, dt_s=cfg.dt_s)

            build_neighbor_tables(nodes, comm_radius_m=comm_radius_m)

            alive_ids = [i for i, n in nodes.items() if n.e_j > 0]
            if len(alive_ids) >= 2:
                for i in alive_ids:
                    if random.random() < cfg.packet_gen_prob_per_s * cfg.dt_s:
                        dst_choices = [x for x in alive_ids if x != i]
                        if not dst_choices:
                            continue
                        dst = random.choice(dst_choices)
                        packets_generated += 1
                        result = router.route(nodes, src=i, dst=dst)

                        path = list(result.path)
                        for u, v in zip(path, path[1:]):
                            if nodes[u].e_j > 0:
                                nodes[u].e_j -= cfg.e_tx_j
                            if nodes[v].e_j > 0:
                                nodes[v].e_j -= cfg.e_rx_j
                            if nodes[u].role == Role.CH and nodes[u].e_j > 0:
                                nodes[u].e_j -= cfg.e_ch_proc_j

                        if result.delivered:
                            packets_delivered += 1
                            delay_sum_s += result.delay_s

            for node in nodes.values():
                if node.e_j <= 0:
                    continue

                if node.role in (Role.CH, Role.FORWARDER):
                    node.e_j -= cfg.ehf_j_per_s * cfg.dt_s
                else:
                    node.e_j -= cfg.en_j_per_s * cfg.dt_s

                if node.e_j <= 0 and not dead_flag[node.node_id]:
                    dead_flag[node.node_id] = True
                    dead_time[node.node_id] = t

            t += cfg.dt_s

        if protocol == "icra":
            assert q_strategy is not None
            if prev_state is not None and prev_action is not None:
                alive_ids = [i for i, n in nodes.items() if n.e_j > 0]

                if alive_ids:
                    Rc_vals = [
                        1.0 if interval_role_changes_by_node.get(i, 0) < cfg.role_change_threshold else -1.0
                        for i in alive_ids
                    ]
                    Rc = sum(Rc_vals) / len(Rc_vals)
                else:
                    Rc = 0.0

                deltas = []
                for i in alive_ids:
                    node = nodes[i]
                    e_start = interval_energy_start.get(i, node.e_j)
                    if node.e0_j <= 0:
                        continue
                    delta = (e_start - node.e_j) / node.e0_j
                    deltas.append(delta)

                deltaE = sum(deltas) / len(deltas) if deltas else 0.0
                Ec = 1.0 - 2.0 * deltaE
                Ec = clamp(Ec, -1.0, 1.0)

                r = cfg.reward_lambda * Rc + (1.0 - cfg.reward_lambda) * Ec
                r = clamp(r, -1.0, 1.0)
                reward = reward_transform(r)
                q_strategy.update(prev_state, prev_action, reward)

    cluster_creation_cost_s = (cluster_cost_sum_s / cluster_cost_samples) if cluster_cost_samples > 0 else 0.0
    avg_role = avg_role_changes(nodes)
    lifetime = first_dead_time(dead_time, sim_time_s=cfg.sim_time_s)
    dead_nodes = sum(1 for n in nodes.values() if n.e_j <= 0)

    pdr = (packets_delivered / packets_generated) if packets_generated > 0 else 0.0
    avg_delay = (delay_sum_s / packets_delivered) if packets_delivered > 0 else 0.0
    avg_iso = (isolation_cluster_sum / isolation_cluster_samples) if isolation_cluster_samples > 0 else 0.0

    metrics = RunMetrics(
        cluster_creation_time_s=cluster_creation_cost_s,
        avg_role_changes=avg_role,
        network_lifetime_s=lifetime,
        dead_nodes=dead_nodes,
        isolation_clusters=int(round(avg_iso)),
        avg_end_to_end_delay_s=avg_delay,
        packet_delivery_ratio=pdr,
    )
    return SimulationResult(metrics=metrics, weight_history=weight_history)