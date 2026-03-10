from __future__ import annotations

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
from .utils import clamp, set_seed


def _init_positions_random(n: int, width_m: float, height_m: float) -> List[Tuple[float, float]]:
    return [(random.uniform(0.0, width_m), random.uniform(0.0, height_m)) for _ in range(n)]


def _sanitize_neighbors(nodes: Dict[int, Node]) -> None:
    valid_ids = set(nodes.keys())
    alive_ids = {i for i, n in nodes.items() if n.e_j > 0}
    for i, node in nodes.items():
        cleaned: List[int] = []
        seen = set()
        for j in node.neighbors:
            if j == i:
                continue
            if j not in valid_ids:
                continue
            if j not in alive_ids:
                continue
            if j in seen:
                continue
            seen.add(j)
            cleaned.append(j)
        node.neighbors = cleaned


@dataclass
class SimulationResult:
    metrics: RunMetrics
    weight_history: List[Tuple[float, float, float, float]]


def _apply_control_overhead(
    nodes: Dict[int, Node],
    cfg: SimConfig,
    clusters: Dict[int, List[int]],
    forwarders: set[int],
    is_icra: bool,
) -> float:
    time_cost_s = 0.0
    alive_nodes = [n for n in nodes.values() if n.e_j > 0]

    for n in alive_nodes:
        n.e_j -= cfg.e_ctrl_tx_j
        time_cost_s += cfg.ctrl_proc_delay_s

        for j in n.neighbors:
            if j in nodes and nodes[j].e_j > 0:
                nodes[j].e_j -= 0.25 * cfg.e_ctrl_rx_j
                time_cost_s += 0.25 * cfg.ctrl_proc_delay_s

    for ch, members in clusters.items():
        if ch not in nodes or nodes[ch].e_j <= 0:
            continue
        for m in members:
            if m == ch or m not in nodes or nodes[m].e_j <= 0:
                continue
            nodes[m].e_j -= 0.6 * cfg.e_ctrl_tx_j
            time_cost_s += 0.5 * cfg.ctrl_proc_delay_s
            nodes[ch].e_j -= 0.5 * cfg.e_ctrl_rx_j
            time_cost_s += 0.5 * cfg.ctrl_proc_delay_s

    for f in forwarders:
        if f in nodes and nodes[f].e_j > 0:
            nodes[f].e_j -= 0.5 * (cfg.e_ctrl_tx_j + cfg.e_ctrl_rx_j)
            time_cost_s += 0.5 * cfg.ctrl_proc_delay_s

    if is_icra:
        for n in alive_nodes:
            n.e_j -= 0.25 * (cfg.e_ctrl_tx_j + cfg.e_ctrl_rx_j)
            time_cost_s += 0.25 * cfg.ctrl_proc_delay_s

    for n in nodes.values():
        n.e_j = max(0.0, n.e_j)

    return time_cost_s


def _pick_random_pair(alive_ids: List[int]) -> Optional[Tuple[int, int]]:
    if len(alive_ids) < 2:
        return None
    src = random.choice(alive_ids)
    dst = random.choice(alive_ids)
    while dst == src:
        dst = random.choice(alive_ids)
    return src, dst


def run_simulation(
    n_nodes: int,
    protocol: ProtocolName,
    scenario_cfg: ScenarioConfig,
    cfg: SimConfig,
) -> SimulationResult:
    set_seed(cfg.seed)

    positions = _init_positions_random(n_nodes, cfg.width_m, cfg.height_m)

    nodes: Dict[int, Node] = {}
    for i, (x, y) in enumerate(positions):
        e0 = random.uniform(scenario_cfg.init_energy_low_j, scenario_cfg.init_energy_high_j)
        speed = (
            scenario_cfg.speed_low_m_s
            if scenario_cfg.constant_speed
            else random.uniform(scenario_cfg.speed_low_m_s, scenario_cfg.speed_high_m_s)
        )
        heading = random.uniform(-3.141592653589793, 3.141592653589793)
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
        speed_range=(scenario_cfg.speed_low_m_s, scenario_cfg.speed_high_m_s),
        area_m=(cfg.width_m, cfg.height_m),
        speed_noise_std=cfg.speed_noise_std,
        heading_noise_std=cfg.heading_noise_std,
    )

    router = Router(
        comm_radius_m=cfg.comm_radius_m,
        data_rate_kbps=cfg.data_rate_kbps,
        packet_size_bytes=cfg.packet_size_bytes,
        per_hop_processing_delay_s=cfg.per_hop_processing_delay_s,
        mac_contention_delay_s=cfg.mac_contention_delay_s,
        queueing_delay_s=cfg.queueing_delay_s,
        max_hops=cfg.max_hops,
    )

    icra_clusterer = ICRAClusterer(
        comm_radius_m=cfg.comm_radius_m,
        lht_threshold_s=cfg.lht_threshold_s,
        lht_cap_s=cfg.lht_cap_s,
        v_max=max(scenario_cfg.speed_high_m_s, 1.0),
        join_hysteresis_margin=cfg.join_hysteresis_margin,
        ch_retain_margin=cfg.ch_retain_margin,
        min_ch_tenure_s=cfg.min_ch_tenure_s,
        max_cluster_members=cfg.max_cluster_members,
        min_gateway_lht_s=cfg.min_gateway_lht_s,
        degree_soft_target=cfg.utility_degree_soft_target,
    )
    wca_clusterer = WCAClusterer(cfg.comm_radius_m)
    dca_clusterer = DCAClusterer()

    q_strategy: Optional[QLearningStrategy] = None
    current_weights = (0.25, 0.25, 0.25, 0.25)
    prev_state = None
    prev_action = None

    if protocol == "icra":
        q_strategy = QLearningStrategy(
            actions=generate_action_space(step=cfg.q_step),
            alpha=cfg.q_alpha,
            gamma=cfg.q_gamma,
            epsilon=cfg.q_epsilon,
            epsilon_min=cfg.q_epsilon_min,
            epsilon_decay=cfg.q_epsilon_decay,
        )

    weight_history: List[Tuple[float, float, float, float]] = []
    dead_time: Dict[int, float] = {}

    packets_generated = 0
    packets_delivered = 0
    delay_sum_s = 0.0

    cluster_cost_sum_s = 0.0
    cluster_cost_samples = 0
    isolation_cluster_sum = 0.0
    isolation_cluster_samples = 0

    interval_role_changes_by_node: Dict[int, int] = {}
    interval_energy_start: Dict[int, float] = {}
    interval_packets_generated = 0
    interval_packets_delivered = 0
    interval_delay_sum_s = 0.0
    interval_isolation_sum = 0.0
    interval_isolation_samples = 0

    active_clusters: Dict[int, List[int]] = {}
    active_forwarders: set[int] = set()

    for t in range(0, cfg.sim_time_s, int(cfg.dt_s)):
        for node in nodes.values():
            if node.e_j <= 0:
                continue

            mobility.step(node, cfg.dt_s)

            if scenario_cfg.constant_speed:
                node.speed_m_s = scenario_cfg.speed_low_m_s
                node.avg_speed.update(node.speed_m_s)

        build_neighbor_tables(nodes, cfg.comm_radius_m)
        _sanitize_neighbors(nodes)

        alive = {i: n for i, n in nodes.items() if n.e_j > 0}

        for node in alive.values():
            factors = compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=cfg.comm_radius_m,
                n_total=max(1, len(alive)),
                lht_cap_s=cfg.lht_cap_s,
                v_max=max(scenario_cfg.speed_high_m_s, 1.0),
                degree_soft_target=cfg.utility_degree_soft_target,
            )
            node.s1 = factors.s1_energy
            node.s2 = factors.s2_degree
            node.s3 = factors.s3_vel_sim
            node.s4 = factors.s4_lht

        if t % cfg.clustering_interval_s == 0:
            interval_role_changes_by_node = {i: 0 for i in nodes.keys()}
            interval_energy_start = {i: n.e_j for i, n in nodes.items() if n.e_j > 0}
            interval_packets_generated = 0
            interval_packets_delivered = 0
            interval_delay_sum_s = 0.0
            interval_isolation_sum = 0.0
            interval_isolation_samples = 0

            role_counts_before = {i: n.role_change_count for i, n in nodes.items()}

            if protocol == "icra":
                assert q_strategy is not None
                s = network_state(nodes)
                a = q_strategy.select_action(s)
                current_weights = a
                prev_state = s
                prev_action = a
                weight_history.append(a)
                result = icra_clusterer.cluster(nodes, current_weights, factors_already_set=True)
            elif protocol == "wca":
                result = wca_clusterer.cluster(nodes)
            else:
                result = dca_clusterer.cluster(nodes)

            active_clusters = result.clusters
            active_forwarders = result.forwarders

            for node in nodes.values():
                if node.e_j <= 0:
                    continue
                if node.role == Role.CH:
                    node.ch_tenure_s += cfg.clustering_interval_s
                else:
                    node.ch_tenure_s = 0.0
                node.note_cluster_membership(node.cluster_head, cfg.clustering_interval_s)

            role_counts_after = {i: n.role_change_count for i, n in nodes.items()}
            for i in nodes.keys():
                interval_role_changes_by_node[i] = role_counts_after[i] - role_counts_before[i]

            cluster_cost = _apply_control_overhead(
                nodes=nodes,
                cfg=cfg,
                clusters=active_clusters,
                forwarders=active_forwarders,
                is_icra=(protocol == "icra"),
            )
            cluster_cost_sum_s += cluster_cost
            cluster_cost_samples += 1

        alive_ids = [i for i, n in nodes.items() if n.e_j > 0]
        for _ in alive_ids:
            if random.random() < cfg.packet_gen_prob_per_s:
                pair = _pick_random_pair(alive_ids)
                if pair is None:
                    continue
                src, dst = pair

                packets_generated += 1
                interval_packets_generated += 1

                pkt = router.route_packet(nodes, src, dst)
                if pkt.delivered:
                    packets_delivered += 1
                    interval_packets_delivered += 1
                    delay_sum_s += pkt.delay_s
                    interval_delay_sum_s += pkt.delay_s

                    for k, node_id in enumerate(pkt.path[:-1]):
                        if node_id not in nodes or nodes[node_id].e_j <= 0:
                            continue
                        nodes[node_id].e_j -= cfg.e_tx_j
                        if nodes[node_id].role in (Role.CH, Role.FORWARDER):
                            nodes[node_id].e_j -= cfg.e_ch_proc_j

                        nxt = pkt.path[k + 1]
                        if nxt in nodes and nodes[nxt].e_j > 0:
                            nodes[nxt].e_j -= cfg.e_rx_j

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            if node.role in (Role.CH, Role.FORWARDER):
                node.e_j -= cfg.ehf_j_per_s * cfg.dt_s
            else:
                node.e_j -= cfg.en_j_per_s * cfg.dt_s
            node.e_j = max(0.0, node.e_j)
            if node.e_j <= 0 and node.node_id not in dead_time:
                dead_time[node.node_id] = float(t)

        iso = count_isolation_clusters(active_clusters)
        isolation_cluster_sum += iso
        isolation_cluster_samples += 1
        interval_isolation_sum += iso
        interval_isolation_samples += 1

        if protocol == "icra" and q_strategy is not None and prev_state is not None and prev_action is not None:
            end_of_interval = ((t + int(cfg.dt_s)) % cfg.clustering_interval_s == 0)
            if end_of_interval:
                alive_ids = [i for i, n in nodes.items() if n.e_j > 0]

                if alive_ids:
                    role_penalties = [
                        1.0 if interval_role_changes_by_node.get(i, 0) < cfg.role_change_threshold else -1.0
                        for i in alive_ids
                    ]
                    Rc = sum(role_penalties) / len(role_penalties)
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
                Ec = clamp(1.0 - 2.5 * deltaE, -1.0, 1.0)

                pdr_interval = (
                    interval_packets_delivered / interval_packets_generated
                    if interval_packets_generated > 0 else 0.0
                )
                Pc = 2.0 * pdr_interval - 1.0

                avg_delay = (
                    interval_delay_sum_s / interval_packets_delivered
                    if interval_packets_delivered > 0
                    else (
                        cfg.per_hop_processing_delay_s
                        + cfg.mac_contention_delay_s
                        + cfg.queueing_delay_s
                    ) * cfg.max_hops
                )
                delay_ref = (
                    cfg.per_hop_processing_delay_s
                    + cfg.mac_contention_delay_s
                    + cfg.queueing_delay_s
                ) * 4.0
                Dc = clamp(1.0 - (avg_delay / max(1e-9, delay_ref)), -1.0, 1.0)

                avg_iso = (
                    interval_isolation_sum / interval_isolation_samples
                    if interval_isolation_samples > 0 else 0.0
                )
                Ic = clamp(1.0 - (avg_iso / max(1.0, len(active_clusters) + 1.0)), -1.0, 1.0)

                cluster_sizes = [len(v) for v in active_clusters.values()] if active_clusters else [0]
                mean_sz = sum(cluster_sizes) / len(cluster_sizes)
                imbalance = sum(abs(sz - mean_sz) for sz in cluster_sizes) / max(
                    1.0, len(cluster_sizes) * max(1.0, mean_sz)
                )
                Bc = clamp(1.0 - imbalance, -1.0, 1.0)

                r = (
                    0.22 * Rc
                    + 0.20 * Ec
                    + 0.24 * Pc
                    + 0.16 * Dc
                    + 0.10 * Ic
                    + 0.08 * Bc
                )
                r = clamp(r, -1.0, 1.0)
                reward = reward_transform(r)

                s_next = network_state(nodes)
                q_strategy.update(prev_state, prev_action, reward, s_next)

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