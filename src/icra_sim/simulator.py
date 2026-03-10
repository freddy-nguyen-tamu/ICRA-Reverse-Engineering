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
from .rl.qlearning import (
    QLearningStrategy,
    generate_action_space,
    network_state,
    reward_transform,
    smooth_action,
)
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
            if j not in valid_ids or j not in alive_ids or j in seen:
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
    alive_nodes = [n for n in nodes.values() if n.e_j > 0]
    n_alive = len(alive_nodes)
    if n_alive == 0:
        return 0.0

    time_cost_s = 0.0

    for n in alive_nodes:
        n.e_j -= cfg.e_ctrl_tx_j
        time_cost_s += cfg.ctrl_proc_delay_s
        for j in n.neighbors:
            if j in nodes and nodes[j].e_j > 0:
                nodes[j].e_j -= 0.20 * cfg.e_ctrl_rx_j
                time_cost_s += 0.20 * cfg.ctrl_proc_delay_s

    for ch, members in clusters.items():
        if ch not in nodes or nodes[ch].e_j <= 0:
            continue

        nodes[ch].e_j -= cfg.e_ctrl_tx_j
        time_cost_s += cfg.ctrl_proc_delay_s

        for m in members:
            if m == ch or m not in nodes or nodes[m].e_j <= 0:
                continue
            nodes[m].e_j -= 0.50 * cfg.e_ctrl_tx_j
            nodes[ch].e_j -= 0.50 * cfg.e_ctrl_rx_j
            time_cost_s += cfg.ctrl_proc_delay_s

    for f in forwarders:
        if f in nodes and nodes[f].e_j > 0:
            nodes[f].e_j -= 0.40 * (cfg.e_ctrl_tx_j + cfg.e_ctrl_rx_j)
            time_cost_s += 0.40 * cfg.ctrl_proc_delay_s

    if is_icra:
        time_cost_s += 0.15 * n_alive * cfg.ctrl_proc_delay_s

    for n in nodes.values():
        n.e_j = max(0.0, n.e_j)

    return time_cost_s


def _apply_steady_energy(nodes: Dict[int, Node], cfg: SimConfig, dt_s: float) -> None:
    for node in nodes.values():
        if node.e_j <= 0:
            continue
        if node.role in (Role.CH, Role.FORWARDER):
            node.e_j -= cfg.ehf_j_per_s * dt_s
        else:
            node.e_j -= cfg.en_j_per_s * dt_s
        node.e_j = max(0.0, node.e_j)


def _apply_path_energy(nodes: Dict[int, Node], path: Tuple[int, ...], cfg: SimConfig) -> None:
    if len(path) < 2:
        return

    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]
        if u in nodes and nodes[u].e_j > 0:
            nodes[u].e_j -= cfg.e_tx_j
        if v in nodes and nodes[v].e_j > 0:
            nodes[v].e_j -= cfg.e_rx_j

        if idx + 1 < len(path) - 1:
            mid = path[idx + 1]
            if mid in nodes and nodes[mid].e_j > 0 and nodes[mid].role in (Role.CH, Role.FORWARDER):
                nodes[mid].e_j -= cfg.e_ch_proc_j

    for node in nodes.values():
        node.e_j = max(0.0, node.e_j)


def _interval_reward(
    nodes: Dict[int, Node],
    interval_energy_start: Dict[int, float],
    interval_role_changes: int,
    current_action: Optional[Tuple[float, float, float, float]],
    previous_action: Optional[Tuple[float, float, float, float]],
    cfg: SimConfig,
) -> float:
    rc = 1.0 if interval_role_changes < cfg.role_change_threshold else -1.0

    deltas: List[float] = []
    for i, node in nodes.items():
        if i not in interval_energy_start:
            continue
        if node.e0_j <= 0:
            continue
        delta = (interval_energy_start[i] - node.e_j) / node.e0_j
        deltas.append(delta)

    delta_e = sum(deltas) / len(deltas) if deltas else 0.0
    ec = clamp(1.0 - 2.0 * delta_e, -1.0, 1.0)

    reward = cfg.reward_lambda * rc + (1.0 - cfg.reward_lambda) * ec

    if current_action is not None and previous_action is not None:
        l1 = sum(abs(current_action[i] - previous_action[i]) for i in range(4))
        reward -= 0.20 * l1
        if l1 < 0.10:
            reward += cfg.action_stickiness_bonus

    return reward_transform(clamp(reward, -1.0, 1.0))


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
    )
    wca_clusterer = WCAClusterer(cfg.comm_radius_m)
    dca_clusterer = DCAClusterer()

    q_strategy: Optional[QLearningStrategy] = None
    current_weights = (0.25, 0.25, 0.25, 0.25)
    raw_action_prev: Optional[Tuple[float, float, float, float]] = None
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
            stickiness_bonus=cfg.action_stickiness_bonus,
            min_action_hold_rounds=cfg.min_action_hold_rounds,
            allow_action_jump_l1=cfg.allow_action_jump_l1,
        )

    weight_history: List[Tuple[float, float, float, float]] = []
    dead_time: Dict[int, float] = {}

    packets_generated = 0
    packets_delivered = 0
    delay_sum_s = 0.0

    first_cluster_cost_s: Optional[float] = None
    isolation_cluster_sum = 0.0
    isolation_cluster_samples = 0

    active_clusters: Dict[int, List[int]] = {}
    active_forwarders: set[int] = set()

    interval_energy_start: Dict[int, float] = {}
    interval_role_changes: int = 0
    rounds_since_last_cluster = 0

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
            )
            node.s1 = factors.s1_energy
            node.s2 = factors.s2_degree
            node.s3 = factors.s3_vel_sim
            node.s4 = factors.s4_lht

        if t % cfg.clustering_interval_s == 0:
            role_counts_before = {i: n.role_change_count for i, n in nodes.items()}

            if protocol == "icra":
                assert q_strategy is not None
                s = network_state(nodes)

                if prev_state is not None and prev_action is not None:
                    reward = _interval_reward(
                        nodes=nodes,
                        interval_energy_start=interval_energy_start,
                        interval_role_changes=interval_role_changes,
                        current_action=prev_action,
                        previous_action=raw_action_prev,
                        cfg=cfg,
                    )
                    q_strategy.update(prev_state, prev_action, reward, s)

                raw_action = q_strategy.select_action(s)
                current_weights = smooth_action(
                    prev_action=current_weights,
                    raw_action=raw_action,
                    beta=cfg.weight_smoothing_beta,
                )

                raw_action_prev = prev_action
                prev_state = s
                prev_action = current_weights
                weight_history.append(current_weights)

                result = icra_clusterer.cluster(
                    nodes=nodes,
                    weights=current_weights,
                    dt_s=cfg.clustering_interval_s,
                    factors_already_set=True,
                )
            elif protocol == "wca":
                result = wca_clusterer.cluster(nodes)
            else:
                result = dca_clusterer.cluster(nodes)

            active_clusters = result.clusters
            active_forwarders = result.forwarders

            cluster_cost_s = _apply_control_overhead(
                nodes=nodes,
                cfg=cfg,
                clusters=active_clusters,
                forwarders=active_forwarders,
                is_icra=(protocol == "icra"),
            )

            if first_cluster_cost_s is None:
                first_cluster_cost_s = cluster_cost_s

            interval_energy_start = {i: n.e_j for i, n in nodes.items() if n.e_j > 0}
            interval_role_changes = sum(
                nodes[i].role_change_count - role_counts_before.get(i, 0)
                for i in nodes.keys()
            )
            rounds_since_last_cluster = 0
        else:
            rounds_since_last_cluster += 1

        alive_ids = [i for i, n in nodes.items() if n.e_j > 0]
        if alive_ids:
            for src in alive_ids:
                if random.random() >= cfg.packet_gen_prob_per_s:
                    continue

                dst = random.choice(alive_ids)
                while dst == src and len(alive_ids) > 1:
                    dst = random.choice(alive_ids)

                packets_generated += 1
                pkt = router.route_packet(nodes, src, dst)
                if pkt.delivered:
                    packets_delivered += 1
                    delay_sum_s += pkt.delay_s
                    _apply_path_energy(nodes, pkt.path, cfg)

        _apply_steady_energy(nodes, cfg, cfg.dt_s)

        if active_clusters:
            isolation_cluster_sum += count_isolation_clusters(active_clusters, threshold=2)
            isolation_cluster_samples += 1

        for i, node in nodes.items():
            if node.e_j <= 0 and i not in dead_time:
                dead_time[i] = float(t)

    if protocol == "icra" and q_strategy is not None and prev_state is not None and prev_action is not None:
        s_next = network_state(nodes)
        reward = _interval_reward(
            nodes=nodes,
            interval_energy_start=interval_energy_start,
            interval_role_changes=interval_role_changes,
            current_action=prev_action,
            previous_action=raw_action_prev,
            cfg=cfg,
        )
        q_strategy.update(prev_state, prev_action, reward, s_next)

    metrics = RunMetrics(
        cluster_creation_time_s=first_cluster_cost_s if first_cluster_cost_s is not None else 0.0,
        avg_role_changes=avg_role_changes(nodes),
        network_lifetime_s=first_dead_time(dead_time, sim_time_s=cfg.sim_time_s),
        dead_nodes=sum(1 for n in nodes.values() if n.e_j <= 0),
        isolation_clusters=int(round(isolation_cluster_sum / isolation_cluster_samples)) if isolation_cluster_samples > 0 else 0,
        avg_end_to_end_delay_s=(delay_sum_s / packets_delivered) if packets_delivered > 0 else 0.0,
        packet_delivery_ratio=(packets_delivered / packets_generated) if packets_generated > 0 else 0.0,
    )

    return SimulationResult(metrics=metrics, weight_history=weight_history)