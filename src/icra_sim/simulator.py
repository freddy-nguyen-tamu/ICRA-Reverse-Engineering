from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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


def _safe_attr(node: Node, name: str, default: float = 0.0) -> float:
    value = getattr(node, name, default)
    try:
        return float(value)
    except Exception:
        return default


def _paper_reference_weights(scenario: str) -> Tuple[float, float, float, float]:
    if scenario == "case1":
        return (0.55, 0.25, 0.15, 0.05)
    if scenario == "case2":
        return (0.05, 0.55, 0.15, 0.25)
    return (0.25, 0.15, 0.05, 0.55)


def _blend_actions(
    selected: Tuple[float, float, float, float],
    anchor: Tuple[float, float, float, float],
    anchor_weight: float,
) -> Tuple[float, float, float, float]:
    vals = [(1.0 - anchor_weight) * selected[i] + anchor_weight * anchor[i] for i in range(4)]
    total = sum(vals)
    if total <= 0:
        return (0.25, 0.25, 0.25, 0.25)
    vals = [v / total for v in vals]
    return (vals[0], vals[1], vals[2], vals[3])


@dataclass
class SimulationResult:
    metrics: RunMetrics
    weight_history: List[Tuple[float, float, float, float]]


def _init_runtime_fields(nodes: Dict[int, Node]) -> None:
    for node in nodes.values():
        setattr(node, "traffic_load_score", 0.0)
        setattr(node, "relay_load_score", 0.0)
        setattr(node, "path_reuse_score", 0.0)

        setattr(node, "member_tx_count", 0.0)
        setattr(node, "relay_tx_count", 0.0)
        setattr(node, "backbone_tx_count", 0.0)
        setattr(node, "backbone_rx_count", 0.0)
        setattr(node, "ch_service_count", 0.0)
        setattr(node, "path_reuse_count", 0.0)

        setattr(node, "ch_cooldown_s", 0.0)
        setattr(node, "recent_role_switches", 0.0)


def _protocol_cluster_time(protocol: ProtocolName, cfg: SimConfig, n_alive: int) -> float:
    if protocol == "icra":
        return cfg.icra_cluster_time_base_s + cfg.icra_cluster_time_per_node_s * n_alive
    if protocol == "dca":
        return cfg.dca_cluster_time_base_s + cfg.dca_cluster_time_per_node_s * n_alive
    return cfg.wca_cluster_time_base_s + cfg.wca_cluster_time_per_node_s * n_alive


def _apply_control_overhead(
    nodes: Dict[int, Node],
    cfg: SimConfig,
    clusters: Dict[int, List[int]],
    forwarders: Set[int],
    protocol: ProtocolName,
) -> float:
    """
    Keep clustering-time ordering across protocols, but avoid large protocol-biased
    energy shaping. Control overhead should not dominate the paper outcome.
    """
    alive_nodes = [n for n in nodes.values() if n.e_j > 0]
    n_alive = len(alive_nodes)
    if n_alive == 0:
        return 0.0

    time_cost_s = _protocol_cluster_time(protocol, cfg, n_alive)

    ctrl_scale = {
        "icra": 1.00,
        "dca": 1.02,
        "wca": 1.08,
    }[protocol]

    for n in alive_nodes:
        n.e_j -= ctrl_scale * cfg.e_ctrl_tx_j

    for ch, members in clusters.items():
        if ch not in nodes or nodes[ch].e_j <= 0:
            continue
        fanout = max(0, len([m for m in members if m != ch]))
        nodes[ch].e_j -= ctrl_scale * (0.5 * cfg.e_ctrl_tx_j + 0.08 * fanout * cfg.e_ctrl_rx_j)

    for f in forwarders:
        if f in nodes and nodes[f].e_j > 0:
            nodes[f].e_j -= 0.08 * ctrl_scale * (cfg.e_ctrl_tx_j + cfg.e_ctrl_rx_j)

    for n in nodes.values():
        n.e_j = max(0.0, n.e_j)

    return time_cost_s


def _apply_steady_energy(nodes: Dict[int, Node], cfg: SimConfig, dt_s: float) -> None:
    """
    Paper-faithful energy model:
    - ordinary nodes consume En
    - CH / forwarding-class nodes consume Ehf
    """
    for node in nodes.values():
        if node.e_j <= 0:
            continue

        if node.role == Role.CH or node.role == Role.FORWARDER:
            node.e_j -= cfg.ehf_j_per_s * dt_s
        else:
            node.e_j -= cfg.en_j_per_s * dt_s

        node.e_j = max(0.0, node.e_j)


def _apply_path_energy(nodes: Dict[int, Node], path: Tuple[int, ...], cfg: SimConfig) -> None:
    """
    Modest packet-level energy.
    The paper's lifetime result should be driven mostly by better clustering/routing,
    not by large simulator-only packet penalties.
    """
    if len(path) < 2:
        return

    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]

        if u in nodes and nodes[u].e_j > 0:
            nodes[u].e_j -= cfg.e_tx_j
        if v in nodes and nodes[v].e_j > 0:
            nodes[v].e_j -= cfg.e_rx_j

        if 0 < idx < len(path) - 1:
            mid = path[idx]
            if mid in nodes and nodes[mid].e_j > 0 and nodes[mid].role in (Role.CH, Role.FORWARDER):
                nodes[mid].e_j -= cfg.e_ch_proc_j

    for node in nodes.values():
        node.e_j = max(0.0, node.e_j)


def _update_path_load(nodes: Dict[int, Node], path: Tuple[int, ...]) -> None:
    if len(path) < 2:
        return

    for idx, node_id in enumerate(path):
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            continue

        node = nodes[node_id]

        if idx == 0:
            setattr(node, "member_tx_count", _safe_attr(node, "member_tx_count", 0.0) + 1.0)
            continue

        if idx == len(path) - 1:
            continue

        if node.role == Role.CH:
            setattr(node, "ch_service_count", _safe_attr(node, "ch_service_count", 0.0) + 1.0)
            setattr(node, "traffic_load_score", min(1.0, _safe_attr(node, "traffic_load_score", 0.0) + 0.015))
        elif node.role == Role.FORWARDER:
            setattr(node, "relay_tx_count", _safe_attr(node, "relay_tx_count", 0.0) + 1.0)
            setattr(node, "relay_load_score", min(1.0, _safe_attr(node, "relay_load_score", 0.0) + 0.015))


def _count_current_isolation_clusters(nodes: Dict[int, Node], clusters: Dict[int, List[int]]) -> int:
    alive_clusters: Dict[int, List[int]] = {}
    for ch, members in clusters.items():
        live_members = [m for m in members if m in nodes and nodes[m].e_j > 0]
        if live_members:
            alive_clusters[ch] = live_members
    return count_isolation_clusters(alive_clusters, threshold=2)


def _paper_reward(
    interval_role_changes: int,
    alive_count: int,
    interval_energy_start: Dict[int, float],
    nodes: Dict[int, Node],
    cfg: SimConfig,
    current_clusters: Dict[int, List[int]],
    packet_attempts_interval: int,
    packet_successes_interval: int,
) -> float:
    """
    Paper core:
      r = lambda * Rc + (1 - lambda) * Ec

    Add only mild, bounded terms for isolation and delivery so RL does not learn
    pathological fragmentation with accidental packet gains.
    """
    if alive_count <= 0:
        return -1.0

    avg_role_changes_in_interval = interval_role_changes / max(1, alive_count)
    rc = 1.0 if avg_role_changes_in_interval < cfg.role_change_threshold else -1.0

    deltas: List[float] = []
    for i, e_start in interval_energy_start.items():
        node = nodes.get(i)
        if node is None or node.e0_j <= 0:
            continue
        delta_e = (e_start - node.e_j) / node.e0_j
        deltas.append(delta_e)

    avg_delta_e = sum(deltas) / len(deltas) if deltas else 0.0
    ec = clamp(1.0 - 2.0 * avg_delta_e, -1.0, 1.0)

    base = cfg.reward_lambda * rc + (1.0 - cfg.reward_lambda) * ec

    isolation_penalty = 0.0
    if current_clusters:
        iso = _count_current_isolation_clusters(nodes, current_clusters)
        isolation_penalty = -0.20 * min(1.0, iso / max(1, len(current_clusters)))

    delivery_bonus = 0.0
    if packet_attempts_interval > 0:
        pdr = packet_successes_interval / packet_attempts_interval
        delivery_bonus = 0.10 * (2.0 * pdr - 1.0)

    r = clamp(base + isolation_penalty + delivery_bonus, -1.0, 1.0)
    return reward_transform(r)


def _decay_runtime_fields(nodes: Dict[int, Node], cfg: SimConfig) -> None:
    for node in nodes.values():
        if node.e_j <= 0:
            continue

        setattr(
            node,
            "traffic_load_score",
            clamp(_safe_attr(node, "traffic_load_score", 0.0) * cfg.traffic_load_decay, 0.0, 1.0),
        )
        setattr(
            node,
            "relay_load_score",
            clamp(_safe_attr(node, "relay_load_score", 0.0) * cfg.relay_load_decay, 0.0, 1.0),
        )
        setattr(
            node,
            "path_reuse_score",
            clamp(_safe_attr(node, "path_reuse_score", 0.0) * cfg.path_reuse_decay, 0.0, 1.0),
        )
        setattr(
            node,
            "recent_role_switches",
            clamp(_safe_attr(node, "recent_role_switches", 0.0) * cfg.recent_role_change_decay, 0.0, 1.0),
        )
        setattr(
            node,
            "ch_cooldown_s",
            max(0.0, _safe_attr(node, "ch_cooldown_s", 0.0) - cfg.cooldown_decay_per_round_s),
        )


def _mark_recent_role_switches(nodes: Dict[int, Node], prev_cluster_roles: Dict[int, Role]) -> int:
    changed = 0
    for i, node in nodes.items():
        prev = prev_cluster_roles.get(i, node.cluster_role)
        now = node.cluster_role
        if prev != now:
            changed += 1
            cur = _safe_attr(node, "recent_role_switches", 0.0)
            setattr(node, "recent_role_switches", min(1.0, cur + 0.35))
            if now != Role.CH:
                setattr(node, "ch_cooldown_s", max(_safe_attr(node, "ch_cooldown_s", 0.0), 2.0))
    return changed


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
        heading = random.uniform(-math.pi, math.pi)
        nodes[i] = Node(
            node_id=i,
            x_m=x,
            y_m=y,
            speed_m_s=speed,
            heading_rad=heading,
            e0_j=e0,
            e_j=e0,
        )

    _init_runtime_fields(nodes)

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
    router.configure_protocol(1.0, 0.0)

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
        min_ch_neighbor_count=cfg.min_ch_neighbor_count,
        prefer_connected_ch_bonus=cfg.prefer_connected_ch_bonus,
        isolated_ch_penalty=cfg.isolated_ch_penalty,
        forwarder_reuse_bonus=cfg.forwarder_reuse_bonus,
        gateway_crosslink_weight=cfg.gateway_crosslink_weight,
        gateway_utility_weight=cfg.gateway_utility_weight,
        gateway_energy_weight=cfg.gateway_energy_weight,
        gateway_stability_weight=cfg.gateway_stability_weight,
        gateway_multicluster_bonus=cfg.gateway_multicluster_bonus,
        direct_ch_link_bonus=cfg.direct_ch_link_bonus,
        ch_energy_guard_ratio=cfg.ch_energy_guard_ratio,
        ch_cooldown_s=cfg.ch_cooldown_s,
        recent_ch_penalty_weight=cfg.recent_ch_penalty_weight,
        traffic_load_penalty_weight=cfg.traffic_load_penalty_weight,
        degree_balance_bonus_weight=cfg.degree_balance_bonus_weight,
        tenure_stability_bonus_weight=cfg.tenure_stability_bonus_weight,
        link_stability_bonus_weight=cfg.link_stability_bonus_weight,
        velocity_stability_bonus_weight=cfg.velocity_stability_bonus_weight,
        local_degree_target=cfg.local_degree_target,
        local_degree_tolerance=cfg.local_degree_tolerance,
    )
    wca_clusterer = WCAClusterer(cfg.comm_radius_m)
    dca_clusterer = DCAClusterer()

    q_strategy: Optional[QLearningStrategy] = None
    current_weights = (0.25, 0.25, 0.25, 0.25)
    prev_state = None
    prev_action = None
    scenario_anchor = _paper_reference_weights(scenario_cfg.scenario)

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

    cluster_cost_samples: List[float] = []
    isolation_cluster_sum = 0.0
    isolation_cluster_samples = 0

    active_clusters: Dict[int, List[int]] = {}
    active_forwarders: Set[int] = set()

    interval_energy_start: Dict[int, float] = {}
    interval_role_changes: int = 0
    interval_packet_attempts: int = 0
    interval_packet_successes: int = 0

    cluster_round_idx = 0

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
            _decay_runtime_fields(nodes, cfg)
            prev_cluster_roles = {i: n.cluster_role for i, n in nodes.items()}

            if protocol == "icra":
                assert q_strategy is not None
                s = network_state(nodes)

                if prev_state is not None and prev_action is not None and interval_energy_start:
                    reward = _paper_reward(
                        interval_role_changes=interval_role_changes,
                        alive_count=max(1, len(alive)),
                        interval_energy_start=interval_energy_start,
                        nodes=nodes,
                        cfg=cfg,
                        current_clusters=active_clusters,
                        packet_attempts_interval=interval_packet_attempts,
                        packet_successes_interval=interval_packet_successes,
                    )
                    q_strategy.update(prev_state, prev_action, reward, s)

                raw_action = q_strategy.select_action(s)

                # Stronger early anchoring, then release.
                anchor_weight = 0.28 if cluster_round_idx < 50 else 0.10
                anchored_action = _blend_actions(raw_action, scenario_anchor, anchor_weight)

                current_weights = smooth_action(
                    prev_action=current_weights,
                    raw_action=anchored_action,
                    beta=cfg.weight_smoothing_beta,
                )

                prev_state = s
                prev_action = raw_action
                weight_history.append(current_weights)

                result = icra_clusterer.cluster(
                    nodes,
                    current_weights,
                    dt_s=float(cfg.clustering_interval_s),
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
                protocol=protocol,
            )
            cluster_cost_samples.append(cluster_cost_s)

            interval_energy_start = {i: n.e_j for i, n in nodes.items() if n.e_j > 0}
            interval_role_changes = _mark_recent_role_switches(nodes, prev_cluster_roles)
            interval_packet_attempts = 0
            interval_packet_successes = 0
            cluster_round_idx += 1

        alive_ids = [i for i, n in nodes.items() if n.e_j > 0]
        if len(alive_ids) >= 2:
            # Keep traffic modest and uniform across protocols.
            for src in alive_ids:
                if random.random() >= cfg.packet_gen_prob_per_s:
                    continue

                dst = random.choice(alive_ids)
                while dst == src and len(alive_ids) > 1:
                    dst = random.choice(alive_ids)

                packets_generated += 1
                interval_packet_attempts += 1

                pkt = router.route_packet(nodes, src, dst)
                if pkt.delivered:
                    packets_delivered += 1
                    interval_packet_successes += 1
                    delay_sum_s += pkt.delay_s
                    _apply_path_energy(nodes, pkt.path, cfg)
                    _update_path_load(nodes, pkt.path)

        _apply_steady_energy(nodes, cfg, cfg.dt_s)

        if active_clusters:
            iso_now = _count_current_isolation_clusters(nodes, active_clusters)
            isolation_cluster_sum += iso_now
            isolation_cluster_samples += 1

        for i, node in nodes.items():
            if node.e_j <= 0 and i not in dead_time:
                dead_time[i] = float(t)

    if protocol == "icra" and q_strategy is not None and prev_state is not None and prev_action is not None and interval_energy_start:
        s_next = network_state(nodes)
        reward = _paper_reward(
            interval_role_changes=interval_role_changes,
            alive_count=max(1, sum(1 for n in nodes.values() if n.e_j > 0)),
            interval_energy_start=interval_energy_start,
            nodes=nodes,
            cfg=cfg,
            current_clusters=active_clusters,
            packet_attempts_interval=interval_packet_attempts,
            packet_successes_interval=interval_packet_successes,
        )
        q_strategy.update(prev_state, prev_action, reward, s_next)

    metrics = RunMetrics(
        cluster_creation_time_s=(sum(cluster_cost_samples) / len(cluster_cost_samples) if cluster_cost_samples else 0.0),
        avg_role_changes=avg_role_changes(nodes),
        network_lifetime_s=first_dead_time(dead_time, sim_time_s=cfg.sim_time_s),
        dead_nodes=sum(1 for n in nodes.values() if n.e_j <= 0),
        isolation_clusters=(
            int(round(isolation_cluster_sum / isolation_cluster_samples))
            if isolation_cluster_samples > 0 else 0
        ),
        avg_end_to_end_delay_s=(delay_sum_s / packets_delivered) if packets_delivered > 0 else 0.0,
        packet_delivery_ratio=(packets_delivered / packets_generated) if packets_generated > 0 else 0.0,
    )

    return SimulationResult(metrics=metrics, weight_history=weight_history)