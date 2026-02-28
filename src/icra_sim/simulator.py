from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import ProtocolName, ScenarioConfig, SimConfig
from .metrics import RunMetrics, avg_role_changes, count_isolation_clusters, first_dead_time
from .mobility.gauss_markov import GaussMarkovMobility
from .node import Node, Role
from .radio import build_neighbor_tables
from .routing.router import Router
from .clustering.clusterer import ICRAClusterer, WCAClusterer, DCAClusterer
from .clustering.utility import compute_factors, weighted_utility
from .rl.qlearning import QLearningStrategy, generate_action_space, network_state, reward_transform
from .utils import clamp


def _init_positions_grid(n: int, width_m: float, height_m: float) -> List[Tuple[float, float]]:
    """Place nodes roughly evenly in the area using a jittered grid."""
    side = math.ceil(math.sqrt(n))
    cell_w = width_m / side
    cell_h = height_m / side

    positions: List[Tuple[float, float]] = []
    for idx in range(n):
        r = idx // side
        c = idx % side
        x0 = (c + 0.5) * cell_w
        y0 = (r + 0.5) * cell_h
        # jitter up to 10% of cell size
        x = x0 + random.uniform(-0.1 * cell_w, 0.1 * cell_w)
        y = y0 + random.uniform(-0.1 * cell_h, 0.1 * cell_h)
        positions.append((clamp(x, 0.0, width_m), clamp(y, 0.0, height_m)))
    random.shuffle(positions)
    return positions


@dataclass
class SimulationResult:
    metrics: RunMetrics
    weight_history: List[Tuple[float, float, float, float]]


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

    # --- Initialize nodes ---
    positions = _init_positions_grid(n_nodes, width_m, height_m)
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

    # --- Models ---
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
        max_hops=cfg.max_hops,
    )

    # Clusterers
    icra_clusterer = ICRAClusterer(
        comm_radius_m=comm_radius_m,
        lht_threshold_s=cfg.lht_threshold_s,
        lht_cap_s=cfg.lht_cap_s,
        v_max=scenario.speed_high_m_s,
    )
    wca_clusterer = WCAClusterer(comm_radius_m=comm_radius_m)
    dca_clusterer = DCAClusterer(comm_radius_m=comm_radius_m, lht_cap_s=cfg.lht_cap_s)

    # RL strategy for ICRA
    weight_history: List[Tuple[float, float, float, float]] = []
    q_strategy: Optional[QLearningStrategy] = None
    if protocol == "icra":
        actions = generate_action_space(step=0.05)
        q_strategy = QLearningStrategy(actions=actions, alpha=cfg.q_alpha, gamma=cfg.q_gamma)

    # Metrics accumulators
    cluster_creation_cpu_s: Optional[float] = None
    isolation_cluster_sum = 0.0
    isolation_cluster_samples = 0

    packets_generated = 0
    packets_delivered = 0
    delay_sum_s = 0.0

    # Dead-node tracking
    dead_time: Dict[int, float] = {}
    dead_flag = {i: False for i in nodes.keys()}

    # Role-change tracking per interval
# Role-change tracking per interval
    last_roles: Dict[int, Tuple[Role, Optional[int], bool]] = {}
    interval_role_changes_by_node: Dict[int, int] = {}

    # For RL reward: energy at interval start
    interval_energy_start: Dict[int, float] = {}

    # RL: last (state, action) for update
    last_state = None
    last_action = None

    # Initial neighbor table
    build_neighbor_tables(nodes, comm_radius_m=comm_radius_m)

    # --- Main loop (interval-based) ---
    t = 0.0
    while t < cfg.sim_time_s:
        # ---- (Re)clustering round ----
        # (1) Update neighbor tables at this moment
        build_neighbor_tables(nodes, comm_radius_m=comm_radius_m)

        # (2) For ICRA: compute state (entropy over factor distributions)
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
                node.s1, node.s2, node.s3, node.s4 = (
                    factors.s1_energy, factors.s2_degree, factors.s3_vel_sim, factors.s4_lht
                )

            s = network_state(nodes)
            a = q_strategy.select_action(s)
            weights = a
            last_state, last_action = s, a
            weight_history.append(weights)

            # measure clustering CPU time only for first clustering
            start = time.perf_counter()
            cluster_res = icra_clusterer.cluster(nodes, weights=weights)
            end = time.perf_counter()
            if cluster_creation_cpu_s is None:
                cluster_creation_cpu_s = end - start

        elif protocol == "wca":
            start = time.perf_counter()
            cluster_res = wca_clusterer.cluster(nodes)
            end = time.perf_counter()
            if cluster_creation_cpu_s is None:
                cluster_creation_cpu_s = end - start
        elif protocol == "dca":
            start = time.perf_counter()
            cluster_res = dca_clusterer.cluster(nodes)
            end = time.perf_counter()
            if cluster_creation_cpu_s is None:
                cluster_creation_cpu_s = end - start
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        # Isolation clusters sample
        iso = count_isolation_clusters(cluster_res.clusters, threshold=2)
        isolation_cluster_sum += iso
        isolation_cluster_samples += 1

        # Role-change count since last clustering (per node)
        interval_role_changes_by_node = {i: 0 for i, n in nodes.items() if n.e_j > 0}

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            current_tuple = (node.role, node.cluster_head, node.is_forwarder)
            if node.node_id in last_roles and last_roles[node.node_id] != current_tuple:
                node.role_change_count += 1
                interval_role_changes_by_node[node.node_id] += 1
            last_roles[node.node_id] = current_tuple
            
        # Store interval-start energy for reward calculation
        interval_energy_start = {i: n.e_j for i, n in nodes.items()}

        # ---- Simulate within this clustering interval ----
        steps = int(cfg.clustering_interval_s / cfg.dt_s)
        for _ in range(steps):
            if t >= cfg.sim_time_s:
                break

            # Mobility and neighbor discovery
            for node in nodes.values():
                if node.e_j <= 0:
                    continue
                mobility.step(node, dt_s=cfg.dt_s)

            build_neighbor_tables(nodes, comm_radius_m=comm_radius_m)

            # Traffic & routing (only alive nodes)
            alive_ids = [i for i, n in nodes.items() if n.e_j > 0]
            if len(alive_ids) >= 2:
                for i in alive_ids:
                    if random.random() < cfg.packet_gen_prob_per_s * cfg.dt_s:
                        dst = random.choice([x for x in alive_ids if x != i])
                        packets_generated += 1
                        result = router.route(nodes, src=i, dst=dst)
                        if result.delivered:
                            packets_delivered += 1
                            delay_sum_s += result.delay_s

            # Energy consumption
            for node in nodes.values():
                if node.e_j <= 0:
                    continue
                drain = cfg.en_j_per_s
                if node.role in (Role.CH, Role.FORWARDER):
                    drain = cfg.ehf_j_per_s
                node.e_j -= drain * cfg.dt_s
                if node.e_j <= 0 and not dead_flag[node.node_id]:
                    dead_flag[node.node_id] = True
                    dead_time[node.node_id] = t

            t += cfg.dt_s

        # ---- RL update at end of interval ----
        if protocol == "icra":
            assert q_strategy is not None
            if last_state is not None and last_action is not None:
                alive_ids = [i for i, n in nodes.items() if n.e_j > 0]
                if alive_ids:
                    Rc_vals = [
                        1.0 if interval_role_changes_by_node.get(i, 0) < cfg.role_change_threshold else -1.0
                        for i in alive_ids
                    ]
                    Rc = sum(Rc_vals) / len(Rc_vals)
                else:
                    Rc = 0.0
                # Ec: energy change rate (average over nodes)
                deltas = []
                for i, node in nodes.items():
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
                q_strategy.update(last_state, last_action, reward)

    # ---- Final metrics ----
    cluster_creation_cpu_s = cluster_creation_cpu_s if cluster_creation_cpu_s is not None else 0.0
    avg_role = avg_role_changes(nodes)
    lifetime = first_dead_time(dead_time, sim_time_s=cfg.sim_time_s)
    dead_nodes = sum(1 for n in nodes.values() if n.e_j <= 0)

    pdr = (packets_delivered / packets_generated) if packets_generated > 0 else 0.0
    avg_delay = (delay_sum_s / packets_delivered) if packets_delivered > 0 else 0.0
    avg_iso = (isolation_cluster_sum / isolation_cluster_samples) if isolation_cluster_samples > 0 else 0.0

    metrics = RunMetrics(
        cluster_creation_time_s=cluster_creation_cpu_s,
        avg_role_changes=avg_role,
        network_lifetime_s=lifetime,
        dead_nodes=dead_nodes,
        isolation_clusters=int(round(avg_iso)),
        avg_end_to_end_delay_s=avg_delay,
        packet_delivery_ratio=pdr,
    )
    return SimulationResult(metrics=metrics, weight_history=weight_history)