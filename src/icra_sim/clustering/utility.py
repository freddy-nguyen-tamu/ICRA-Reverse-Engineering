from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..link import link_holding_time_s
from ..node import Node
from ..utils import clamp, mean


def _valid_neighbor_ids(node: Node, nodes: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in nodes and nodes[j].e_j > 0]


def velocity_distance(node_i: Node, node_j: Node) -> float:
    dtheta = abs(node_i.heading_rad - node_j.heading_rad)
    dtheta = min(dtheta, 2.0 * math.pi - dtheta)
    return math.sqrt(
        (node_i.speed_m_s - node_j.speed_m_s) ** 2
        + dtheta ** 2
    )


def velocity_similarity(node_i: Node, node_j: Node) -> float:
    d_ij = velocity_distance(node_i, node_j)
    return 1.0 / (1.0 + d_ij)


def mobility_stability_factor(node: Node, nodes: Dict[int, Node]) -> float:
    nbrs = _valid_neighbor_ids(node, nodes)
    if not nbrs:
        return 0.0

    sims = [velocity_similarity(node, nodes[j]) for j in nbrs]
    sims_sorted = sorted(sims)

    p25 = sims_sorted[max(0, len(sims_sorted) // 4)]
    p50 = sims_sorted[len(sims_sorted) // 2]
    worst = sims_sorted[0]

    score = 0.45 * mean(sims) + 0.30 * p50 + 0.15 * p25 + 0.10 * worst
    return clamp(score, 0.0, 1.0)


def link_stability_factor(
    node: Node,
    nodes: Dict[int, Node],
    comm_radius_m: float,
    lht_cap_s: float,
) -> float:
    nbrs = _valid_neighbor_ids(node, nodes)
    if not nbrs:
        return 0.0

    lhts = [link_holding_time_s(node, nodes[j], comm_radius_m) for j in nbrs]
    stable = [min(x, lht_cap_s) / max(1e-9, lht_cap_s) for x in lhts]
    stable_sorted = sorted(stable)

    p25 = stable_sorted[max(0, len(stable_sorted) // 4)]
    p50 = stable_sorted[len(stable_sorted) // 2]
    worst = stable_sorted[0]

    score = 0.40 * mean(stable) + 0.25 * p50 + 0.20 * p25 + 0.15 * worst
    return clamp(score, 0.0, 1.0)


def degree_centrality_factor(node: Node, nodes: Dict[int, Node]) -> float:
    nbrs = _valid_neighbor_ids(node, nodes)
    deg = len(nbrs)
    if deg <= 0:
        return 0.0

    alive_nodes = [n for n in nodes.values() if n.e_j > 0]
    if not alive_nodes:
        return 0.0

    max_deg = max(len(_valid_neighbor_ids(n, nodes)) for n in alive_nodes)
    global_deg = deg / max(1, len(alive_nodes) - 1)
    local_deg = deg / max(1, max_deg)

    return clamp(0.40 * global_deg + 0.60 * local_deg, 0.0, 1.0)


def connectivity_support_factor(
    node: Node,
    nodes: Dict[int, Node],
    comm_radius_m: float,
    lht_cap_s: float,
) -> float:
    nbrs = _valid_neighbor_ids(node, nodes)
    if not nbrs:
        return 0.0

    supports: List[float] = []
    for j in nbrs:
        other = nodes[j]
        if other.e_j <= 0:
            continue
        lht = link_holding_time_s(node, other, comm_radius_m)
        lht_norm = min(lht, lht_cap_s) / max(1e-9, lht_cap_s)
        energy = clamp(other.e_j / max(1e-9, other.e0_j), 0.0, 1.0)
        supports.append(0.65 * lht_norm + 0.35 * energy)

    if not supports:
        return 0.0

    supports.sort(reverse=True)
    best = supports[0]
    second = supports[1] if len(supports) > 1 else supports[0]
    mean_support = mean(supports)

    return clamp(0.30 * best + 0.25 * second + 0.45 * mean_support, 0.0, 1.0)


@dataclass(frozen=True)
class UtilityFactors:
    s1_energy: float
    s2_degree: float
    s3_vel_sim: float
    s4_lht: float


def compute_factors(
    node: Node,
    nodes: Dict[int, Node],
    comm_radius_m: float,
    n_total: int,
    lht_cap_s: float,
    v_max: float,
) -> UtilityFactors:
    s1 = clamp(node.e_j / max(1e-9, node.e0_j), 0.0, 1.0)

    base_degree = degree_centrality_factor(node, nodes)
    connectivity = connectivity_support_factor(node, nodes, comm_radius_m, lht_cap_s)
    s2 = clamp(0.78 * base_degree + 0.22 * connectivity, 0.0, 1.0)

    s3 = mobility_stability_factor(node, nodes)
    s4 = link_stability_factor(node, nodes, comm_radius_m, lht_cap_s)

    return UtilityFactors(s1, s2, s3, s4)


def weighted_utility(
    factors: UtilityFactors,
    weights: Tuple[float, float, float, float],
) -> float:
    w1, w2, w3, w4 = weights
    return (
        w1 * factors.s1_energy
        + w2 * factors.s2_degree
        + w3 * factors.s3_vel_sim
        + w4 * factors.s4_lht
    )