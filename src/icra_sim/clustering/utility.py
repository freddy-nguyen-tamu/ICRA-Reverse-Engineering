from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..link import link_holding_time_s
from ..node import Node
from ..utils import clamp, mean


def _alive_neighbors(node: Node, nodes: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in nodes and nodes[j].e_j > 0]


def velocity_distance(node_i: Node, node_j: Node) -> float:
    """
    Paper Eq. (9) uses the Euclidean distance between the two velocity states.
    In this lightweight implementation, the velocity state is represented by
    (speed, heading).
    """
    dtheta = abs(node_i.heading_rad - node_j.heading_rad)
    dtheta = min(dtheta, 2.0 * math.pi - dtheta)
    return math.sqrt((node_i.speed_m_s - node_j.speed_m_s) ** 2 + dtheta ** 2)


def velocity_similarity(node_i: Node, node_j: Node) -> float:
    # Eq. (10)
    return 1.0 / (1.0 + velocity_distance(node_i, node_j))


def residual_energy_factor(node: Node) -> float:
    # Eq. (7)
    return clamp(node.e_j / max(1e-9, node.e0_j), 0.0, 1.0)


def degree_centrality_factor(node: Node, nodes: Dict[int, Node], n_total: int) -> float:
    # Eq. (8)
    deg = len(_alive_neighbors(node, nodes))
    return clamp(deg / max(1, n_total - 1), 0.0, 1.0)


def velocity_similarity_factor(node: Node, nodes: Dict[int, Node]) -> float:
    """
    The paper text calls s3 a velocity-similarity factor, but the printed Eq. (12)
    is a variance-like dispersion term. To preserve the paper's stated intent
    (higher utility for more similar motion), we compute the Eq. (10)/(11)
    similarities and convert the Eq. (12) dispersion into a monotone stability
    score in [0, 1].
    """
    nbrs = _alive_neighbors(node, nodes)
    if not nbrs:
        return 0.0

    sims = [velocity_similarity(node, nodes[j]) for j in nbrs]
    avg_sim = mean(sims)  # Eq. (11)
    dispersion = mean((sim - avg_sim) ** 2 for sim in sims)  # Eq. (12) core term
    score = avg_sim * (1.0 - min(1.0, dispersion))
    return clamp(score, 0.0, 1.0)


def link_holding_time_factor(
    node: Node,
    nodes: Dict[int, Node],
    comm_radius_m: float,
    lht_cap_s: float,
) -> float:
    """
    Paper Eq. (14) is the average link holding time to neighbors. Because this
    simulator combines factors in a bounded weighted utility and the RL state is
    discretized on [0, 1], we retain the paper's averaging structure but cap the
    raw LHT before normalization.
    """
    nbrs = _alive_neighbors(node, nodes)
    if not nbrs:
        return 0.0

    lhts = [link_holding_time_s(node, nodes[j], comm_radius_m) for j in nbrs]
    avg_lht = mean(min(x, lht_cap_s) for x in lhts)
    return clamp(avg_lht / max(1e-9, lht_cap_s), 0.0, 1.0)


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
    del v_max  # kept for backward-compatible signature

    s1 = residual_energy_factor(node)
    s2 = degree_centrality_factor(node, nodes, n_total)
    s3 = velocity_similarity_factor(node, nodes)
    s4 = link_holding_time_factor(node, nodes, comm_radius_m, lht_cap_s)
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
