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
    """
    Paper Eq. 9:
    d_ij = sqrt((v_i - v_j)^2 + (theta_i - theta_j)^2)
    """
    return math.sqrt(
        (node_i.speed_m_s - node_j.speed_m_s) ** 2
        + (node_i.heading_rad - node_j.heading_rad) ** 2
    )


def velocity_similarity(node_i: Node, node_j: Node) -> float:
    """
    Paper Eq. 10:
    V_ij = 1 / (1 + d_ij)
    """
    d_ij = velocity_distance(node_i, node_j)
    return 1.0 / (1.0 + d_ij)


def mobility_stability_factor(node: Node, nodes: Dict[int, Node]) -> float:
    """
    Paper Eq. 11-12 exactly:
      Vbar = mean(V_ij)
      s3 = mean((V_ij - Vbar)^2)

    Note:
    This is intentionally the paper's formula, even though it behaves
    like a dispersion term rather than a "higher-is-better" similarity term.
    """
    nbrs = _valid_neighbor_ids(node, nodes)
    if not nbrs:
        return 0.0

    sims = [velocity_similarity(node, nodes[j]) for j in nbrs]
    vbar = mean(sims)
    s3 = mean((v - vbar) ** 2 for v in sims)
    return clamp(s3, 0.0, 1.0)


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
    v_max: float,  # kept for API compatibility
) -> UtilityFactors:
    # Eq. 7
    s1 = clamp(node.e_j / max(1e-9, node.e0_j), 0.0, 1.0)

    nbrs = _valid_neighbor_ids(node, nodes)
    deg = len(nbrs)

    # Eq. 8
    s2 = clamp(deg / max(1, (n_total - 1)), 0.0, 1.0)

    # Eq. 12
    s3 = mobility_stability_factor(node, nodes)

    # Eq. 14
    if not nbrs:
        s4 = 0.0
    else:
        lhts = [link_holding_time_s(node, nodes[j], comm_radius_m) for j in nbrs]
        s4 = clamp(mean(lhts) / max(1e-9, lht_cap_s), 0.0, 1.0)

    return UtilityFactors(s1, s2, s3, s4)


def weighted_utility(factors: UtilityFactors, weights: Tuple[float, float, float, float]) -> float:
    w1, w2, w3, w4 = weights
    return (
        w1 * factors.s1_energy
        + w2 * factors.s2_degree
        + w3 * factors.s3_vel_sim
        + w4 * factors.s4_lht
    )