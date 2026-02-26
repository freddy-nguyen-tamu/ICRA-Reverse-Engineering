from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..link import link_holding_time_s
from ..node import Node
from ..utils import clamp, euclidean, mean, wrap_angle_rad


def velocity_similarity(node_i: Node, node_j: Node, v_max: float) -> float:
    """Larger = bigger similarity  in [0,1]
    Normalized distance d = sqrt( (Δv/v_max)^2 + (Δθ/pi)^2 ),
    map to similarity with 1/(1+d) (Eq.(10))
    """
    dv = abs(node_i.speed_m_s - node_j.speed_m_s) / max(1e-9, v_max)
    dtheta = abs(wrap_angle_rad(node_i.heading_rad - node_j.heading_rad)) / math.pi
    d = math.sqrt(dv * dv + dtheta * dtheta)
    return 1.0 / (1.0 + d)


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
    # s1: residual energy ratio
    s1 = clamp(node.e_j / max(1e-9, node.e0_j), 0.0, 1.0)

    # s2: degree centrality in [0,1]
    deg = len(node.neighbors)
    s2 = deg / max(1, (n_total - 1))

    # s3: velocity similarity with neighbors (mean similarity) in [0,1]
    if deg == 0:
        s3 = 0.0
    else:
        sims = [velocity_similarity(node, nodes[j], v_max=v_max) for j in node.neighbors]
        s3 = clamp(mean(sims), 0.0, 1.0)

    # s4: average link holding time with neighbors normalized to [0,1]
    if deg == 0:
        s4 = 0.0
    else:
        lhts = [link_holding_time_s(node, nodes[j], comm_radius_m) for j in node.neighbors]
        s4 = clamp(mean(lhts) / max(1e-9, lht_cap_s), 0.0, 1.0)

    return UtilityFactors(s1, s2, s3, s4)


def weighted_utility(factors: UtilityFactors, weights: Tuple[float, float, float, float]) -> float:
    w1, w2, w3, w4 = weights
    return w1 * factors.s1_energy + w2 * factors.s2_degree + w3 * factors.s3_vel_sim + w4 * factors.s4_lht