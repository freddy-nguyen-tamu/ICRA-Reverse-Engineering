from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..link import link_holding_time_s
from ..node import Node
from ..utils import clamp, mean, wrap_angle_rad


def _valid_neighbor_ids(node: Node, nodes: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in nodes and nodes[j].e_j > 0]


def velocity_similarity(node_i: Node, node_j: Node, v_max: float) -> float:
    """
    Pairwise mobility similarity in [0, 1].
    Higher is better.
    """
    dv = abs(node_i.speed_m_s - node_j.speed_m_s) / max(1e-9, v_max)
    dtheta = abs(wrap_angle_rad(node_i.heading_rad - node_j.heading_rad)) / math.pi
    d = math.sqrt(dv * dv + dtheta * dtheta)
    return 1.0 / (1.0 + d)


def mobility_stability_factor(node: Node, nodes: Dict[int, Node], v_max: float) -> float:
    """
    Stronger s3 than plain mean similarity.

    We reward:
    - high mean similarity to neighbors
    - low dispersion of similarity values
    - some support from speed smoothness

    Robust against stale/dead neighbor IDs.
    """
    nbrs = _valid_neighbor_ids(node, nodes)
    if not nbrs:
        return 0.0

    sims = [velocity_similarity(node, nodes[j], v_max=v_max) for j in nbrs]
    mu = mean(sims)
    var = mean([(x - mu) ** 2 for x in sims])
    dispersion_penalty = 1.0 - clamp(math.sqrt(var), 0.0, 1.0)

    avg_hist_speed = node.avg_speed.update(node.speed_m_s)
    speed_dev = abs(node.speed_m_s - avg_hist_speed) / max(1e-9, v_max)
    temporal_smoothness = 1.0 - clamp(speed_dev, 0.0, 1.0)

    s3 = 0.60 * mu + 0.25 * dispersion_penalty + 0.15 * temporal_smoothness
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
    v_max: float,
    degree_soft_target: float = 0.22,
) -> UtilityFactors:
    # s1: residual energy ratio
    s1 = clamp(node.e_j / max(1e-9, node.e0_j), 0.0, 1.0)

    # count only valid/alive neighbors
    nbrs = _valid_neighbor_ids(node, nodes)
    deg = len(nbrs)

    # s2: softened degree desirability
    raw_degree = deg / max(1, (n_total - 1))
    s2 = 1.0 - abs(raw_degree - degree_soft_target) / max(degree_soft_target, 1e-9)
    s2 = clamp(s2, 0.0, 1.0)

    # s3: mobility stability
    s3 = mobility_stability_factor(node, nodes, v_max=v_max)

    # s4: average link holding time over valid neighbors only
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