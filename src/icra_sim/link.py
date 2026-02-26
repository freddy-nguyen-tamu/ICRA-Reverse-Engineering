from __future__ import annotations

import math
from typing import Tuple

from .node import Node
from .utils import euclidean


def link_holding_time_s(node_i: Node, node_j: Node, comm_radius_m: float) -> float:
    ''' LET / link expiration time computing in seconds
    Responding to Eq.(13)-(18) in og article
    Nodes out of range => 0
    Relative speed ~0 within range => Larger num
    '''
    # If already disconnected
    if euclidean(node_i.pos(), node_j.pos()) > comm_radius_m:
        return 0.0

    Xi, Yi = node_i.x_m, node_i.y_m
    Xj, Yj = node_j.x_m, node_j.y_m
    vi, thetai = node_i.speed_m_s, node_i.heading_rad
    vj, thetaj = node_j.speed_m_s, node_j.heading_rad

    a = vi * math.cos(thetai) - vj * math.cos(thetaj)
    b = Xi - Xj
    c = vi * math.sin(thetai) - vj * math.sin(thetaj)
    d = Yi - Yj

    denom = a * a + c * c
    if denom < 1e-9:
        # Relative motion is negligible; treat as stable link.
        return 1e9

    disc = (a * a + c * c) * (comm_radius_m ** 2) - (a * d - b * c) ** 2
    if disc < 0:
        return 0.0

    t = (-(a * b + c * d) + math.sqrt(disc)) / denom
    return max(0.0, t)