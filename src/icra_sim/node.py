from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .utils import RunningMean


class Role(str, Enum):
    CH = "CH"
    FORWARDER = "FORWARDER"
    MEMBER = "MEMBER"


@dataclass
class Node:
    node_id: int

    x_m: float
    y_m: float

    speed_m_s: float
    heading_rad: float

    e0_j: float
    e_j: float

    role: Role = Role.MEMBER
    cluster_head: Optional[int] = None
    is_forwarder: bool = False

    neighbors: List[int] = field(default_factory=list)

    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    s4: float = 0.0
    utility: float = 0.0

    role_change_count: int = 0

    avg_speed: RunningMean = field(default_factory=lambda: RunningMean(window=10))

    # Stability / hysteresis state
    ch_tenure_s: float = 0.0
    time_in_cluster_s: float = 0.0
    last_cluster_head: Optional[int] = None

    # Cached neighbor relationship quality for clustering / routing
    neighbor_lht: Dict[int, float] = field(default_factory=dict)
    neighbor_vel_sim: Dict[int, float] = field(default_factory=dict)

    # Optional role bookkeeping
    last_role: Role = Role.MEMBER

    def pos(self) -> Tuple[float, float]:
        return (self.x_m, self.y_m)

    def velocity_vec(self) -> Tuple[float, float]:
        return (
            self.speed_m_s * math.cos(self.heading_rad),
            self.speed_m_s * math.sin(self.heading_rad),
        )

    def set_role(self, new_role: Role) -> None:
        if new_role != self.role:
            self.role_change_count += 1
            self.last_role = self.role
            self.role = new_role
        if self.role == Role.CH:
            self.cluster_head = self.node_id

    def reset_clustering_flags(self) -> None:
        self.is_forwarder = False
        if self.role != Role.CH:
            self.set_role(Role.MEMBER)
            self.cluster_head = None
        else:
            self.cluster_head = self.node_id

    def note_cluster_membership(self, ch_id: Optional[int], dt_s: float) -> None:
        if ch_id == self.cluster_head:
            self.time_in_cluster_s += dt_s
        else:
            self.time_in_cluster_s = 0.0
            self.last_cluster_head = self.cluster_head
            self.cluster_head = ch_id

    def note_role_tenure(self, dt_s: float) -> None:
        if self.role == Role.CH:
            self.ch_tenure_s += dt_s
        else:
            self.ch_tenure_s = 0.0