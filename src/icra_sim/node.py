from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

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

    def pos(self) -> Tuple[float, float]:
        return (self.x_m, self.y_m)

    def velocity_vec(self) -> Tuple[float, float]:
        return (
            self.speed_m_s * math.cos(self.heading_rad),
            self.speed_m_s * math.sin(self.heading_rad),
        )

    def reset_clustering_flags(self) -> None:
        self.is_forwarder = False
        if self.role != Role.CH:
            self.role = Role.MEMBER
        if self.role == Role.CH:
            self.cluster_head = self.node_id
        else:
            self.cluster_head = None