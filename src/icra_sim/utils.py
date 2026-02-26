from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple


def set_seed(seed: int) -> None:
    random.seed(seed)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_angle_rad(theta: float) -> float:
    # Wrap to [-pi, pi]
    while theta <= -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi
    return theta


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0


@dataclass
class RunningMean:
    # mobility metric used by WCA baseline
    window: int
    values: List[float]

    def __init__(self, window: int = 10):
        self.window = window
        self.values = []

    def update(self, x: float) -> float:
        self.values.append(x)
        if len(self.values) > self.window:
            self.values.pop(0)
        return sum(self.values) / len(self.values)