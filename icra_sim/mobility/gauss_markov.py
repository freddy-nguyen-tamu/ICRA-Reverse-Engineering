from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

from ..node import Node
from ..utils import clamp, wrap_angle_rad


@dataclass
class GaussMarkovMobility:
    """Gauss-Markov mobility model
    discrete-time because lightweight simulator
    """
    alpha: float
    speed_range: Tuple[float, float]  # (min,max) in m/s
    area_m: Tuple[float, float]       # (width,height) in meters
    speed_noise_std: float = 1.0
    heading_noise_std: float = 0.05

    def step(self, node: Node, dt_s: float) -> None:
        vmin, vmax = self.speed_range
        width_m, height_m = self.area_m

        v_mean = (vmin + vmax) / 2.0

        # speed update
        noise_v = random.gauss(0.0, self.speed_noise_std)
        v_new = (
            self.alpha * node.speed_m_s
            + (1.0 - self.alpha) * v_mean
            + math.sqrt(max(0.0, 1.0 - self.alpha ** 2)) * noise_v
        )
        v_new = clamp(v_new, vmin, vmax)

        # heading update
        noise_h = random.gauss(0.0, self.heading_noise_std)
        heading_new = node.heading_rad + math.sqrt(max(0.0, 1.0 - self.alpha ** 2)) * noise_h
        heading_new = wrap_angle_rad(heading_new)

        # integrate position
        x_new = node.x_m + v_new * math.cos(heading_new) * dt_s
        y_new = node.y_m + v_new * math.sin(heading_new) * dt_s

        # reflect at boundaries
        bounced = False
        if x_new < 0:
            x_new = -x_new
            heading_new = math.pi - heading_new
            bounced = True
        elif x_new > width_m:
            x_new = 2 * width_m - x_new
            heading_new = math.pi - heading_new
            bounced = True

        if y_new < 0:
            y_new = -y_new
            heading_new = -heading_new
            bounced = True
        elif y_new > height_m:
            y_new = 2 * height_m - y_new
            heading_new = -heading_new
            bounced = True

        if bounced:
            heading_new = wrap_angle_rad(heading_new)

        node.speed_m_s = v_new
        node.heading_rad = heading_new
        node.x_m = clamp(x_new, 0.0, width_m)
        node.y_m = clamp(y_new, 0.0, height_m)
        node.avg_speed.update(v_new)