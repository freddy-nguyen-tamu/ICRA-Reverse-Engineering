from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple

from ..node import Node
from ..utils import clamp, wrap_angle_rad


@dataclass
class GaussMarkovMobility:
    """Gauss-Markov mobility with a weak anchor pull.

    The original paper used a reference-point-group model together with
    Gauss-Markov motion. This lightweight simulator does not implement the full
    group model, but it can approximate the same effect by initializing nodes in
    groups and pulling each node weakly toward its group anchor while preserving
    Gauss-Markov local motion.
    """

    alpha: float
    speed_range: Tuple[float, float]  # (min,max) in m/s
    area_m: Tuple[float, float]       # (width,height) in meters
    speed_noise_std: float = 1.0
    heading_noise_std: float = 0.05
    anchor_pull: float = 0.0

    def step(self, node: Node, dt_s: float) -> None:
        vmin, vmax = self.speed_range
        width_m, height_m = self.area_m

        v_mean = (vmin + vmax) / 2.0

        noise_v = random.gauss(0.0, self.speed_noise_std)
        v_new = (
            self.alpha * node.speed_m_s
            + (1.0 - self.alpha) * v_mean
            + math.sqrt(max(0.0, 1.0 - self.alpha ** 2)) * noise_v
        )
        v_new = clamp(v_new, vmin, vmax)

        noise_h = random.gauss(0.0, self.heading_noise_std)
        heading_new = node.heading_rad + math.sqrt(max(0.0, 1.0 - self.alpha ** 2)) * noise_h
        heading_new = wrap_angle_rad(heading_new)

        x_new = node.x_m + v_new * math.cos(heading_new) * dt_s
        y_new = node.y_m + v_new * math.sin(heading_new) * dt_s

        anchor_x = getattr(node, "anchor_x_m", None)
        anchor_y = getattr(node, "anchor_y_m", None)
        node_pull = float(getattr(node, "anchor_pull", self.anchor_pull))
        if anchor_x is not None and anchor_y is not None and node_pull > 0.0:
            x_new = (1.0 - node_pull) * x_new + node_pull * float(anchor_x)
            y_new = (1.0 - node_pull) * y_new + node_pull * float(anchor_y)

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
