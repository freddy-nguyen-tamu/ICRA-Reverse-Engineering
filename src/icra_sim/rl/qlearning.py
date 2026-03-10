from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..node import Node
from ..utils import clamp, safe_log

State = Tuple[float, float, float, float]
Action = Tuple[float, float, float, float]


def generate_action_space(step: float = 0.05) -> List[Action]:
    """Generate all 4-tuples of weights in increments of `step` that sum to 1."""
    denom = int(round(1.0 / step))
    total = denom
    actions: List[Action] = []
    for a in range(total + 1):
        for b in range(total - a + 1):
            for c in range(total - a - b + 1):
                d = total - a - b - c
                actions.append((a * step, b * step, c * step, d * step))
    return actions


def entropy_from_values(values: List[float]) -> float:
    """Entropy over values quantized to {0.0, 0.1, ..., 1.0}."""
    if not values:
        return 0.0
    bins = [0] * 11
    for v in values:
        q = int(round(clamp(v, 0.0, 1.0) * 10))
        bins[q] += 1
    n = len(values)
    H = 0.0
    for count in bins:
        if count == 0:
            continue
        p = count / n
        H -= p * safe_log(p)
    return H


def network_state(nodes: Dict[int, Node]) -> State:
    """
    Paper-like state:
    normalize factor entropy by log(11), then quantize to 0.1 resolution.
    """
    Hmax = math.log(11.0)
    alive = [n for n in nodes.values() if n.e_j > 0]

    s1_vals = [n.s1 for n in alive]
    s2_vals = [n.s2 for n in alive]
    s3_vals = [n.s3 for n in alive]
    s4_vals = [n.s4 for n in alive]

    def norm_round(H: float) -> float:
        x = 0.0 if Hmax <= 0 else clamp(H / Hmax, 0.0, 1.0)
        return round(x * 10.0) / 10.0

    return (
        norm_round(entropy_from_values(s1_vals)),
        norm_round(entropy_from_values(s2_vals)),
        norm_round(entropy_from_values(s3_vals)),
        norm_round(entropy_from_values(s4_vals)),
    )


def reward_transform(r: float) -> float:
    """Paper Eq.(22)-style transform."""
    if abs(r - 1.0) < 1e-12:
        return 1.0
    if abs(r + 1.0) < 1e-12:
        return -1.0
    return (1.0 - math.exp(-3.0 * r)) / (1.0 + math.exp(-3.0 * r))


@dataclass
class QLearningStrategy:
    actions: List[Action]
    alpha: float = 0.8
    gamma: float = 0.0
    default_action: Action = (0.25, 0.25, 0.25, 0.25)

    epsilon: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999

    Q: Dict[State, Dict[Action, float]] = field(default_factory=dict)

    def get_q(self, s: State, a: Action) -> float:
        if s not in self.Q:
            self.Q[s] = {}
        if a in self.Q[s]:
            return self.Q[s][a]
        return 1.0 if a == self.default_action else 0.0

    def set_q(self, s: State, a: Action, v: float) -> None:
        self.Q.setdefault(s, {})[a] = v

    def select_action(self, s: State) -> Action:
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        best_val = -1e18
        best_actions: List[Action] = []
        for a in self.actions:
            q = self.get_q(s, a)
            if q > best_val + 1e-12:
                best_val = q
                best_actions = [a]
            elif abs(q - best_val) < 1e-12:
                best_actions.append(a)

        return random.choice(best_actions) if best_actions else self.default_action

    def update(self, s: State, a: Action, reward: float) -> None:
        old = self.get_q(s, a)
        target = reward
        new = self.alpha * target + (1.0 - self.alpha) * old
        self.set_q(s, a, new)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)