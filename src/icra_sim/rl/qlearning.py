from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Tuple

from ..node import Node
from ..utils import clamp

Action = Tuple[float, float, float, float]
State = Tuple[float, float, float, float]


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else default


def _quantize_to_bin(x: float, bin_width: float = 0.10) -> float:
    x = clamp(x, 0.0, 1.0)
    return round(round(x / bin_width) * bin_width, 1)


def _factor_entropy(values: List[float], bin_width: float = 0.10) -> float:
    if not values:
        return 0.0

    bucket_count = int(round(1.0 / bin_width)) + 1
    buckets = [0 for _ in range(bucket_count)]
    for value in values:
        idx = int(round(clamp(value, 0.0, 1.0) / bin_width))
        idx = max(0, min(bucket_count - 1, idx))
        buckets[idx] += 1

    total = len(values)
    probs = [count / total for count in buckets if count > 0]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(bucket_count)
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0
    return _quantize_to_bin(normalized, bin_width)


def network_state(nodes: Dict[int, Node], bin_width: float = 0.10) -> State:
    """
    Paper Section III-D: the network state is formed from the entropy of the
    distributions of the four utility factors across all alive nodes, discretized
    onto 0.1 bins.
    """
    alive = [n for n in nodes.values() if n.e_j > 0]
    if not alive:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        _factor_entropy([n.s1 for n in alive], bin_width),
        _factor_entropy([n.s2 for n in alive], bin_width),
        _factor_entropy([n.s3 for n in alive], bin_width),
        _factor_entropy([n.s4 for n in alive], bin_width),
    )


def generate_action_space(step: float = 0.05) -> List[Action]:
    step_units = int(round(1.0 / step))
    actions: List[Action] = []
    for a in range(step_units + 1):
        for b in range(step_units + 1 - a):
            for c in range(step_units + 1 - a - b):
                d = step_units - a - b - c
                actions.append(
                    (
                        round(a * step, 10),
                        round(b * step, 10),
                        round(c * step, 10),
                        round(d * step, 10),
                    )
                )
    actions.sort()
    return actions


def reward_transform(r: float) -> float:
    """
    Paper Eq. (22) uses a bounded nonlinear transform for r in [-1, 1].
    The exact typeset formula is hard to parse from the PDF text extraction,
    but it is a symmetric saturation around +/-1. tanh(1.5 r) matches that role.
    """
    if r >= 1.0:
        return 1.0
    if r <= -1.0:
        return -1.0
    return math.tanh(1.5 * r)


class QLearningStrategy:
    def __init__(
        self,
        actions: List[Action],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        default_action: Action = (0.25, 0.25, 0.25, 0.25),
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.default_action = default_action
        self.q: Dict[Tuple[State, Action], float] = {}

        # Algorithm 2 initialization: Q[s, equal-weights] = 1.
        for s1 in [round(i * 0.1, 1) for i in range(11)]:
            for s2 in [round(i * 0.1, 1) for i in range(11)]:
                for s3 in [round(i * 0.1, 1) for i in range(11)]:
                    for s4 in [round(i * 0.1, 1) for i in range(11)]:
                        self.q[((s1, s2, s3, s4), default_action)] = 1.0

    def get_q(self, s: State, a: Action) -> float:
        return self.q.get((s, a), 0.0)

    def set_q(self, s: State, a: Action, v: float) -> None:
        self.q[(s, a)] = v

    def best_action_value(self, s: State) -> float:
        return max(self.get_q(s, a) for a in self.actions)

    def select_action(self, s: State) -> Action:
        # Practical epsilon-greedy exploration. The paper does not spell out the
        # exploration rule, so this is the smallest necessary implementation detail.
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        best_q = self.best_action_value(s)
        best_actions = [a for a in self.actions if abs(self.get_q(s, a) - best_q) < 1e-12]
        return random.choice(best_actions)

    def update(self, s: State, a: Action, reward: float, s_next: State) -> None:
        old = self.get_q(s, a)
        target = reward + self.gamma * self.best_action_value(s_next)
        new = self.alpha * target + (1.0 - self.alpha) * old
        self.set_q(s, a, new)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
