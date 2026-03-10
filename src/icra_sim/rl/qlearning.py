from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..node import Node
from ..utils import clamp, safe_log

State = Tuple[float, float, float, float]
Action = Tuple[float, float, float, float]


def generate_action_space(step: float = 0.10) -> List[Action]:
    """
    Generate all 4-tuples of weights in increments of `step` that sum to 1.
    """
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
    State derived from factor dispersion / entropy.
    """
    Hmax = math.log(11.0)
    alive = [n for n in nodes.values() if n.e_j > 0]
    if not alive:
        return (0.0, 0.0, 0.0, 0.0)

    s1_vals = [n.s1 for n in alive]
    s2_vals = [n.s2 for n in alive]
    s3_vals = [n.s3 for n in alive]
    s4_vals = [n.s4 for n in alive]

    def norm_round(H: float) -> float:
        x = 0.0 if Hmax <= 0 else clamp(H / Hmax, 0.0, 1.0)
        return round(x, 1)

    return (
        norm_round(entropy_from_values(s1_vals)),
        norm_round(entropy_from_values(s2_vals)),
        norm_round(entropy_from_values(s3_vals)),
        norm_round(entropy_from_values(s4_vals)),
    )


def reward_transform(r: float) -> float:
    """
    Smooth monotonic transform from [-1, 1] to roughly [-1, 1].
    """
    r = clamp(r, -1.0, 1.0)
    return math.tanh(2.5 * r)


@dataclass
class QLearningStrategy:
    actions: List[Action]
    alpha: float = 0.35
    gamma: float = 0.80
    default_action: Action = (0.25, 0.25, 0.25, 0.25)

    epsilon: float = 0.18
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.992

    optimistic_init: float = 0.15
    Q: Dict[State, Dict[Action, float]] = field(default_factory=dict)

    def get_q(self, s: State, a: Action) -> float:
        if s not in self.Q:
            self.Q[s] = {}
        return self.Q[s].get(a, self.optimistic_init)

    def set_q(self, s: State, a: Action, v: float) -> None:
        self.Q.setdefault(s, {})[a] = v

    def best_action_value(self, s: State) -> float:
        if not self.actions:
            return 0.0
        return max(self.get_q(s, a) for a in self.actions)

    def select_action(self, s: State) -> Action:
        if not self.actions:
            return self.default_action

        if random.random() < self.epsilon:
            # biased exploration: favor non-uniform actions a bit
            non_uniform = [a for a in self.actions if a != self.default_action]
            pool = non_uniform if non_uniform and random.random() < 0.70 else self.actions
            return random.choice(pool)

        best_val = -1e18
        best_actions: List[Action] = []
        for a in self.actions:
            q = self.get_q(s, a)
            if q > best_val + 1e-12:
                best_val = q
                best_actions = [a]
            elif abs(q - best_val) < 1e-12:
                best_actions.append(a)

        if not best_actions:
            return self.default_action

        # prefer more decisive actions among ties
        def peakedness(a: Action) -> float:
            return max(a) - min(a)

        best_actions.sort(key=peakedness, reverse=True)
        top = [a for a in best_actions if abs(peakedness(a) - peakedness(best_actions[0])) < 1e-12]
        return random.choice(top)

    def update(self, s: State, a: Action, reward: float, s_next: State) -> None:
        old = self.get_q(s, a)
        target = reward + self.gamma * self.best_action_value(s_next)
        new = old + self.alpha * (target - old)
        self.set_q(s, a, new)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)