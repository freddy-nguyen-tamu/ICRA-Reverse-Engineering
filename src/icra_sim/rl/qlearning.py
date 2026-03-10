from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from ..node import Node
from ..utils import clamp

Action = Tuple[float, float, float, float]
State = Tuple[float, float, float, float]


def _quantize_tenth(x: float) -> float:
    return round(clamp(round(x * 10.0) / 10.0, 0.0, 1.0), 1)


def _entropy_from_values(values: Iterable[float]) -> float:
    vals = [_quantize_tenth(v) for v in values]
    if not vals:
        return 0.0
    c = Counter(vals)
    total = len(vals)
    h = 0.0
    for count in c.values():
        p = count / total
        if p > 0:
            h -= p * math.log(p)
    return _quantize_tenth(min(1.0, h))


def network_state(nodes: Dict[int, Node]) -> State:
    alive = [n for n in nodes.values() if n.e_j > 0]
    if not alive:
        return (0.0, 0.0, 0.0, 0.0)

    return (
        _entropy_from_values(n.s1 for n in alive),
        _entropy_from_values(n.s2 for n in alive),
        _entropy_from_values(n.s3 for n in alive),
        _entropy_from_values(n.s4 for n in alive),
    )


def generate_action_space(step: float = 0.05) -> List[Action]:
    step_units = int(round(1.0 / step))
    actions: List[Action] = []
    for a in range(step_units + 1):
        for b in range(step_units + 1 - a):
            for c in range(step_units + 1 - a - b):
                d = step_units - a - b - c
                action = (
                    round(a * step, 10),
                    round(b * step, 10),
                    round(c * step, 10),
                    round(d * step, 10),
                )
                actions.append(action)
    actions.sort()
    return actions


def reward_transform(r: float) -> float:
    return clamp(r, -1.0, 1.0)


def smooth_action(
    prev_action: Optional[Action],
    raw_action: Action,
    beta: float,
) -> Action:
    if prev_action is None:
        return raw_action

    smoothed = tuple(
        beta * prev_action[i] + (1.0 - beta) * raw_action[i]
        for i in range(4)
    )
    total = sum(smoothed)
    if total <= 0:
        return (0.25, 0.25, 0.25, 0.25)

    normalized = tuple(x / total for x in smoothed)

    snapped = tuple(round(x / 0.05) * 0.05 for x in normalized)
    total2 = sum(snapped)
    if total2 <= 0:
        return (0.25, 0.25, 0.25, 0.25)

    fixed = [x / total2 for x in snapped]
    fixed = [round(x, 10) for x in fixed]

    diff = round(1.0 - sum(fixed), 10)
    if abs(diff) > 1e-12:
        idx = max(range(4), key=lambda i: fixed[i])
        fixed[idx] = round(fixed[idx] + diff, 10)

    return (fixed[0], fixed[1], fixed[2], fixed[3])


class QLearningStrategy:
    def __init__(
        self,
        actions: List[Action],
        alpha: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        stickiness_bonus: float = 0.08,
        min_action_hold_rounds: int = 6,
        allow_action_jump_l1: float = 0.40,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.stickiness_bonus = stickiness_bonus
        self.min_action_hold_rounds = min_action_hold_rounds
        self.allow_action_jump_l1 = allow_action_jump_l1

        self.q: Dict[Tuple[State, Action], float] = {}
        self.default_action: Action = (0.25, 0.25, 0.25, 0.25)

        self.last_action: Optional[Action] = None
        self.last_action_rounds: int = 0

    def get_q(self, s: State, a: Action) -> float:
        return self.q.get((s, a), 0.0)

    def set_q(self, s: State, a: Action, v: float) -> None:
        self.q[(s, a)] = v

    def best_action_value(self, s: State) -> float:
        return max(self.get_q(s, a) for a in self.actions)

    def _action_distance(self, a: Action, b: Action) -> float:
        return sum(abs(a[i] - b[i]) for i in range(4))

    def _eligible_actions(self) -> List[Action]:
        if self.last_action is None:
            return self.actions

        eligible = [
            a for a in self.actions
            if self._action_distance(a, self.last_action) <= self.allow_action_jump_l1 + 1e-12
        ]
        return eligible if eligible else self.actions

    def select_action(self, s: State) -> Action:
        if self.last_action is not None and self.last_action_rounds < self.min_action_hold_rounds:
            self.last_action_rounds += 1
            return self.last_action

        candidates = self._eligible_actions()

        if random.random() < self.epsilon:
            chosen = random.choice(candidates)
        else:
            scored: List[Tuple[float, Action]] = []
            for a in candidates:
                q = self.get_q(s, a)
                if self.last_action is not None and a == self.last_action:
                    q += self.stickiness_bonus
                if a == self.default_action:
                    q += 0.01
                scored.append((q, a))

            best_q = max(q for q, _ in scored)
            best_actions = [a for q, a in scored if abs(q - best_q) < 1e-12]
            chosen = random.choice(best_actions)

        if chosen == self.last_action:
            self.last_action_rounds += 1
        else:
            self.last_action = chosen
            self.last_action_rounds = 1

        return chosen

    def update(self, s: State, a: Action, reward: float, s_next: State) -> None:
        old = self.get_q(s, a)
        target = reward + self.gamma * self.best_action_value(s_next)
        new = old + self.alpha * (target - old)
        self.set_q(s, a, new)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)