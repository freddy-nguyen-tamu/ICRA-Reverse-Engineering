from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from ..node import Node
from ..utils import clamp

Action = Tuple[float, float, float, float]
State = Tuple[float, float, float, float]


def _quantize_tenth(x: float) -> float:
    return round(clamp(round(x * 10.0) / 10.0, 0.0, 1.0), 1)


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else default


def _std(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / len(vals)
    return var ** 0.5


def _role_mix_score(nodes: List[Node]) -> float:
    if not nodes:
        return 0.0
    counts = Counter(n.role.value for n in nodes)
    total = len(nodes)
    probs = [c / total for c in counts.values()]
    if len(probs) <= 1:
        return 0.0
    import math
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return clamp(h / math.log(3.0), 0.0, 1.0)


def network_state(nodes: Dict[int, Node]) -> State:
    alive = [n for n in nodes.values() if n.e_j > 0]
    if not alive:
        return (0.0, 0.0, 0.0, 0.0)

    energy_ratios = [n.s1 for n in alive]
    lhts = [n.s4 for n in alive]
    vels = [n.s3 for n in alive]
    recent_switch = [_quantize_tenth(clamp(getattr(n, "recent_role_switches", 0.0), 0.0, 1.0)) for n in alive]

    avg_energy = clamp(_safe_mean(energy_ratios), 0.0, 1.0)
    energy_balance = clamp(1.0 - min(1.0, 2.5 * _std(energy_ratios)), 0.0, 1.0)
    topo_stability = clamp(
        0.55 * (1.0 - _safe_mean(recent_switch, 0.0))
        + 0.25 * _safe_mean(lhts, 0.0)
        + 0.20 * _safe_mean(vels, 0.0),
        0.0,
        1.0,
    )
    role_mix = _role_mix_score(alive)

    return (
        _quantize_tenth(avg_energy),
        _quantize_tenth(energy_balance),
        _quantize_tenth(topo_stability),
        _quantize_tenth(role_mix),
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


def _snap_to_simplex(raw: Tuple[float, float, float, float], step: float = 0.05) -> Action:
    vals = [max(0.0, x) for x in raw]
    total = sum(vals)
    if total <= 0:
        return (0.25, 0.25, 0.25, 0.25)

    vals = [x / total for x in vals]
    snapped = [round(x / step) * step for x in vals]
    total2 = sum(snapped)

    if total2 <= 0:
        return (0.25, 0.25, 0.25, 0.25)

    snapped = [x / total2 for x in snapped]
    snapped = [round(x, 10) for x in snapped]

    diff = round(1.0 - sum(snapped), 10)
    if abs(diff) > 1e-12:
        idx = max(range(4), key=lambda i: snapped[i])
        snapped[idx] = round(snapped[idx] + diff, 10)

    return (snapped[0], snapped[1], snapped[2], snapped[3])


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
    return _snap_to_simplex(smoothed, step=0.05)


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
        self.default_action: Action = (0.30, 0.15, 0.20, 0.35)

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
            a
            for a in self.actions
            if self._action_distance(a, self.last_action) <= self.allow_action_jump_l1 + 1e-12
        ]
        return eligible if eligible else self.actions

    def _prior_score(self, a: Action) -> float:
        w1, w2, w3, w4 = a

        score = 0.0
        score += 0.035 * min(w1, 0.40)
        score += 0.040 * min(w4, 0.40)
        score += 0.020 * min(w3, 0.30)
        score += 0.010 * min(w2, 0.25)

        if w1 >= 0.55:
            score -= 0.060
        if w4 < 0.15:
            score -= 0.045
        if w3 < 0.10:
            score -= 0.030
        if w2 > 0.35:
            score -= 0.015

        score -= 0.025 * abs(w1 - 0.30)
        score -= 0.020 * abs(w4 - 0.30)
        return score

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
                q = self.get_q(s, a) + self._prior_score(a)

                if self.last_action is not None and a == self.last_action:
                    q += self.stickiness_bonus

                if a == self.default_action:
                    q += 0.010

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