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
        state_floor: float = 0.02,
        state_gamma: float = 1.15,
        sample_scale: float = 8.0,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.default_action = default_action
        self.state_floor = state_floor
        self.state_gamma = state_gamma
        self.sample_scale = sample_scale
        self.context_target: Optional[Action] = None
        self.q: Dict[Tuple[State, Action], float] = {}
        self.state_updates: Dict[State, int] = {}
        self.action_updates: Dict[Tuple[State, Action], int] = {}

        # Algorithm 2 initialization: Q[s, equal-weights] = 1.
        for s1 in [round(i * 0.1, 1) for i in range(11)]:
            for s2 in [round(i * 0.1, 1) for i in range(11)]:
                for s3 in [round(i * 0.1, 1) for i in range(11)]:
                    for s4 in [round(i * 0.1, 1) for i in range(11)]:
                        self.q[((s1, s2, s3, s4), default_action)] = 1.0

    def set_context_target(self, target: Optional[Action]) -> None:
        self.context_target = target

    def get_q(self, s: State, a: Action) -> float:
        return self.q.get((s, a), 0.0)

    def set_q(self, s: State, a: Action, v: float) -> None:
        self.q[(s, a)] = v

    def _has_only_default_prior(self, s: State) -> bool:
        known_actions = [a for a in self.actions if (s, a) in self.q]
        return (
            len(known_actions) <= 1
            and abs(self.get_q(s, self.default_action) - 1.0) < 1e-12
            and self.action_updates.get((s, self.default_action), 0) == 0
        )

    def _effective_q(self, s: State, a: Action) -> float:
        q = self.get_q(s, a)
        if a == self.default_action and self.action_updates.get((s, a), 0) == 0:
            # Preserve the paper-inspired equal-weight initialization as a
            # prior, but do not let it permanently dominate greedy selection
            # before the state has received any real feedback.
            return 0.0
        return q

    def best_action_value(self, s: State) -> float:
        return max(self._effective_q(s, a) for a in self.actions)

    def _state_target_action(self, s: State) -> Action:
        if max(s) <= 0.0 and self.context_target is None:
            return self.default_action

        vals = [max(self.state_floor, x) ** self.state_gamma for x in s]
        total = sum(vals)
        if total > 0.0:
            target = [v / total for v in vals]
        else:
            target = list(self.default_action)

        # When the quantized entropy state is nearly uniform, use a weak
        # scenario/context prior only as a disambiguation hint. This avoids the
        # old hard anchors while still reflecting the paper's scenario logic.
        if self.context_target is not None:
            mix = 0.60 if (max(s) - min(s) < 0.05) else 0.25
            target = [(1.0 - mix) * target[i] + mix * self.context_target[i] for i in range(4)]
            tsum = sum(target)
            if tsum > 0.0:
                target = [x / tsum for x in target]

        target_t = tuple(target)
        return min(
            self.actions,
            key=lambda a: (
                sum(abs(a[i] - target_t[i]) for i in range(4)),
                -sum(a[i] * target_t[i] for i in range(4)),
                a,
            ),
        )

    def _sample_guided_action(self, s: State) -> Action:
        target = self._state_target_action(s)
        weights = []
        for action in self.actions:
            l1 = sum(abs(action[i] - target[i]) for i in range(4))
            weights.append(math.exp(-self.sample_scale * l1))
        return random.choices(self.actions, weights=weights, k=1)[0]

    def select_action(self, s: State) -> Action:
        # The paper states that factors with wider value distributions deserve
        # more attention. Before enough feedback has been collected for a state,
        # use the entropy-state vector itself to choose the nearest simplex
        # action instead of getting stuck forever at equal weights.
        if random.random() < self.epsilon:
            return self._sample_guided_action(s)

        if self._has_only_default_prior(s):
            return self._sample_guided_action(s) if random.random() < 0.35 else self._state_target_action(s)

        best_q = self.best_action_value(s)
        best_actions = [a for a in self.actions if abs(self._effective_q(s, a) - best_q) < 1e-12]
        if len(best_actions) == 1:
            return best_actions[0]

        target = self._state_target_action(s)
        return min(
            best_actions,
            key=lambda a: (
                sum(abs(a[i] - target[i]) for i in range(4)),
                -sum(a[i] * target[i] for i in range(4)),
                a,
            ),
        )

    def update(self, s: State, a: Action, reward: float, s_next: State) -> None:
        old = self._effective_q(s, a)
        target = reward + self.gamma * self.best_action_value(s_next)
        new = self.alpha * target + (1.0 - self.alpha) * old
        self.set_q(s, a, new)
        self.state_updates[s] = self.state_updates.get(s, 0) + 1
        self.action_updates[(s, a)] = self.action_updates.get((s, a), 0) + 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)




