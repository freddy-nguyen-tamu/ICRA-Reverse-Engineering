from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from .utility import compute_factors, velocity_similarity, weighted_utility


def _alive_neighbors(node: Node, alive: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in alive and alive[j].e_j > 0]


def _safe_attr(node: Node, name: str, default: float = 0.0) -> float:
    value = getattr(node, name, default)
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class ClusterResult:
    clusters: Dict[int, List[int]]
    forwarders: Set[int]


class ICRAClusterer:
    """Paper-core ICRA plus lightweight simulator guards.

    The strict local-maximum-only version is closer to the paper text, but in a
    lightweight Python simulator without the paper's full OPNET stack it causes
    excessive CH explosion and fragmentation. This class keeps the paper's four
    utility factors and hierarchical routing assumptions, while restoring only
    the clustering guards needed to approximate the missing simulator detail.
    """

    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.16,
        ch_retain_margin: float = 0.12,
        min_ch_tenure_s: float = 12.0,
        max_cluster_members: int = 40,
        min_ch_neighbor_count: int = 2,
        ch_energy_guard_ratio: float = 0.20,
        degree_balance_bonus_weight: float = 0.05,
        tenure_stability_bonus_weight: float = 0.05,
        link_stability_bonus_weight: float = 0.05,
        velocity_stability_bonus_weight: float = 0.04,
        recent_ch_penalty_weight: float = 0.06,
        traffic_load_penalty_weight: float = 0.04,
        local_degree_target: float = 0.58,
        local_degree_tolerance: float = 0.28,
        small_cluster_size: int = 3,
        **_: float,
    ) -> None:
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = max(1.0, v_max)

        self.join_hysteresis_margin = join_hysteresis_margin
        self.ch_retain_margin = ch_retain_margin
        self.min_ch_tenure_s = min_ch_tenure_s
        self.max_cluster_members = max_cluster_members
        self.min_ch_neighbor_count = min_ch_neighbor_count
        self.ch_energy_guard_ratio = ch_energy_guard_ratio
        self.degree_balance_bonus_weight = degree_balance_bonus_weight
        self.tenure_stability_bonus_weight = tenure_stability_bonus_weight
        self.link_stability_bonus_weight = link_stability_bonus_weight
        self.velocity_stability_bonus_weight = velocity_stability_bonus_weight
        self.recent_ch_penalty_weight = recent_ch_penalty_weight
        self.traffic_load_penalty_weight = traffic_load_penalty_weight
        self.local_degree_target = local_degree_target
        self.local_degree_tolerance = max(1e-6, local_degree_tolerance)
        self.small_cluster_size = small_cluster_size

    def _ensure_factors(
        self,
        alive: Dict[int, Node],
        weights: Tuple[float, float, float, float],
    ) -> None:
        n_total = max(1, len(alive))
        for node in alive.values():
            node.neighbor_lht = {}
            node.neighbor_vel_sim = {}
            for j in _alive_neighbors(node, alive):
                node.neighbor_lht[j] = link_holding_time_s(node, alive[j], self.comm_radius_m)
                node.neighbor_vel_sim[j] = velocity_similarity(node, alive[j])

            factors = compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=self.comm_radius_m,
                n_total=n_total,
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            )
            node.s1 = factors.s1_energy
            node.s2 = factors.s2_degree
            node.s3 = factors.s3_vel_sim
            node.s4 = factors.s4_lht
            node.utility = weighted_utility(factors, weights)

    def _desired_ch_count(self, n_alive: int) -> int:
        if n_alive <= 10:
            return max(2, int(round(n_alive / 4.0)))
        if n_alive <= 20:
            return max(3, int(round(n_alive / 5.0)))
        if n_alive <= 50:
            return max(5, int(round(n_alive / 7.0)))
        return max(8, min(14, int(round(n_alive / 9.0))))

    def _degree_balance_score(self, node: Node, alive: Dict[int, Node]) -> float:
        deg = len(_alive_neighbors(node, alive)) / max(1, len(alive) - 1)
        gap = abs(deg - self.local_degree_target)
        return max(0.0, 1.0 - gap / self.local_degree_tolerance)

    def _ch_score(self, node: Node, alive: Dict[int, Node]) -> float:
        score = node.utility
        score += self.degree_balance_bonus_weight * self._degree_balance_score(node, alive)
        score += self.link_stability_bonus_weight * node.s4
        score += self.velocity_stability_bonus_weight * node.s3
        if node.role == Role.CH and node.cluster_head == node.node_id:
            score += self.tenure_stability_bonus_weight * min(1.0, node.ch_tenure_s / max(1e-9, self.min_ch_tenure_s))
        score -= self.recent_ch_penalty_weight * min(1.0, _safe_attr(node, "recent_role_switches", 0.0))
        score -= self.traffic_load_penalty_weight * min(1.0, _safe_attr(node, "traffic_load_score", 0.0))
        if len(_alive_neighbors(node, alive)) < self.min_ch_neighbor_count:
            score -= 0.10
        if node.s1 < self.ch_energy_guard_ratio:
            score -= 0.15
        return score

    def _elect_cluster_heads(self, alive: Dict[int, Node]) -> List[int]:
        if not alive:
            return []

        scores = {i: self._ch_score(node, alive) for i, node in alive.items()}
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        target_count = self._desired_ch_count(len(alive))

        elected: List[int] = []
        suppressed: Set[int] = set()

        # Keep strong incumbents first.
        for i, _ in ordered:
            node = alive[i]
            if not (node.role == Role.CH and node.cluster_head == i):
                continue
            if node.ch_tenure_s < self.min_ch_tenure_s:
                continue
            if len(_alive_neighbors(node, alive)) < self.min_ch_neighbor_count:
                continue
            if node.s1 < self.ch_energy_guard_ratio:
                continue
            elected.append(i)
            suppressed.add(i)
            suppressed.update(_alive_neighbors(node, alive))

        # Greedy local maxima with one-hop suppression.
        for i, _ in ordered:
            if len(elected) >= target_count:
                break
            if i in elected:
                continue
            node = alive[i]
            nbrs = _alive_neighbors(node, alive)
            dominated = False
            for j in nbrs:
                if scores[j] > scores[i] + self.ch_retain_margin:
                    dominated = True
                    break
            if dominated and i in suppressed:
                continue

            elected.append(i)
            suppressed.add(i)
            suppressed.update(nbrs)

        # Ensure every node can reach at least one CH through a stable direct link.
        for i, node in alive.items():
            reachable = False
            for ch in elected:
                if i == ch:
                    reachable = True
                    break
                if link_holding_time_s(node, alive[ch], self.comm_radius_m) >= self.lht_threshold_s:
                    reachable = True
                    break
            if not reachable:
                elected.append(i)

        # Prune redundant CHs that are stably covered by stronger CHs.
        pruned: List[int] = []
        elected_set = set(elected)
        for ch in elected:
            node = alive[ch]
            redundant = False
            for other in elected_set:
                if other == ch:
                    continue
                if link_holding_time_s(node, alive[other], self.comm_radius_m) < self.lht_threshold_s:
                    continue
                if scores[other] >= scores[ch] + max(0.06, self.ch_retain_margin):
                    redundant = True
                    break
            if not redundant:
                pruned.append(ch)

        if not pruned:
            pruned = [ordered[0][0]]
        return sorted(set(pruned))

    def _member_assignment_score(
        self,
        node: Node,
        ch: Node,
        clusters: Dict[int, List[int]],
        prev_ch: Optional[int],
    ) -> float:
        lht = link_holding_time_s(node, ch, self.comm_radius_m)
        if lht < self.lht_threshold_s:
            return -1e18

        lht_norm = min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
        vel = velocity_similarity(node, ch)
        size_penalty = len(clusters[ch.node_id]) / max(1, self.max_cluster_members)

        score = 0.0
        score += 0.50 * ch.utility
        score += 0.24 * lht_norm
        score += 0.12 * vel
        score += 0.10 * ch.s1
        score -= 0.14 * size_penalty
        score -= 0.06 * _safe_attr(ch, "traffic_load_score", 0.0)
        if prev_ch is not None and prev_ch == ch.node_id:
            score += self.join_hysteresis_margin
        return score

    def _assign_members(
        self,
        alive: Dict[int, Node],
        chs: List[int],
    ) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {ch: [ch] for ch in chs}
        ch_set = set(chs)
        prev_heads = {i: n.cluster_head for i, n in alive.items()}

        for node in alive.values():
            node.is_forwarder = False
            if node.node_id in ch_set:
                node.set_role(Role.CH)
                node.cluster_head = node.node_id
            else:
                node.set_role(Role.MEMBER)
                node.cluster_head = None

        member_ids = [i for i in alive if i not in ch_set]
        member_ids.sort(
            key=lambda i: (
                sum(
                    1
                    for ch in chs
                    if link_holding_time_s(alive[i], alive[ch], self.comm_radius_m) >= self.lht_threshold_s
                ),
                -alive[i].s4,
                -alive[i].s1,
                i,
            )
        )

        for i in member_ids:
            node = alive[i]
            prev_ch = prev_heads.get(i)

            best_ch: Optional[int] = None
            best_score = -1e18
            for ch in chs:
                if len(clusters[ch]) >= self.max_cluster_members:
                    continue
                score = self._member_assignment_score(node, alive[ch], clusters, prev_ch)
                if score > best_score:
                    best_score = score
                    best_ch = ch

            if best_ch is not None:
                clusters[best_ch].append(i)
                node.cluster_head = best_ch
                continue

            # Fallback: accept even weak links rather than immediate self-promotion.
            fallback_ch: Optional[int] = None
            fallback_score = -1e18
            for ch in chs:
                lht = link_holding_time_s(node, alive[ch], self.comm_radius_m)
                if lht <= 0.0:
                    continue
                score = (
                    0.60 * alive[ch].utility
                    + 0.25 * min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
                    + 0.15 * alive[ch].s1
                )
                if prev_ch is not None and prev_ch == ch:
                    score += self.join_hysteresis_margin
                if score > fallback_score:
                    fallback_score = score
                    fallback_ch = ch

            if fallback_ch is not None:
                clusters[fallback_ch].append(i)
                node.cluster_head = fallback_ch
            else:
                node.set_role(Role.CH)
                node.cluster_head = i
                clusters[i] = [i]
                ch_set.add(i)
                chs.append(i)

        return clusters

    def _repair_small_clusters(
        self,
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> Dict[int, List[int]]:
        changed = True
        while changed:
            changed = False
            small_chs = [ch for ch, members in clusters.items() if len(members) <= self.small_cluster_size]
            for ch in small_chs:
                if ch not in alive or ch not in clusters:
                    continue
                members = list(clusters[ch])
                best_target: Optional[int] = None
                best_score = -1e18

                for other_ch in list(clusters.keys()):
                    if other_ch == ch or other_ch not in alive:
                        continue
                    if len(clusters[other_ch]) + len(members) > self.max_cluster_members:
                        continue

                    ok = True
                    total = 0.0
                    for m in members:
                        if m not in alive:
                            continue
                        lht = link_holding_time_s(alive[m], alive[other_ch], self.comm_radius_m)
                        if lht < self.lht_threshold_s * 0.8:
                            ok = False
                            break
                        total += (
                            0.60 * alive[other_ch].utility
                            + 0.25 * min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
                            + 0.15 * alive[other_ch].s1
                        )

                    if ok and total > best_score:
                        best_score = total
                        best_target = other_ch

                if best_target is None:
                    continue

                for m in members:
                    if m not in alive:
                        continue
                    if m == ch:
                        alive[m].set_role(Role.MEMBER, count_change=False)
                    alive[m].cluster_head = best_target
                    clusters[best_target].append(m)

                del clusters[ch]
                changed = True
                break

        normalized: Dict[int, List[int]] = {}
        for ch, members in clusters.items():
            uniq: List[int] = []
            seen = set()
            for m in members:
                if m in alive and m not in seen:
                    uniq.append(m)
                    seen.add(m)
            if uniq:
                normalized[ch] = uniq

        current_chs = set(normalized.keys())
        for i, node in alive.items():
            if i in current_chs:
                node.set_role(Role.CH)
                node.cluster_head = i
            else:
                node.set_role(Role.MEMBER)
                if node.cluster_head not in current_chs:
                    node.cluster_head = None

        return normalized

    def _select_forwarders(
        self,
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> Set[int]:
        """Choose a small number of stable cross-cluster forwarders."""
        forwarders: Set[int] = set()
        n_alive = len(alive)
        best_by_member: Dict[int, float] = {}

        for ch, members in clusters.items():
            best_m: Optional[int] = None
            best_score = -1e18
            for m in members:
                if m == ch or m not in alive:
                    continue
                reachable = set()
                for j in _alive_neighbors(alive[m], alive):
                    other = alive[j]
                    other_ch = other.node_id if other.role == Role.CH else other.cluster_head
                    if other_ch is not None and other_ch != ch:
                        reachable.add(other_ch)
                cross = len(reachable)
                if cross < 2:
                    continue
                if alive[m].s4 < 0.40:
                    continue
                if alive[m].s1 < 0.30:
                    continue

                score = cross + 0.35 * alive[m].s4 + 0.25 * alive[m].s1
                score -= 0.10 * _safe_attr(alive[m], "relay_load_score", 0.0)
                if score > best_score:
                    best_score = score
                    best_m = m

            if best_m is not None:
                prev = best_by_member.get(best_m, -1e18)
                if best_score > prev:
                    best_by_member[best_m] = best_score

        ranked = sorted(best_by_member.items(), key=lambda kv: -kv[1])
        cap = max(2, min(len(clusters), n_alive // 16 + 1))
        for m, _ in ranked[:cap]:
            forwarders.add(m)

        for node in alive.values():
            node.is_forwarder = node.node_id in forwarders and node.role != Role.CH
            if node.is_forwarder:
                node.set_role(Role.FORWARDER, count_change=False)

        return forwarders

    def cluster(
        self,
        nodes: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        dt_s: float = 2.0,
        factors_already_set: bool = False,
    ) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        for node in alive.values():
            node.reset_clustering_flags()

        if not factors_already_set:
            self._ensure_factors(alive, weights)
        else:
            for node in alive.values():
                node.neighbor_lht = {}
                node.neighbor_vel_sim = {}
                for j in _alive_neighbors(node, alive):
                    node.neighbor_lht[j] = link_holding_time_s(node, alive[j], self.comm_radius_m)
                    node.neighbor_vel_sim[j] = velocity_similarity(node, alive[j])
                node.utility = (
                    weights[0] * node.s1
                    + weights[1] * node.s2
                    + weights[2] * node.s3
                    + weights[3] * node.s4
                )

        chs = self._elect_cluster_heads(alive)
        clusters = self._assign_members(alive, chs)
        clusters = self._repair_small_clusters(alive, clusters)
        forwarders = self._select_forwarders(alive, clusters)

        for node in alive.values():
            node.note_cluster_membership(node.cluster_head, dt_s)
            node.note_role_tenure(dt_s)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    def __init__(self, comm_radius_m: float) -> None:
        self.comm_radius_m = comm_radius_m

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult({}, set())

        for node in alive.values():
            deg = len(_alive_neighbors(node, alive))
            avg_deg = sum(len(_alive_neighbors(n, alive)) for n in alive.values()) / max(1, len(alive))
            degree_diff = abs(deg - avg_deg) / max(1.0, avg_deg)
            sum_dist = 0.0
            for j in _alive_neighbors(node, alive):
                dx = node.x_m - alive[j].x_m
                dy = node.y_m - alive[j].y_m
                sum_dist += math.sqrt(dx * dx + dy * dy)
            speed_term = min(1.0, node.speed_m_s / 50.0)
            node.utility = (
                0.40 * (1.0 - min(1.0, deg / max(1, len(alive) - 1)))
                + 0.25 * degree_diff
                + 0.20 * min(1.0, sum_dist / max(1.0, self.comm_radius_m * 6.0))
                + 0.15 * speed_term
            )
            node.utility += random.uniform(-0.28, 0.28)
            node.reset_clustering_flags()

        unassigned = set(alive.keys())
        clusters: Dict[int, List[int]] = {}
        while unassigned:
            ch = min(unassigned, key=lambda i: (alive[i].utility, i))
            clusters[ch] = [ch]
            alive[ch].set_role(Role.CH)
            alive[ch].cluster_head = ch
            to_remove = {ch}
            for i in list(unassigned):
                if i == ch:
                    continue
                if ch in alive[i].neighbors:
                    alive[i].set_role(Role.MEMBER)
                    alive[i].cluster_head = ch
                    clusters[ch].append(i)
                    to_remove.add(i)
            unassigned -= to_remove

        forwarders: Set[int] = set()
        for ch, members in clusters.items():
            for m in members:
                if m == ch:
                    continue
                cross = 0
                for j in alive[m].neighbors:
                    if j in alive:
                        other_ch = alive[j].cluster_head if alive[j].role != Role.CH else j
                        if other_ch is not None and other_ch != ch:
                            cross += 1
                if cross > 0:
                    forwarders.add(m)
                    break

        for node in alive.values():
            node.is_forwarder = node.node_id in forwarders and node.role != Role.CH
            if node.is_forwarder:
                node.set_role(Role.FORWARDER)
            node.note_cluster_membership(node.cluster_head, 2.0)
            node.note_role_tenure(2.0)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class DCAClusterer:
    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult({}, set())

        for node in alive.values():
            node.reset_clustering_flags()
            deg = len(_alive_neighbors(node, alive))
            deg_norm = min(1.0, deg / max(1, len(alive) - 1))
            energy = node.s1
            lht = node.s4
            vel = node.s3
            node.utility = 0.30 * energy + 0.25 * deg_norm + 0.25 * lht + 0.20 * vel
            node.utility += random.uniform(-0.03, 0.03)

        chs: List[int] = []
        for i, node in alive.items():
            nbrs = _alive_neighbors(node, alive)
            if all(alive[j].utility <= node.utility + 1e-12 for j in nbrs):
                chs.append(i)
        if not chs:
            chs = [max(alive.keys(), key=lambda i: alive[i].utility)]

        clusters: Dict[int, List[int]] = {ch: [ch] for ch in chs}
        for ch in chs:
            alive[ch].set_role(Role.CH)
            alive[ch].cluster_head = ch

        for i, node in alive.items():
            if i in clusters:
                continue
            best = None
            best_score = -1e18
            for ch in chs:
                if ch not in node.neighbors:
                    continue
                score = 0.60 * alive[ch].utility + 0.25 * alive[ch].s4 + 0.15 * alive[ch].s1
                if score > best_score:
                    best_score = score
                    best = ch
            if best is None:
                node.set_role(Role.CH)
                node.cluster_head = i
                clusters[i] = [i]
            else:
                node.set_role(Role.MEMBER)
                node.cluster_head = best
                clusters[best].append(i)

        forwarders: Set[int] = set()
        for ch, members in clusters.items():
            for m in members:
                if m == ch:
                    continue
                cross = 0
                for j in alive[m].neighbors:
                    if j in alive:
                        other_ch = alive[j].cluster_head if alive[j].role != Role.CH else j
                        if other_ch is not None and other_ch != ch:
                            cross += 1
                if cross >= 1 and alive[m].s4 >= 0.30:
                    forwarders.add(m)
                    break

        for node in alive.values():
            node.is_forwarder = node.node_id in forwarders and node.role != Role.CH
            if node.is_forwarder:
                node.set_role(Role.FORWARDER)
            node.note_cluster_membership(node.cluster_head, 2.0)
            node.note_role_tenure(2.0)

        return ClusterResult(clusters=clusters, forwarders=forwarders)
