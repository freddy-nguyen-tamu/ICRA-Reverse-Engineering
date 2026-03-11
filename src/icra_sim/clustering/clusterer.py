from __future__ import annotations

import math
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


@dataclass(frozen=True)
class GatewayCandidate:
    node_id: int
    own_ch: int
    reachable_chs: Tuple[int, ...]
    score: float


class ICRAClusterer:
    """
    Tuned for fewer CHs, stronger retention, aggressive merging.
    """

    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.20,
        ch_retain_margin: float = 0.25,
        min_ch_tenure_s: float = 16.0,
        max_cluster_members: int = 40,
        min_gateway_lht_s: float = 0.20,
        min_ch_neighbor_count: int = 2,
        prefer_connected_ch_bonus: float = 0.10,
        isolated_ch_penalty: float = 0.20,
        forwarder_reuse_bonus: float = 0.01,
        gateway_crosslink_weight: float = 0.50,
        gateway_utility_weight: float = 0.15,
        gateway_energy_weight: float = 0.15,
        gateway_stability_weight: float = 0.20,
        gateway_multicluster_bonus: float = 0.03,
        direct_ch_link_bonus: float = 0.02,
        ch_energy_guard_ratio: float = 0.20,
        ch_cooldown_s: float = 8.0,
        recent_ch_penalty_weight: float = 0.08,
        traffic_load_penalty_weight: float = 0.03,
        degree_balance_bonus_weight: float = 0.05,
        tenure_stability_bonus_weight: float = 0.06,
        link_stability_bonus_weight: float = 0.06,
        velocity_stability_bonus_weight: float = 0.05,
        local_degree_target: float = 0.60,
        local_degree_tolerance: float = 0.25,
    ) -> None:
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = max(1e-9, v_max)

        self.join_hysteresis_margin = join_hysteresis_margin
        self.ch_retain_margin = ch_retain_margin
        self.min_ch_tenure_s = min_ch_tenure_s
        self.max_cluster_members = max_cluster_members
        self.min_gateway_lht_s = min_gateway_lht_s

        self.min_ch_neighbor_count = min_ch_neighbor_count
        self.prefer_connected_ch_bonus = prefer_connected_ch_bonus
        self.isolated_ch_penalty = isolated_ch_penalty

        self.forwarder_reuse_bonus = forwarder_reuse_bonus
        self.gateway_crosslink_weight = gateway_crosslink_weight
        self.gateway_utility_weight = gateway_utility_weight
        self.gateway_energy_weight = gateway_energy_weight
        self.gateway_stability_weight = gateway_stability_weight
        self.gateway_multicluster_bonus = gateway_multicluster_bonus
        self.direct_ch_link_bonus = direct_ch_link_bonus

        self.ch_energy_guard_ratio = ch_energy_guard_ratio
        self.ch_cooldown_s = max(1e-9, ch_cooldown_s)
        self.recent_ch_penalty_weight = recent_ch_penalty_weight
        self.traffic_load_penalty_weight = traffic_load_penalty_weight
        self.degree_balance_bonus_weight = degree_balance_bonus_weight
        self.tenure_stability_bonus_weight = tenure_stability_bonus_weight
        self.link_stability_bonus_weight = link_stability_bonus_weight
        self.velocity_stability_bonus_weight = velocity_stability_bonus_weight
        self.local_degree_target = local_degree_target
        self.local_degree_tolerance = max(1e-6, local_degree_tolerance)

    def _ensure_factors(self, alive: Dict[int, Node]) -> None:
        for node in alive.values():
            factors = compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=self.comm_radius_m,
                n_total=max(1, len(alive)),
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            )
            node.s1 = factors.s1_energy
            node.s2 = factors.s2_degree
            node.s3 = factors.s3_vel_sim
            node.s4 = factors.s4_lht

            node.neighbor_lht = {}
            node.neighbor_vel_sim = {}
            for j in _alive_neighbors(node, alive):
                node.neighbor_lht[j] = link_holding_time_s(node, alive[j], self.comm_radius_m)
                node.neighbor_vel_sim[j] = velocity_similarity(node, alive[j])

    def _degree_balance_score(self, node: Node, alive: Dict[int, Node]) -> float:
        deg = len(_alive_neighbors(node, alive))
        if deg <= 0:
            return 0.0
        max_deg = max(1, max(len(_alive_neighbors(n, alive)) for n in alive.values()))
        local_deg = deg / max_deg
        z = (local_deg - self.local_degree_target) / self.local_degree_tolerance
        return max(0.0, math.exp(-(z * z)))

    def _candidate_utility(
        self,
        node: Node,
        weights: Tuple[float, float, float, float],
        alive: Dict[int, Node],
    ) -> float:
        base = weighted_utility(
            compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=self.comm_radius_m,
                n_total=max(1, len(alive)),
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            ),
            weights,
        )

        nbrs = _alive_neighbors(node, alive)
        deg = len(nbrs)

        avg_lht = 0.0
        avg_vel = 0.0
        if deg > 0:
            avg_lht = sum(min(node.neighbor_lht.get(j, 0.0), self.lht_cap_s) for j in nbrs) / (
                deg * max(1e-9, self.lht_cap_s)
            )
            avg_vel = sum(node.neighbor_vel_sim.get(j, 0.0) for j in nbrs) / deg

        retention_bonus = 0.0
        if node.role == Role.CH and node.cluster_head == node.node_id:
            retention_bonus += self.ch_retain_margin
            retention_bonus += self.tenure_stability_bonus_weight * min(
                1.0, node.ch_tenure_s / max(1e-9, self.min_ch_tenure_s)
            )

        score = base
        score += self.degree_balance_bonus_weight * self._degree_balance_score(node, alive)
        score += self.link_stability_bonus_weight * avg_lht
        score += self.velocity_stability_bonus_weight * avg_vel
        score += retention_bonus

        if deg >= self.min_ch_neighbor_count:
            score += self.prefer_connected_ch_bonus * min(1.0, deg / 5.0)
        else:
            score -= self.isolated_ch_penalty

        if node.s1 < self.ch_energy_guard_ratio:
            score -= 0.18

        score -= self.recent_ch_penalty_weight * _safe_attr(node, "recent_role_switches", 0.0)
        score -= self.traffic_load_penalty_weight * _safe_attr(node, "traffic_load_score", 0.0)
        score -= 0.05 * min(1.0, _safe_attr(node, "ch_cooldown_s", 0.0) / self.ch_cooldown_s)

        node.utility = score
        return score

    def _desired_ch_count(self, n_alive: int) -> int:
        # Target: about 8-10 CHs at N=100
        if n_alive <= 10:
            return 1
        if n_alive <= 20:
            return max(2, int(math.ceil(n_alive / 8.0)))
        return max(3, int(math.ceil(n_alive / 12.0)))   # N=100 => 9

    def _elect_cluster_heads(
        self,
        alive: Dict[int, Node],
        weights: Tuple[float, float, float, float],
    ) -> List[int]:
        scores = {i: self._candidate_utility(node, weights, alive) for i, node in alive.items()}
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        target_count = self._desired_ch_count(len(alive))

        elected: List[int] = []
        suppressed: Set[int] = set()

        # Keep strong incumbents
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

        # Greedy local maxima with suppression
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

        # Ensure every node has a reachable CH
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

        # Prune redundant CHs covered by stronger neighbours
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
        dt_s: float,
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
                node.note_cluster_membership(best_ch, dt_s)
                continue

            # Fallback: even weak links are accepted to avoid self-promotion
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
                node.note_cluster_membership(fallback_ch, dt_s)
            else:
                # Truly isolated – self-promote
                node.set_role(Role.CH)
                node.cluster_head = i
                clusters[i] = [i]
                ch_set.add(i)
                chs.append(i)

        for ch in clusters:
            if ch in alive and alive[ch].role == Role.CH:
                alive[ch].note_role_tenure(dt_s)

        return clusters

    def _repair_small_clusters(
        self,
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
        dt_s: float,
    ) -> Dict[int, List[int]]:
        """
        Aggressively merge tiny clusters (size <= 3) into larger ones.
        """
        changed = True
        while changed:
            changed = False
            small_chs = [ch for ch, members in clusters.items() if len(members) <= 3]
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
                        if lht < self.lht_threshold_s * 0.8:   # lower threshold for merging
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
                        alive[m].set_role(Role.MEMBER)
                    alive[m].cluster_head = best_target
                    alive[m].note_cluster_membership(best_target, dt_s)
                    clusters[best_target].append(m)

                del clusters[ch]
                changed = True
                break

        # Normalize
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

    def _candidate_gateway_score(
        self,
        node: Node,
        own_ch: int,
        target_chs: List[int],
        alive: Dict[int, Node],
    ) -> Optional[GatewayCandidate]:
        if node.e_j <= 0 or node.role == Role.CH:
            return None
        if own_ch not in alive:
            return None

        own_lht = link_holding_time_s(node, alive[own_ch], self.comm_radius_m)
        if own_lht < self.lht_threshold_s:
            return None

        reachable: List[int] = []
        mean_cross_lht = 0.0
        for ch in target_chs:
            if ch == own_ch or ch not in alive:
                continue
            lht = link_holding_time_s(node, alive[ch], self.comm_radius_m)
            if lht >= self.min_gateway_lht_s:
                reachable.append(ch)
                mean_cross_lht += min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)

        if not reachable:
            return None

        mean_cross_lht /= max(1, len(reachable))

        score = 0.0
        score += self.gateway_crosslink_weight * min(1.0, len(reachable) / 3.0)
        score += self.gateway_utility_weight * max(0.0, min(1.0, node.utility))
        score += self.gateway_energy_weight * node.s1
        score += self.gateway_stability_weight * (0.5 * node.s3 + 0.5 * node.s4)
        score += 0.08 * mean_cross_lht

        if node.is_forwarder:
            score += self.forwarder_reuse_bonus

        score -= 0.05 * _safe_attr(node, "traffic_load_score", 0.0)
        score -= 0.05 * _safe_attr(node, "relay_load_score", 0.0)

        return GatewayCandidate(
            node_id=node.node_id,
            own_ch=own_ch,
            reachable_chs=tuple(sorted(set(reachable))),
            score=score,
        )

    def _select_forwarders(
        self,
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> Set[int]:
        """
        Forwarder selection – allow a node that reaches at least one other cluster
        and has decent stability/energy.
        """
        forwarders: Set[int] = set()
        chs = sorted(clusters.keys())

        for ch, members in clusters.items():
            best_m: Optional[int] = None
            best_score = -1e18
            for m in members:
                if m == ch or m not in alive:
                    continue
                # Count other clusters reachable directly from m
                reachable = set()
                for j in alive[m].neighbors:
                    if j not in alive:
                        continue
                    other_node = alive[j]
                    other_ch = other_node.node_id if other_node.role == Role.CH else other_node.cluster_head
                    if other_ch is not None and other_ch != ch:
                        reachable.add(other_ch)
                cross = len(reachable)
                if cross == 0:
                    continue

                # Require at least moderate stability and energy
                if alive[m].s4 < 0.40:
                    continue
                if alive[m].s1 < 0.30:
                    continue

                score = cross + 0.3 * alive[m].s4 + 0.2 * alive[m].s1
                if score > best_score:
                    best_score = score
                    best_m = m

            if best_m is not None:
                forwarders.add(best_m)

        for node in alive.values():
            node.is_forwarder = node.node_id in forwarders and node.role != Role.CH
            if node.is_forwarder:
                node.set_role(Role.FORWARDER)

        return forwarders

    def cluster(
        self,
        nodes: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        dt_s: float = 8.0,
        factors_already_set: bool = False,
    ) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        if not factors_already_set:
            self._ensure_factors(alive)
        else:
            for node in alive.values():
                node.neighbor_lht = {}
                node.neighbor_vel_sim = {}
                for j in _alive_neighbors(node, alive):
                    node.neighbor_lht[j] = link_holding_time_s(node, alive[j], self.comm_radius_m)
                    node.neighbor_vel_sim[j] = velocity_similarity(node, alive[j])

        chs = self._elect_cluster_heads(alive, weights)
        clusters = self._assign_members(alive, chs, dt_s)
        clusters = self._repair_small_clusters(alive, clusters, dt_s)
        forwarders = self._select_forwarders(alive, clusters)

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
            degree_diff = 0.0
            if alive:
                avg_deg = sum(len(_alive_neighbors(n, alive)) for n in alive.values()) / len(alive)
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

        unassigned = set(alive.keys())
        clusters: Dict[int, List[int]] = {}

        while unassigned:
            ch = min(unassigned, key=lambda i: (alive[i].utility, i))
            clusters[ch] = [ch]
            alive[ch].set_role(Role.CH)
            alive[ch].cluster_head = ch

            members_to_remove = {ch}
            for i in list(unassigned):
                if i == ch:
                    continue
                if ch in alive[i].neighbors:
                    alive[i].set_role(Role.MEMBER)
                    alive[i].cluster_head = ch
                    clusters[ch].append(i)
                    members_to_remove.add(i)

            unassigned -= members_to_remove

        forwarders: Set[int] = set()
        for ch, members in clusters.items():
            best_m = None
            best_score = -1.0
            for m in members:
                if m == ch:
                    continue
                cross = 0
                for j in alive[m].neighbors:
                    if j in alive:
                        other_ch = alive[j].cluster_head if alive[j].role != Role.CH else j
                        if other_ch is not None and other_ch != ch:
                            cross += 1
                score = cross + 0.2 * alive[m].s1
                if score > best_score:
                    best_score = score
                    best_m = m
            if best_m is not None and best_score > 0:
                forwarders.add(best_m)

        for node in alive.values():
            node.is_forwarder = node.node_id in forwarders
            if node.is_forwarder:
                node.set_role(Role.FORWARDER)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class DCAClusterer:
    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult({}, set())

        for node in alive.values():
            deg = len(_alive_neighbors(node, alive))
            deg_norm = min(1.0, deg / max(1, len(alive) - 1))
            energy = node.s1
            lht = node.s4
            vel = node.s3
            recent = 1.0 - min(1.0, _safe_attr(node, "recent_role_switches", 0.0))
            load = 1.0 - min(1.0, _safe_attr(node, "traffic_load_score", 0.0))
            node.utility = 0.22 * energy + 0.18 * deg_norm + 0.18 * lht + 0.14 * vel + 0.14 * recent + 0.14 * load

        chs: List[int] = []
        for i, node in alive.items():
            nbrs = _alive_neighbors(node, alive)
            local_best = True
            for j in nbrs:
                if alive[j].utility > node.utility + 1e-12:
                    local_best = False
                    break
            if local_best:
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
                score = 0.65 * alive[ch].utility + 0.20 * alive[ch].s4 + 0.15 * alive[ch].s1
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
                if cross >= 2 and alive[m].s4 >= 0.35:
                    forwarders.add(m)
                    break

        for node in alive.values():
            node.is_forwarder = node.node_id in forwarders and node.role != Role.CH
            if node.is_forwarder:
                node.set_role(Role.FORWARDER)

        return ClusterResult(clusters=clusters, forwarders=forwarders)