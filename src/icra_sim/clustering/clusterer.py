from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from .utility import compute_factors, weighted_utility, velocity_similarity


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
    Paper-faithful direction:
    - stable CH election driven by utility, not aggressive self-promotion
    - strong CH retention to reduce role churn
    - member assignment prefers stable/strong CHs and avoids fragmentation
    - post-pass repairs tiny clusters by merging them whenever possible
    - forwarding node selection is coverage-based but conservative
    """

    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.08,
        ch_retain_margin: float = 0.10,
        min_ch_tenure_s: float = 8.0,
        max_cluster_members: int = 18,
        min_gateway_lht_s: float = 0.10,
        min_ch_neighbor_count: int = 1,
        prefer_connected_ch_bonus: float = 0.04,
        isolated_ch_penalty: float = 0.08,
        forwarder_reuse_bonus: float = 0.01,
        gateway_crosslink_weight: float = 0.45,
        gateway_utility_weight: float = 0.15,
        gateway_energy_weight: float = 0.15,
        gateway_stability_weight: float = 0.15,
        gateway_multicluster_bonus: float = 0.04,
        direct_ch_link_bonus: float = 0.04,
        ch_energy_guard_ratio: float = 0.15,
        ch_cooldown_s: float = 4.0,
        recent_ch_penalty_weight: float = 0.04,
        traffic_load_penalty_weight: float = 0.02,
        degree_balance_bonus_weight: float = 0.04,
        tenure_stability_bonus_weight: float = 0.02,
        link_stability_bonus_weight: float = 0.03,
        velocity_stability_bonus_weight: float = 0.03,
        local_degree_target: float = 0.50,
        local_degree_tolerance: float = 0.35,
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

    def _base_utility(
        self,
        node: Node,
        weights: Tuple[float, float, float, float],
        alive: Dict[int, Node],
    ) -> float:
        return weighted_utility(
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

    def _candidate_utility(
        self,
        node: Node,
        weights: Tuple[float, float, float, float],
        alive: Dict[int, Node],
    ) -> float:
        nbrs = _alive_neighbors(node, alive)
        deg = len(nbrs)
        avg_lht_norm = 0.0
        avg_vel = 0.0
        if deg > 0:
            avg_lht_norm = (
                sum(min(node.neighbor_lht.get(j, 0.0), self.lht_cap_s) for j in nbrs)
                / (deg * max(1e-9, self.lht_cap_s))
            )
            avg_vel = sum(node.neighbor_vel_sim.get(j, 0.0) for j in nbrs) / deg

        retention_bonus = 0.0
        if node.role == Role.CH and node.cluster_head == node.node_id:
            retention_bonus += self.ch_retain_margin
            retention_bonus += self.tenure_stability_bonus_weight * min(
                1.0, node.ch_tenure_s / max(1e-9, self.min_ch_tenure_s)
            )

        cooldown_penalty = min(1.0, _safe_attr(node, "ch_cooldown_s", 0.0) / self.ch_cooldown_s)
        switch_penalty = _safe_attr(node, "recent_role_switches", 0.0)
        traffic_penalty = _safe_attr(node, "traffic_load_score", 0.0)

        score = self._base_utility(node, weights, alive)
        score += self.degree_balance_bonus_weight * self._degree_balance_score(node, alive)
        score += self.link_stability_bonus_weight * avg_lht_norm
        score += self.velocity_stability_bonus_weight * avg_vel
        score += retention_bonus

        if deg >= self.min_ch_neighbor_count:
            score += self.prefer_connected_ch_bonus * min(1.0, deg / max(1.0, self.min_ch_neighbor_count + 2.0))
        else:
            score -= self.isolated_ch_penalty

        if node.s1 < self.ch_energy_guard_ratio:
            score -= 0.18

        score -= self.recent_ch_penalty_weight * switch_penalty
        score -= self.traffic_load_penalty_weight * traffic_penalty
        score -= 0.05 * cooldown_penalty

        return score

    def _elect_cluster_heads(
        self,
        alive: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        dt_s: float,
    ) -> List[int]:
        candidates: Dict[int, float] = {}
        for i, node in alive.items():
            candidates[i] = self._candidate_utility(node, weights, alive)
            node.utility = candidates[i]

        if not candidates:
            return []

        ordered = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        candidate_ids = [i for i, _ in ordered]

        # Build a restrained CH set.
        # A node becomes CH if it is a strong local maximum and not already covered
        # by a clearly better CH candidate.
        elected: List[int] = []
        covered: Set[int] = set()

        strong_threshold_rank = max(1, int(math.ceil(0.30 * len(candidate_ids))))
        strong_threshold = ordered[min(len(ordered) - 1, strong_threshold_rank - 1)][1]

        for i in candidate_ids:
            node = alive[i]
            nbrs = _alive_neighbors(node, alive)
            better_neighbor = False
            for j in nbrs:
                if candidates[j] > candidates[i] + self.ch_retain_margin:
                    better_neighbor = True
                    break

            is_existing_ch = node.role == Role.CH and node.cluster_head == node.node_id
            retain_existing = (
                is_existing_ch
                and node.ch_tenure_s >= self.min_ch_tenure_s
                and len(nbrs) >= self.min_ch_neighbor_count
                and node.s1 >= self.ch_energy_guard_ratio
            )

            if retain_existing:
                elected.append(i)
                covered.add(i)
                covered.update(nbrs)
                continue

            if candidates[i] < strong_threshold and i in covered:
                continue

            if better_neighbor and i in covered:
                continue

            elected.append(i)
            covered.add(i)
            covered.update(nbrs)

        # Safety: ensure no node is completely unserviceable.
        uncovered = [i for i in alive if i not in covered]
        uncovered.sort(key=lambda x: (-candidates[x], x))
        for i in uncovered:
            # only promote if it really has no reachable elected CH
            has_reachable_ch = False
            for ch in elected:
                if ch == i:
                    has_reachable_ch = True
                    break
                if link_holding_time_s(alive[i], alive[ch], self.comm_radius_m) >= self.lht_threshold_s:
                    has_reachable_ch = True
                    break
            if not has_reachable_ch:
                elected.append(i)

        elected = sorted(set(elected))

        # Prune weak redundant CHs that are fully dominated by a nearby stronger CH.
        pruned: List[int] = []
        elected_set = set(elected)
        for ch in elected:
            node = alive[ch]
            redundant = False
            for other in elected:
                if other == ch:
                    continue
                if other not in elected_set:
                    continue
                lht = link_holding_time_s(node, alive[other], self.comm_radius_m)
                if lht < self.lht_threshold_s:
                    continue
                if candidates[other] >= candidates[ch] + max(0.03, 0.5 * self.ch_retain_margin):
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

        vel = velocity_similarity(node, ch)
        lht_norm = min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
        size_penalty = len(clusters[ch.node_id]) / max(1, self.max_cluster_members)
        ch_load = _safe_attr(ch, "traffic_load_score", 0.0)

        score = 0.0
        score += 0.42 * ch.utility
        score += 0.24 * lht_norm
        score += 0.16 * vel
        score += 0.10 * ch.s1
        score += 0.05 * self._degree_balance_score(ch, {n.node_id: n for n in [ch]})
        score -= 0.13 * size_penalty
        score -= 0.08 * ch_load

        if prev_ch is not None and prev_ch == ch.node_id:
            score += self.join_hysteresis_margin

        return score

    def _assign_members(
        self,
        alive: Dict[int, Node],
        chs: List[int],
        weights: Tuple[float, float, float, float],
        dt_s: float,
    ) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {ch: [ch] for ch in chs}
        ch_set = set(chs)
        prev_cluster_heads = {i: n.cluster_head for i, n in alive.items()}

        for node in alive.values():
            node.is_forwarder = False
            if node.node_id in ch_set:
                node.set_role(Role.CH)
                node.cluster_head = node.node_id
            else:
                node.set_role(Role.MEMBER)
                node.cluster_head = None

        member_ids = [i for i in alive if i not in ch_set]
        # Assign difficult nodes first so they are not stranded by capacity.
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
            prev_ch = prev_cluster_heads.get(i)
            best_ch: Optional[int] = None
            best_score = -1e18

            for ch in chs:
                if len(clusters[ch]) >= self.max_cluster_members:
                    continue
                score = self._member_assignment_score(node, alive[ch], clusters, prev_ch)
                if score > best_score:
                    best_score = score
                    best_ch = ch

            if best_ch is None:
                # Best-effort merge instead of self-promotion whenever possible.
                fallback_ch: Optional[int] = None
                fallback_score = -1e18
                for ch in chs:
                    lht = link_holding_time_s(node, alive[ch], self.comm_radius_m)
                    if lht <= 0.0:
                        continue
                    score = (
                        0.55 * alive[ch].utility
                        + 0.25 * min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
                        + 0.20 * alive[ch].s1
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
                    # Truly isolated; only then self-promote.
                    node.set_role(Role.CH)
                    node.cluster_head = i
                    clusters[i] = [i]
                    ch_set.add(i)
                    chs.append(i)
                continue

            clusters[best_ch].append(i)
            node.cluster_head = best_ch
            node.note_cluster_membership(best_ch, dt_s)

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
        Merge tiny clusters whenever a stable alternative CH exists.
        This directly attacks the paper mismatch on isolation clusters.
        """
        changed = True
        while changed:
            changed = False
            small_chs = [ch for ch, members in clusters.items() if len(members) <= 2]
            for ch in small_chs:
                if ch not in clusters or ch not in alive:
                    continue

                members = list(clusters[ch])
                target_scores: List[Tuple[float, int]] = []

                for other_ch in list(clusters.keys()):
                    if other_ch == ch or other_ch not in alive:
                        continue
                    if len(clusters[other_ch]) >= self.max_cluster_members:
                        continue

                    ok = True
                    total_score = 0.0
                    for m in members:
                        if m not in alive:
                            continue
                        lht = link_holding_time_s(alive[m], alive[other_ch], self.comm_radius_m)
                        if lht < self.lht_threshold_s:
                            ok = False
                            break
                        total_score += (
                            0.55 * alive[other_ch].utility
                            + 0.30 * min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
                            + 0.15 * alive[other_ch].s1
                        )

                    if ok:
                        target_scores.append((total_score, other_ch))

                if not target_scores:
                    continue

                target_scores.sort(reverse=True)
                target_ch = target_scores[0][1]

                for m in members:
                    if m not in alive:
                        continue
                    if m == ch:
                        alive[m].set_role(Role.MEMBER)
                    alive[m].cluster_head = target_ch
                    alive[m].note_cluster_membership(target_ch, dt_s)
                    clusters[target_ch].append(m)

                del clusters[ch]
                changed = True
                break

        # Normalize cluster lists
        normalized: Dict[int, List[int]] = {}
        for ch, members in clusters.items():
            uniq = []
            seen = set()
            for m in members:
                if m in alive and m not in seen:
                    uniq.append(m)
                    seen.add(m)
            if uniq:
                normalized[ch] = uniq

        # Re-mark CHs after merges.
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
        direct_ch_links = 0
        mean_cross_lht = 0.0

        for ch in target_chs:
            if ch == own_ch or ch not in alive:
                continue
            lht = link_holding_time_s(node, alive[ch], self.comm_radius_m)
            if lht >= self.min_gateway_lht_s:
                reachable.append(ch)
                direct_ch_links += 1
                mean_cross_lht += min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)

        if not reachable:
            return None

        mean_cross_lht /= max(1, len(reachable))
        utility_norm = max(0.0, min(1.0, node.utility))
        energy_norm = node.s1
        stability_norm = 0.5 * node.s3 + 0.5 * node.s4
        load_penalty = _safe_attr(node, "traffic_load_score", 0.0)
        relay_penalty = _safe_attr(node, "relay_load_score", 0.0)

        score = 0.0
        score += self.gateway_crosslink_weight * min(1.0, len(reachable) / 3.0)
        score += self.gateway_utility_weight * utility_norm
        score += self.gateway_energy_weight * energy_norm
        score += self.gateway_stability_weight * stability_norm
        score += self.gateway_multicluster_bonus * max(0.0, min(1.0, (len(reachable) - 1) / 2.0))
        score += self.direct_ch_link_bonus * min(1.0, direct_ch_links / 2.0)
        score += 0.08 * mean_cross_lht

        if node.is_forwarder:
            score += self.forwarder_reuse_bonus

        score -= 0.08 * load_penalty
        score -= 0.08 * relay_penalty

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
        forwarders: Set[int] = set()
        chs = sorted(clusters.keys())

        direct_ch_reach: Dict[int, Set[int]] = {}
        for ch in chs:
            direct_ch_reach[ch] = set()
            for other_ch in chs:
                if other_ch == ch:
                    continue
                lht = link_holding_time_s(alive[ch], alive[other_ch], self.comm_radius_m)
                if lht >= self.min_gateway_lht_s:
                    direct_ch_reach[ch].add(other_ch)

        for ch, members in clusters.items():
            uncovered = (set(chs) - {ch}) - direct_ch_reach.get(ch, set())
            if not uncovered:
                continue

            candidates: List[GatewayCandidate] = []
            for m in members:
                if m == ch or m not in alive:
                    continue
                cand = self._candidate_gateway_score(alive[m], ch, chs, alive)
                if cand is not None:
                    candidates.append(cand)

            selected_local: List[GatewayCandidate] = []
            covered = set(direct_ch_reach.get(ch, set()))

            while candidates and len(selected_local) < 2:
                best_idx = None
                best_value = -1e18

                for idx, cand in enumerate(candidates):
                    adds = set(cand.reachable_chs) - covered
                    if not adds:
                        continue
                    value = cand.score + 0.22 * len(adds)
                    if value > best_value:
                        best_value = value
                        best_idx = idx

                if best_idx is None:
                    break

                chosen = candidates.pop(best_idx)
                selected_local.append(chosen)
                covered.update(chosen.reachable_chs)
                if uncovered <= covered:
                    break

            for cand in selected_local:
                forwarders.add(cand.node_id)

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

        chs = self._elect_cluster_heads(alive, weights, dt_s)
        clusters = self._assign_members(alive, chs, weights, dt_s)
        clusters = self._repair_small_clusters(alive, clusters, dt_s)
        forwarders = self._select_forwarders(alive, clusters)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    """
    Kept simple as baseline.
    """

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
    """
    Kept simple as baseline.
    """

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

            node.utility = (
                0.22 * energy
                + 0.18 * deg_norm
                + 0.18 * lht
                + 0.14 * vel
                + 0.14 * recent
                + 0.14 * load
            )

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