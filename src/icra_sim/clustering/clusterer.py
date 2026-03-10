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
    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.16,
        ch_retain_margin: float = 0.20,
        min_ch_tenure_s: float = 28.0,
        max_cluster_members: int = 16,
        min_gateway_lht_s: float = 0.80,
        min_ch_neighbor_count: int = 2,
        prefer_connected_ch_bonus: float = 0.20,
        isolated_ch_penalty: float = 0.28,
        forwarder_reuse_bonus: float = 0.16,
        gateway_crosslink_weight: float = 0.56,
        gateway_utility_weight: float = 0.14,
        gateway_energy_weight: float = 0.14,
        gateway_stability_weight: float = 0.16,
        gateway_multicluster_bonus: float = 0.16,
        direct_ch_link_bonus: float = 0.10,
        ch_energy_guard_ratio: float = 0.35,
        ch_cooldown_s: float = 24.0,
        recent_ch_penalty_weight: float = 0.18,
        traffic_load_penalty_weight: float = 0.18,
        degree_balance_bonus_weight: float = 0.18,
        tenure_stability_bonus_weight: float = 0.12,
        link_stability_bonus_weight: float = 0.12,
        velocity_stability_bonus_weight: float = 0.10,
        local_degree_target: float = 0.58,
        local_degree_tolerance: float = 0.26,
    ) -> None:
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = v_max

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
        self.ch_cooldown_s = ch_cooldown_s
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

    def _count_direct_ch_neighbors(self, node: Node, alive: Dict[int, Node]) -> int:
        count = 0
        for j in _alive_neighbors(node, alive):
            other = alive[j]
            if other.role == Role.CH:
                count += 1
        return count

    def _ch_quality_bonus(self, node: Node, alive: Dict[int, Node]) -> float:
        nbrs = _alive_neighbors(node, alive)
        if not nbrs:
            return -self.isolated_ch_penalty

        avg_lht = sum(node.neighbor_lht.get(j, 0.0) for j in nbrs) / len(nbrs)
        avg_lht_norm = min(avg_lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)

        avg_vel = sum(node.neighbor_vel_sim.get(j, 0.0) for j in nbrs) / len(nbrs)
        degree_balance = self._degree_balance_score(node, alive)

        tenure_bonus = min(1.0, node.ch_tenure_s / max(1e-9, self.min_ch_tenure_s))
        cooldown_penalty = min(1.0, _safe_attr(node, "ch_cooldown_s", 0.0) / max(1e-9, self.ch_cooldown_s))
        switch_penalty = _safe_attr(node, "recent_role_switches", 0.0)
        traffic_penalty = _safe_attr(node, "traffic_load_score", 0.0)
        direct_ch_degree = _safe_attr(node, "candidate_direct_ch_degree", 0.0)

        bonus = 0.0
        bonus += self.degree_balance_bonus_weight * degree_balance
        bonus += self.tenure_stability_bonus_weight * tenure_bonus
        bonus += self.link_stability_bonus_weight * avg_lht_norm
        bonus += self.velocity_stability_bonus_weight * avg_vel
        bonus += self.prefer_connected_ch_bonus * min(1.0, len(nbrs) / max(1, self.min_ch_neighbor_count + 2))
        bonus += 0.10 * min(1.0, direct_ch_degree / max(1.0, self.min_ch_neighbor_count + 1.0))

        if len(nbrs) < self.min_ch_neighbor_count:
            bonus -= self.isolated_ch_penalty

        if node.s1 < self.ch_energy_guard_ratio:
            bonus -= 0.30

        bonus -= self.recent_ch_penalty_weight * switch_penalty
        bonus -= self.traffic_load_penalty_weight * traffic_penalty
        bonus -= 0.10 * cooldown_penalty
        return bonus

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
        return base + self._ch_quality_bonus(node, alive)

    def _elect_cluster_heads(
        self,
        alive: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        dt_s: float,
    ) -> List[int]:
        for node in alive.values():
            setattr(node, "candidate_direct_ch_degree", 0.0)

        # First compute a provisional base utility.
        provisional: Dict[int, float] = {}
        for i, node in alive.items():
            provisional[i] = weighted_utility(
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

        # Estimate how well each node sits in a potential CH backbone.
        ch_like_threshold = max(0.25, sorted(provisional.values(), reverse=True)[max(0, min(len(provisional) - 1, len(provisional) // 4))])
        for i, node in alive.items():
            count = 0
            for j in _alive_neighbors(node, alive):
                if provisional.get(j, 0.0) >= ch_like_threshold:
                    if node.neighbor_lht.get(j, 0.0) >= self.lht_threshold_s:
                        count += 1
            setattr(node, "candidate_direct_ch_degree", float(count))

        candidates: Dict[int, float] = {}
        for i, node in alive.items():
            candidates[i] = self._candidate_utility(node, weights, alive)
            node.utility = candidates[i]

        elected: Set[int] = set()

        for i, node in alive.items():
            nbrs = _alive_neighbors(node, alive)

            local_best = candidates[i]
            local_best_id = i
            for j in nbrs:
                if candidates[j] > local_best + 1e-12:
                    local_best = candidates[j]
                    local_best_id = j
                elif abs(candidates[j] - local_best) < 1e-12 and j < local_best_id:
                    local_best_id = j

            keep_current = (
                node.role == Role.CH
                and node.node_id == i
                and node.ch_tenure_s >= self.min_ch_tenure_s
            )

            if keep_current:
                if candidates[i] + self.ch_retain_margin >= local_best:
                    elected.add(i)
                    continue

            if local_best_id == i:
                elected.add(i)

        covered = set(elected)
        for ch in list(elected):
            covered.update(_alive_neighbors(alive[ch], alive))

        uncovered = [i for i in alive.keys() if i not in covered]
        uncovered.sort(key=lambda x: candidates[x], reverse=True)
        for i in uncovered:
            elected.add(i)

        return sorted(elected)

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

        ch_backbone_connectivity: Dict[int, float] = {}
        for ch in chs:
            ch_node = alive[ch]
            direct_chs = 0
            stable_direct_chs = 0
            for other_ch in chs:
                if other_ch == ch:
                    continue
                lht = link_holding_time_s(ch_node, alive[other_ch], self.comm_radius_m)
                if lht >= self.lht_threshold_s:
                    direct_chs += 1
                if lht >= self.min_gateway_lht_s:
                    stable_direct_chs += 1
            ch_backbone_connectivity[ch] = min(1.0, 0.45 * direct_chs + 0.55 * stable_direct_chs)

        for node in alive.values():
            node.is_forwarder = False
            if node.node_id in ch_set:
                node.set_role(Role.CH)
                node.cluster_head = node.node_id
                node.note_role_tenure(dt_s)
            else:
                node.set_role(Role.MEMBER)
                node.cluster_head = None

        for i, node in alive.items():
            if i in ch_set:
                continue

            options: List[Tuple[float, int]] = []
            prev_ch = prev_cluster_heads.get(i)

            for ch in chs:
                ch_node = alive[ch]
                lht = link_holding_time_s(node, ch_node, self.comm_radius_m)
                if lht < self.lht_threshold_s:
                    continue
                if len(clusters[ch]) >= self.max_cluster_members:
                    continue

                vel = velocity_similarity(node, ch_node)
                util_gap = abs(ch_node.utility - node.utility)
                load_penalty = _safe_attr(ch_node, "traffic_load_score", 0.0)
                size_penalty = len(clusters[ch]) / max(1, self.max_cluster_members)
                connectivity_bonus = ch_backbone_connectivity.get(ch, 0.0)
                energy_bonus = ch_node.s1

                score = 0.0
                score += 0.33 * ch_node.utility
                score += 0.24 * min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
                score += 0.13 * vel
                score += 0.10 * energy_bonus
                score += 0.14 * connectivity_bonus
                score -= 0.08 * util_gap
                score -= 0.12 * load_penalty
                score -= 0.14 * size_penalty

                if prev_ch is not None and ch == prev_ch:
                    score += self.join_hysteresis_margin

                options.append((score, ch))

            if not options:
                stable_nbrs = 0
                for j in _alive_neighbors(node, alive):
                    if node.neighbor_lht.get(j, 0.0) >= self.lht_threshold_s:
                        stable_nbrs += 1

                # Much stricter fallback: only self-promote when genuinely isolated / unusable.
                if stable_nbrs <= 1 and node.s1 >= self.ch_energy_guard_ratio:
                    node.set_role(Role.CH)
                    node.cluster_head = i
                    clusters[i] = [i]
                else:
                    # Best-effort fallback to nearest reachable CH by raw LHT, even if overloaded.
                    best_ch = None
                    best_lht = -1.0
                    for ch in chs:
                        lht = link_holding_time_s(node, alive[ch], self.comm_radius_m)
                        if lht > best_lht:
                            best_lht = lht
                            best_ch = ch
                    if best_ch is None:
                        node.set_role(Role.CH)
                        node.cluster_head = i
                        clusters[i] = [i]
                    else:
                        clusters[best_ch].append(i)
                        node.cluster_head = best_ch
                        node.note_cluster_membership(best_ch, dt_s)
                continue

            options.sort(reverse=True)
            best_ch = options[0][1]
            clusters[best_ch].append(i)
            node.cluster_head = best_ch
            node.note_cluster_membership(best_ch, dt_s)

        for ch in clusters:
            if ch in alive:
                alive[ch].note_role_tenure(dt_s)

        return clusters

    def _candidate_gateway_score(
        self,
        node: Node,
        own_ch: int,
        target_chs: List[int],
        alive: Dict[int, Node],
    ) -> Optional[GatewayCandidate]:
        if node.e_j <= 0:
            return None
        if node.role == Role.CH:
            return None

        own_ch_node = alive.get(own_ch)
        if own_ch_node is None:
            return None

        own_lht = link_holding_time_s(node, own_ch_node, self.comm_radius_m)
        if own_lht < self.lht_threshold_s:
            return None

        reachable: List[int] = []
        direct_ch_links = 0
        min_cross_lht = 1.0
        mean_cross_lht = 0.0

        for ch in target_chs:
            if ch == own_ch:
                continue
            if ch not in alive:
                continue
            lht = link_holding_time_s(node, alive[ch], self.comm_radius_m)
            if lht >= self.min_gateway_lht_s:
                reachable.append(ch)
                direct_ch_links += 1
                norm = min(lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)
                mean_cross_lht += norm
                min_cross_lht = min(min_cross_lht, norm)

        if not reachable:
            return None

        mean_cross_lht /= max(1, len(reachable))
        utility_norm = max(0.0, min(1.0, node.utility))
        energy_norm = node.s1
        stability_norm = 0.5 * node.s3 + 0.5 * node.s4
        load_penalty = _safe_attr(node, "traffic_load_score", 0.0)
        relay_penalty = _safe_attr(node, "relay_load_score", 0.0)
        own_link_norm = min(own_lht, self.lht_cap_s) / max(1e-9, self.lht_cap_s)

        score = 0.0
        score += self.gateway_crosslink_weight * min(1.0, len(reachable) / 3.0)
        score += self.gateway_utility_weight * utility_norm
        score += self.gateway_energy_weight * energy_norm
        score += self.gateway_stability_weight * stability_norm
        score += self.gateway_multicluster_bonus * max(0.0, min(1.0, (len(reachable) - 1) / 2.0))
        score += self.direct_ch_link_bonus * min(1.0, direct_ch_links / 2.0)
        score += 0.08 * min_cross_lht
        score += 0.10 * mean_cross_lht
        score += 0.10 * own_link_norm

        if node.is_forwarder:
            score += self.forwarder_reuse_bonus

        score -= 0.16 * load_penalty
        score -= 0.16 * relay_penalty
        score -= 0.14 * max(0.0, 1.0 - own_link_norm)

        return GatewayCandidate(
            node_id=node.node_id,
            own_ch=own_ch,
            reachable_chs=tuple(sorted(reachable)),
            score=score,
        )

    def _select_forwarders(
        self,
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> Set[int]:
        forwarders: Set[int] = set()
        chs = sorted(clusters.keys())

        # Precompute direct CH reachability.
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
            candidates: List[GatewayCandidate] = []
            uncovered = (set(chs) - {ch}) - direct_ch_reach.get(ch, set())

            for m in members:
                if m == ch or m not in alive:
                    continue
                cand = self._candidate_gateway_score(alive[m], ch, chs, alive)
                if cand is not None:
                    candidates.append(cand)

            selected_local: List[GatewayCandidate] = []
            covered = set(direct_ch_reach.get(ch, set()))
            used_target_count: Dict[int, int] = {}

            while candidates and len(selected_local) < 3:
                best_idx = None
                best_value = -1e18

                for idx, cand in enumerate(candidates):
                    adds = set(cand.reachable_chs) - covered
                    overlap = len(set(cand.reachable_chs) & covered)
                    redundancy_penalty = sum(used_target_count.get(x, 0) for x in cand.reachable_chs)

                    value = cand.score
                    value += 0.30 * len(adds)
                    value -= 0.10 * overlap
                    value -= 0.10 * redundancy_penalty

                    if not adds and len(selected_local) >= 1:
                        value -= 0.30

                    if value > best_value:
                        best_value = value
                        best_idx = idx

                if best_idx is None:
                    break

                chosen = candidates.pop(best_idx)
                adds = set(chosen.reachable_chs) - covered

                if not adds and len(selected_local) >= 1:
                    continue

                selected_local.append(chosen)
                for x in chosen.reachable_chs:
                    used_target_count[x] = used_target_count.get(x, 0) + 1
                covered.update(chosen.reachable_chs)

                if uncovered <= covered:
                    break

            for cand in selected_local:
                forwarders.add(cand.node_id)

            # If CH is disconnected from all other CHs, try hard to keep at least one bridge.
            if not direct_ch_reach.get(ch) and not selected_local and candidates:
                fallback = max(candidates, key=lambda c: c.score)
                forwarders.add(fallback.node_id)

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
        forwarders = self._select_forwarders(alive, clusters)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    """
    Intentionally less stable than ICRA:
    - no RL
    - weaker retention
    - repeated local elections
    - no good gateway coordination
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

            avg_speed = _safe_attr(node, "avg_speed", None)
            if hasattr(avg_speed, "update"):
                speed_term = min(1.0, node.speed_m_s / 50.0)
            else:
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
    More stable than WCA but still clearly worse than ICRA.
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