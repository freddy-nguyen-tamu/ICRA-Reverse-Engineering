from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from ..utils import euclidean, mean
from .utility import compute_factors, weighted_utility


def _alive_neighbors(node: Node, alive: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in alive and alive[j].e_j > 0]


@dataclass
class ClusterResult:
    clusters: Dict[int, List[int]]
    forwarders: Set[int]


class ICRAClusterer:
    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.08,
        ch_retain_margin: float = 0.10,
        min_ch_tenure_s: float = 12.0,
        max_cluster_members: int = 18,
        min_gateway_lht_s: float = 0.25,
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

    def _compute_utilities(
        self,
        alive: Dict[int, Node],
        weights: Tuple[float, float, float, float],
    ) -> None:
        for node in alive.values():
            factors = compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=self.comm_radius_m,
                n_total=max(1, len(alive)),
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            )
            node.utility = weighted_utility(factors, weights)

    def _retain_existing_chs(self, alive: Dict[int, Node]) -> Set[int]:
        retained: Set[int] = set()

        for i, node in alive.items():
            if node.role != Role.CH:
                continue
            if getattr(node, "ch_tenure_s", 0.0) < self.min_ch_tenure_s:
                retained.add(i)
                continue

            nbrs = _alive_neighbors(node, alive)
            if not nbrs:
                retained.add(i)
                continue

            best_neighbor_utility = max(alive[j].utility for j in nbrs)
            if node.utility + self.ch_retain_margin >= best_neighbor_utility:
                retained.add(i)

        return retained

    def _elect_new_chs(self, alive: Dict[int, Node], retained: Set[int]) -> Set[int]:
        chs: Set[int] = set(retained)

        covered: Set[int] = set()
        for ch in retained:
            covered.add(ch)
            covered.update(_alive_neighbors(alive[ch], alive))

        remaining = [i for i in alive.keys() if i not in covered]

        while remaining:
            best = max(remaining, key=lambda i: (alive[i].utility, -i))
            chs.add(best)
            newly_covered = {best}
            newly_covered.update(_alive_neighbors(alive[best], alive))
            remaining = [i for i in remaining if i not in newly_covered]

        return chs

    def _best_candidate_ch(
        self,
        node: Node,
        alive: Dict[int, Node],
        chs: Set[int],
        current_ch: Optional[int],
        clusters: Dict[int, List[int]],
    ) -> Optional[int]:
        candidates: List[Tuple[float, float, int]] = []

        for ch in chs:
            if ch == node.node_id:
                continue
            if ch not in _alive_neighbors(node, alive):
                continue

            ch_node = alive[ch]
            lht = link_holding_time_s(node, ch_node, self.comm_radius_m)
            if lht < self.lht_threshold_s:
                continue
            if len(clusters.get(ch, [])) >= self.max_cluster_members + 1:
                continue

            candidates.append((ch_node.utility, lht, ch))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_score, _, best_ch = candidates[0]

        if current_ch is not None and current_ch in alive and current_ch in chs:
            if current_ch in _alive_neighbors(node, alive):
                old_lht = link_holding_time_s(node, alive[current_ch], self.comm_radius_m)
                old_score = alive[current_ch].utility
                if old_lht >= self.lht_threshold_s and old_score + self.join_hysteresis_margin >= best_score:
                    return current_ch

        return best_ch

    def _assign_members_with_retention(
        self,
        nodes: Dict[int, Node],
        alive: Dict[int, Node],
        chs: Set[int],
    ) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {ch: [ch] for ch in chs}

        for i, node in alive.items():
            if i in chs:
                continue

            current_ch = node.cluster_head if node.cluster_head in chs else None
            chosen = self._best_candidate_ch(
                node=node,
                alive=alive,
                chs=chs,
                current_ch=current_ch,
                clusters=clusters,
            )

            if chosen is None:
                chosen = i
                chs.add(i)
                clusters.setdefault(i, [i])

            clusters.setdefault(chosen, [])
            if i not in clusters[chosen]:
                clusters[chosen].append(i)

        return clusters

    def _select_forwarders(
        self,
        nodes: Dict[int, Node],
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> Set[int]:
        old_forwarders = {
            i for i, n in nodes.items()
            if n.e_j > 0 and n.role == Role.FORWARDER
        }

        forwarders: Set[int] = set()
        chs = list(clusters.keys())

        for ch_a in chs:
            for ch_b in chs:
                if ch_a >= ch_b:
                    continue

                candidates_a: List[Tuple[float, int]] = []
                candidates_b: List[Tuple[float, int]] = []

                for u in clusters[ch_a]:
                    if u == ch_a:
                        continue
                    if ch_a not in _alive_neighbors(alive[u], alive):
                        continue

                    lht_ua = link_holding_time_s(alive[u], alive[ch_a], self.comm_radius_m)
                    if lht_ua < self.min_gateway_lht_s:
                        continue

                    best_cross = 0.0
                    for v in clusters[ch_b]:
                        if v not in _alive_neighbors(alive[u], alive):
                            continue
                        lht_uv = link_holding_time_s(alive[u], alive[v], self.comm_radius_m)
                        best_cross = max(best_cross, lht_uv)

                    if best_cross >= self.min_gateway_lht_s:
                        score = 0.55 * best_cross + 0.35 * alive[u].utility + 0.10 * alive[u].s1
                        if u in old_forwarders:
                            score += 0.05
                        candidates_a.append((score, u))

                for v in clusters[ch_b]:
                    if v == ch_b:
                        continue
                    if ch_b not in _alive_neighbors(alive[v], alive):
                        continue

                    lht_vb = link_holding_time_s(alive[v], alive[ch_b], self.comm_radius_m)
                    if lht_vb < self.min_gateway_lht_s:
                        continue

                    best_cross = 0.0
                    for u in clusters[ch_a]:
                        if u not in _alive_neighbors(alive[v], alive):
                            continue
                        lht_vu = link_holding_time_s(alive[v], alive[u], self.comm_radius_m)
                        best_cross = max(best_cross, lht_vu)

                    if best_cross >= self.min_gateway_lht_s:
                        score = 0.55 * best_cross + 0.35 * alive[v].utility + 0.10 * alive[v].s1
                        if v in old_forwarders:
                            score += 0.05
                        candidates_b.append((score, v))

                if candidates_a:
                    candidates_a.sort(reverse=True)
                    forwarders.add(candidates_a[0][1])

                if candidates_b:
                    candidates_b.sort(reverse=True)
                    forwarders.add(candidates_b[0][1])

        return forwarders

    def _apply_roles(
        self,
        nodes: Dict[int, Node],
        clusters: Dict[int, List[int]],
        forwarders: Set[int],
        dt_s: float,
    ) -> None:
        alive_ids = {i for i, n in nodes.items() if n.e_j > 0}

        old_roles = {i: nodes[i].role for i in alive_ids}
        old_ch = {i: nodes[i].cluster_head for i in alive_ids}

        new_role: Dict[int, Role] = {}
        new_ch: Dict[int, Optional[int]] = {}

        for ch, members in clusters.items():
            new_role[ch] = Role.CH
            new_ch[ch] = ch
            for m in members:
                if m == ch:
                    continue
                new_role[m] = Role.MEMBER
                new_ch[m] = ch

        for f in forwarders:
            if f in new_role and new_role[f] != Role.CH:
                new_role[f] = Role.FORWARDER

        for i, node in nodes.items():
            if i not in alive_ids:
                continue

            nr = new_role.get(i, Role.MEMBER)
            nch = new_ch.get(i, None)

            changed = (old_roles[i] != nr) or (old_ch[i] != nch)
            if changed:
                node.role_change_count += 1

            node.role = nr
            node.cluster_head = nch
            node.is_forwarder = (nr == Role.FORWARDER)

            if nr == Role.CH:
                if old_roles[i] == Role.CH:
                    node.ch_tenure_s = getattr(node, "ch_tenure_s", 0.0) + dt_s
                else:
                    node.ch_tenure_s = dt_s
            else:
                node.ch_tenure_s = 0.0

            if old_ch[i] == nch and nch is not None:
                node.time_in_cluster_s = getattr(node, "time_in_cluster_s", 0.0) + dt_s
            else:
                node.time_in_cluster_s = dt_s if nch is not None else 0.0

    def cluster(
        self,
        nodes: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        dt_s: float = 1.0,
        factors_already_set: bool = False,
    ) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        if not factors_already_set:
            self._ensure_factors(alive)

        self._compute_utilities(alive, weights)
        retained = self._retain_existing_chs(alive)
        chs = self._elect_new_chs(alive, retained)
        clusters = self._assign_members_with_retention(nodes, alive, chs)
        forwarders = self._select_forwarders(nodes, alive, clusters)
        self._apply_roles(nodes, clusters, forwarders, dt_s=dt_s)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    def __init__(self, comm_radius_m: float):
        self.comm_radius_m = comm_radius_m

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        scores: Dict[int, float] = {}
        n_total = len(alive)

        for i, node in alive.items():
            nbrs = _alive_neighbors(node, alive)
            deg = len(nbrs) / max(1, n_total - 1)
            energy = node.e_j / max(1e-9, node.e0_j)
            avg_dist = mean(euclidean(node.pos(), alive[j].pos()) for j in nbrs) if nbrs else self.comm_radius_m
            avg_dist_norm = min(1.0, avg_dist / max(1e-9, self.comm_radius_m))
            mobility = abs(node.speed_m_s - node.avg_speed.update(node.speed_m_s)) / max(1.0, node.speed_m_s)
            scores[i] = 0.35 * deg + 0.30 * energy - 0.20 * avg_dist_norm - 0.15 * mobility

        remaining = set(alive.keys())
        clusters: Dict[int, List[int]] = {}
        while remaining:
            ch = max(remaining, key=lambda x: (scores[x], -x))
            members = [m for m in remaining if m == ch or ch in _alive_neighbors(alive[m], alive)]
            clusters[ch] = members
            remaining -= set(members)

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for ch, members in clusters.items():
            nodes[ch].role = Role.CH
            nodes[ch].cluster_head = ch
            for m in members:
                if m != ch:
                    nodes[m].role = Role.MEMBER
                    nodes[m].cluster_head = ch

        return ClusterResult(clusters=clusters, forwarders=set())


class DCAClusterer:
    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        remaining = set(alive.keys())
        clusters: Dict[int, List[int]] = {}

        while remaining:
            ch = max(
                remaining,
                key=lambda i: (
                    len(_alive_neighbors(alive[i], alive)),
                    alive[i].e_j / max(1e-9, alive[i].e0_j),
                    -i,
                ),
            )
            members = [m for m in remaining if m == ch or ch in _alive_neighbors(alive[m], alive)]
            clusters[ch] = members
            remaining -= set(members)

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for ch, members in clusters.items():
            nodes[ch].role = Role.CH
            nodes[ch].cluster_head = ch
            for m in members:
                if m != ch:
                    nodes[m].role = Role.MEMBER
                    nodes[m].cluster_head = ch

        return ClusterResult(clusters=clusters, forwarders=set())