from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from ..utils import clamp, euclidean, mean
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
        join_hysteresis_margin: float = 0.04,
        ch_retain_margin: float = 0.03,
        min_ch_tenure_s: float = 40.0,
        max_cluster_members: int = 12,
        min_gateway_lht_s: float = 3.0,
        degree_soft_target: float = 0.22,
    ):
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = v_max
        self.join_hysteresis_margin = join_hysteresis_margin
        self.ch_retain_margin = ch_retain_margin
        self.min_ch_tenure_s = min_ch_tenure_s
        self.max_cluster_members = max_cluster_members
        self.min_gateway_lht_s = min_gateway_lht_s
        self.degree_soft_target = degree_soft_target

    def cluster(
        self,
        nodes: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        *,
        factors_already_set: bool = False,
    ) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        n_total = len(alive)

        if n_total == 0:
            return ClusterResult(clusters={}, forwarders=set())

        if not factors_already_set:
            for node in alive.values():
                factors = compute_factors(
                    node=node,
                    nodes=alive,
                    comm_radius_m=self.comm_radius_m,
                    n_total=n_total,
                    lht_cap_s=self.lht_cap_s,
                    v_max=self.v_max,
                    degree_soft_target=self.degree_soft_target,
                )
                node.s1 = factors.s1_energy
                node.s2 = factors.s2_degree
                node.s3 = factors.s3_vel_sim
                node.s4 = factors.s4_lht
                node.utility = weighted_utility(factors, weights)

        for i, ni in alive.items():
            ni.neighbor_lht = {}
            ni.neighbor_vel_sim = {}
            for j in _alive_neighbors(ni, alive):
                lht = link_holding_time_s(ni, alive[j], self.comm_radius_m)
                ni.neighbor_lht[j] = lht

        scores: Dict[int, float] = {}
        for i, node in alive.items():
            tenure_bonus = 0.02 if node.role == Role.CH and node.ch_tenure_s >= self.min_ch_tenure_s else 0.0
            scores[i] = node.utility + tenure_bonus

        sorted_ids = sorted(scores.keys(), key=lambda i: (-scores[i], -alive[i].e_j, i))

        clusters: Dict[int, List[int]] = {}
        assigned: Set[int] = set()
        ch_set: Set[int] = set()

        for i in sorted_ids:
            node = alive[i]
            if node.role != Role.CH:
                continue
            if i in assigned:
                continue

            nbrs = _alive_neighbors(node, alive)
            best_neighbor_score = max([scores[j] for j in nbrs] + [-1e9])

            if node.ch_tenure_s >= self.min_ch_tenure_s and scores[i] + self.ch_retain_margin >= best_neighbor_score:
                ch_set.add(i)
                clusters[i] = [i]
                assigned.add(i)

        for i in sorted_ids:
            if i in assigned:
                continue
            node = alive[i]

            covered_by_better_ch = False
            for ch in ch_set:
                if ch not in _alive_neighbors(node, alive):
                    continue
                lht = node.neighbor_lht.get(ch, 0.0)
                if lht >= self.lht_threshold_s and scores[ch] >= scores[i] - self.join_hysteresis_margin:
                    covered_by_better_ch = True
                    break

            if not covered_by_better_ch:
                ch_set.add(i)
                clusters[i] = [i]
                assigned.add(i)

        for i in sorted_ids:
            if i in ch_set:
                continue
            node = alive[i]

            candidate_chs: List[Tuple[float, int]] = []
            node_nbrs = set(_alive_neighbors(node, alive))

            for ch in ch_set:
                if ch not in node_nbrs:
                    continue

                lht = node.neighbor_lht.get(ch, 0.0)
                if lht < self.lht_threshold_s:
                    continue

                load_penalty = len(clusters[ch]) / max(1, self.max_cluster_members)
                affinity = 0.55 * scores[ch] + 0.45 * clamp(lht / self.lht_cap_s, 0.0, 1.0) - 0.12 * load_penalty

                if node.cluster_head == ch:
                    affinity += self.join_hysteresis_margin

                candidate_chs.append((affinity, ch))

            if not candidate_chs:
                ch_set.add(i)
                clusters[i] = [i]
                assigned.add(i)
                continue

            candidate_chs.sort(reverse=True)
            _, best_ch = candidate_chs[0]

            if len(clusters[best_ch]) >= self.max_cluster_members:
                ch_set.add(i)
                clusters[i] = [i]
                assigned.add(i)
            else:
                clusters[best_ch].append(i)
                assigned.add(i)

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            node.reset_clustering_flags()

        for ch, members in clusters.items():
            nodes[ch].set_role(Role.CH)
            nodes[ch].cluster_head = ch
            for m in members:
                if m == ch:
                    continue
                nodes[m].set_role(Role.MEMBER)
                nodes[m].cluster_head = ch

        forwarders = self._select_forwarders(nodes, clusters, ch_set)

        for f in forwarders:
            if f not in ch_set and nodes[f].e_j > 0:
                nodes[f].set_role(Role.FORWARDER)
                nodes[f].is_forwarder = True

        return ClusterResult(clusters=clusters, forwarders=forwarders)

    def _select_forwarders(
        self,
        nodes: Dict[int, Node],
        clusters: Dict[int, List[int]],
        ch_set: Set[int],
    ) -> Set[int]:
        forwarders: Set[int] = set()
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        ch_list = sorted([ch for ch in ch_set if ch in alive])

        for idx, ch_a in enumerate(ch_list):
            for ch_b in ch_list[idx + 1:]:
                if ch_b in _alive_neighbors(alive[ch_a], alive):
                    continue

                best_pair: Optional[Tuple[float, int, int]] = None

                for u in clusters.get(ch_a, []):
                    if u == ch_a or u not in alive:
                        continue
                    if ch_a not in _alive_neighbors(alive[u], alive):
                        continue

                    lht_ua = link_holding_time_s(alive[u], alive[ch_a], self.comm_radius_m)
                    if lht_ua < self.min_gateway_lht_s:
                        continue

                    for v in clusters.get(ch_b, []):
                        if v == ch_b or v not in alive:
                            continue
                        if ch_b not in _alive_neighbors(alive[v], alive):
                            continue
                        if v not in _alive_neighbors(alive[u], alive):
                            continue

                        lht_vb = link_holding_time_s(alive[v], alive[ch_b], self.comm_radius_m)
                        lht_uv = link_holding_time_s(alive[u], alive[v], self.comm_radius_m)
                        if min(lht_vb, lht_uv) < self.min_gateway_lht_s:
                            continue

                        score = (
                            0.40 * min(lht_ua, lht_uv, lht_vb)
                            + 0.20 * alive[u].utility
                            + 0.20 * alive[v].utility
                            + 0.10 * (alive[u].e_j / max(1e-9, alive[u].e0_j))
                            + 0.10 * (alive[v].e_j / max(1e-9, alive[v].e0_j))
                        )

                        candidate = (score, u, v)
                        if best_pair is None or candidate > best_pair:
                            best_pair = candidate

                if best_pair is not None:
                    _, u, v = best_pair
                    forwarders.add(u)
                    forwarders.add(v)

        return forwarders


class WCAClusterer:
    def __init__(self, comm_radius_m: float):
        self.comm_radius_m = comm_radius_m

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        n_total = len(alive)
        scores: Dict[int, float] = {}

        for i, node in alive.items():
            nbrs = _alive_neighbors(node, alive)
            deg = len(nbrs)
            deg_term = abs((deg / max(1, n_total - 1)) - 0.20)
            energy_term = 1.0 - (node.e_j / max(1e-9, node.e0_j))
            mobility_term = abs(node.speed_m_s - node.avg_speed.update(node.speed_m_s)) / max(1e-9, node.speed_m_s + 1.0)
            center_term = mean(
                [euclidean(node.pos(), alive[j].pos()) / max(1e-9, self.comm_radius_m) for j in nbrs]
            ) if nbrs else 1.0
            scores[i] = 0.35 * deg_term + 0.30 * energy_term + 0.20 * mobility_term + 0.15 * center_term

        remaining = set(alive.keys())
        clusters: Dict[int, List[int]] = {}

        while remaining:
            ch = min(remaining, key=lambda i: (scores[i], i))
            members = [m for m in remaining if m == ch or ch in _alive_neighbors(alive[m], alive)]
            clusters[ch] = members
            remaining -= set(members)

        for node in nodes.values():
            if node.e_j > 0:
                node.reset_clustering_flags()

        for ch, members in clusters.items():
            nodes[ch].set_role(Role.CH)
            nodes[ch].cluster_head = ch
            for m in members:
                if m != ch:
                    nodes[m].set_role(Role.MEMBER)
                    nodes[m].cluster_head = ch

        return ClusterResult(clusters=clusters, forwarders=set())


class DCAClusterer:
    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        uncovered: Set[int] = set(alive.keys())
        clusters: Dict[int, List[int]] = {}

        while uncovered:
            ch = max(
                uncovered,
                key=lambda i: (
                    len([j for j in _alive_neighbors(alive[i], alive) if j in uncovered]),
                    alive[i].e_j,
                    -i,
                ),
            )
            covered = {ch} | {j for j in _alive_neighbors(alive[ch], alive) if j in uncovered}
            clusters[ch] = sorted(covered)
            uncovered -= covered

        for node in nodes.values():
            if node.e_j > 0:
                node.reset_clustering_flags()

        for ch, members in clusters.items():
            nodes[ch].set_role(Role.CH)
            nodes[ch].cluster_head = ch
            for m in members:
                if m != ch:
                    nodes[m].set_role(Role.MEMBER)
                    nodes[m].cluster_head = ch

        return ClusterResult(clusters=clusters, forwarders=set())