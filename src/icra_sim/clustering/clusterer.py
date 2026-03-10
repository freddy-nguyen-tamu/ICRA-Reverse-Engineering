from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from ..utils import clamp, euclidean, mean
from .utility import compute_factors, weighted_utility


@dataclass
class ClusterResult:
    clusters: Dict[int, List[int]]
    forwarders: Set[int]


class ICRAClusterer:
    """Paper-inspired clustering, but still lightweight."""

    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.03,
    ):
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = v_max
        self.join_hysteresis_margin = join_hysteresis_margin

    def cluster(
        self,
        nodes: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        *,
        factors_already_set: bool = False,
    ) -> ClusterResult:
        n_total = len(nodes)

        if not factors_already_set:
            for node in nodes.values():
                factors = compute_factors(
                    node=node,
                    nodes=nodes,
                    comm_radius_m=self.comm_radius_m,
                    n_total=n_total,
                    lht_cap_s=self.lht_cap_s,
                    v_max=self.v_max,
                )
                node.s1 = factors.s1_energy
                node.s2 = factors.s2_degree
                node.s3 = factors.s3_vel_sim
                node.s4 = factors.s4_lht

        for node in nodes.values():
            node.utility = weighted_utility(
                type("F", (), {
                    "s1_energy": node.s1,
                    "s2_degree": node.s2,
                    "s3_vel_sim": node.s3,
                    "s4_lht": node.s4,
                })(),
                weights,
            )

        prev_heads = {n.node_id: n.cluster_head for n in nodes.values()}

        for node in nodes.values():
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        # CH election: local max utility
        for node in nodes.values():
            if node.e_j <= 0:
                continue

            if not node.neighbors:
                node.role = Role.CH
                node.cluster_head = node.node_id
                continue

            best_id = node.node_id
            best_u = node.utility
            for j in node.neighbors:
                if nodes[j].e_j <= 0:
                    continue
                uj = nodes[j].utility
                if uj > best_u or (abs(uj - best_u) < 1e-12 and j < best_id):
                    best_u = uj
                    best_id = j

            if best_id == node.node_id:
                node.role = Role.CH
                node.cluster_head = node.node_id

        ch_ids = {n.node_id for n in nodes.values() if n.role == Role.CH and n.e_j > 0}

        # Members join reachable CH with best utility, but keep previous CH if nearly as good
        for node in nodes.values():
            if node.e_j <= 0 or node.role == Role.CH:
                continue

            best_ch: Optional[int] = None
            best_u = -1e18

            candidates: List[int] = []
            for j in node.neighbors:
                if j not in ch_ids:
                    continue
                lht = link_holding_time_s(node, nodes[j], self.comm_radius_m)
                if lht < self.lht_threshold_s:
                    continue
                candidates.append(j)
                if nodes[j].utility > best_u:
                    best_u = nodes[j].utility
                    best_ch = j

            prev = prev_heads.get(node.node_id)
            if prev is not None and prev in candidates:
                prev_u = nodes[prev].utility
                if prev_u >= best_u - self.join_hysteresis_margin:
                    best_ch = prev

            if best_ch is None:
                node.role = Role.CH
                node.cluster_head = node.node_id
                ch_ids.add(node.node_id)
            else:
                node.role = Role.MEMBER
                node.cluster_head = best_ch

        # Forwarders: members that are 1-hop neighbors of CH and can hear another cluster
        for ch_id in list(ch_ids):
            ch = nodes[ch_id]
            if ch.e_j <= 0:
                continue
            for nbr in ch.neighbors:
                n = nodes[nbr]
                if n.e_j <= 0:
                    continue
                if n.cluster_head != ch_id:
                    continue
                if n.role == Role.CH:
                    continue

                is_gateway = False
                for nn in n.neighbors:
                    if nodes[nn].e_j <= 0:
                        continue
                    other_ch = nodes[nn].cluster_head
                    if other_ch is not None and other_ch != ch_id:
                        is_gateway = True
                        break

                if is_gateway:
                    n.is_forwarder = True
                    n.role = Role.FORWARDER

        clusters: Dict[int, List[int]] = {}
        for node in nodes.values():
            if node.e_j <= 0:
                continue
            ch = node.cluster_head if node.cluster_head is not None else node.node_id
            clusters.setdefault(ch, []).append(node.node_id)

        forwarders = {n.node_id for n in nodes.values() if n.role == Role.FORWARDER and n.e_j > 0}
        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    """Simplified WCA-like baseline, but cleaned up and made consistent."""

    def __init__(self, comm_radius_m: float):
        self.comm_radius_m = comm_radius_m
        self.w_neighbors = 0.70
        self.w_degree_diff = 0.20
        self.w_dist_sum = 0.05
        self.w_speed = 0.05
        self.desired_degree = 10

    def _utility(self, node: Node, nodes: Dict[int, Node]) -> float:
        deg = len(node.neighbors)
        n1 = clamp(deg / 20.0, 0.0, 1.0)
        degree_diff = abs(deg - self.desired_degree) / max(1, self.desired_degree)
        dist_sum = 0.0
        if deg > 0:
            dist_sum = sum(euclidean(node.pos(), nodes[j].pos()) for j in node.neighbors) / (deg * self.comm_radius_m)
        speed_norm = clamp(node.avg_speed.update(node.speed_m_s) / 50.0, 0.0, 1.0)

        return (
            self.w_neighbors * n1
            - self.w_degree_diff * clamp(degree_diff, 0.0, 1.0)
            - self.w_dist_sum * clamp(dist_sum, 0.0, 1.0)
            - self.w_speed * speed_norm
        )

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        for node in nodes.values():
            node.utility = self._utility(node, nodes)

        for node in nodes.values():
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            best_id = node.node_id
            best_u = node.utility
            for j in node.neighbors:
                if nodes[j].e_j <= 0:
                    continue
                uj = nodes[j].utility
                if uj > best_u or (abs(uj - best_u) < 1e-12 and j < best_id):
                    best_u = uj
                    best_id = j
            if best_id == node.node_id:
                node.role = Role.CH
                node.cluster_head = node.node_id

        ch_ids = {n.node_id for n in nodes.values() if n.role == Role.CH and n.e_j > 0}

        for node in nodes.values():
            if node.e_j <= 0 or node.role == Role.CH:
                continue

            best_ch = None
            best_u = -1e18
            for j in node.neighbors:
                if j in ch_ids and nodes[j].utility > best_u:
                    best_u = nodes[j].utility
                    best_ch = j

            if best_ch is None:
                node.role = Role.CH
                node.cluster_head = node.node_id
                ch_ids.add(node.node_id)
            else:
                node.role = Role.MEMBER
                node.cluster_head = best_ch

        clusters: Dict[int, List[int]] = {}
        for node in nodes.values():
            if node.e_j <= 0:
                continue
            ch = node.cluster_head if node.cluster_head is not None else node.node_id
            clusters.setdefault(ch, []).append(node.node_id)

        return ClusterResult(clusters=clusters, forwarders=set())


class DCAClusterer:
    """Simplified DCA-like baseline."""

    def __init__(self, comm_radius_m: float, lht_cap_s: float):
        self.comm_radius_m = comm_radius_m
        self.lht_cap_s = lht_cap_s
        self.w_energy = 0.45
        self.w_lht = 0.45
        self.w_degree = 0.10

    def _utility(self, node: Node, nodes: Dict[int, Node], n_total: int) -> float:
        s1 = clamp(node.e_j / max(1e-9, node.e0_j), 0.0, 1.0)
        deg = len(node.neighbors)
        sdeg = deg / max(1, (n_total - 1))

        if deg == 0:
            slht = 0.0
        else:
            lhts = [link_holding_time_s(node, nodes[j], self.comm_radius_m) for j in node.neighbors]
            slht = clamp(mean(lhts) / max(1e-9, self.lht_cap_s), 0.0, 1.0)

        return self.w_energy * s1 + self.w_lht * slht + self.w_degree * sdeg

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        n_total = len(nodes)

        for node in nodes.values():
            node.utility = self._utility(node, nodes, n_total)

        for node in nodes.values():
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            best_id = node.node_id
            best_u = node.utility
            for j in node.neighbors:
                if nodes[j].e_j <= 0:
                    continue
                uj = nodes[j].utility
                if uj > best_u or (abs(uj - best_u) < 1e-12 and j < best_id):
                    best_u = uj
                    best_id = j
            if best_id == node.node_id:
                node.role = Role.CH
                node.cluster_head = node.node_id

        ch_ids = {n.node_id for n in nodes.values() if n.role == Role.CH and n.e_j > 0}

        for node in nodes.values():
            if node.e_j <= 0 or node.role == Role.CH:
                continue

            best_ch = None
            best_u = -1e18
            for j in node.neighbors:
                if j in ch_ids and nodes[j].utility > best_u:
                    best_u = nodes[j].utility
                    best_ch = j

            if best_ch is None:
                node.role = Role.CH
                node.cluster_head = node.node_id
                ch_ids.add(node.node_id)
            else:
                node.role = Role.MEMBER
                node.cluster_head = best_ch

        clusters: Dict[int, List[int]] = {}
        for node in nodes.values():
            if node.e_j <= 0:
                continue
            ch = node.cluster_head if node.cluster_head is not None else node.node_id
            clusters.setdefault(ch, []).append(node.node_id)

        return ClusterResult(clusters=clusters, forwarders=set())