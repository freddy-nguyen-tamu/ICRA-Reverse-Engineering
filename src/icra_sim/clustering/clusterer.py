from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from ..utils import clamp, euclidean, mean
from .utility import UtilityFactors, compute_factors, weighted_utility


@dataclass
class ClusterResult:
    # CH id -> list of member ids
    clusters: Dict[int, List[int]]
    # nodes = inter-cluster forwarders
    forwarders: Set[int]


class ICRAClusterer:
    """Clustering procedure Section III-C"""

    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
    ):
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = v_max

    def cluster(self, nodes: Dict[int, Node], weights: Tuple[float, float, float, float]) -> ClusterResult:
        n_total = len(nodes)

        for node in nodes.values():
            factors = compute_factors(
                node=node,
                nodes=nodes,
                comm_radius_m=self.comm_radius_m,
                n_total=n_total,
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            )
            node.s1, node.s2, node.s3, node.s4 = (
                factors.s1_energy,
                factors.s2_degree,
                factors.s3_vel_sim,
                factors.s4_lht,
            )
            node.utility = weighted_utility(factors, weights)

        # CH election local max utility
        for node in nodes.values():
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for node in nodes.values():
            if not node.neighbors:
                node.role = Role.CH
                node.cluster_head = node.node_id
                continue

            best_id = node.node_id
            best_u = node.utility
            for j in node.neighbors:
                uj = nodes[j].utility
                if uj > best_u or (abs(uj - best_u) < 1e-12 and j < best_id):
                    best_u = uj
                    best_id = j

            if best_id == node.node_id:
                node.role = Role.CH
                node.cluster_head = node.node_id

        # candidate CHs, cluster formation (members join best CH neighbor with LET threshold)
        ch_ids = {n.node_id for n in nodes.values() if n.role == Role.CH}

        # non-CH nodes: choose CH neighbor with highest utility + adequate LET
        for node in nodes.values():
            if node.role == Role.CH:
                continue

            best_ch: Optional[int] = None
            best_u = -1e18
            for j in node.neighbors:
                if j not in ch_ids:
                    continue
                lht = link_holding_time_s(node, nodes[j], self.comm_radius_m)
                if lht < self.lht_threshold_s:
                    continue
                uj = nodes[j].utility
                if uj > best_u:
                    best_u = uj
                    best_ch = j

            if best_ch is None:
                node.role = Role.CH
                node.cluster_head = node.node_id
                ch_ids.add(node.node_id)
            else:
                node.role = Role.MEMBER
                node.cluster_head = best_ch

        # topology establishment; choose inter-cluster forwarding nodes
        for node in nodes.values():
            if node.role == Role.CH:
                node.is_forwarder = False
                continue
            my_ch = node.cluster_head
            if my_ch is None:
                continue
            for j in node.neighbors:
                other_ch = nodes[j].cluster_head if nodes[j].cluster_head is not None else nodes[j].node_id
                if other_ch != my_ch:
                    node.is_forwarder = True
                    node.role = Role.FORWARDER
                    break

        clusters: Dict[int, List[int]] = {}
        for node in nodes.values():
            ch = node.cluster_head if node.cluster_head is not None else node.node_id
            clusters.setdefault(ch, []).append(node.node_id)

        forwarders = {n.node_id for n in nodes.values() if n.role == Role.FORWARDER}
        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    """Simplified WCA-like baseline"""

    def __init__(self, comm_radius_m: float):
        self.comm_radius_m = comm_radius_m
        # Fig. 10(b) weights: w1=0.7, w2=0.2, w3=0.05, w4=0.05
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

        # Larger utility -> better CH candidate
        return (
            self.w_neighbors * n1
            - self.w_degree_diff * clamp(degree_diff, 0.0, 1.0)
            - self.w_dist_sum * clamp(dist_sum, 0.0, 1.0)
            - self.w_speed * speed_norm
        )

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        for node in nodes.values():
            node.utility = self._utility(node, nodes)

        # CH election (local max)
        for node in nodes.values():
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for node in nodes.values():
            best_id = node.node_id
            best_u = node.utility
            for j in node.neighbors:
                uj = nodes[j].utility
                if uj > best_u or (abs(uj - best_u) < 1e-12 and j < best_id):
                    best_u = uj
                    best_id = j
            if best_id == node.node_id:
                node.role = Role.CH
                node.cluster_head = node.node_id

        ch_ids = {n.node_id for n in nodes.values() if n.role == Role.CH}

        # Join closest CH by utility
        for node in nodes.values():
            if node.role == Role.CH:
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
                node.cluster_head = best_ch
                node.role = Role.MEMBER

        # WCA baseline no explicit forwarder nodes
        clusters: Dict[int, List[int]] = {}
        for node in nodes.values():
            ch = node.cluster_head if node.cluster_head is not None else node.node_id
            clusters.setdefault(ch, []).append(node.node_id)
        return ClusterResult(clusters=clusters, forwarders=set())


class DCAClusterer:
    """Simplified DCA-like baseline"""

    def __init__(self, comm_radius_m: float, lht_cap_s: float):
        self.comm_radius_m = comm_radius_m
        self.lht_cap_s = lht_cap_s
        # Fig. 10(c) weights: w1=0.45,w2=0.45,w3=0.10
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
            best_id = node.node_id
            best_u = node.utility
            for j in node.neighbors:
                uj = nodes[j].utility
                if uj > best_u or (abs(uj - best_u) < 1e-12 and j < best_id):
                    best_u = uj
                    best_id = j
            if best_id == node.node_id:
                node.role = Role.CH
                node.cluster_head = node.node_id

        ch_ids = {n.node_id for n in nodes.values() if n.role == Role.CH}

        for node in nodes.values():
            if node.role == Role.CH:
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
                node.cluster_head = best_ch
                node.role = Role.MEMBER

        clusters: Dict[int, List[int]] = {}
        for node in nodes.values():
            ch = node.cluster_head if node.cluster_head is not None else node.node_id
            clusters.setdefault(ch, []).append(node.node_id)
        return ClusterResult(clusters=clusters, forwarders=set())