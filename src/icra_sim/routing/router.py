from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..utils import euclidean
from ..node import Node, Role


def _alive_neighbors(node: Node, nodes: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in nodes and nodes[j].e_j > 0]


@dataclass
class PacketResult:
    delivered: bool
    hops: int
    delay_s: float
    path: Tuple[int, ...]


class Router:
    """
    Paper-faithful router:
    - member sends to its CH directly
    - if sender/receiver are in same cluster, CH delivers directly
    - otherwise CH forwards toward the destination through:
        * neighboring CHs, or
        * forwarding nodes in its cluster
    - forwarding node forwards to the closest eligible next hop among:
        * its own CH
        * neighbors in other clusters
    """

    def __init__(
        self,
        comm_radius_m: float,
        data_rate_kbps: int,
        packet_size_bytes: int,
        per_hop_processing_delay_s: float,
        mac_contention_delay_s: float,
        queueing_delay_s: float,
        max_hops: int = 30,
    ):
        self.comm_radius_m = comm_radius_m
        self.data_rate_bps = data_rate_kbps * 1000
        self.packet_size_bits = packet_size_bytes * 8
        self.per_hop_processing_delay_s = per_hop_processing_delay_s
        self.mac_contention_delay_s = mac_contention_delay_s
        self.queueing_delay_s = queueing_delay_s
        self.max_hops = max_hops

        # Retained only for API compatibility with simulator.
        self.backbone_queue_scale: float = 1.0
        self.backbone_loss_bias: float = 0.0

    def configure_protocol(self, backbone_queue_scale: float, backbone_loss_bias: float) -> None:
        self.backbone_queue_scale = backbone_queue_scale
        self.backbone_loss_bias = backbone_loss_bias

    def _tx_delay(self) -> float:
        return self.packet_size_bits / max(1.0, self.data_rate_bps)

    def _hop_delay(self, backbone: bool) -> float:
        delay = self._tx_delay() + self.per_hop_processing_delay_s + self.mac_contention_delay_s
        if backbone:
            delay += self.queueing_delay_s
        return delay

    def _same_cluster(self, nodes: Dict[int, Node], a: int, b: int) -> bool:
        if a not in nodes or b not in nodes:
            return False

        a_ch = a if nodes[a].role == Role.CH else nodes[a].cluster_head
        b_ch = b if nodes[b].role == Role.CH else nodes[b].cluster_head
        return a_ch is not None and a_ch == b_ch

    def _access_point(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None

        node = nodes[node_id]
        if node.role == Role.CH:
            return node_id

        if node.cluster_head is not None and node.cluster_head in nodes and nodes[node.cluster_head].e_j > 0:
            return node.cluster_head

        return None

    def _cluster_id(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None
        return node_id if nodes[node_id].role == Role.CH else nodes[node_id].cluster_head

    def _eligible_backbone_neighbors(self, nodes: Dict[int, Node], node_id: int) -> List[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return []

        node = nodes[node_id]
        nbrs = _alive_neighbors(node, nodes)

        if node.role == Role.CH:
            own_cluster = node_id
            eligible: List[int] = []
            for j in nbrs:
                other = nodes[j]
                if other.role == Role.CH:
                    eligible.append(j)
                elif other.role == Role.FORWARDER and other.cluster_head == own_cluster:
                    eligible.append(j)
            return eligible

        if node.role == Role.FORWARDER:
            own_cluster = node.cluster_head
            eligible = []
            if own_cluster is not None and own_cluster in nodes and own_cluster in nbrs:
                eligible.append(own_cluster)

            for j in nbrs:
                other_cluster = self._cluster_id(nodes, j)
                if other_cluster is not None and other_cluster != own_cluster:
                    eligible.append(j)
            return sorted(set(eligible))

        return []

    def _distance_to_dst(self, nodes: Dict[int, Node], node_id: int, dst: int) -> float:
        if node_id not in nodes or dst not in nodes:
            return float("inf")
        return euclidean(nodes[node_id].pos(), nodes[dst].pos())

    def _backbone_next_hop(self, nodes: Dict[int, Node], cur: int, dst: int, visited: set[int]) -> Optional[int]:
        candidates = [j for j in self._eligible_backbone_neighbors(nodes, cur) if j not in visited]
        if not candidates:
            return None

        candidates.sort(
            key=lambda j: (
                self._distance_to_dst(nodes, j, dst),
                0 if nodes[j].role == Role.CH else 1,
                -nodes[j].e_j,
                j,
            )
        )
        return candidates[0]

    def _shortest_backbone_path(self, nodes: Dict[int, Node], src_ap: int, dst_ap: int) -> Optional[Tuple[int, ...]]:
        """
        Used by the simulator's connectivity/isolation accounting.
        Graph is the paper-style backbone:
        CH <-> neighboring CH
        CH <-> forwarding nodes in its cluster
        FORWARDER <-> its CH
        FORWARDER <-> neighbors in other clusters
        """
        if src_ap not in nodes or dst_ap not in nodes:
            return None
        if nodes[src_ap].e_j <= 0 or nodes[dst_ap].e_j <= 0:
            return None
        if src_ap == dst_ap:
            return (src_ap,)

        pq: List[Tuple[int, int, Tuple[int, ...]]] = [(0, src_ap, (src_ap,))]
        best_cost: Dict[int, int] = {src_ap: 0}

        while pq:
            cost, cur, path = heapq.heappop(pq)
            if cur == dst_ap:
                return path
            if cost > best_cost.get(cur, 10**9):
                continue

            for nxt in self._eligible_backbone_neighbors(nodes, cur):
                nxt_cost = cost + 1
                if nxt_cost < best_cost.get(nxt, 10**9):
                    best_cost[nxt] = nxt_cost
                    heapq.heappush(pq, (nxt_cost, nxt, path + (nxt,)))

        return None

    def _delivery_success_prob(self, path: Tuple[int, ...]) -> float:
        """
        Keep packet success moderate and realistic:
        longer paths reduce success, but not via protocol favoritism.
        """
        if len(path) <= 1:
            return 1.0
        hops = len(path) - 1
        prob = 0.97
        for _ in range(hops):
            prob *= 0.965
        return max(0.0, min(1.0, prob))

    def route_packet(self, nodes: Dict[int, Node], src: int, dst: int) -> PacketResult:
        if src not in nodes or dst not in nodes:
            return PacketResult(False, 0, 0.0, tuple())
        if nodes[src].e_j <= 0 or nodes[dst].e_j <= 0:
            return PacketResult(False, 0, 0.0, tuple())
        if src == dst:
            return PacketResult(True, 0, 0.0, (src,))

        src_ap = self._access_point(nodes, src)
        dst_ap = self._access_point(nodes, dst)
        if src_ap is None or dst_ap is None:
            return PacketResult(False, 0, 0.0, (src,))

        path: List[int] = [src]
        hops = 0
        delay_s = 0.0

        # Step 1: member -> CH directly
        if src != src_ap:
            if src_ap not in _alive_neighbors(nodes[src], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            path.append(src_ap)
            hops += 1
            delay_s += self._hop_delay(backbone=False)

        # Same cluster: CH can deliver directly to destination member/CH.
        if self._same_cluster(nodes, src_ap, dst_ap):
            if dst != path[-1]:
                if dst not in _alive_neighbors(nodes[path[-1]], nodes):
                    return PacketResult(False, hops, delay_s, tuple(path))
                path.append(dst)
                hops += 1
                delay_s += self._hop_delay(backbone=False)

            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

            ok_prob = self._delivery_success_prob(tuple(path))
            return PacketResult(ok_prob >= 0.78, hops, delay_s, tuple(path))

        # Different clusters: greedily forward across the backbone
        cur = src_ap
        visited = {src}
        if src_ap in path:
            visited.add(src_ap)

        while cur != dst_ap and hops < self.max_hops:
            nxt = self._backbone_next_hop(nodes, cur, dst, visited)
            if nxt is None:
                return PacketResult(False, hops, delay_s, tuple(path))

            path.append(nxt)
            visited.add(nxt)
            hops += 1
            delay_s += self._hop_delay(backbone=True)
            cur = nxt

        if cur != dst_ap:
            return PacketResult(False, hops, delay_s, tuple(path))

        # Final CH -> destination member, if needed
        if dst != dst_ap:
            if dst not in _alive_neighbors(nodes[dst_ap], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            path.append(dst)
            hops += 1
            delay_s += self._hop_delay(backbone=False)

        if hops > self.max_hops:
            return PacketResult(False, hops, delay_s, tuple(path))

        ok_prob = self._delivery_success_prob(tuple(path))
        return PacketResult(ok_prob >= 0.78, hops, delay_s, tuple(path))