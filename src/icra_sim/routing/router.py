from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..node import Node, Role
from ..utils import euclidean


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
    Simple shortest‑hop backbone router with geographic tie‑break.
    No extra per‑protocol shaping.
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
        self.data_rate_bps = max(1, data_rate_kbps * 1000)
        self.packet_size_bits = packet_size_bytes * 8
        self.per_hop_processing_delay_s = per_hop_processing_delay_s
        self.mac_contention_delay_s = mac_contention_delay_s
        self.queueing_delay_s = queueing_delay_s
        self.max_hops = max_hops

        # kept for compatibility only
        self.backbone_queue_scale = 1.0
        self.backbone_loss_bias = 0.0

    def configure_protocol(self, backbone_queue_scale: float, backbone_loss_bias: float) -> None:
        self.backbone_queue_scale = backbone_queue_scale
        self.backbone_loss_bias = backbone_loss_bias

    def _tx_delay(self) -> float:
        return self.packet_size_bits / self.data_rate_bps

    def _hop_delay(self, backbone: bool) -> float:
        d = self._tx_delay() + self.per_hop_processing_delay_s + self.mac_contention_delay_s
        if backbone:
            d += self.queueing_delay_s
        return d

    def _cluster_id(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None
        n = nodes[node_id]
        return node_id if n.role == Role.CH else n.cluster_head

    def _access_point(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None
        n = nodes[node_id]
        if n.role == Role.CH:
            return node_id
        if n.cluster_head is not None and n.cluster_head in nodes and nodes[n.cluster_head].e_j > 0:
            return n.cluster_head
        return None

    def _same_cluster(self, nodes: Dict[int, Node], a: int, b: int) -> bool:
        ca = self._cluster_id(nodes, a)
        cb = self._cluster_id(nodes, b)
        return ca is not None and ca == cb

    def _eligible_backbone_neighbors(self, nodes: Dict[int, Node], node_id: int) -> List[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return []
        node = nodes[node_id]
        nbrs = _alive_neighbors(node, nodes)

        if node.role == Role.CH:
            out: List[int] = []
            for j in nbrs:
                other = nodes[j]
                if other.role == Role.CH:
                    out.append(j)
                elif other.role == Role.FORWARDER and other.cluster_head == node_id:
                    out.append(j)
            return sorted(set(out))

        if node.role == Role.FORWARDER:
            out: List[int] = []
            own_cluster = node.cluster_head
            if own_cluster is not None and own_cluster in nbrs and nodes[own_cluster].e_j > 0:
                out.append(own_cluster)
            for j in nbrs:
                other_cluster = self._cluster_id(nodes, j)
                if other_cluster is not None and other_cluster != own_cluster:
                    out.append(j)
            return sorted(set(out))

        return []

    def _shortest_path(
        self,
        nodes: Dict[int, Node],
        src_ap: int,
        dst_ap: int,
        dst: int,
    ) -> Optional[Tuple[int, ...]]:
        """
        Dijkstra with hop count as primary metric, geographic distance as tie‑break.
        """
        if src_ap not in nodes or dst_ap not in nodes:
            return None
        if nodes[src_ap].e_j <= 0 or nodes[dst_ap].e_j <= 0:
            return None
        if src_ap == dst_ap:
            return (src_ap,)

        # priority queue entries: (hops, distance_to_dst, current, path)
        pq: List[Tuple[int, float, int, Tuple[int, ...]]] = []
        heapq.heappush(pq, (0, euclidean(nodes[src_ap].pos(), nodes[dst].pos()), src_ap, (src_ap,)))
        best: Dict[int, Tuple[int, float]] = {src_ap: (0, 0.0)}  # (hops, dist)

        while pq:
            hops, _, cur, path = heapq.heappop(pq)
            if cur == dst_ap:
                return path
            if hops >= self.max_hops:
                continue
            prev_best = best.get(cur)
            if prev_best and (hops > prev_best[0] or (hops == prev_best[0] and False)):
                # the "_" is distance tie‑break, not needed here because we always use the smallest hops first
                continue

            for nxt in self._eligible_backbone_neighbors(nodes, cur):
                nxt_hops = hops + 1
                nxt_dist = euclidean(nodes[nxt].pos(), nodes[dst].pos())
                prev = best.get(nxt)
                if prev is None or nxt_hops < prev[0] or (nxt_hops == prev[0] and nxt_dist < prev[1]):
                    best[nxt] = (nxt_hops, nxt_dist)
                    heapq.heappush(pq, (nxt_hops, nxt_dist, nxt, path + (nxt,)))
        return None

    def _link_quality(self, nodes: Dict[int, Node], a: int, b: int) -> float:
        d = euclidean(nodes[a].pos(), nodes[b].pos())
        if d > self.comm_radius_m:
            return 0.0
        return 1.0 - (d / self.comm_radius_m) ** 1.2

    def _delivery_probability(self, nodes: Dict[int, Node], path: Tuple[int, ...]) -> float:
        if len(path) <= 1:
            return 1.0
        prob = 1.0
        for i in range(len(path) - 1):
            prob *= 0.97 * self._link_quality(nodes, path[i], path[i + 1])
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

        # Step 1: member -> CH
        if src != src_ap:
            if src_ap not in _alive_neighbors(nodes[src], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            path.append(src_ap)
            hops += 1
            delay_s += self._hop_delay(backbone=False)

        # Same cluster
        if self._same_cluster(nodes, src_ap, dst_ap):
            if dst != path[-1]:
                if dst not in _alive_neighbors(nodes[path[-1]], nodes):
                    return PacketResult(False, hops, delay_s, tuple(path))
                path.append(dst)
                hops += 1
                delay_s += self._hop_delay(backbone=False)
            full_path = tuple(path)
            delivered = random.random() < self._delivery_probability(nodes, full_path)
            return PacketResult(delivered, hops, delay_s, full_path)

        # Inter-cluster backbone
        backbone = self._shortest_path(nodes, src_ap, dst_ap, dst)
        if backbone is None:
            return PacketResult(False, hops, delay_s, tuple(path))

        for nxt in backbone[1:]:
            path.append(nxt)
            hops += 1
            delay_s += self._hop_delay(backbone=True)
            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

        # Final CH -> destination
        if dst != dst_ap:
            if dst not in _alive_neighbors(nodes[dst_ap], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            path.append(dst)
            hops += 1
            delay_s += self._hop_delay(backbone=False)

        if hops > self.max_hops:
            return PacketResult(False, hops, delay_s, tuple(path))

        full_path = tuple(path)
        delivered = random.random() < self._delivery_probability(nodes, full_path)
        return PacketResult(delivered, hops, delay_s, full_path)