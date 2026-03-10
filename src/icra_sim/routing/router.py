from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    def _hop_delay(self, backbone: bool = False) -> float:
        tx = self.packet_size_bits / max(1.0, self.data_rate_bps)
        queue = 0.7 * self.queueing_delay_s if backbone else self.queueing_delay_s
        return tx + self.per_hop_processing_delay_s + self.mac_contention_delay_s + queue

    def _access_point(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None

        node = nodes[node_id]
        if node.role == Role.CH:
            return node_id
        if node.cluster_head is not None and node.cluster_head in nodes and nodes[node.cluster_head].e_j > 0:
            return node.cluster_head
        return None

    def _same_cluster(self, nodes: Dict[int, Node], a: int, b: int) -> bool:
        na = nodes[a]
        nb = nodes[b]
        cha = na.node_id if na.role == Role.CH else na.cluster_head
        chb = nb.node_id if nb.role == Role.CH else nb.cluster_head
        return cha is not None and cha == chb

    def _backbone_neighbors(self, nodes: Dict[int, Node], node_id: int) -> List[int]:
        node = nodes[node_id]
        nbrs = _alive_neighbors(node, nodes)

        out: List[int] = []
        for j in nbrs:
            other = nodes[j]
            if node.role == Role.CH:
                # CH connects to CHs in range and forwarders in its cluster
                if other.role == Role.CH:
                    out.append(j)
                elif other.role == Role.FORWARDER and other.cluster_head == node_id:
                    out.append(j)
            elif node.role == Role.FORWARDER:
                # forwarder connects to its CH and nodes outside its own cluster
                if node.cluster_head is not None and j == node.cluster_head:
                    out.append(j)
                elif other.role in (Role.CH, Role.FORWARDER):
                    if other.cluster_head != node.cluster_head and other.node_id != node.cluster_head:
                        out.append(j)
        return list(dict.fromkeys(out))

    def _shortest_backbone_path(self, nodes: Dict[int, Node], src: int, dst: int) -> Optional[List[int]]:
        pq: List[Tuple[float, int]] = [(0.0, src)]
        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, Optional[int]] = {src: None}

        while pq:
            cur_d, u = heapq.heappop(pq)
            if u == dst:
                break
            if cur_d > dist.get(u, float("inf")):
                continue

            for v in self._backbone_neighbors(nodes, u):
                nd = cur_d + self._hop_delay(backbone=True)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dst not in prev:
            return None

        rev: List[int] = []
        cur: Optional[int] = dst
        while cur is not None:
            rev.append(cur)
            cur = prev.get(cur)
        rev.reverse()
        return rev

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

        # Step 4: member -> CH
        if src != src_ap:
            if src_ap not in _alive_neighbors(nodes[src], nodes):
                return PacketResult(False, 0, 0.0, tuple(path))
            path.append(src_ap)
            hops += 1
            delay_s += self._hop_delay(backbone=False)

        # intra-cluster
        if self._same_cluster(nodes, src_ap, dst_ap):
            if dst != src_ap:
                if dst not in _alive_neighbors(nodes[src_ap], nodes):
                    return PacketResult(False, hops, delay_s, tuple(path))
                path.append(dst)
                hops += 1
                delay_s += self._hop_delay(backbone=False)
            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))
            return PacketResult(True, hops, delay_s, tuple(path))

        # inter-cluster: backbone route
        bb_path = self._shortest_backbone_path(nodes, src_ap, dst_ap)
        if bb_path is None:
            return PacketResult(False, hops, delay_s, tuple(path))

        for node_id in bb_path[1:]:
            path.append(node_id)
            hops += 1
            delay_s += self._hop_delay(backbone=True)
            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

        # final CH -> dst member
        if dst != dst_ap:
            if dst not in _alive_neighbors(nodes[dst_ap], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            path.append(dst)
            hops += 1
            delay_s += self._hop_delay(backbone=False)

        if hops > self.max_hops:
            return PacketResult(False, hops, delay_s, tuple(path))

        return PacketResult(True, hops, delay_s, tuple(path))