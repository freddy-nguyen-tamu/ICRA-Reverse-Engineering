from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

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

    def _hop_delay(self) -> float:
        tx = self.packet_size_bits / max(1.0, self.data_rate_bps)
        return tx + self.per_hop_processing_delay_s + self.mac_contention_delay_s + self.queueing_delay_s

    def route_packet(self, nodes: Dict[int, Node], src: int, dst: int) -> PacketResult:
        if src not in nodes or dst not in nodes:
            return PacketResult(False, 0, 0.0, tuple())

        if nodes[src].e_j <= 0 or nodes[dst].e_j <= 0:
            return PacketResult(False, 0, 0.0, tuple())

        if src == dst:
            return PacketResult(True, 0, 0.0, (src,))

        src_entry = self._access_point(nodes, src)
        dst_entry = self._access_point(nodes, dst)

        if src_entry is None or dst_entry is None:
            return PacketResult(False, 0, 0.0, (src,))

        path: List[int] = [src]
        hops = 0

        if src != src_entry:
            if src_entry not in _alive_neighbors(nodes[src], nodes):
                return PacketResult(False, 0, 0.0, tuple(path))
            path.append(src_entry)
            hops += 1

        if src_entry == dst_entry:
            if dst != dst_entry:
                if dst not in _alive_neighbors(nodes[dst_entry], nodes):
                    return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))
                path.append(dst)
                hops += 1
            return PacketResult(True, hops, hops * self._hop_delay(), tuple(path))

        backbone_path = self._shortest_backbone_path(nodes, src_entry, dst_entry)
        if not backbone_path:
            return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))

        for u in backbone_path[1:]:
            path.append(u)
            hops += 1
            if hops > self.max_hops:
                return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))

        if dst != dst_entry:
            if dst not in _alive_neighbors(nodes[dst_entry], nodes):
                return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))
            path.append(dst)
            hops += 1

        if hops > self.max_hops:
            return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))

        return PacketResult(True, hops, hops * self._hop_delay(), tuple(path))

    def _access_point(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        node = nodes[node_id]
        if node.role == Role.CH:
            return node_id
        if node.cluster_head is None:
            return None
        ch = node.cluster_head
        if ch not in nodes or nodes[ch].e_j <= 0:
            return None
        if ch not in _alive_neighbors(node, nodes):
            return None
        return ch

    def _backbone_nodes(self, nodes: Dict[int, Node]) -> Set[int]:
        return {
            i for i, n in nodes.items()
            if n.e_j > 0 and n.role in (Role.CH, Role.FORWARDER)
        }

    def _edge_cost(self, nodes: Dict[int, Node], u: int, v: int) -> float:
        nu = nodes[u]
        nv = nodes[v]
        energy_u = nu.e_j / max(1e-9, nu.e0_j)
        energy_v = nv.e_j / max(1e-9, nv.e0_j)
        stability = 0.5 * (
            nu.neighbor_lht.get(v, 0.0) / 60.0 +
            nv.neighbor_lht.get(u, 0.0) / 60.0
        )
        stability = max(0.0, min(1.0, stability))
        return 1.0 + 0.35 * (1.0 - stability) + 0.15 * (1.0 - min(energy_u, energy_v))

    def _shortest_backbone_path(self, nodes: Dict[int, Node], src: int, dst: int) -> Optional[List[int]]:
        backbone = self._backbone_nodes(nodes)
        if src not in backbone or dst not in backbone:
            return None

        pq: List[Tuple[float, int]] = [(0.0, src)]
        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, Optional[int]] = {src: None}

        while pq:
            cur_dist, u = heapq.heappop(pq)
            if cur_dist > dist.get(u, 1e18):
                continue
            if u == dst:
                break

            for v in _alive_neighbors(nodes[u], nodes):
                if v not in backbone:
                    continue

                nd = cur_dist + self._edge_cost(nodes, u, v)
                if nd + 1e-12 < dist.get(v, 1e18):
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