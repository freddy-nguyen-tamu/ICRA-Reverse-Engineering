from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from ..node import Node, Role
from ..utils import euclidean


@dataclass
class PacketResult:
    delivered: bool
    hops: int
    delay_s: float


class Router:
    """Implements the routing steps in paper Section III-E."""

    def __init__(
        self,
        comm_radius_m: float,
        data_rate_kbps: int,
        packet_size_bytes: int,
        per_hop_processing_delay_s: float,
        max_hops: int = 30,
    ):
        self.comm_radius_m = comm_radius_m
        self.data_rate_bps = data_rate_kbps * 1000
        self.packet_size_bits = packet_size_bytes * 8
        self.per_hop_processing_delay_s = per_hop_processing_delay_s
        self.max_hops = max_hops

        self.tx_time_s = self.packet_size_bits / max(1, self.data_rate_bps)

    def route(self, nodes: Dict[int, Node], src: int, dst: int) -> PacketResult:
        if src == dst:
            return PacketResult(delivered=True, hops=0, delay_s=0.0)

        current = src
        hops = 0
        visited: Set[int] = {src}

        while hops < self.max_hops:
            node = nodes[current]
            if dst in node.neighbors:
                # Direct delivery
                hops += 1
                return PacketResult(delivered=True, hops=hops, delay_s=hops * (self.tx_time_s + self.per_hop_processing_delay_s))

            # Decide next hop according to role
            next_hop: Optional[int] = None

            if node.role == Role.CH:
                # candidates: forwarders in same cluster that are neighbors + neighboring CHs
                candidates = []
                for j in node.neighbors:
                    nj = nodes[j]
                    if nj.role == Role.CH:
                        candidates.append(j)
                    elif nj.role == Role.FORWARDER and nj.cluster_head == node.node_id:
                        candidates.append(j)

                next_hop = self._closest_to_dst(nodes, candidates, dst, visited)

            elif node.role == Role.FORWARDER:
                candidates = []
                # include its CH if reachable
                if node.cluster_head is not None and node.cluster_head in node.neighbors:
                    candidates.append(node.cluster_head)
                # plus neighbors in other clusters
                my_ch = node.cluster_head
                for j in node.neighbors:
                    if nodes[j].cluster_head is None:
                        continue
                    if nodes[j].cluster_head != my_ch:
                        candidates.append(j)

                next_hop = self._closest_to_dst(nodes, candidates, dst, visited)

            else:
                # MEMBER -> forward to CH
                if node.cluster_head is not None and node.cluster_head in node.neighbors:
                    next_hop = node.cluster_head
                else:
                    next_hop = None

            if next_hop is None:
                return PacketResult(delivered=False, hops=hops, delay_s=hops * (self.tx_time_s + self.per_hop_processing_delay_s))

            visited.add(next_hop)
            current = next_hop
            hops += 1

        return PacketResult(delivered=False, hops=hops, delay_s=hops * (self.tx_time_s + self.per_hop_processing_delay_s))

    def _closest_to_dst(self, nodes: Dict[int, Node], candidates: list[int], dst: int, visited: Set[int]) -> Optional[int]:
        if not candidates:
            return None
        dst_pos = nodes[dst].pos()

        best = None
        best_dist = 1e18
        for j in candidates:
            if j in visited:
                continue
            d = euclidean(nodes[j].pos(), dst_pos)
            if d < best_dist:
                best_dist = d
                best = j
        return best