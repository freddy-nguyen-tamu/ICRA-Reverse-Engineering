from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..node import Node, Role
from ..utils import euclidean


@dataclass
class PacketResult:
    delivered: bool
    hops: int
    delay_s: float
    path: Tuple[int, ...]


class Router:
    """
    Paper-faithful lightweight router:
    - member -> its CH directly
    - CHs and forwarding nodes form the inter-cluster backbone
    - no custom intra-cluster BFS repair
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
        self.tx_time_s = self.packet_size_bits / max(1, self.data_rate_bps)

    def _hop_delay(self) -> float:
        return (
            self.tx_time_s
            + self.per_hop_processing_delay_s
            + self.mac_contention_delay_s
            + self.queueing_delay_s
        )

    def route(self, nodes: Dict[int, Node], src: int, dst: int) -> PacketResult:
        if src == dst:
            return PacketResult(True, 0, 0.0, (src,))

        if nodes[src].e_j <= 0 or nodes[dst].e_j <= 0:
            return PacketResult(False, 0, 0.0, (src,))

        src_node = nodes[src]
        dst_node = nodes[dst]

        # Same cluster and one-hop direct neighbor
        if (
            src_node.cluster_head is not None
            and dst_node.cluster_head is not None
            and src_node.cluster_head == dst_node.cluster_head
            and dst in src_node.neighbors
        ):
            return PacketResult(True, 1, self._hop_delay(), (src, dst))

        # Member must reach CH in one hop
        if src_node.role == Role.MEMBER:
            if src_node.cluster_head is None or src_node.cluster_head not in src_node.neighbors:
                return PacketResult(False, 0, 0.0, (src,))
            path = [src, src_node.cluster_head]
        elif src_node.role in (Role.CH, Role.FORWARDER):
            path = [src]
        else:
            return PacketResult(False, 0, 0.0, (src,))

        current = path[-1]
        hops = len(path) - 1
        visited: Set[int] = set(path)

        # If current can directly reach destination, finish
        if dst in nodes[current].neighbors:
            path.append(dst)
            hops += 1
            return PacketResult(True, hops, hops * self._hop_delay(), tuple(path))

        # Otherwise, route through CH/forwarder backbone toward destination CH
        dst_ch = dst_node.cluster_head if dst_node.cluster_head is not None else dst

        while hops < self.max_hops:
            current_node = nodes[current]

            if current == dst_ch:
                if dst in current_node.neighbors:
                    path.append(dst)
                    hops += 1
                    return PacketResult(True, hops, hops * self._hop_delay(), tuple(path))
                return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))

            candidates: List[int] = []
            for j in current_node.neighbors:
                nj = nodes[j]
                if nj.e_j <= 0 or j in visited:
                    continue
                if nj.role in (Role.CH, Role.FORWARDER):
                    candidates.append(j)

            next_hop = self._closest_to_target(nodes, candidates, dst_ch)
            if next_hop is None:
                return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))

            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            hops += 1

            if dst in nodes[current].neighbors:
                path.append(dst)
                hops += 1
                return PacketResult(True, hops, hops * self._hop_delay(), tuple(path))

        return PacketResult(False, hops, hops * self._hop_delay(), tuple(path))

    def _closest_to_target(
        self,
        nodes: Dict[int, Node],
        candidates: List[int],
        target: int,
    ) -> Optional[int]:
        if not candidates:
            return None

        target_pos = nodes[target].pos()
        best = None
        best_dist = 1e18
        for j in candidates:
            d = euclidean(nodes[j].pos(), target_pos)
            if d < best_dist:
                best_dist = d
                best = j
        return best