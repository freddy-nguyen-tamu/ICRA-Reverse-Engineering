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
    Paper-faithful routing model:
    - member -> own CH directly
    - same-cluster delivery goes directly through CH
    - otherwise packets traverse a backbone made of CHs and forwarding nodes
    - backbone path selection prefers short, stable, forwarder-assisted paths
    - failed delivery returns the actually traversed prefix, not a fictional full path
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

        # kept only for API compatibility
        self.backbone_queue_scale: float = 1.0
        self.backbone_loss_bias: float = 0.0

    def configure_protocol(self, backbone_queue_scale: float, backbone_loss_bias: float) -> None:
        self.backbone_queue_scale = backbone_queue_scale
        self.backbone_loss_bias = backbone_loss_bias

    def _tx_delay(self) -> float:
        return self.packet_size_bits / self.data_rate_bps

    def _hop_delay(self, backbone: bool) -> float:
        delay = self._tx_delay() + self.per_hop_processing_delay_s + self.mac_contention_delay_s
        if backbone:
            delay += self.queueing_delay_s * max(0.7, self.backbone_queue_scale)
        return delay

    def _cluster_id(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None
        node = nodes[node_id]
        return node_id if node.role == Role.CH else node.cluster_head

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
        ca = self._cluster_id(nodes, a)
        cb = self._cluster_id(nodes, b)
        return ca is not None and ca == cb

    def _link_distance(self, nodes: Dict[int, Node], a: int, b: int) -> float:
        if a not in nodes or b not in nodes:
            return float("inf")
        return euclidean(nodes[a].pos(), nodes[b].pos())

    def _link_quality(self, nodes: Dict[int, Node], a: int, b: int) -> float:
        d = self._link_distance(nodes, a, b)
        if d > self.comm_radius_m:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (d / max(1.0, self.comm_radius_m)) ** 1.35))

    def _eligible_backbone_neighbors(self, nodes: Dict[int, Node], node_id: int) -> List[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return []

        node = nodes[node_id]
        nbrs = _alive_neighbors(node, nodes)
        own_cluster = self._cluster_id(nodes, node_id)

        eligible: Set[int] = set()

        if node.role == Role.CH:
            for j in nbrs:
                other = nodes[j]
                if other.role == Role.CH:
                    eligible.add(j)
                elif other.role == Role.FORWARDER and other.cluster_head == node_id:
                    eligible.add(j)
        elif node.role == Role.FORWARDER:
            if own_cluster is not None and own_cluster in nbrs and nodes[own_cluster].e_j > 0:
                eligible.add(own_cluster)
            for j in nbrs:
                other_cluster = self._cluster_id(nodes, j)
                if other_cluster is not None and other_cluster != own_cluster:
                    eligible.add(j)

        return sorted(eligible)

    def _edge_cost(self, nodes: Dict[int, Node], a: int, b: int, dst: int) -> float:
        q = self._link_quality(nodes, a, b)
        geo = self._link_distance(nodes, b, dst) / max(1.0, self.comm_radius_m)

        # Mild preference for forwarder-assisted edges, because the paper's
        # routing explicitly uses forwarding nodes for inter-cluster relay.
        forwarder_bonus = 0.0
        if nodes[a].role == Role.FORWARDER or nodes[b].role == Role.FORWARDER:
            forwarder_bonus = 0.08

        return 1.0 - 0.20 * q - forwarder_bonus + 0.03 * geo

    def _backbone_path(self, nodes: Dict[int, Node], src_ap: int, dst_ap: int, dst: int) -> Optional[Tuple[int, ...]]:
        if src_ap not in nodes or dst_ap not in nodes:
            return None
        if nodes[src_ap].e_j <= 0 or nodes[dst_ap].e_j <= 0:
            return None
        if src_ap == dst_ap:
            return (src_ap,)

        pq: List[Tuple[float, int, int, Tuple[int, ...]]] = []
        heapq.heappush(pq, (0.0, 0, src_ap, (src_ap,)))
        best: Dict[int, Tuple[float, int]] = {src_ap: (0.0, 0)}

        while pq:
            cost, hops, cur, path = heapq.heappop(pq)
            if cur == dst_ap:
                return path
            if hops >= self.max_hops:
                continue

            prev = best.get(cur)
            if prev is not None and (cost > prev[0] + 1e-12 or hops > prev[1]):
                continue

            for nxt in self._eligible_backbone_neighbors(nodes, cur):
                step = self._edge_cost(nodes, cur, nxt, dst)
                nxt_cost = cost + step
                nxt_hops = hops + 1

                prev_best = best.get(nxt)
                if prev_best is None or nxt_cost < prev_best[0] - 1e-12 or (
                    abs(nxt_cost - prev_best[0]) <= 1e-12 and nxt_hops < prev_best[1]
                ):
                    best[nxt] = (nxt_cost, nxt_hops)
                    heapq.heappush(pq, (nxt_cost, nxt_hops, nxt, path + (nxt,)))

        return None

    def _hop_success_probability(self, nodes: Dict[int, Node], a: int, b: int, backbone: bool) -> float:
        q = self._link_quality(nodes, a, b)

        # Similar physical model for all protocols.
        # Backbone hops are slightly less reliable than direct intra-cluster hops,
        # but forwarding-node assistance offsets some of that.
        base = 0.92 + 0.06 * q
        if backbone:
            base -= 0.01
            if nodes[a].role == Role.FORWARDER or nodes[b].role == Role.FORWARDER:
                base += 0.015

        return max(0.0, min(1.0, base))

    def _traverse_path(
        self,
        nodes: Dict[int, Node],
        path: List[int],
        backbone_flags: List[bool],
    ) -> PacketResult:
        traversed: List[int] = [path[0]]
        delay_s = 0.0
        hops = 0

        for idx in range(len(path) - 1):
            a = path[idx]
            b = path[idx + 1]
            backbone = backbone_flags[idx]

            delay_s += self._hop_delay(backbone=backbone)
            hops += 1
            traversed.append(b)

            success_prob = self._hop_success_probability(nodes, a, b, backbone)
            if random.random() >= success_prob:
                # return the actually traversed prefix
                return PacketResult(False, hops, delay_s, tuple(traversed))

        return PacketResult(True, hops, delay_s, tuple(traversed))

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
        backbone_flags: List[bool] = []

        # member -> CH
        if src != src_ap:
            if src_ap not in _alive_neighbors(nodes[src], nodes):
                return PacketResult(False, 0, 0.0, (src,))
            path.append(src_ap)
            backbone_flags.append(False)

        # same-cluster delivery
        if self._same_cluster(nodes, src_ap, dst_ap):
            if dst != path[-1]:
                if dst not in _alive_neighbors(nodes[path[-1]], nodes):
                    return PacketResult(False, len(path) - 1, 0.0, tuple(path))
                path.append(dst)
                backbone_flags.append(False)
            return self._traverse_path(nodes, path, backbone_flags)

        # backbone routing
        backbone = self._backbone_path(nodes, src_ap, dst_ap, dst)
        if backbone is None:
            return PacketResult(False, len(path) - 1, 0.0, tuple(path))

        for nxt in backbone[1:]:
            path.append(nxt)
            backbone_flags.append(True)

        # final CH -> destination
        if dst != dst_ap:
            if dst not in _alive_neighbors(nodes[dst_ap], nodes):
                return PacketResult(False, len(path) - 1, 0.0, tuple(path))
            path.append(dst)
            backbone_flags.append(False)

        return self._traverse_path(nodes, path, backbone_flags)