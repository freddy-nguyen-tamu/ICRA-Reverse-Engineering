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

        self.backbone_queue_scale: float = 1.0
        self.backbone_loss_bias: float = 0.0

    def configure_protocol(self, backbone_queue_scale: float, backbone_loss_bias: float) -> None:
        self.backbone_queue_scale = backbone_queue_scale
        self.backbone_loss_bias = backbone_loss_bias

    def _hop_delay(self, backbone: bool = False, stability_bonus: float = 0.0) -> float:
        tx = self.packet_size_bits / max(1.0, self.data_rate_bps)
        queue = self.queueing_delay_s * (self.backbone_queue_scale if backbone else 1.0)
        base = tx + self.per_hop_processing_delay_s + self.mac_contention_delay_s + queue
        return max(0.0, base * (1.0 - 0.18 * stability_bonus))

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
                if other.role == Role.CH:
                    out.append(j)
                elif other.role == Role.FORWARDER and other.cluster_head == node_id:
                    out.append(j)

            elif node.role == Role.FORWARDER:
                if node.cluster_head is not None and j == node.cluster_head:
                    out.append(j)
                elif other.role == Role.CH:
                    other_ch = other.node_id
                    if other_ch != node.cluster_head:
                        out.append(j)
                elif other.role == Role.FORWARDER:
                    if other.cluster_head != node.cluster_head:
                        out.append(j)

        return list(dict.fromkeys(out))

    def _backbone_edge_cost(self, nodes: Dict[int, Node], u: int, v: int) -> float:
        nu = nodes[u]
        nv = nodes[v]

        energy_term = min(nu.s1, nv.s1)
        lht_term = min(
            nu.neighbor_lht.get(v, 0.0) if hasattr(nu, "neighbor_lht") else 0.0,
            nv.neighbor_lht.get(u, 0.0) if hasattr(nv, "neighbor_lht") else 0.0,
        )
        lht_norm = min(1.0, lht_term / 90.0)

        stability_bonus = 0.55 * lht_norm + 0.25 * min(nu.s4, nv.s4) + 0.20 * min(nu.s3, nv.s3)

        load_penalty = 0.5 * (
            min(1.0, getattr(nu, "traffic_load_score", 0.0))
            + min(1.0, getattr(nv, "traffic_load_score", 0.0))
        )

        delay = self._hop_delay(backbone=True, stability_bonus=stability_bonus)
        cost = delay
        cost *= (1.0 + 0.50 * load_penalty)
        cost *= (1.0 + 0.30 * self.backbone_loss_bias)
        cost *= (1.0 + 0.10 * (1.0 - energy_term))
        return cost

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
                nd = cur_d + self._backbone_edge_cost(nodes, u, v)
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

    def _delivery_success_prob(self, nodes: Dict[int, Node], path: List[int]) -> float:
        if len(path) <= 1:
            return 1.0

        prob = 1.0
        for i in range(len(path) - 1):
            u = nodes[path[i]]
            v = nodes[path[i + 1]]

            lht_uv = 0.0
            if hasattr(u, "neighbor_lht"):
                lht_uv = max(lht_uv, u.neighbor_lht.get(v.node_id, 0.0))
            if hasattr(v, "neighbor_lht"):
                lht_uv = max(lht_uv, v.neighbor_lht.get(u.node_id, 0.0))

            lht_norm = min(1.0, lht_uv / 90.0)
            energy_norm = min(u.s1, v.s1)
            hop_bias = 0.98 - self.backbone_loss_bias
            hop_ok = max(0.60, hop_bias + 0.08 * lht_norm + 0.04 * energy_norm)
            prob *= hop_ok

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

        if src != src_ap:
            if src_ap not in _alive_neighbors(nodes[src], nodes):
                return PacketResult(False, 0, 0.0, tuple(path))
            stability_bonus = 0.5 * (nodes[src].s3 + nodes[src_ap].s4)
            path.append(src_ap)
            hops += 1
            delay_s += self._hop_delay(backbone=False, stability_bonus=stability_bonus)

        if self._same_cluster(nodes, src_ap, dst_ap):
            if dst != src_ap:
                if dst not in _alive_neighbors(nodes[src_ap], nodes):
                    return PacketResult(False, hops, delay_s, tuple(path))
                stability_bonus = 0.5 * (nodes[src_ap].s4 + nodes[dst].s3)
                path.append(dst)
                hops += 1
                delay_s += self._hop_delay(backbone=False, stability_bonus=stability_bonus)

            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

            ok_prob = self._delivery_success_prob(nodes, path)
            return PacketResult(ok_prob >= 0.80, hops, delay_s, tuple(path))

        bb_path = self._shortest_backbone_path(nodes, src_ap, dst_ap)
        if bb_path is None:
            return PacketResult(False, hops, delay_s, tuple(path))

        for node_id in bb_path[1:]:
            prev = path[-1]
            stability_bonus = min(nodes[prev].s4, nodes[node_id].s4)
            path.append(node_id)
            hops += 1
            delay_s += self._hop_delay(backbone=True, stability_bonus=stability_bonus)
            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

        if dst != dst_ap:
            if dst not in _alive_neighbors(nodes[dst_ap], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            stability_bonus = 0.5 * (nodes[dst_ap].s4 + nodes[dst].s3)
            path.append(dst)
            hops += 1
            delay_s += self._hop_delay(backbone=False, stability_bonus=stability_bonus)

        if hops > self.max_hops:
            return PacketResult(False, hops, delay_s, tuple(path))

        ok_prob = self._delivery_success_prob(nodes, path)
        return PacketResult(ok_prob >= 0.80, hops, delay_s, tuple(path))