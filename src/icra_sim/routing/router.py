from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..node import Node, Role


def _alive_neighbors(node: Node, nodes: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in nodes and nodes[j].e_j > 0]


def _safe_attr(node: Node, name: str, default: float = 0.0) -> float:
    value = getattr(node, name, default)
    try:
        return float(value)
    except Exception:
        return default


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

        # Internal routing-shape knobs.
        self.direct_ch_bonus: float = 0.22
        self.forwarder_fallback_penalty: float = 0.18
        self.forwarder_chain_penalty: float = 0.40
        self.low_energy_penalty: float = 0.22
        self.load_penalty_weight: float = 0.55
        self.path_reuse_penalty_weight: float = 0.34
        self.weak_link_penalty_weight: float = 0.28

    def configure_protocol(self, backbone_queue_scale: float, backbone_loss_bias: float) -> None:
        self.backbone_queue_scale = backbone_queue_scale
        self.backbone_loss_bias = backbone_loss_bias

    def _hop_delay(
        self,
        backbone: bool = False,
        stability_bonus: float = 0.0,
        queue_scale_override: Optional[float] = None,
    ) -> float:
        tx = self.packet_size_bits / max(1.0, self.data_rate_bps)
        queue_scale = self.backbone_queue_scale if backbone else 1.0
        if queue_scale_override is not None:
            queue_scale *= max(0.25, queue_scale_override)
        queue = self.queueing_delay_s * queue_scale
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

    def _neighbor_link_quality(self, nodes: Dict[int, Node], u: int, v: int) -> float:
        nu = nodes[u]
        nv = nodes[v]

        lht_uv = 0.0
        if hasattr(nu, "neighbor_lht"):
            lht_uv = max(lht_uv, nu.neighbor_lht.get(v, 0.0))
        if hasattr(nv, "neighbor_lht"):
            lht_uv = max(lht_uv, nv.neighbor_lht.get(u, 0.0))
        lht_norm = min(1.0, lht_uv / 90.0)

        vel = 0.5 * (_safe_attr(nu, "s3", 0.0) + _safe_attr(nv, "s3", 0.0))
        stab = 0.5 * (_safe_attr(nu, "s4", 0.0) + _safe_attr(nv, "s4", 0.0))
        return max(0.0, min(1.0, 0.55 * lht_norm + 0.20 * vel + 0.25 * stab))

    def _backbone_neighbors(self, nodes: Dict[int, Node], node_id: int) -> List[int]:
        node = nodes[node_id]
        nbrs = _alive_neighbors(node, nodes)

        direct_chs: List[int] = []
        own_forwarders: List[int] = []
        foreign_chs: List[int] = []
        foreign_forwarders: List[int] = []

        for j in nbrs:
            other = nodes[j]

            if node.role == Role.CH:
                if other.role == Role.CH:
                    direct_chs.append(j)
                elif other.role == Role.FORWARDER and other.cluster_head == node_id:
                    own_forwarders.append(j)
                elif other.role == Role.FORWARDER:
                    foreign_forwarders.append(j)

            elif node.role == Role.FORWARDER:
                if node.cluster_head is not None and j == node.cluster_head:
                    own_forwarders.append(j)  # own CH always first
                elif other.role == Role.CH:
                    foreign_chs.append(j)
                elif other.role == Role.FORWARDER and other.cluster_head != node.cluster_head:
                    foreign_forwarders.append(j)

        ordered = direct_chs + own_forwarders + foreign_chs + foreign_forwarders
        return list(dict.fromkeys(ordered))

    def _backbone_edge_cost(self, nodes: Dict[int, Node], u: int, v: int) -> float:
        nu = nodes[u]
        nv = nodes[v]

        quality = self._neighbor_link_quality(nodes, u, v)
        min_energy = min(_safe_attr(nu, "s1", 0.0), _safe_attr(nv, "s1", 0.0))

        load_u = _safe_attr(nu, "traffic_load_score", 0.0)
        load_v = _safe_attr(nv, "traffic_load_score", 0.0)
        relay_load_u = _safe_attr(nu, "relay_load_score", 0.0)
        relay_load_v = _safe_attr(nv, "relay_load_score", 0.0)
        reuse_u = _safe_attr(nu, "path_reuse_score", 0.0)
        reuse_v = _safe_attr(nv, "path_reuse_score", 0.0)

        avg_load = 0.30 * (load_u + load_v) + 0.35 * (relay_load_u + relay_load_v) + 0.35 * (reuse_u + reuse_v)
        avg_load = max(0.0, min(1.0, 0.5 * avg_load))

        is_ch_ch = nu.role == Role.CH and nv.role == Role.CH
        is_fwd_fwd = nu.role == Role.FORWARDER and nv.role == Role.FORWARDER

        queue_scale = 1.0 + 0.90 * avg_load
        delay = self._hop_delay(backbone=True, stability_bonus=quality, queue_scale_override=queue_scale)

        cost = delay
        cost *= 1.0 + self.load_penalty_weight * avg_load
        cost *= 1.0 + self.low_energy_penalty * max(0.0, 1.0 - min_energy)
        cost *= 1.0 + self.weak_link_penalty_weight * max(0.0, 1.0 - quality)
        cost *= 1.0 + self.path_reuse_penalty_weight * 0.5 * (reuse_u + reuse_v)

        # De-bias protocol effect: keep this small.
        cost *= 1.0 + 0.12 * self.backbone_loss_bias

        if is_ch_ch:
            cost *= max(0.60, 1.0 - self.direct_ch_bonus * quality)

        if is_fwd_fwd:
            cost *= 1.0 + self.forwarder_chain_penalty

        if (nu.role == Role.CH and nv.role == Role.FORWARDER) or (nu.role == Role.FORWARDER and nv.role == Role.CH):
            if not is_ch_ch:
                cost *= 1.0 + self.forwarder_fallback_penalty * max(0.0, 1.0 - quality)

        return max(1e-9, cost)

    def _shortest_backbone_path(self, nodes: Dict[int, Node], src: int, dst: int) -> Optional[List[int]]:
        pq: List[Tuple[float, int]] = [(0.0, src)]
        dist: Dict[int, float] = {src: 0.0}
        prev: Dict[int, Optional[int]] = {src: None}
        hops_used: Dict[int, int] = {src: 0}

        while pq:
            cur_d, u = heapq.heappop(pq)
            if u == dst:
                break
            if cur_d > dist.get(u, float("inf")):
                continue

            for v in self._backbone_neighbors(nodes, u):
                next_hops = hops_used[u] + 1
                if next_hops > self.max_hops:
                    continue

                nd = cur_d + self._backbone_edge_cost(nodes, u, v)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = u
                    hops_used[v] = next_hops
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
        forwarder_hops = 0

        for i in range(len(path) - 1):
            u = nodes[path[i]]
            v = nodes[path[i + 1]]

            quality = self._neighbor_link_quality(nodes, u.node_id, v.node_id)
            energy_norm = min(_safe_attr(u, "s1", 0.0), _safe_attr(v, "s1", 0.0))

            load = 0.5 * (
                _safe_attr(u, "relay_load_score", _safe_attr(u, "traffic_load_score", 0.0))
                + _safe_attr(v, "relay_load_score", _safe_attr(v, "traffic_load_score", 0.0))
            )
            reuse = 0.5 * (_safe_attr(u, "path_reuse_score", 0.0) + _safe_attr(v, "path_reuse_score", 0.0))
            queue_penalty = min(1.0, 0.65 * load + 0.35 * reuse)

            hop_ok = 0.90
            hop_ok += 0.07 * quality
            hop_ok += 0.03 * energy_norm
            hop_ok -= 0.10 * queue_penalty
            hop_ok -= 0.04 * self.backbone_loss_bias

            if u.role == Role.FORWARDER or v.role == Role.FORWARDER:
                forwarder_hops += 1
                hop_ok -= 0.03

            if u.role == Role.FORWARDER and v.role == Role.FORWARDER:
                hop_ok -= 0.05

            hop_ok = max(0.58, min(0.995, hop_ok))
            prob *= hop_ok

        length_penalty = max(0.0, len(path) - 2) * 0.008
        relay_penalty = forwarder_hops * 0.012
        prob *= max(0.55, 1.0 - length_penalty - relay_penalty)

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
            stability_bonus = self._neighbor_link_quality(nodes, src, src_ap)
            path.append(src_ap)
            hops += 1
            delay_s += self._hop_delay(backbone=False, stability_bonus=stability_bonus)

        if self._same_cluster(nodes, src_ap, dst_ap):
            if dst != src_ap:
                if dst not in _alive_neighbors(nodes[src_ap], nodes):
                    return PacketResult(False, hops, delay_s, tuple(path))
                stability_bonus = self._neighbor_link_quality(nodes, src_ap, dst)
                path.append(dst)
                hops += 1
                delay_s += self._hop_delay(backbone=False, stability_bonus=stability_bonus)

            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

            ok_prob = self._delivery_success_prob(nodes, path)
            return PacketResult(ok_prob >= 0.78, hops, delay_s, tuple(path))

        bb_path = self._shortest_backbone_path(nodes, src_ap, dst_ap)
        if bb_path is None:
            return PacketResult(False, hops, delay_s, tuple(path))

        for node_id in bb_path[1:]:
            prev = path[-1]
            quality = self._neighbor_link_quality(nodes, prev, node_id)
            queue_scale = 1.0 + 0.75 * max(
                _safe_attr(nodes[prev], "traffic_load_score", 0.0),
                _safe_attr(nodes[node_id], "traffic_load_score", 0.0),
            )
            path.append(node_id)
            hops += 1
            delay_s += self._hop_delay(backbone=True, stability_bonus=quality, queue_scale_override=queue_scale)

            if nodes[prev].role == Role.FORWARDER and nodes[node_id].role == Role.FORWARDER:
                delay_s += 0.20 * self.queueing_delay_s

            if hops > self.max_hops:
                return PacketResult(False, hops, delay_s, tuple(path))

        if dst != dst_ap:
            if dst not in _alive_neighbors(nodes[dst_ap], nodes):
                return PacketResult(False, hops, delay_s, tuple(path))
            stability_bonus = self._neighbor_link_quality(nodes, dst_ap, dst)
            path.append(dst)
            hops += 1
            delay_s += self._hop_delay(backbone=False, stability_bonus=stability_bonus)

        if hops > self.max_hops:
            return PacketResult(False, hops, delay_s, tuple(path))

        ok_prob = self._delivery_success_prob(nodes, path)
        return PacketResult(ok_prob >= 0.78, hops, delay_s, tuple(path))