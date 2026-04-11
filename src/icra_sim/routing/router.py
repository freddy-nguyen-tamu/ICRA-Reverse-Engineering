from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

from ..node import Node, Role
from ..utils import clamp, euclidean


def _alive_neighbors(node: Node, nodes: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in nodes and nodes[j].e_j > 0]


@dataclass
class PacketResult:
    delivered: bool
    hops: int
    delay_s: float
    path: Tuple[int, ...]


class Router:
    """Topology-aware routing with lightweight QoS realism.

    The paper specifies the hierarchical forwarding logic, but not a full packet
    error / queueing model. OPNET would naturally produce losses and delay spread;
    this lightweight simulator therefore adds a modest link-quality and congestion
    model on top of the paper's CH/forwarder routing structure.
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
        distance_delay_scale_s: float = 0.0,
        congestion_delay_scale_s: float = 0.0,
        link_success_floor: float = 0.72,
        link_success_exponent: float = 1.2,
        hop_congestion_loss_scale: float = 0.08,
    ) -> None:
        self.comm_radius_m = comm_radius_m
        self.data_rate_bps = max(1, data_rate_kbps * 1000)
        self.packet_size_bits = packet_size_bytes * 8
        self.per_hop_processing_delay_s = per_hop_processing_delay_s
        self.mac_contention_delay_s = mac_contention_delay_s
        self.queueing_delay_s = queueing_delay_s
        self.max_hops = max_hops
        self.distance_delay_scale_s = distance_delay_scale_s
        self.congestion_delay_scale_s = congestion_delay_scale_s
        self.link_success_floor = link_success_floor
        self.link_success_exponent = link_success_exponent
        self.hop_congestion_loss_scale = hop_congestion_loss_scale
        self.backbone_queue_scale = 1.0
        self.success_bias = 0.0
        self.route_success_scale = 1.0

    def configure_protocol(self, backbone_queue_scale: float, backbone_loss_bias: float) -> None:
        self.backbone_queue_scale = max(0.5, backbone_queue_scale)
        self.success_bias = backbone_loss_bias
        # Lightweight proxy for end-to-end route-maintenance quality omitted by
        # the simplified simulator. Positive values help stable protocols a bit;
        # negative values hurt churn-prone protocols more noticeably.
        self.route_success_scale = clamp(1.0 + 1.0 * backbone_loss_bias, 0.82, 1.08)

    def _tx_delay(self) -> float:
        return self.packet_size_bits / self.data_rate_bps

    def _node_load(self, node: Node) -> float:
        if node.role == Role.CH:
            return min(1.0, float(getattr(node, "traffic_load_score", 0.0)))
        if node.role == Role.FORWARDER:
            return min(1.0, float(getattr(node, "relay_load_score", 0.0)))
        return min(1.0, 0.2 * float(getattr(node, "member_tx_count", 0.0)))

    def _hop_delay(self, nodes: Dict[int, Node], a: int, b: int, backbone: bool) -> float:
        delay = self._tx_delay() + self.per_hop_processing_delay_s + self.mac_contention_delay_s
        if backbone:
            delay += self.queueing_delay_s * self.backbone_queue_scale
        dist = euclidean(nodes[a].pos(), nodes[b].pos())
        delay += self.distance_delay_scale_s * min(1.0, dist / max(1e-9, self.comm_radius_m))
        load = 0.5 * (self._node_load(nodes[a]) + self._node_load(nodes[b]))
        instability = 0.5 * (float(getattr(nodes[a], "recent_role_switches", 0.0)) + float(getattr(nodes[b], "recent_role_switches", 0.0)))
        delay += self.congestion_delay_scale_s * load
        delay += 0.0060 * instability
        delay += 0.0040 * 0.5 * (self._fragmentation_penalty(nodes[a]) + self._fragmentation_penalty(nodes[b]))
        return delay

    def _fragmentation_penalty(self, node: Node) -> float:
        size_ratio = float(getattr(node, "cluster_size_ratio", 1.0))
        isolation = float(getattr(node, "cluster_isolation_penalty", 0.0))
        return 0.14 * (1.0 - clamp(size_ratio, 0.0, 1.0)) + 0.08 * clamp(isolation, 0.0, 1.0)

    def _link_success_prob(self, nodes: Dict[int, Node], a: int, b: int) -> float:
        dist = euclidean(nodes[a].pos(), nodes[b].pos())
        dist_ratio = min(1.0, dist / max(1e-9, self.comm_radius_m))
        quality = 1.0 - (dist_ratio ** self.link_success_exponent)
        quality = max(self.link_success_floor, quality)
        load_penalty = self.hop_congestion_loss_scale * 0.5 * (self._node_load(nodes[a]) + self._node_load(nodes[b]))
        instability_penalty = 0.28 * 0.5 * (float(getattr(nodes[a], "recent_role_switches", 0.0)) + float(getattr(nodes[b], "recent_role_switches", 0.0)))
        frag_penalty = 0.5 * (self._fragmentation_penalty(nodes[a]) + self._fragmentation_penalty(nodes[b]))
        return clamp(quality - load_penalty - instability_penalty - frag_penalty + self.success_bias, 0.45, 0.995)

    def _cluster_id(self, nodes: Dict[int, Node], node_id: int) -> Optional[int]:
        if node_id not in nodes or nodes[node_id].e_j <= 0:
            return None
        node = nodes[node_id]
        return node_id if node.role == Role.CH else node.cluster_head

    def _is_one_hop_neighbor(self, nodes: Dict[int, Node], a: int, b: int) -> bool:
        return a in nodes and b in nodes and b in _alive_neighbors(nodes[a], nodes)

    def _choose_from_candidates(
        self,
        nodes: Dict[int, Node],
        candidates: List[int],
        dst: int,
        visited: Tuple[int, ...],
    ) -> Optional[int]:
        filtered = [c for c in candidates if c not in visited and c in nodes and nodes[c].e_j > 0]
        if not filtered:
            return None
        return min(
            filtered,
            key=lambda node_id: (
                euclidean(nodes[node_id].pos(), nodes[dst].pos()),
                self._node_load(nodes[node_id]),
                float(getattr(nodes[node_id], "recent_role_switches", 0.0)),
                node_id,
            ),
        )

    def _next_hop(self, nodes: Dict[int, Node], current: int, dst: int, path: Tuple[int, ...]) -> Optional[int]:
        node = nodes[current]

        if self._is_one_hop_neighbor(nodes, current, dst):
            return dst

        if node.role == Role.CH:
            own_cluster = current
            candidates: List[int] = []
            for nbr in _alive_neighbors(node, nodes):
                other = nodes[nbr]
                if other.role == Role.CH:
                    candidates.append(nbr)
                elif other.role == Role.FORWARDER and other.cluster_head == own_cluster:
                    candidates.append(nbr)
            return self._choose_from_candidates(nodes, candidates, dst, path)

        if node.role == Role.FORWARDER:
            own_cluster = node.cluster_head
            candidates: List[int] = []
            if own_cluster is not None and self._is_one_hop_neighbor(nodes, current, own_cluster):
                candidates.append(own_cluster)
            for nbr in _alive_neighbors(node, nodes):
                other_cluster = self._cluster_id(nodes, nbr)
                if other_cluster is not None and other_cluster != own_cluster:
                    candidates.append(nbr)
            return self._choose_from_candidates(nodes, candidates, dst, path)

        if node.cluster_head is not None and self._is_one_hop_neighbor(nodes, current, node.cluster_head):
            return node.cluster_head
        return None

    def route_packet(self, nodes: Dict[int, Node], src: int, dst: int) -> PacketResult:
        if src not in nodes or dst not in nodes:
            return PacketResult(False, 0, 0.0, tuple())
        if nodes[src].e_j <= 0 or nodes[dst].e_j <= 0:
            return PacketResult(False, 0, 0.0, tuple())
        if src == dst:
            return PacketResult(True, 0, 0.0, (src,))

        path: List[int] = [src]
        hops = 0
        delay_s = 0.0
        current = src
        hop_probs: List[float] = []

        while current != dst and hops < self.max_hops:
            nxt = self._next_hop(nodes, current, dst, tuple(path))
            if nxt is None:
                return PacketResult(False, hops, delay_s, tuple(path))

            backbone = nodes[current].role in (Role.CH, Role.FORWARDER) or nodes[nxt].role in (Role.CH, Role.FORWARDER)
            path.append(nxt)
            hops += 1
            delay_s += self._hop_delay(nodes, current, nxt, backbone=backbone)
            hop_probs.append(self._link_success_prob(nodes, current, nxt))
            current = nxt

        if current != dst:
            return PacketResult(False, hops, delay_s, tuple(path))

        success_prob = 1.0
        for p in hop_probs:
            success_prob *= p
        success_prob = clamp(success_prob * self.route_success_scale, 0.0, 1.0)
        delivered = random.random() < success_prob
        return PacketResult(delivered, hops, delay_s, tuple(path))


