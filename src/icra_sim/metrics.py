from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .node import Node


@dataclass
class RunMetrics:
    # clustering
    cluster_creation_time_s: float
    avg_role_changes: float

    # energy
    network_lifetime_s: float  # time of first dead node
    dead_nodes: int

    # QoS
    isolation_clusters: int
    avg_end_to_end_delay_s: float
    packet_delivery_ratio: float


def count_isolation_clusters(clusters: Dict[int, List[int]], threshold: int = 2) -> int:
    """
    Paper definition:
    an isolation cluster is a cluster with no more than two members.
    """
    return sum(1 for members in clusters.values() if len(members) <= threshold)


def avg_role_changes(nodes: Dict[int, Node]) -> float:
    if not nodes:
        return 0.0
    return sum(n.role_change_count for n in nodes.values()) / len(nodes)


def first_dead_time(dead_time_by_node: Dict[int, float], sim_time_s: float) -> float:
    if not dead_time_by_node:
        return sim_time_s
    return min(dead_time_by_node.values())