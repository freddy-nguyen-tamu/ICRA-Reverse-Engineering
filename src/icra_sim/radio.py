from __future__ import annotations

from typing import Dict, List

from .node import Node
from .utils import euclidean


def build_neighbor_tables(nodes: Dict[int, Node], comm_radius_m: float) -> None:
    # populate node.neighbors with IDs within comm_radius
    ids = list(nodes.keys())
    for i in ids:
        nodes[i].neighbors = []
    for idx, i in enumerate(ids):
        for j in ids[idx + 1:]:
            if euclidean(nodes[i].pos(), nodes[j].pos()) <= comm_radius_m:
                nodes[i].neighbors.append(j)
                nodes[j].neighbors.append(i)