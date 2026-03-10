from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..link import link_holding_time_s
from ..node import Node, Role
from ..utils import euclidean, mean
from .utility import compute_factors, weighted_utility


def _alive_neighbors(node: Node, alive: Dict[int, Node]) -> List[int]:
    return [j for j in node.neighbors if j in alive and alive[j].e_j > 0]


def _safe_attr(node: Node, name: str, default: float = 0.0) -> float:
    value = getattr(node, name, default)
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class ClusterResult:
    clusters: Dict[int, List[int]]
    forwarders: Set[int]


@dataclass(frozen=True)
class GatewayCandidate:
    node_id: int
    own_ch: int
    reachable_chs: Tuple[int, ...]
    score: float


class ICRAClusterer:
    def __init__(
        self,
        comm_radius_m: float,
        lht_threshold_s: float,
        lht_cap_s: float,
        v_max: float,
        join_hysteresis_margin: float = 0.08,
        ch_retain_margin: float = 0.10,
        min_ch_tenure_s: float = 12.0,
        max_cluster_members: int = 18,
        min_gateway_lht_s: float = 0.18,
        min_ch_neighbor_count: int = 1,
        prefer_connected_ch_bonus: float = 0.10,
        isolated_ch_penalty: float = 0.12,
        forwarder_reuse_bonus: float = 0.07,
        gateway_crosslink_weight: float = 0.50,
        gateway_utility_weight: float = 0.18,
        gateway_energy_weight: float = 0.12,
        gateway_stability_weight: float = 0.10,
        gateway_multicluster_bonus: float = 0.10,
        direct_ch_link_bonus: float = 0.08,
        ch_energy_guard_ratio: float = 0.35,
        ch_cooldown_s: float = 24.0,
        recent_ch_penalty_weight: float = 0.18,
        traffic_load_penalty_weight: float = 0.18,
        degree_balance_bonus_weight: float = 0.16,
        tenure_stability_bonus_weight: float = 0.10,
        link_stability_bonus_weight: float = 0.10,
        velocity_stability_bonus_weight: float = 0.08,
        local_degree_target: float = 0.58,
        local_degree_tolerance: float = 0.28,
    ) -> None:
        self.comm_radius_m = comm_radius_m
        self.lht_threshold_s = lht_threshold_s
        self.lht_cap_s = lht_cap_s
        self.v_max = v_max

        self.join_hysteresis_margin = join_hysteresis_margin
        self.ch_retain_margin = ch_retain_margin
        self.min_ch_tenure_s = min_ch_tenure_s
        self.max_cluster_members = max_cluster_members
        self.min_gateway_lht_s = min_gateway_lht_s

        self.min_ch_neighbor_count = min_ch_neighbor_count
        self.prefer_connected_ch_bonus = prefer_connected_ch_bonus
        self.isolated_ch_penalty = isolated_ch_penalty

        self.forwarder_reuse_bonus = forwarder_reuse_bonus
        self.gateway_crosslink_weight = gateway_crosslink_weight
        self.gateway_utility_weight = gateway_utility_weight
        self.gateway_energy_weight = gateway_energy_weight
        self.gateway_stability_weight = gateway_stability_weight
        self.gateway_multicluster_bonus = gateway_multicluster_bonus
        self.direct_ch_link_bonus = direct_ch_link_bonus

        self.ch_energy_guard_ratio = ch_energy_guard_ratio
        self.ch_cooldown_s = ch_cooldown_s
        self.recent_ch_penalty_weight = recent_ch_penalty_weight
        self.traffic_load_penalty_weight = traffic_load_penalty_weight
        self.degree_balance_bonus_weight = degree_balance_bonus_weight
        self.tenure_stability_bonus_weight = tenure_stability_bonus_weight
        self.link_stability_bonus_weight = link_stability_bonus_weight
        self.velocity_stability_bonus_weight = velocity_stability_bonus_weight
        self.local_degree_target = local_degree_target
        self.local_degree_tolerance = max(1e-6, local_degree_tolerance)

    def _ensure_factors(self, alive: Dict[int, Node]) -> None:
        for node in alive.values():
            factors = compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=self.comm_radius_m,
                n_total=max(1, len(alive)),
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            )
            node.s1 = factors.s1_energy
            node.s2 = factors.s2_degree
            node.s3 = factors.s3_vel_sim
            node.s4 = factors.s4_lht

    def _degree_balance_score(self, node: Node, alive: Dict[int, Node]) -> float:
        deg = len(_alive_neighbors(node, alive))
        if deg <= 0:
            return 0.0

        max_deg = max(1, max(len(_alive_neighbors(n, alive)) for n in alive.values()))
        local_deg = deg / max_deg
        z = (local_deg - self.local_degree_target) / self.local_degree_tolerance
        return max(0.0, math.exp(-(z * z)))

    def _compute_base_utilities(
        self,
        alive: Dict[int, Node],
        weights: Tuple[float, float, float, float],
    ) -> None:
        for node in alive.values():
            factors = compute_factors(
                node=node,
                nodes=alive,
                comm_radius_m=self.comm_radius_m,
                n_total=max(1, len(alive)),
                lht_cap_s=self.lht_cap_s,
                v_max=self.v_max,
            )

            base = weighted_utility(factors, weights)
            degree_balance = self._degree_balance_score(node, alive)
            traffic_load = _safe_attr(node, "traffic_load_score", 0.0)
            recent_switch = _safe_attr(node, "recent_role_switches", 0.0)
            cooldown_left = _safe_attr(node, "ch_cooldown_s", 0.0)
            cluster_tenure = min(_safe_attr(node, "time_in_cluster_s", 0.0), 40.0) / 40.0

            low_degree_penalty = 0.12 if len(_alive_neighbors(node, alive)) < self.min_ch_neighbor_count else 0.0
            energy_guard_penalty = 0.25 if factors.s1_energy < self.ch_energy_guard_ratio else 0.0
            recent_ch_penalty = self.recent_ch_penalty_weight * min(1.0, cooldown_left / max(1.0, self.ch_cooldown_s))
            load_penalty = self.traffic_load_penalty_weight * min(1.0, traffic_load)
            switch_penalty = 0.10 * min(1.0, recent_switch)

            utility = base
            utility += self.degree_balance_bonus_weight * degree_balance
            utility += self.tenure_stability_bonus_weight * cluster_tenure
            utility += self.link_stability_bonus_weight * factors.s4_lht
            utility += self.velocity_stability_bonus_weight * factors.s3_vel_sim
            utility -= low_degree_penalty
            utility -= energy_guard_penalty
            utility -= recent_ch_penalty
            utility -= load_penalty
            utility -= switch_penalty

            node.s1 = factors.s1_energy
            node.s2 = factors.s2_degree
            node.s3 = factors.s3_vel_sim
            node.s4 = factors.s4_lht
            node.utility = utility

    def _would_be_connected_ch(self, node_id: int, alive: Dict[int, Node]) -> bool:
        nbrs = _alive_neighbors(alive[node_id], alive)
        if len(nbrs) < self.min_ch_neighbor_count:
            return False

        strong_links = 0
        for j in nbrs:
            lht = link_holding_time_s(alive[node_id], alive[j], self.comm_radius_m)
            if lht >= self.min_gateway_lht_s:
                strong_links += 1
        return strong_links >= self.min_ch_neighbor_count

    def _apply_connectivity_bonus(self, alive: Dict[int, Node]) -> None:
        for i, node in alive.items():
            if self._would_be_connected_ch(i, alive):
                node.utility += self.prefer_connected_ch_bonus
            else:
                node.utility -= self.isolated_ch_penalty

    def _retain_existing_chs(self, alive: Dict[int, Node]) -> Set[int]:
        retained: Set[int] = set()

        for i, node in alive.items():
            if node.role != Role.CH:
                continue

            if node.s1 < self.ch_energy_guard_ratio:
                continue

            if getattr(node, "ch_tenure_s", 0.0) < self.min_ch_tenure_s:
                retained.add(i)
                continue

            nbrs = _alive_neighbors(node, alive)
            if not nbrs:
                retained.add(i)
                continue

            best_neighbor_utility = max(alive[j].utility for j in nbrs)
            retain_margin = self.ch_retain_margin

            if self._would_be_connected_ch(i, alive):
                retain_margin += 0.05
            if node.s1 >= 0.55:
                retain_margin += 0.03
            if node.s4 >= 0.55:
                retain_margin += 0.03
            if _safe_attr(node, "traffic_load_score", 0.0) >= 0.75:
                retain_margin -= 0.05

            if node.utility + retain_margin >= best_neighbor_utility:
                retained.add(i)

        return retained

    def _elect_new_chs(self, alive: Dict[int, Node], retained: Set[int]) -> Set[int]:
        chs: Set[int] = set(retained)

        covered: Set[int] = set()
        for ch in retained:
            covered.add(ch)
            covered.update(_alive_neighbors(alive[ch], alive))

        remaining = [i for i in alive.keys() if i not in covered]

        while remaining:
            candidates = [
                i for i in remaining
                if alive[i].s1 >= self.ch_energy_guard_ratio
            ]
            if not candidates:
                candidates = remaining

            best = max(
                candidates,
                key=lambda i: (
                    alive[i].utility,
                    alive[i].s4,
                    alive[i].s3,
                    alive[i].s1,
                    alive[i].s2,
                    -i,
                ),
            )
            chs.add(best)
            newly_covered = {best}
            newly_covered.update(_alive_neighbors(alive[best], alive))
            remaining = [i for i in remaining if i not in newly_covered]

        return chs

    def _ch_connectivity_score(self, ch_id: int, candidate_members: List[int], alive: Dict[int, Node]) -> float:
        ch_node = alive[ch_id]
        score = 0.0

        strong_neighbor_count = 0
        for j in _alive_neighbors(ch_node, alive):
            lht = link_holding_time_s(ch_node, alive[j], self.comm_radius_m)
            if lht >= self.min_gateway_lht_s:
                strong_neighbor_count += 1
                score += min(lht, 10.0) / 10.0

        size_penalty = max(0, len(candidate_members) - self.max_cluster_members // 2) / max(1, self.max_cluster_members)
        score += min(len(candidate_members), 4) * 0.04
        score += min(strong_neighbor_count, 4) * 0.10
        score += 0.08 * ch_node.s1
        score += 0.10 * ch_node.s4
        score += 0.06 * ch_node.s3
        score -= 0.10 * size_penalty
        score -= 0.12 * min(1.0, _safe_attr(ch_node, "traffic_load_score", 0.0))
        return score

    def _best_candidate_ch(
        self,
        node: Node,
        alive: Dict[int, Node],
        chs: Set[int],
        current_ch: Optional[int],
        clusters: Dict[int, List[int]],
    ) -> Optional[int]:
        candidates: List[Tuple[float, float, float, int]] = []

        for ch in chs:
            if ch == node.node_id:
                continue
            if ch not in _alive_neighbors(node, alive):
                continue

            ch_node = alive[ch]
            lht = link_holding_time_s(node, ch_node, self.comm_radius_m)
            if lht < self.lht_threshold_s:
                continue
            if len(clusters.get(ch, [])) >= self.max_cluster_members + 1:
                continue
            if ch_node.s1 < self.ch_energy_guard_ratio:
                continue

            projected_members = list(clusters.get(ch, [])) + [node.node_id]
            conn_score = self._ch_connectivity_score(ch, projected_members, alive)
            size_score = 1.0 - min(1.0, len(projected_members) / max(1, self.max_cluster_members))

            score = (
                ch_node.utility
                + 0.32 * min(lht, 5.0) / 5.0
                + 0.22 * conn_score
                + 0.10 * size_score
                + 0.06 * ch_node.s1
                + 0.08 * ch_node.s4
                + 0.04 * ch_node.s3
            )
            candidates.append((score, ch_node.utility, lht, ch))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_score, _, _, best_ch = candidates[0]

        if current_ch is not None and current_ch in alive and current_ch in chs:
            if current_ch in _alive_neighbors(node, alive):
                old_lht = link_holding_time_s(node, alive[current_ch], self.comm_radius_m)
                projected_members = list(clusters.get(current_ch, []))
                old_conn = self._ch_connectivity_score(current_ch, projected_members, alive)
                old_size_score = 1.0 - min(1.0, len(projected_members) / max(1, self.max_cluster_members))
                old_score = (
                    alive[current_ch].utility
                    + 0.32 * min(old_lht, 5.0) / 5.0
                    + 0.22 * old_conn
                    + 0.10 * old_size_score
                    + 0.06 * alive[current_ch].s1
                    + 0.08 * alive[current_ch].s4
                    + 0.04 * alive[current_ch].s3
                )

                if old_lht >= self.lht_threshold_s and old_score + self.join_hysteresis_margin >= best_score:
                    return current_ch

        return best_ch

    def _assign_members_with_retention(
        self,
        alive: Dict[int, Node],
        chs: Set[int],
    ) -> Dict[int, List[int]]:
        clusters: Dict[int, List[int]] = {ch: [ch] for ch in chs}

        non_ch_nodes = sorted(
            [i for i in alive.keys() if i not in chs],
            key=lambda i: (
                alive[i].s4,
                alive[i].s3,
                alive[i].s1,
                alive[i].utility,
            ),
            reverse=True,
        )

        for i in non_ch_nodes:
            node = alive[i]
            current_ch = node.cluster_head if node.cluster_head in chs else None
            chosen = self._best_candidate_ch(
                node=node,
                alive=alive,
                chs=chs,
                current_ch=current_ch,
                clusters=clusters,
            )

            if chosen is None:
                chosen = i
                chs.add(i)
                clusters.setdefault(i, [i])

            clusters.setdefault(chosen, [])
            if i not in clusters[chosen]:
                clusters[chosen].append(i)

        return clusters

    def _ch_direct_neighbors(self, chs: List[int], alive: Dict[int, Node]) -> Dict[int, Set[int]]:
        graph: Dict[int, Set[int]] = {ch: set() for ch in chs}
        for i, ch_a in enumerate(chs):
            for ch_b in chs[i + 1:]:
                if ch_b in _alive_neighbors(alive[ch_a], alive):
                    lht = link_holding_time_s(alive[ch_a], alive[ch_b], self.comm_radius_m)
                    if lht >= self.min_gateway_lht_s:
                        graph[ch_a].add(ch_b)
                        graph[ch_b].add(ch_a)
        return graph

    def _reachable_other_chs_for_member(
        self,
        node_id: int,
        own_ch: int,
        clusters: Dict[int, List[int]],
        alive: Dict[int, Node],
    ) -> Set[int]:
        node = alive[node_id]
        if own_ch not in _alive_neighbors(node, alive):
            return set()

        lht_to_ch = link_holding_time_s(node, alive[own_ch], self.comm_radius_m)
        if lht_to_ch < self.min_gateway_lht_s:
            return set()

        reachable: Set[int] = set()
        for other_ch, members in clusters.items():
            if other_ch == own_ch:
                continue

            if other_ch in _alive_neighbors(node, alive):
                lht = link_holding_time_s(node, alive[other_ch], self.comm_radius_m)
                if lht >= self.min_gateway_lht_s:
                    reachable.add(other_ch)
                    continue

            for v in members:
                if v == node_id:
                    continue
                if v not in _alive_neighbors(node, alive):
                    continue
                lht = link_holding_time_s(node, alive[v], self.comm_radius_m)
                if lht >= self.min_gateway_lht_s:
                    reachable.add(other_ch)
                    break

        return reachable

    def _gateway_candidate_score(
        self,
        node_id: int,
        own_ch: int,
        reachable_chs: Set[int],
        alive: Dict[int, Node],
        old_forwarders: Set[int],
    ) -> float:
        node = alive[node_id]
        lht_to_ch = link_holding_time_s(node, alive[own_ch], self.comm_radius_m)
        stability_term = min(getattr(node, "time_in_cluster_s", 0.0), 20.0) / 20.0
        best_cross = 0.0

        for other_ch in reachable_chs:
            if other_ch in _alive_neighbors(node, alive):
                lht = link_holding_time_s(node, alive[other_ch], self.comm_radius_m)
                best_cross = max(best_cross, lht)

        score = (
            self.gateway_crosslink_weight * min(best_cross, 10.0) / 10.0
            + self.gateway_utility_weight * node.utility
            + self.gateway_energy_weight * node.s1
            + self.gateway_stability_weight * stability_term
            + self.gateway_multicluster_bonus * min(len(reachable_chs), 3)
            + 0.07 * min(lht_to_ch, 5.0) / 5.0
        )

        if node_id in old_forwarders:
            score += self.forwarder_reuse_bonus

        score -= 0.08 * min(1.0, _safe_attr(node, "traffic_load_score", 0.0))
        return score

    def _build_gateway_candidates(
        self,
        nodes: Dict[int, Node],
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> List[GatewayCandidate]:
        old_forwarders = {
            i for i, n in nodes.items()
            if n.e_j > 0 and n.role == Role.FORWARDER
        }

        candidates: List[GatewayCandidate] = []

        for own_ch, members in clusters.items():
            for node_id in members:
                if node_id == own_ch:
                    continue

                reachable_chs = self._reachable_other_chs_for_member(
                    node_id=node_id,
                    own_ch=own_ch,
                    clusters=clusters,
                    alive=alive,
                )
                if not reachable_chs:
                    continue

                score = self._gateway_candidate_score(
                    node_id=node_id,
                    own_ch=own_ch,
                    reachable_chs=reachable_chs,
                    alive=alive,
                    old_forwarders=old_forwarders,
                )

                candidates.append(
                    GatewayCandidate(
                        node_id=node_id,
                        own_ch=own_ch,
                        reachable_chs=tuple(sorted(reachable_chs)),
                        score=score,
                    )
                )

        return candidates

    def _components(self, chs: List[int], graph: Dict[int, Set[int]]) -> List[Set[int]]:
        seen: Set[int] = set()
        comps: List[Set[int]] = []

        for ch in chs:
            if ch in seen:
                continue
            stack = [ch]
            comp: Set[int] = set()
            seen.add(ch)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in graph.get(u, set()):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)

        return comps

    def _graph_with_candidate(
        self,
        graph: Dict[int, Set[int]],
        candidate: GatewayCandidate,
    ) -> Dict[int, Set[int]]:
        new_graph = {k: set(v) for k, v in graph.items()}
        for other_ch in candidate.reachable_chs:
            if other_ch == candidate.own_ch:
                continue
            new_graph.setdefault(candidate.own_ch, set()).add(other_ch)
            new_graph.setdefault(other_ch, set()).add(candidate.own_ch)
        return new_graph

    def _select_forwarders(
        self,
        nodes: Dict[int, Node],
        alive: Dict[int, Node],
        clusters: Dict[int, List[int]],
    ) -> Set[int]:
        chs = sorted(clusters.keys())
        if not chs:
            return set()

        direct_graph = self._ch_direct_neighbors(chs, alive)
        graph = {k: set(v) for k, v in direct_graph.items()}

        old_forwarders = {
            i for i, n in nodes.items()
            if n.e_j > 0 and n.role == Role.FORWARDER
        }

        selected: Set[int] = set()
        used_chs: Set[int] = set()
        candidates = self._build_gateway_candidates(nodes, alive, clusters)

        while True:
            cur_components = self._components(chs, graph)
            cur_count = len(cur_components)

            best_pick: Optional[Tuple[int, int, float, GatewayCandidate]] = None

            for cand in candidates:
                if cand.node_id in selected:
                    continue
                if cand.own_ch in used_chs:
                    continue

                new_graph = self._graph_with_candidate(graph, cand)
                new_components = self._components(chs, new_graph)
                new_count = len(new_components)

                component_gain = cur_count - new_count
                new_neighbors = len(set(cand.reachable_chs) - graph.get(cand.own_ch, set()))
                score = cand.score

                key = (component_gain, new_neighbors, score, cand)
                if best_pick is None or (component_gain, new_neighbors, score) > (
                    best_pick[0],
                    best_pick[1],
                    best_pick[2],
                ):
                    best_pick = key

            if best_pick is None:
                break

            component_gain, new_neighbors, _, cand = best_pick
            if component_gain <= 0 and new_neighbors <= 0:
                break

            selected.add(cand.node_id)
            used_chs.add(cand.own_ch)
            graph = self._graph_with_candidate(graph, cand)

        comps = self._components(chs, graph)
        isolated_or_weak = {
            ch for comp in comps for ch in comp if len(comp) == 1 and len(graph.get(ch, set())) == 0
        }

        for ch in chs:
            if ch not in isolated_or_weak:
                continue

            best_fallback: Optional[Tuple[float, GatewayCandidate]] = None
            for cand in candidates:
                if cand.own_ch != ch:
                    continue
                if cand.node_id in selected:
                    continue

                new_neighbors = len(set(cand.reachable_chs) - graph.get(ch, set()))
                if new_neighbors <= 0:
                    continue

                fallback_score = cand.score + 0.15 * new_neighbors
                if best_fallback is None or fallback_score > best_fallback[0]:
                    best_fallback = (fallback_score, cand)

            if best_fallback is not None:
                cand = best_fallback[1]
                selected.add(cand.node_id)
                graph = self._graph_with_candidate(graph, cand)

        for ch in chs:
            if len(graph.get(ch, set())) >= 2:
                continue

            best_extra: Optional[Tuple[float, GatewayCandidate]] = None
            for cand in candidates:
                if cand.own_ch != ch:
                    continue
                if cand.node_id in selected:
                    continue
                if cand.node_id not in old_forwarders:
                    continue

                new_neighbors = len(set(cand.reachable_chs) - graph.get(ch, set()))
                if new_neighbors <= 0:
                    continue

                extra_score = cand.score + 0.20 * new_neighbors
                if best_extra is None or extra_score > best_extra[0]:
                    best_extra = (extra_score, cand)

            if best_extra is not None:
                cand = best_extra[1]
                selected.add(cand.node_id)
                graph = self._graph_with_candidate(graph, cand)

        return selected

    def _apply_roles(
        self,
        nodes: Dict[int, Node],
        clusters: Dict[int, List[int]],
        forwarders: Set[int],
        dt_s: float,
    ) -> None:
        alive_ids = {i for i, n in nodes.items() if n.e_j > 0}

        old_roles = {i: nodes[i].role for i in alive_ids}
        old_ch = {i: nodes[i].cluster_head for i in alive_ids}

        new_role: Dict[int, Role] = {}
        new_ch: Dict[int, Optional[int]] = {}

        for ch, members in clusters.items():
            new_role[ch] = Role.CH
            new_ch[ch] = ch
            for m in members:
                if m == ch:
                    continue
                new_role[m] = Role.MEMBER
                new_ch[m] = ch

        for f in forwarders:
            if f in new_role and new_role[f] != Role.CH:
                new_role[f] = Role.FORWARDER

        for i, node in nodes.items():
            if i not in alive_ids:
                continue

            nr = new_role.get(i, Role.MEMBER)
            nch = new_ch.get(i, None)

            changed = (old_roles[i] != nr) or (old_ch[i] != nch)
            if changed:
                node.role_change_count += 1
                setattr(node, "recent_role_switches", min(1.0, _safe_attr(node, "recent_role_switches", 0.0) + 0.35))
            else:
                setattr(node, "recent_role_switches", max(0.0, _safe_attr(node, "recent_role_switches", 0.0) * 0.80))

            if old_roles[i] == Role.CH and nr != Role.CH:
                setattr(node, "ch_cooldown_s", self.ch_cooldown_s)
            else:
                setattr(node, "ch_cooldown_s", max(0.0, _safe_attr(node, "ch_cooldown_s", 0.0) - dt_s))

            node.role = nr
            node.cluster_head = nch
            node.is_forwarder = (nr == Role.FORWARDER)

            if nr == Role.CH:
                if old_roles[i] == Role.CH:
                    node.ch_tenure_s = getattr(node, "ch_tenure_s", 0.0) + dt_s
                else:
                    node.ch_tenure_s = dt_s
            else:
                node.ch_tenure_s = 0.0

            if old_ch[i] == nch and nch is not None:
                node.time_in_cluster_s = getattr(node, "time_in_cluster_s", 0.0) + dt_s
            else:
                node.time_in_cluster_s = dt_s if nch is not None else 0.0

    def cluster(
        self,
        nodes: Dict[int, Node],
        weights: Tuple[float, float, float, float],
        dt_s: float = 1.0,
        factors_already_set: bool = False,
    ) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        if not factors_already_set:
            self._ensure_factors(alive)

        self._compute_base_utilities(alive, weights)
        self._apply_connectivity_bonus(alive)

        retained = self._retain_existing_chs(alive)
        chs = self._elect_new_chs(alive, retained)
        clusters = self._assign_members_with_retention(alive, chs)
        forwarders = self._select_forwarders(nodes, alive, clusters)
        self._apply_roles(nodes, clusters, forwarders, dt_s=dt_s)

        return ClusterResult(clusters=clusters, forwarders=forwarders)


class WCAClusterer:
    def __init__(self, comm_radius_m: float):
        self.comm_radius_m = comm_radius_m

    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        old_roles = {i: nodes[i].role for i in alive}
        old_ch = {i: nodes[i].cluster_head for i in alive}

        scores: Dict[int, float] = {}
        n_total = len(alive)

        for i, node in alive.items():
            nbrs = _alive_neighbors(node, alive)
            deg = len(nbrs) / max(1, n_total - 1)
            energy = node.e_j / max(1e-9, node.e0_j)
            avg_dist = mean(euclidean(node.pos(), alive[j].pos()) for j in nbrs) if nbrs else self.comm_radius_m
            avg_dist_norm = min(1.0, avg_dist / max(1e-9, self.comm_radius_m))
            mobility = abs(node.speed_m_s - node.avg_speed.update(node.speed_m_s)) / max(1.0, node.speed_m_s)
            scores[i] = 0.35 * deg + 0.30 * energy - 0.20 * avg_dist_norm - 0.15 * mobility

        remaining = set(alive.keys())
        clusters: Dict[int, List[int]] = {}
        while remaining:
            ch = max(remaining, key=lambda x: (scores[x], -x))
            members = [m for m in remaining if m == ch or ch in _alive_neighbors(alive[m], alive)]
            clusters[ch] = members
            remaining -= set(members)

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for ch, members in clusters.items():
            nodes[ch].role = Role.CH
            nodes[ch].cluster_head = ch
            for m in members:
                if m != ch:
                    nodes[m].role = Role.MEMBER
                    nodes[m].cluster_head = ch

        for i in alive:
            if nodes[i].role != old_roles[i] or nodes[i].cluster_head != old_ch[i]:
                nodes[i].role_change_count += 1

        return ClusterResult(clusters=clusters, forwarders=set())


class DCAClusterer:
    def cluster(self, nodes: Dict[int, Node]) -> ClusterResult:
        alive = {i: n for i, n in nodes.items() if n.e_j > 0}
        if not alive:
            return ClusterResult(clusters={}, forwarders=set())

        old_roles = {i: nodes[i].role for i in alive}
        old_ch = {i: nodes[i].cluster_head for i in alive}

        remaining = set(alive.keys())
        clusters: Dict[int, List[int]] = {}

        while remaining:
            ch = max(
                remaining,
                key=lambda i: (
                    len(_alive_neighbors(alive[i], alive)),
                    alive[i].e_j / max(1e-9, alive[i].e0_j),
                    -i,
                ),
            )
            members = [m for m in remaining if m == ch or ch in _alive_neighbors(alive[m], alive)]
            clusters[ch] = members
            remaining -= set(members)

        for node in nodes.values():
            if node.e_j <= 0:
                continue
            node.role = Role.MEMBER
            node.cluster_head = None
            node.is_forwarder = False

        for ch, members in clusters.items():
            nodes[ch].role = Role.CH
            nodes[ch].cluster_head = ch
            for m in members:
                if m != ch:
                    nodes[m].role = Role.MEMBER
                    nodes[m].cluster_head = ch

        for i in alive:
            if nodes[i].role != old_roles[i] or nodes[i].cluster_head != old_ch[i]:
                nodes[i].role_change_count += 1

        return ClusterResult(clusters=clusters, forwarders=set())