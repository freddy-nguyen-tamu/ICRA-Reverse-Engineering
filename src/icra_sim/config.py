from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

ProtocolName = Literal["icra", "wca", "dca"]
ScenarioName = Literal["case1", "case2", "case3"]


@dataclass(frozen=True)
class SimConfig:
    # ------------------------------------------------------------------
    # Core time model
    # ------------------------------------------------------------------
    sim_time_s: int = 1500
    dt_s: float = 1.0
    clustering_interval_s: int = 2  # paper feedback / strategy cycle

    # ------------------------------------------------------------------
    # Area and radio
    # ------------------------------------------------------------------
    area_km: Tuple[float, float] = (10.0, 10.0)
    comm_radius_km: float = 1.0  # paper Table II

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------
    gauss_markov_alpha: float = 0.8  # paper Table II
    speed_noise_std: float = 0.25
    heading_noise_std: float = 0.015

    # ------------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------------
    packet_gen_prob_per_s: float = 0.020
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000

    # Keep the router simple. The paper compares qualitatively rather
    # than through a large custom queue model.
    per_hop_processing_delay_s: float = 0.00075
    mac_contention_delay_s: float = 0.00060
    queueing_delay_s: float = 0.00080
    max_hops: int = 30

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------
    # Paper: forwarding / CH nodes consume faster than common members.
    ehf_j_per_s: float = 2.0
    en_j_per_s: float = 1.0

    # Packet-level communication energy.
    e_tx_j: float = 0.040
    e_rx_j: float = 0.018
    e_ch_proc_j: float = 0.005

    # Simple steady-state role energy. Keep this close to the paper:
    # CHs and forwarding nodes use Ehf, ordinary members use En.
    ch_idle_extra_j_per_s: float = 1.0     # Ehf - En
    forwarder_idle_extra_j_per_s: float = 1.0  # treat forwarding node same class as paper's forwarding entity

    # Keep path extras small; avoid simulator-only shaping.
    e_ch_backbone_tx_j: float = 0.006
    e_forwarder_backbone_tx_j: float = 0.006
    e_ch_service_rx_j: float = 0.003
    e_forwarder_backbone_rx_j: float = 0.003
    e_ch_service_proc_j: float = 0.003
    e_forwarder_proc_j: float = 0.003
    e_path_reuse_surcharge_j: float = 0.0

    load_energy_scale_j: float = 0.0
    relay_load_energy_scale_j: float = 0.0
    path_reuse_energy_scale_j: float = 0.0

    # control packets
    control_packet_size_bytes: int = 64
    e_ctrl_tx_j: float = 0.0035
    e_ctrl_rx_j: float = 0.0018
    ctrl_proc_delay_s: float = 0.00012

    # ------------------------------------------------------------------
    # Utility / clustering
    # ------------------------------------------------------------------
    # The paper's CH formation rule only requires positive, stable links.
    # Keep thresholds permissive and avoid extra topology shaping.
    lht_threshold_s: float = 0.10
    lht_cap_s: float = 120.0

    join_hysteresis_margin: float = 0.08
    ch_retain_margin: float = 0.10
    min_ch_tenure_s: float = 8.0
    max_cluster_members: int = 18

    # ICRA connectivity shaping - greatly reduced from custom version.
    min_ch_neighbor_count: int = 1
    prefer_connected_ch_bonus: float = 0.04
    isolated_ch_penalty: float = 0.08

    # gateway / inter-cluster forwarding
    min_gateway_lht_s: float = 0.10
    forwarder_reuse_bonus: float = 0.01
    gateway_crosslink_weight: float = 0.45
    gateway_utility_weight: float = 0.15
    gateway_energy_weight: float = 0.15
    gateway_stability_weight: float = 0.15
    gateway_multicluster_bonus: float = 0.04
    direct_ch_link_bonus: float = 0.04

    # CH quality shaping - kept but softened so weights do the work
    ch_energy_guard_ratio: float = 0.15
    ch_cooldown_s: float = 4.0
    recent_ch_penalty_weight: float = 0.04
    traffic_load_penalty_weight: float = 0.02
    degree_balance_bonus_weight: float = 0.04
    tenure_stability_bonus_weight: float = 0.02
    link_stability_bonus_weight: float = 0.03
    velocity_stability_bonus_weight: float = 0.03
    local_degree_target: float = 0.50
    local_degree_tolerance: float = 0.35

    # ------------------------------------------------------------------
    # RL
    # ------------------------------------------------------------------
    # Paper reward is only the weighted combination of Rc and Ec.
    reward_lambda: float = 0.8
    role_change_threshold: int = 2

    q_alpha: float = 0.20
    q_gamma: float = 0.0  # paper explicitly sets gamma = 0
    q_epsilon: float = 0.08
    q_epsilon_min: float = 0.01
    q_epsilon_decay: float = 0.995
    q_step: float = 0.05

    # Minimize simulator-imposed action smoothing.
    action_stickiness_bonus: float = 0.02
    min_action_hold_rounds: int = 1
    weight_smoothing_beta: float = 0.15
    allow_action_jump_l1: float = 1.0

    # Retained only for compatibility with existing code paths.
    reward_role_changes_weight: float = 0.80
    reward_energy_weight: float = 0.20
    reward_pdr_weight: float = 0.0
    reward_delay_weight: float = 0.0
    reward_isolation_weight: float = 0.0
    reward_balance_weight: float = 0.0
    reward_survival_weight: float = 0.0

    # runtime memory
    recent_role_change_decay: float = 0.85
    traffic_load_decay: float = 0.80
    relay_load_decay: float = 0.80
    path_reuse_decay: float = 0.80
    cooldown_decay_per_round_s: float = 2.0
    clustering_warmup_rounds: int = 1

    # ------------------------------------------------------------------
    # Metric model knobs
    # ------------------------------------------------------------------
    icra_cluster_time_base_s: float = 0.30
    icra_cluster_time_per_node_s: float = 0.0012

    dca_cluster_time_base_s: float = 0.36
    dca_cluster_time_per_node_s: float = 0.0011

    wca_cluster_time_base_s: float = 0.30
    wca_cluster_time_per_node_s: float = 0.0120

    # Router protocol knobs are kept for API compatibility only.
    icra_backbone_queue_scale: float = 1.0
    dca_backbone_queue_scale: float = 1.0
    wca_backbone_queue_scale: float = 1.0

    icra_backbone_loss_bias: float = 0.0
    dca_backbone_loss_bias: float = 0.0
    wca_backbone_loss_bias: float = 0.0

    # reproducibility
    seed: int = 7

    @property
    def width_m(self) -> float:
        return self.area_km[0] * 1000.0

    @property
    def height_m(self) -> float:
        return self.area_km[1] * 1000.0

    @property
    def comm_radius_m(self) -> float:
        return self.comm_radius_km * 1000.0


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: ScenarioName
    init_energy_low_j: float
    init_energy_high_j: float
    speed_low_m_s: float
    speed_high_m_s: float
    constant_speed: bool = False

    @staticmethod
    def from_name(name: ScenarioName) -> "ScenarioConfig":
        if name == "case1":
            return ScenarioConfig(
                scenario=name,
                init_energy_low_j=500.0,
                init_energy_high_j=2000.0,
                speed_low_m_s=40.0,
                speed_high_m_s=40.0,
                constant_speed=True,
            )
        if name == "case2":
            return ScenarioConfig(
                scenario=name,
                init_energy_low_j=2000.0,
                init_energy_high_j=2000.0,
                speed_low_m_s=40.0,
                speed_high_m_s=40.0,
                constant_speed=True,
            )
        if name == "case3":
            return ScenarioConfig(
                scenario=name,
                init_energy_low_j=2000.0,
                init_energy_high_j=2000.0,
                speed_low_m_s=30.0,
                speed_high_m_s=50.0,
                constant_speed=False,
            )
        raise ValueError(f"Unknown scenario: {name}")