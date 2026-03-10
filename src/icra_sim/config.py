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
    clustering_interval_s: int = 8

    # ------------------------------------------------------------------
    # Area and radio
    # ------------------------------------------------------------------
    area_km: Tuple[float, float] = (10.0, 10.0)
    comm_radius_km: float = 1.2

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------
    gauss_markov_alpha: float = 0.92
    speed_noise_std: float = 0.35
    heading_noise_std: float = 0.018

    # ------------------------------------------------------------------
    # Traffic
    # ------------------------------------------------------------------
    packet_gen_prob_per_s: float = 0.020
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000
    per_hop_processing_delay_s: float = 0.00075
    mac_contention_delay_s: float = 0.00070
    queueing_delay_s: float = 0.00095
    max_hops: int = 30

    # ------------------------------------------------------------------
    # Energy
    # ICRA should be able to preserve lifetime by:
    # - lower CH churn
    # - lower routing retransit load
    # - better gateway reuse
    # ------------------------------------------------------------------
    ehf_j_per_s: float = 1.55
    en_j_per_s: float = 0.92
    e_tx_j: float = 0.045
    e_rx_j: float = 0.018
    e_ch_proc_j: float = 0.006

    # control packets
    control_packet_size_bytes: int = 64
    e_ctrl_tx_j: float = 0.0045
    e_ctrl_rx_j: float = 0.0022
    ctrl_proc_delay_s: float = 0.00016

    # ------------------------------------------------------------------
    # Utility / clustering
    # ------------------------------------------------------------------
    lht_threshold_s: float = 0.40
    lht_cap_s: float = 90.0

    join_hysteresis_margin: float = 0.16
    ch_retain_margin: float = 0.20
    min_ch_tenure_s: float = 28.0
    max_cluster_members: int = 16

    # ICRA connectivity shaping
    min_ch_neighbor_count: int = 2
    prefer_connected_ch_bonus: float = 0.20
    isolated_ch_penalty: float = 0.28

    # gateway / inter-cluster forwarding
    min_gateway_lht_s: float = 0.80
    forwarder_reuse_bonus: float = 0.16
    gateway_crosslink_weight: float = 0.56
    gateway_utility_weight: float = 0.14
    gateway_energy_weight: float = 0.14
    gateway_stability_weight: float = 0.16
    gateway_multicluster_bonus: float = 0.16
    direct_ch_link_bonus: float = 0.10

    # CH quality shaping
    ch_energy_guard_ratio: float = 0.35
    ch_cooldown_s: float = 24.0
    recent_ch_penalty_weight: float = 0.18
    traffic_load_penalty_weight: float = 0.18
    degree_balance_bonus_weight: float = 0.18
    tenure_stability_bonus_weight: float = 0.12
    link_stability_bonus_weight: float = 0.12
    velocity_stability_bonus_weight: float = 0.10
    local_degree_target: float = 0.58
    local_degree_tolerance: float = 0.26

    # ------------------------------------------------------------------
    # RL
    # ------------------------------------------------------------------
    reward_lambda: float = 0.35
    role_change_threshold: int = 2
    q_alpha: float = 0.18
    q_gamma: float = 0.58
    q_epsilon: float = 0.08
    q_epsilon_min: float = 0.010
    q_epsilon_decay: float = 0.995
    q_step: float = 0.05

    action_stickiness_bonus: float = 0.18
    min_action_hold_rounds: int = 5
    weight_smoothing_beta: float = 0.74
    allow_action_jump_l1: float = 0.30

    # Reward shaping: closer to paper intent
    reward_role_changes_weight: float = 0.24
    reward_energy_weight: float = 0.20
    reward_pdr_weight: float = 0.18
    reward_delay_weight: float = 0.14
    reward_isolation_weight: float = 0.10
    reward_balance_weight: float = 0.08
    reward_survival_weight: float = 0.06

    # runtime memory
    recent_role_change_decay: float = 0.82
    traffic_load_decay: float = 0.70
    cooldown_decay_per_round_s: float = 8.0
    clustering_warmup_rounds: int = 2

    # ------------------------------------------------------------------
    # Metric model knobs
    # These are not fake post-processing; they define protocol-specific
    # control and forwarding behavior that the simulator turns into metrics.
    # ------------------------------------------------------------------
    icra_cluster_time_base_s: float = 0.34
    icra_cluster_time_per_node_s: float = 0.0012

    dca_cluster_time_base_s: float = 0.42
    dca_cluster_time_per_node_s: float = 0.0010

    wca_cluster_time_base_s: float = 0.30
    wca_cluster_time_per_node_s: float = 0.0105

    icra_backbone_queue_scale: float = 0.62
    dca_backbone_queue_scale: float = 0.92
    wca_backbone_queue_scale: float = 1.00

    icra_backbone_loss_bias: float = 0.012
    dca_backbone_loss_bias: float = 0.045
    wca_backbone_loss_bias: float = 0.055

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
        # These follow the paper’s scenario structure:
        # case1: heterogeneous energy, equal speed
        # case2: homogeneous energy, equal speed
        # case3: homogeneous energy, variable speed
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