from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

ProtocolName = Literal["icra", "wca", "dca"]
ScenarioName = Literal["case1", "case2", "case3"]


@dataclass(frozen=True)
class SimConfig:
    # core
    sim_time_s: int = 1500
    dt_s: float = 1.0

    # re-clustering
    clustering_interval_s: int = 8

    # area and radio
    area_km: Tuple[float, float] = (10.0, 10.0)
    comm_radius_km: float = 1.2

    # mobility
    gauss_markov_alpha: float = 0.92
    speed_noise_std: float = 0.45
    heading_noise_std: float = 0.025

    # traffic
    packet_gen_prob_per_s: float = 0.02
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000
    per_hop_processing_delay_s: float = 0.0008
    mac_contention_delay_s: float = 0.0008
    queueing_delay_s: float = 0.0010
    max_hops: int = 30

    # energy
    ehf_j_per_s: float = 1.75
    en_j_per_s: float = 0.95
    e_tx_j: float = 0.05
    e_rx_j: float = 0.02
    e_ch_proc_j: float = 0.007

    # control plane
    control_packet_size_bytes: int = 64
    e_ctrl_tx_j: float = 0.005
    e_ctrl_rx_j: float = 0.0025
    ctrl_proc_delay_s: float = 0.0002

    # clustering / utility
    lht_threshold_s: float = 0.40
    lht_cap_s: float = 90.0

    # retention / hysteresis
    join_hysteresis_margin: float = 0.18
    ch_retain_margin: float = 0.22
    min_ch_tenure_s: float = 28.0
    max_cluster_members: int = 16

    # gateway layer
    min_gateway_lht_s: float = 0.60

    # connectivity-aware CH scoring
    min_ch_neighbor_count: int = 2
    prefer_connected_ch_bonus: float = 0.18
    isolated_ch_penalty: float = 0.24

    # forwarder / gateway selection
    forwarder_reuse_bonus: float = 0.10
    gateway_crosslink_weight: float = 0.58
    gateway_utility_weight: float = 0.14
    gateway_energy_weight: float = 0.14
    gateway_stability_weight: float = 0.14
    gateway_multicluster_bonus: float = 0.12
    direct_ch_link_bonus: float = 0.10

    # CH quality shaping
    ch_energy_guard_ratio: float = 0.35
    ch_cooldown_s: float = 24.0
    recent_ch_penalty_weight: float = 0.18
    traffic_load_penalty_weight: float = 0.18
    degree_balance_bonus_weight: float = 0.16
    tenure_stability_bonus_weight: float = 0.10
    link_stability_bonus_weight: float = 0.10
    velocity_stability_bonus_weight: float = 0.08
    local_degree_target: float = 0.58
    local_degree_tolerance: float = 0.28

    # RL
    reward_lambda: float = 0.35
    role_change_threshold: int = 2
    q_alpha: float = 0.18
    q_gamma: float = 0.55
    q_epsilon: float = 0.08
    q_epsilon_min: float = 0.010
    q_epsilon_decay: float = 0.995
    q_step: float = 0.05

    # RL stability controls
    action_stickiness_bonus: float = 0.16
    min_action_hold_rounds: int = 5
    weight_smoothing_beta: float = 0.72
    allow_action_jump_l1: float = 0.30

    # reward weights for paper-like behavior
    reward_role_changes_weight: float = 0.24
    reward_energy_weight: float = 0.20
    reward_pdr_weight: float = 0.18
    reward_delay_weight: float = 0.12
    reward_isolation_weight: float = 0.08
    reward_balance_weight: float = 0.10
    reward_survival_weight: float = 0.08

    # temporal smoothing / role-memory
    recent_role_change_decay: float = 0.82
    traffic_load_decay: float = 0.70
    cooldown_decay_per_round_s: float = 8.0
    clustering_warmup_rounds: int = 2

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