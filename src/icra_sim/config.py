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

    # slower re-clustering to reduce role churn
    clustering_interval_s: int = 5

    # area and radio
    area_km: Tuple[float, float] = (10.0, 10.0)
    comm_radius_km: float = 1.0

    # mobility
    gauss_markov_alpha: float = 0.8
    speed_noise_std: float = 0.80
    heading_noise_std: float = 0.04

    # traffic
    packet_gen_prob_per_s: float = 0.02
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000
    per_hop_processing_delay_s: float = 0.0008
    mac_contention_delay_s: float = 0.0008
    queueing_delay_s: float = 0.0012
    max_hops: int = 30

    # energy
    ehf_j_per_s: float = 2.0
    en_j_per_s: float = 1.0

    e_tx_j: float = 0.05
    e_rx_j: float = 0.02
    e_ch_proc_j: float = 0.008

    # control plane
    control_packet_size_bytes: int = 64
    e_ctrl_tx_j: float = 0.006
    e_ctrl_rx_j: float = 0.003
    ctrl_proc_delay_s: float = 0.0002

    # clustering / utility
    lht_threshold_s: float = 0.20
    lht_cap_s: float = 60.0

    # stronger retention
    join_hysteresis_margin: float = 0.08
    ch_retain_margin: float = 0.10
    min_ch_tenure_s: float = 12.0
    max_cluster_members: int = 18
    min_gateway_lht_s: float = 0.25

    # RL
    reward_lambda: float = 0.50
    role_change_threshold: int = 2
    q_alpha: float = 0.35
    q_gamma: float = 0.0
    q_epsilon: float = 0.06
    q_epsilon_min: float = 0.01
    q_epsilon_decay: float = 0.997
    q_step: float = 0.05

    # RL stability controls
    action_stickiness_bonus: float = 0.08
    min_action_hold_rounds: int = 6
    weight_smoothing_beta: float = 0.82
    allow_action_jump_l1: float = 0.40

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