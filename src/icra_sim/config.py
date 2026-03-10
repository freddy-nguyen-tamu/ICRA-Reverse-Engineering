from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal


ProtocolName = Literal["icra", "wca", "dca"]
ScenarioName = Literal["case1", "case2", "case3"]


@dataclass(frozen=True)
class SimConfig:
    # core
    sim_time_s: int = 1500
    dt_s: float = 1.0

    # Re-clustering / RL strategy update interval
    clustering_interval_s: int = 20

    # area and radio
    area_km: Tuple[float, float] = (10.0, 10.0)  # (width, height) in km
    comm_radius_km: float = 1.0

    # --- Mobility ---
    gauss_markov_alpha: float = 0.85
    speed_noise_std: float = 1.0
    heading_noise_std: float = 0.05

    # --- Traffic ---
    packet_gen_prob_per_s: float = 0.02
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000
    per_hop_processing_delay_s: float = 0.001
    mac_contention_delay_s: float = 0.0015
    queueing_delay_s: float = 0.0020
    max_hops: int = 30

    # --- Energy model ---
    # steady-state role drain
    ehf_j_per_s: float = 2.0       # CH / forwarder
    en_j_per_s: float = 1.0        # member

    # data packet radio costs
    e_tx_j: float = 0.05
    e_rx_j: float = 0.02
    e_ch_proc_j: float = 0.01

    # control plane overhead
    control_packet_size_bytes: int = 64
    e_ctrl_tx_j: float = 0.010
    e_ctrl_rx_j: float = 0.005
    e_rl_feedback_j: float = 0.004

    # approximate control-message delays
    control_msg_proc_delay_s: float = 0.0005

    # --- Clustering thresholds ---
    lht_threshold_s: float = 0.1
    role_change_threshold: int = 2

    # join hysteresis to reduce unnecessary churn
    join_hysteresis_margin: float = 0.03

    # --- RL parameters ---
    q_alpha: float = 0.8
    q_gamma: float = 0.0
    reward_lambda: float = 0.5

    # --- Normalization caps ---
    lht_cap_s: float = 60.0

    # --- Randomness ---
    seed: int = 42

    # ε-greedy
    q_epsilon: float = 0.20
    q_epsilon_min: float = 0.02
    q_epsilon_decay: float = 0.995


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: ScenarioName

    # Initial energy settings
    init_energy_low_j: float
    init_energy_high_j: float

    # Speed settings (m/s)
    speed_low_m_s: float
    speed_high_m_s: float
    constant_speed: bool

    @staticmethod
    def for_scenario(name: ScenarioName) -> "ScenarioConfig":
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