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
    area_km: Tuple[float, float] = (10.0, 10.0) # (width, height) in km
    comm_radius_km: float = 1.0

    # --- Mobility (Gauss-Markov) ---
    gauss_markov_alpha: float = 0.85
    speed_noise_std: float = 1.0 # in m/s
    heading_noise_std: float = 0.05 # in radian

    # --- Traffic ---
    # packets generated per node per second (Bernoulli with this probability)
    packet_gen_prob_per_s: float = 0.02
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000
    per_hop_processing_delay_s: float = 0.001
    max_hops: int = 30

    # --- Energy model (Table II in the paper) ---
    # CH / forwarder energy drain (J/s)
    ehf_j_per_s: float = 2.0
    # ordinary member energy drain (J/s)
    en_j_per_s: float = 1.0

    # --- Clustering thresholds (Table II) ---
    lht_threshold_s: float = 0.1 # σ
    role_change_threshold: int = 2 # φ

    # --- RL parameters (paper uses α=0.8, γ=0) ---
    q_alpha: float = 0.8
    q_gamma: float = 0.0
    reward_lambda: float = 0.5  # λ in Eq.(23)

    # --- Normalization caps ---
    lht_cap_s: float = 60.0

    # --- Randomness ---
    seed: int = 42


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: ScenarioName

    # Initial energy settings
    # case1: uniform energies [500, 2000] J
    # case2/3: constant energies 2000 J
    init_energy_low_j: float
    init_energy_high_j: float

    # Speed settings (m/s)
    # case1/2: constant speed
    # case3: random speed in [30,50]
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