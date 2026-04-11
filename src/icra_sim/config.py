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
    clustering_interval_s: int = 2

    # ------------------------------------------------------------------
    # Area and radio (Table II)
    # ------------------------------------------------------------------
    area_km: Tuple[float, float] = (10.0, 10.0)
    comm_radius_km: float = 1.0

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------
    gauss_markov_alpha: float = 0.8
    speed_noise_std: float = 0.25
    heading_noise_std: float = 0.015

    # A lightweight approximation of the paper's reference-point-group motion:
    # nodes are initialized around a small number of group centers and their
    # Gauss-Markov motion is weakly pulled back toward those centers.
    use_grouped_init: bool = True
    group_anchor_pull: float = 0.035
    group_spread_frac_of_radius: float = 0.34

    # ------------------------------------------------------------------
    # Traffic / packet model
    # ------------------------------------------------------------------
    packet_gen_prob_per_s: float = 0.020
    packet_size_bytes: int = 512
    data_rate_kbps: int = 1000

    # These delays are implementation choices for the lightweight simulator.
    # They are intentionally larger than the old millisecond-scale model so the
    # QoS curves do not collapse to a nearly constant band.
    per_hop_processing_delay_s: float = 0.0025
    mac_contention_delay_s: float = 0.0030
    queueing_delay_s: float = 0.0150
    ctrl_proc_delay_s: float = 0.0012
    distance_delay_scale_s: float = 0.0020
    congestion_delay_scale_s: float = 0.0040
    max_hops: int = 30

    # Moderate packet success model: OPNET would naturally induce losses from
    # congestion / topology; this lightweight simulator needs an explicit proxy.
    link_success_floor: float = 0.72
    link_success_exponent: float = 1.2
    hop_congestion_loss_scale: float = 0.08

    # ------------------------------------------------------------------
    # Energy (Table II)
    # ------------------------------------------------------------------
    ehf_j_per_s: float = 2.0   # CH / forwarding node energy rate E_hf
    en_j_per_s: float = 1.0    # common member energy rate E_n

    # Packet-level energy is no longer forced to be negligible.
    e_tx_j: float = 0.020
    e_rx_j: float = 0.010
    e_proc_j: float = 0.005

    control_packet_size_bytes: int = 64
    e_ctrl_tx_j: float = 0.002
    e_ctrl_rx_j: float = 0.001

    # ------------------------------------------------------------------
    # Clustering / RL (paper notation)
    # ------------------------------------------------------------------
    sigma_lht_threshold_s: float = 0.10
    phi_role_change_threshold: int = 2

    # Hybrid reward: keep the paper's role-change / energy emphasis, but add
    # routing feedback because the paper explicitly says strategy adjustment uses
    # the previous routing result as feedback.
    reward_lambda: float = 0.62
    reward_qos_weight: float = 0.20
    reward_isolation_weight: float = 0.10
    reward_cluster_penalty_weight: float = 0.08

    # Practical paper-aligned additions for the lightweight simulator.
    # Frequent cluster-role or membership reconfiguration should consume
    # additional control energy; otherwise DCA can unrealistically win the
    # first-dead-node lifetime metric by spreading CH duty despite very high
    # topology churn.
    e_membership_change_j: float = 0.050
    e_cluster_role_change_j: float = 0.180
    e_cluster_head_change_j: float = 0.120

    # The paper says the importance of each factor should vary with the value
    # distribution of that factor in the network. The RL implementation uses the
    # entropy-state vector to guide early action choice before enough feedback is
    # collected for a state.
    guided_action_state_floor: float = 0.02
    guided_action_gamma: float = 1.15
    guided_action_sample_scale: float = 8.0

    e_reconfig_service_ch_j: float = 0.030
    e_reconfig_service_fwd_j: float = 0.015

    instability_energy_scale_ch: float = 0.55
    instability_energy_scale_member: float = 0.20

    q_alpha: float = 0.18
    q_gamma: float = 0.0
    q_epsilon: float = 0.10
    q_epsilon_min: float = 0.02
    q_epsilon_decay: float = 0.997
    q_step: float = 0.05
    state_bin: float = 0.10

    # A practical implementation detail omitted by the paper: start from equal weights.
    initial_icra_weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    # To keep the four utility terms on a comparable scale in this lightweight
    # implementation, the link holding time term is clipped to a finite window
    # before normalization. This removes the old percentile heuristic but still
    # avoids unbounded dominance by very large LHT values.
    lht_cap_s: float = 120.0

    # Practical clustering guards that approximate missing simulator detail
    # without reverting to anchored RL or non-paper utility factors.
    icra_min_cluster_size: int = 3
    icra_max_cluster_members: int = 40
    icra_join_hysteresis_margin: float = 0.16
    icra_ch_retain_margin: float = 0.12
    icra_min_ch_tenure_s: float = 12.0
    icra_min_ch_neighbor_count: int = 2
    icra_ch_energy_guard_ratio: float = 0.20
    icra_degree_balance_bonus_weight: float = 0.05
    icra_tenure_stability_bonus_weight: float = 0.05
    icra_link_stability_bonus_weight: float = 0.05
    icra_velocity_stability_bonus_weight: float = 0.04
    icra_recent_ch_penalty_weight: float = 0.06
    icra_traffic_load_penalty_weight: float = 0.04
    icra_local_degree_target: float = 0.58
    icra_local_degree_tolerance: float = 0.28

    # Runtime-field decay between clustering rounds.
    traffic_load_decay: float = 0.92
    relay_load_decay: float = 0.90
    recent_role_change_decay: float = 0.85

    # Reproducibility
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


