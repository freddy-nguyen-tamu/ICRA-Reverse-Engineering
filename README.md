# ICRA Reverse Engineering

## About it.

This project is a reverse engineering and simulation-based reproduction of the paper:

"ICRA: An Intelligent Clustering Routing Approach for UAV Ad Hoc Networks"

The codebase implements a Python simulator for comparing three clustering/routing protocols in UAV ad hoc networks:

- ICRA (Intelligent Clustering Routing Approach)
- WCA (Weighted Clustering Algorithm – baseline)
- DCA (Distributed Clustering Algorithm – baseline)

The simulator reproduces the paper's three experimental scenarios and evaluates the protocols using the same main metrics discussed in the paper:

- cluster creation time (protocol cost)
- average role changes
- network lifetime (first dead node)
- isolation clusters
- end-to-end delay
- packet delivery ratio

The implementation is organized as a reusable Python package under `src/icra_sim` and experiment scripts under `scripts/`.

## Project Structure

```
ICRA/
├── scripts/
│   ├── run_experiments.py
│   └── check_consistency.py
├── src/
│   └── icra_sim/
│       ├── __init__.py
│       ├── config.py
│       ├── link.py
│       ├── metrics.py
│       ├── node.py
│       ├── radio.py
│       ├── simulator.py
│       ├── utils.py
│       ├── clustering/
│       │   ├── clusterer.py
│       │   └── utility.py
│       ├── mobility/
│       │   └── gauss_markov.py
│       ├── rl/
│       │   └── qlearning.py
│       └── routing/
│           └── router.py
└── results/
```

## Implemented Features

### Core simulation
- Discrete-time simulation over 1500 seconds (default)
- 10 km × 10 km simulation area
- 1 km communication radius
- Gauss-Markov mobility model with weak anchor pull (grouped initialization)
- Packet-based communication with per‑hop delay and probabilistic success
- Energy consumption model for cluster heads, forwarders, and members
- Reconfiguration energy costs and control overhead penalties

### Protocols
- `icra`
- `wca`
- `dca`

### ICRA‑specific logic
- Weighted utility‑based cluster head election
- Four utility factors:
  - residual energy (`s1`)
  - degree / connectivity (`s2`)
  - velocity similarity (`s3`)
  - link holding time (`s4`)
- Q‑learning‑based clustering strategy adjustment
- State discretization via entropy of factor distributions
- Hybrid reward combining role‑change frequency, energy consumption, routing QoS, and isolation clusters
- Guided action selection and entropy‑state prior for exploration
- Intra‑cluster and inter‑cluster routing support
- Fragmentation penalties in routing to reflect clustering quality

### Evaluation metrics
- `cluster_protocol_cost_s` – average cluster control overhead time
- `avg_role_changes` – average number of cluster‑role changes per node
- `network_lifetime_s` – time until first node depletes energy
- `dead_nodes` – number of nodes that ran out of energy
- `isolation_clusters_avg` – average number of isolation clusters (≤2 members)
- `avg_end_to_end_delay_s` – mean end‑to‑end delay of successfully delivered packets
- `packet_delivery_ratio` – fraction of generated packets delivered

## Setup

1. Clone this repository
   ```bash
   git clone <your-repository-url>
   cd ICRA
   ```

2. Create and activate a Python virtual environment (recommended)
   ```bash
   python3 -m venv icra_env
   source icra_env/bin/activate   # Linux / Git Bash
   # or .venv\Scripts\activate    # Windows
   ```

3. Install dependencies
   ```bash
   python -m pip install --upgrade pip
   pip install matplotlib
   ```

## Running Experiments

The main experiment driver is `scripts/run_experiments.py`.

### Run all protocols, all scenarios, and all node counts
```bash
python scripts/run_experiments.py
```

### Run one protocol only
```bash
python scripts/run_experiments.py --protocol icra
python scripts/run_experiments.py --protocol wca
python scripts/run_experiments.py --protocol dca
```

### Run one scenario only
```bash
python scripts/run_experiments.py --scenario case1
python scripts/run_experiments.py --scenario case2
python scripts/run_experiments.py --scenario case3
```

### Run one node count only
```bash
python scripts/run_experiments.py --n 100
```

### Run a specific protocol/scenario combination
```bash
python scripts/run_experiments.py --protocol icra --scenario case3 --n 100
```

### Write outputs to a custom folder
```bash
python scripts/run_experiments.py --out results_custom
```

## Command Line Arguments

### `run_experiments.py`

| Argument      | Choices / Type                | Meaning                                             |
|---------------|-------------------------------|-----------------------------------------------------|
| `--protocol`  | `icra`, `wca`, `dca`, `all`   | Selects which protocol to run. Default is `all`.    |
| `--scenario`  | `case1`, `case2`, `case3`, `all` | Selects which scenario to run. Default is `all`. |
| `--n`         | int                           | Runs only one node count if set. Default is all.    |
| `--out`       | str                           | Output directory for CSV files and plots. Default is `results`. |

### Supported node counts
The experiment script uses:
```
10, 20, 50, 100
```

## Simulation Scenarios

The simulator reproduces the three scenarios used in the paper:

### case1
- Initial node energy varies uniformly between 500 J and 2000 J
- Speed is fixed at 40 m/s

### case2
- Initial node energy is fixed at 2000 J
- Speed is fixed at 40 m/s

### case3
- Initial node energy is fixed at 2000 J
- Node speed varies uniformly between 30 m/s and 50 m/s

## Outputs

After running experiments, the output folder contains:

### Main metrics file
```
metrics.csv
```
This file stores one row per `(scenario, N, protocol)` combination with the following fields:
- `scenario`
- `N`
- `protocol`
- `cluster_protocol_cost_s`
- `avg_role_changes`
- `network_lifetime_s`
- `dead_nodes`
- `isolation_clusters_avg`
- `avg_end_to_end_delay_s`
- `packet_delivery_ratio`

### Generated plots
For each scenario, the script generates six PNG plots:
- `cluster_cost_{scenario}.png`
- `role_changes_{scenario}.png`
- `lifetime_{scenario}.png`
- `isolation_{scenario}.png`
- `delay_{scenario}.png`
- `pdr_{scenario}.png`

### ICRA weight history files
When running `icra`, the script also writes per‑(scenario, N) CSV files:
- `weights_icra_case1_N10.csv`, `weights_icra_case1_N20.csv`, …
- `weights_icra_case2_N10.csv`, …
- `weights_icra_case3_N10.csv`, …

And a summary plot:
- `icra_weight_summary.png`

## Consistency Checking

The repository includes a result checker in `scripts/check_consistency.py`.

It reads `metrics.csv` and the ICRA weight history files, then compares the generated results against the qualitative trends described in the paper.

### Run the checker on the default results folder
```bash
python scripts/check_consistency.py
```

### Run the checker on a custom results folder
```bash
python scripts/check_consistency.py --results-dir results_custom
```

## Main Modules

### `src/icra_sim/config.py`
Defines:
- supported protocol and scenario names
- simulation constants (time, area, radio)
- mobility parameters
- traffic and packet model
- energy consumption rates
- clustering and RL parameters
- scenario‑specific energy/speed settings

### `src/icra_sim/simulator.py`
Main simulation loop:
- initializes nodes with group‑based positions
- updates mobility (Gauss‑Markov)
- runs clustering rounds every 2 seconds
- routes packets and updates energy
- records weight evolution for ICRA
- computes final run metrics

### `src/icra_sim/clustering/clusterer.py`
Contains clustering logic for ICRA, WCA, and DCA, including:
- ICRA CH election with stability/load penalties
- member assignment with LET thresholds and hysteresis
- small‑cluster repair and forwarder selection

### `src/icra_sim/clustering/utility.py`
Computes the four utility factors used during cluster head election.

### `src/icra_sim/link.py`
Computes link holding time based on relative motion and communication radius.

### `src/icra_sim/routing/router.py`
Implements hierarchical routing with:
- member‑to‑CH access
- intra‑cluster forwarding
- backbone path search (CH ↔ forwarder)
- hop‑based delay accumulation and probabilistic delivery
- fragmentation penalties that degrade QoS for unstable clusters

### `src/icra_sim/metrics.py`
Defines the output metrics and helper functions.

## Notes

* This is a reverse‑engineering reproduction in Python, not the original OPNET simulator.
* The project compares qualitative trends against the paper; absolute values may differ because the routing model, reward design, and mobility details are implementation‑specific.
* The results directory can be regenerated at any time by rerunning `scripts/run_experiments.py`.

## Results

* Clean the previous results folder before running a new full experiment so the generated CSV files and figures only reflect the latest run.
* The current workflow is designed for scenario‑level comparison across `icra`, `wca`, and `dca`.
* The generated plots are intended to be used directly in the reverse engineering report and comparison discussion.
``