# ICRA Reverse Engineering

## About it.

This project is a reverse engineering and simulation-based reproduction of the paper:

"ICRA: An Intelligent Clustering Routing Approach for UAV Ad Hoc Networks"

The codebase implements a Python simulator for comparing three clustering/routing protocols in UAV ad hoc networks:

- ICRA
- WCA
- DCA

The simulator reproduces the paper's three experimental scenarios and evaluates the protocols using the same main metrics discussed in the paper:

- cluster creation time
- average role changes
- network lifetime
- isolation clusters
- end-to-end delay
- packet delivery ratio

The implementation is organized as a reusable Python package under `src/icra_sim` and experiment scripts under `scripts/`.

## Project Structure

~~~text
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
│       ├── simulator.py
│       ├── clustering/
│       │   ├── clusterer.py
│       │   └── utility.py
│       └── routing/
│           └── router.py
└── results/
~~~

## Implemented Features

### Core simulation
- Discrete-time simulation over 1500 seconds
- 10 km x 10 km simulation area
- 1 km communication radius
- Gauss-Markov mobility model
- Packet-based communication with per-hop delay model
- Energy consumption model for cluster heads, forwarders, and members

### Protocols
- `icra`
- `wca`
- `dca`

### ICRA-specific logic
- Weighted utility-based cluster head election
- Four utility factors:
  - residual energy
  - degree / connectivity
  - velocity similarity
  - link holding time
- Q-learning-based clustering strategy adjustment
- Per-scenario adaptive weight evolution
- Intra-cluster and inter-cluster routing support

### Evaluation metrics
- `cluster_protocol_cost_s`
- `avg_role_changes`
- `network_lifetime_s`
- `dead_nodes`
- `isolation_clusters_avg`
- `avg_end_to_end_delay_s`
- `packet_delivery_ratio`

## Setup

* Clone this repository
~~~bash
git clone <your-repository-url>
cd ICRA
~~~

* Create and activate a Python virtual environment (optional, but recommended)
* For Linux/Git bash:
~~~bash
python3 -m venv icra_env
source icra_env/bin/activate
~~~

* Install the dependency used by the experiment script
~~~bash
python -m pip install --upgrade pip
pip install matplotlib
~~~

## Running Experiments

The main experiment driver is `scripts/run_experiments.py`.

### Run all protocols, all scenarios, and all node counts
~~~bash
python scripts/run_experiments.py
~~~

### Run one protocol only
~~~bash
python scripts/run_experiments.py --protocol icra
python scripts/run_experiments.py --protocol wca
python scripts/run_experiments.py --protocol dca
~~~

### Run one scenario only
~~~bash
python scripts/run_experiments.py --scenario case1
python scripts/run_experiments.py --scenario case2
python scripts/run_experiments.py --scenario case3
~~~

### Run one node count only
~~~bash
python scripts/run_experiments.py --n 100
~~~

### Run a specific protocol/scenario combination
~~~bash
python scripts/run_experiments.py --protocol icra --scenario case3 --n 100
~~~

### Write outputs to a custom folder
~~~bash
python scripts/run_experiments.py --out results_custom
~~~

## Command Line Arguments

### `run_experiments.py`
| Argument | Choices / Type | Meaning |
|----------|----------------|---------|
| `--protocol` | `icra`, `wca`, `dca`, `all` | Selects which protocol to run. Default is `all`. |
| `--scenario` | `case1`, `case2`, `case3`, `all` | Selects which scenario to run. Default is `all`. |
| `--n` | int | Runs only one node count if set. Default is all supported node counts. |
| `--out` | str | Output directory for CSV files, plots, and learned weights. Default is `results`. |

### Supported node counts
The experiment script uses:
~~~text
10, 20, 50, 100
~~~

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
- Node speed varies between 30 m/s and 50 m/s

## Outputs

After running experiments, the output folder contains:

### Main metrics file
~~~text
metrics.csv
~~~

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
For each scenario, the script generates:
- `cluster_cost_case1.png`
- `cluster_cost_case2.png`
- `cluster_cost_case3.png`

- `role_changes_case1.png`
- `role_changes_case2.png`
- `role_changes_case3.png`

- `lifetime_case1.png`
- `lifetime_case2.png`
- `lifetime_case3.png`

- `isolation_case1.png`
- `isolation_case2.png`
- `isolation_case3.png`

- `delay_case1.png`
- `delay_case2.png`
- `delay_case3.png`

- `pdr_case1.png`
- `pdr_case2.png`
- `pdr_case3.png`

### ICRA weight history files
When running `icra`, the script also writes:
- `weights_icra_case1_N10.csv`
- `weights_icra_case1_N20.csv`
- `weights_icra_case1_N50.csv`
- `weights_icra_case1_N100.csv`

- `weights_icra_case2_N10.csv`
- `weights_icra_case2_N20.csv`
- `weights_icra_case2_N50.csv`
- `weights_icra_case2_N100.csv`

- `weights_icra_case3_N10.csv`
- `weights_icra_case3_N20.csv`
- `weights_icra_case3_N50.csv`
- `weights_icra_case3_N100.csv`

And a summary plot:
- `icra_weight_summary.png`

## Consistency Checking

The repository includes a result checker in `scripts/check_consistency.py`.

It reads `metrics.csv` and the ICRA weight history files, then compares the generated results against the qualitative trends described in the paper.

### Run the checker on the default results folder
~~~bash
python scripts/check_consistency.py
~~~

### Run the checker on a custom results folder
~~~bash
python scripts/check_consistency.py --results-dir results_custom
~~~

## Main Modules

### `src/icra_sim/config.py`
Defines:
- supported protocol names
- supported scenario names
- simulation constants
- traffic parameters
- mobility parameters
- energy parameters
- scenario-specific configuration

### `src/icra_sim/simulator.py`
Main simulation loop:
- initializes nodes
- applies mobility updates
- runs clustering rounds
- routes packets
- updates energy state
- records weight evolution for ICRA
- computes final run metrics

### `src/icra_sim/clustering/clusterer.py`
Contains the clustering logic for:
- ICRA cluster head election
- member assignment
- forwarder selection
- cluster construction

### `src/icra_sim/clustering/utility.py`
Computes the utility factors used during cluster head election:
- energy factor
- degree / connectivity factor
- mobility stability / velocity similarity factor
- link holding time factor

### `src/icra_sim/link.py`
Computes link holding time based on relative motion and communication radius.

### `src/icra_sim/routing/router.py`
Implements routing with:
- member-to-cluster-head access
- intra-cluster forwarding
- backbone path search
- hop-based delay accumulation
- delivery probability calculation

### `src/icra_sim/metrics.py`
Defines the output metrics and helper functions for:
- isolation cluster counting
- average role changes
- first dead node time

## Notes

* The implementation is a reverse engineering reproduction in Python, not the original OPNET simulator used in the paper.
* The project compares qualitative trends against the paper, but some absolute values may differ because the simulator, routing model, and reward design are implementation-specific.
* The results directory can be regenerated at any time by rerunning `scripts/run_experiments.py`.

## Results

* Clean the previous results folder before running a new full experiment so the generated CSV files and figures only reflect the latest run.
* The current workflow is designed for scenario-level comparison across `icra`, `wca`, and `dca`.
* The generated plots are intended to be used directly in the reverse engineering report and comparison discussion.