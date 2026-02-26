# ICRA Simulator (Python)

## About it.

This is a lightweight, from-scratch discrete-time simulator that implements the core ideas of:

ICRA: An Intelligent Clustering Routing Approach for UAV Ad Hoc Networks  
Jingjing Guo et al., IEEE T-ITS, 2023.

The original paper evaluated the protocol in OPNET (a discrete-event network simulator).  
This repository re-implements the algorithmic components (clustering, reinforcement learning-based weight adjustment, and routing) in pure Python so you can reproduce qualitatively similar behaviors and trends such as:

- Cluster-head role changes  
- Packet Delivery Ratio (PDR)  
- End-to-end delay  
- Isolation clusters  
- Network lifetime  
- Weight evolution over time  

The goal is algorithmic clarity and reproducibility without requiring OPNET.

---

## What is implemented

### Clustering Module (Section III-C)

Implements the clustering logic described in the paper:

- Neighbor discovery within communication radius  
- Utility factor computation:
  - `s1`: residual energy ratio  
  - `s2`: degree centrality  
  - `s3`: velocity similarity (implemented as mean similarity for “higher is better”)  
  - `s4`: link expiration time (LET), normalized  
- Cluster-head (CH) election using local maximum utility  
- Cluster formation (members join best CH with LET threshold)  
- Inter-cluster forwarding node selection  

Note:  
The paper’s Eq.(12) resembles a variance formulation although the text refers to it as velocity similarity.  
To maintain consistency with “maximum utility wins”, `s3` is implemented as mean similarity (Eq.(11)).

---

### RL-Based Clustering Strategy Adjustment (Section III-D)

Implements Q-learning to dynamically adjust weight vectors `(w1, w2, w3, w4)`.

- Network state is discretized using entropy of factor distributions  
- Action space: all weight vectors in increments of 0.05 that sum to 1  
- Reward combines:
  - Topology stability (role-change frequency)
  - Energy consumption rate  

This enables adaptive clustering behavior under dynamic UAV mobility.

---

### Routing Approach (Section III-E)

Implements greedy forwarding based on node roles:

- Cluster Head (CH)
- Forwarding node
- Member node

Packets are forwarded hop-by-hop over current one-hop connectivity.

---

### Baselines (for comparison)

Includes simplified implementations of:

- WCA-like weighted clustering  
- DCA-like weighted clustering  

These are not exact reproductions of the original WCA/DCA papers, but are included to generate comparison trends similar to those shown in the ICRA paper.

---

## Setup

* Clone this repository
```
git clone <your-repository-url>
cd <your-repository-folder>
```

* If you don't use Python 3.10+ (recommended), create a dedicated environment:

Using conda:
```bash
conda create -n icra_env python=3.10
conda activate icra_env
```

* Create and activate a Python virtual environment (optional, but recommended)

For Linux / Git Bash:
```
python -m venv .venv
source .venv/bin/activate
```

For Windows:
```
python -m venv .venv
.venv\Scripts\activate
```

* Install the dependencies
```
pip install --upgrade pip
pip install -e .
```

---

## Running Experiments

### Run Paper-Style Experiments (All Scenarios)

To reproduce all scenarios used for comparison:

```
python scripts/run_experiments.py --out results
```

This will:

- Run `case1`, `case2`, `case3`
- For number of nodes `N ∈ {10, 20, 50, 100}`
- For protocols:
  - `icra`
  - `wca`
  - `dca`
- Save CSV files and PNG plots to the `results/` directory

---

### Run a Single Experiment

To run a specific protocol, scenario, and network size:

```
python scripts/run_experiments.py --protocol icra --scenario case3 --n 100 --out results
```

This allows focused evaluation of a single configuration.

---

## Outputs

After running experiments, the following files are generated:

- `results/metrics.csv`  
  Aggregated metrics per `(protocol, scenario, N)`

- `results/weights_icra.csv`  
  Chosen weight vectors over time for ICRA (RL evolution)

- Plots (PNG format), including:
  - Role change frequency
  - Packet Delivery Ratio (PDR)
  - Average delay
  - Isolation clusters
  - Network lifetime
  - Weight evolution over time

---

## Simulation Overview

The simulator follows a discrete-time loop:

1. Update UAV positions  
2. Recompute neighbor sets  
3. Compute utility factors  
4. Elect cluster heads  
5. Adjust weights using RL (ICRA only)  
6. Perform routing for active flows  
7. Collect metrics  

Metrics tracked:

- Packet Delivery Ratio (PDR)  
- Average delay  
- Cluster-head change frequency  
- Isolation cluster count  
- Energy consumption rate  
- Network lifetime  
- Weight evolution (ICRA)

---

## Results

* Clear the previous `results/` folder before running new experiments to avoid mixing outputs.
* The generated metrics and plots allow qualitative comparison with the original paper.
* Exact numerical values may differ from OPNET results, but overall behavioral trends should match.

---

## License

This code is provided for research and educational purposes.

---

## ToDo List

* Add configurable mobility models (Gauss-Markov, Random Waypoint variants)
* Add reproducible experiment presets (paper-like scenarios)
* Add more detailed logging and verbose mode
* Improve routing realism (queueing delay, buffer overflow)
* Add additional baselines (e.g., mobility-aware clustering)
* Add visualization of dynamic cluster formation
* Add reproducibility seed control across all modules
* Provide ready-to-run scripts for generating publication-style plots