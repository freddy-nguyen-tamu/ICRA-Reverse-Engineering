#!/usr/bin/env python3
"""
Consistency checker for ICRA simulation results.
Reads metrics.csv and weight files from the results directory and prints
a comparison against the paper's qualitative trends.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def load_metrics(csv_path: Path) -> List[Dict]:
    """Load metrics.csv into a list of dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_weight_summary(results_dir: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Load the last few weight entries for each scenario at N=100 and return average weights.
    Returns dict {scenario: (w1,w2,w3,w4)}.
    """
    summary = {}
    for scenario in ["case1", "case2", "case3"]:
        wfile = results_dir / f"weights_icra_{scenario}_N100.csv"
        if not wfile.exists():
            continue
        with open(wfile, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue
            # average the last 10% or at least 5 rows
            tail_len = max(5, len(rows) // 10)
            tail = rows[-tail_len:]
            w1 = sum(float(r["w1"]) for r in tail) / len(tail)
            w2 = sum(float(r["w2"]) for r in tail) / len(tail)
            w3 = sum(float(r["w3"]) for r in tail) / len(tail)
            w4 = sum(float(r["w4"]) for r in tail) / len(tail)
            summary[scenario] = (w1, w2, w3, w4)
    return summary


def best_key(d: Dict[str, float], maximize: bool = True) -> str:
    """Return key with max (or min) value."""
    return max(d, key=d.get) if maximize else min(d, key=d.get)


def print_consistency_check(
    scenario: str,
    n: int,
    cluster_cost: Dict[str, float],
    role_changes: Dict[str, float],
    lifetime: Dict[str, float],
    isolation: Dict[str, float],
    delay: Dict[str, float],
    pdr: Dict[str, float],
) -> None:
    """Print analysis for a single (scenario, N) group."""
    print(f"\n--- Consistency check for {scenario} N={n} ---")

    # 1. Cluster creation time: ICRA fastest, DCA slightly slower, WCA slowest
    best_cluster = best_key(cluster_cost, maximize=False)
    if best_cluster == "icra":
        print("✓ Cluster cost: ICRA is fastest (good)")
    else:
        print(f"✗ Cluster cost: {best_cluster} is fastest, expected ICRA")

    if cluster_cost["wca"] > cluster_cost["dca"] and cluster_cost["wca"] > cluster_cost["icra"]:
        print("  (WCA is the slowest, as expected)")
    else:
        print("  (WCA is not the slowest – check)")

    # 2. Role changes: ICRA lowest, DCA moderate, WCA highest
    low_role = best_key(role_changes, maximize=False)
    high_role = best_key(role_changes, maximize=True)
    if low_role == "icra":
        print("✓ Role changes: ICRA has the fewest (good)")
    else:
        print(f"✗ Role changes: {low_role} has the fewest, expected ICRA")
    if high_role == "wca":
        print("  (WCA has the most, as expected)")
    else:
        print("  (WCA does not have the most role changes)")

    # 3. Network lifetime: ICRA should be best in all cases
    best_life = best_key(lifetime, maximize=True)
    if best_life == "icra":
        print("✓ Network lifetime: ICRA is the best (good)")
    else:
        print(f"✗ Network lifetime: {best_life} is best, expected ICRA")

    # 4. Isolation clusters: ICRA should be lowest
    low_iso = best_key(isolation, maximize=False)
    if low_iso == "icra":
        print("✓ Isolation clusters: ICRA has the fewest (good)")
    else:
        print(f"✗ Isolation clusters: {low_iso} has the fewest, expected ICRA")

    # 5. Delay: ICRA similar to DCA, not much worse than WCA
    if n >= 50:
        if delay["icra"] <= delay["wca"] * 1.2:
            print("✓ Delay: ICRA delay is reasonable compared to WCA")
        else:
            print("✗ Delay: ICRA delay is much higher than WCA")

        if abs(delay["icra"] - delay["dca"]) / max(delay["icra"], delay["dca"]) < 0.3:
            print("  (ICRA and DCA delays are similar, as expected)")
        else:
            print("  (ICRA and DCA delays differ significantly)")

    # 6. PDR: In case3, ICRA similar to WCA, DCA lower
    if scenario == "case3" and n >= 50:
        if pdr["icra"] > pdr["dca"] * 1.1 and pdr["wca"] > pdr["dca"] * 1.1:
            print("✓ PDR: DCA has the lowest PDR in case3 (expected)")
        else:
            print("✗ PDR: DCA is not clearly the worst in case3")
        if abs(pdr["icra"] - pdr["wca"]) / max(pdr["icra"], pdr["wca"]) < 0.2:
            print("  (ICRA and WCA PDR are similar, good)")
        else:
            print("  (ICRA and WCA PDR differ more than expected)")

    print("----------------------------------------\n")


def print_weight_comment(weight_summary: Dict[str, Tuple[float, float, float, float]]) -> None:
    """Print a short note about learned weights."""
    if not weight_summary:
        return
    print("\n--- Learned weights summary (tail average at N=100) ---")
    for scenario, (w1, w2, w3, w4) in weight_summary.items():
        print(f"{scenario}: w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}, w4={w4:.2f}")
    print("(Expected: case1 energy-heavy, case2 degree-heavy, case3 LHT-heavy)")
    print("-----------------------------------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check consistency of ICRA simulation results with the paper.")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing metrics.csv and weight files (default: results)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    metrics_path = results_dir / "metrics.csv"

    try:
        metrics = load_metrics(metrics_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Group metrics by scenario and N
    groups: Dict[Tuple[str, int], List[Dict]] = {}
    for row in metrics:
        key = (row["scenario"], int(row["N"]))
        groups.setdefault(key, []).append(row)

    # For each group, collect values for the three protocols
    for (scenario, n), rows in sorted(groups.items()):
        if len(rows) != 3:
            print(f"Warning: {scenario} N={n} has {len(rows)} entries, expected 3 (icra, wca, dca)")
            continue

        # Build dictionaries for each metric
        cluster_cost = {}
        role_changes = {}
        lifetime = {}
        isolation = {}
        delay = {}
        pdr = {}

        for row in rows:
            proto = row["protocol"]
            cluster_cost[proto] = float(row["cluster_protocol_cost_s"])
            role_changes[proto] = float(row["avg_role_changes"])
            lifetime[proto] = float(row["network_lifetime_s"])
            isolation[proto] = int(row["isolation_clusters_avg"])
            delay[proto] = float(row["avg_end_to_end_delay_s"])
            pdr[proto] = float(row["packet_delivery_ratio"])

        # Ensure all three protocols present
        if set(cluster_cost.keys()) != {"icra", "wca", "dca"}:
            print(f"Warning: {scenario} N={n} missing some protocols, skipping")
            continue

        print_consistency_check(
            scenario, n,
            cluster_cost, role_changes, lifetime, isolation, delay, pdr
        )

    # Load weight summary and print comment
    weight_summary = load_weight_summary(results_dir)
    print_weight_comment(weight_summary)


if __name__ == "__main__":
    main()