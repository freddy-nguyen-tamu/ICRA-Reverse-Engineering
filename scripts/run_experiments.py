from __future__ import annotations

import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from icra_sim.config import ProtocolName, ScenarioConfig, ScenarioName, SimConfig
from icra_sim.simulator import run_simulation


def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_metric_vs_n(
    out_dir: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    data: Dict[str, Dict[int, float]],
    fname: str,
) -> None:
    plt.figure()
    for proto, series in data.items():
        ns = sorted(series.keys())
        ys = [series[n] for n in ns]
        plt.plot(ns, ys, marker="o", label=proto)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()


def plot_weight_summary(
    out_dir: Path,
    weight_summary: Dict[str, Tuple[float, float, float, float]],
    fname: str,
) -> None:
    scenarios = list(weight_summary.keys())
    w1 = [weight_summary[s][0] for s in scenarios]
    w2 = [weight_summary[s][1] for s in scenarios]
    w3 = [weight_summary[s][2] for s in scenarios]
    w4 = [weight_summary[s][3] for s in scenarios]

    plt.figure()
    xs = range(len(scenarios))
    plt.plot(xs, w1, marker="o", label="w1 (energy)")
    plt.plot(xs, w2, marker="o", label="w2 (degree)")
    plt.plot(xs, w3, marker="o", label="w3 (vel sim)")
    plt.plot(xs, w4, marker="o", label="w4 (LHT)")
    plt.xticks(xs, scenarios)
    plt.ylim(0, 1)
    plt.title("ICRA learned weight summary per scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()


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
    """
    Compare the three protocols for this (scenario, N) against the paper's expected trends.
    Print a short analysis to the terminal.
    """
    print(f"\n--- Consistency check for {scenario} N={n} ---")

    def best_key(d, maximize=True):
        return max(d, key=d.get) if maximize else min(d, key=d.get)

    # 1. Cluster creation time: ICRA should be lowest, DCA slightly higher, WCA highest
    best_cluster = best_key(cluster_cost, maximize=False)
    if best_cluster == "icra":
        print("✓ Cluster cost: ICRA is fastest (good)")
    else:
        print(f"✗ Cluster cost: {best_cluster} is fastest, expected ICRA")

    if cluster_cost["wca"] > cluster_cost["dca"] and cluster_cost["wca"] > cluster_cost["icra"]:
        print("  (WCA is the slowest, as expected)")
    else:
        print("  (WCA is not the slowest – check)")

    # 2. Role changes: ICRA should be lowest, DCA moderate, WCA highest
    low_role = min(role_changes, key=role_changes.get)
    high_role = max(role_changes, key=role_changes.get)
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
    low_iso = min(isolation, key=isolation.get)
    if low_iso == "icra":
        print("✓ Isolation clusters: ICRA has the fewest (good)")
    else:
        print(f"✗ Isolation clusters: {low_iso} has the fewest, expected ICRA")

    # 5. Delay: ICRA should be similar to DCA, not much worse than WCA
    #    (paper only gives case1 plot where ICRA and DCA are close, WCA rises at high N)
    if n >= 50:
        if delay["icra"] <= delay["wca"] * 1.2:
            print("✓ Delay: ICRA delay is reasonable compared to WCA")
        else:
            print("✗ Delay: ICRA delay is much higher than WCA")

        if abs(delay["icra"] - delay["dca"]) / max(delay["icra"], delay["dca"]) < 0.3:
            print("  (ICRA and DCA delays are similar, as expected)")
        else:
            print("  (ICRA and DCA delays differ significantly)")

    # 6. PDR: In case3, ICRA should be similar to WCA, DCA lower (for other cases just note ordering)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["icra", "wca", "dca", "all"], default="all")
    ap.add_argument("--scenario", choices=["case1", "case2", "case3", "all"], default="all")
    ap.add_argument("--n", type=int, default=0, help="If set (>0), run only this N.")
    ap.add_argument("--out", type=str, default="results", help="Output directory.")
    args = ap.parse_args()

    cfg = SimConfig()

    protocols: List[ProtocolName]
    if args.protocol == "all":
        protocols = ["icra", "wca", "dca"]
    else:
        protocols = [args.protocol]  # type: ignore

    scenarios: List[ScenarioName]
    if args.scenario == "all":
        scenarios = ["case1", "case2", "case3"]
    else:
        scenarios = [args.scenario]  # type: ignore

    Ns = [10, 20, 50, 100] if args.n <= 0 else [args.n]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict] = []
    icra_weight_summary: Dict[str, Tuple[float, float, float, float]] = {}

    for scen in scenarios:
        scen_cfg = ScenarioConfig.from_name(scen)

        series_cluster_cost: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_role_changes: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_lifetime: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_iso: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_delay: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_pdr: Dict[str, Dict[int, float]] = {p: {} for p in protocols}

        for n in Ns:
            # Store results for this N to pass to consistency check
            results_this_n: Dict[str, Dict] = {p: {} for p in protocols}

            for proto in protocols:
                print(f"Running: scenario={scen} N={n} protocol={proto}")
                res = run_simulation(
                    protocol=proto,
                    scenario_cfg=scen_cfg,
                    n_nodes=n,
                    cfg=cfg,
                )
                m = res.metrics

                metrics_rows.append(
                    {
                        "scenario": scen,
                        "N": n,
                        "protocol": proto,
                        "cluster_protocol_cost_s": m.cluster_creation_time_s,
                        "avg_role_changes": m.avg_role_changes,
                        "network_lifetime_s": m.network_lifetime_s,
                        "dead_nodes": m.dead_nodes,
                        "isolation_clusters_avg": m.isolation_clusters,
                        "avg_end_to_end_delay_s": m.avg_end_to_end_delay_s,
                        "packet_delivery_ratio": m.packet_delivery_ratio,
                    }
                )

                series_cluster_cost[proto][n] = m.cluster_creation_time_s
                series_role_changes[proto][n] = m.avg_role_changes
                series_lifetime[proto][n] = m.network_lifetime_s
                series_iso[proto][n] = m.isolation_clusters
                series_delay[proto][n] = m.avg_end_to_end_delay_s
                series_pdr[proto][n] = m.packet_delivery_ratio

                # Store for this N
                results_this_n[proto] = {
                    "cluster_cost": m.cluster_creation_time_s,
                    "role_changes": m.avg_role_changes,
                    "lifetime": m.network_lifetime_s,
                    "isolation": m.isolation_clusters,
                    "delay": m.avg_end_to_end_delay_s,
                    "pdr": m.packet_delivery_ratio,
                }

                if proto == "icra":
                    w_path = out_dir / f"weights_icra_{scen}_N{n}.csv"
                    with w_path.open("w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["round", "w1", "w2", "w3", "w4"])
                        for i, (w1, w2, w3, w4) in enumerate(res.weight_history):
                            w.writerow([i, w1, w2, w3, w4])

                    if n == 100 and res.weight_history:
                        tail = res.weight_history[-max(5, len(res.weight_history) // 10):]
                        w1 = sum(x[0] for x in tail) / len(tail)
                        w2 = sum(x[1] for x in tail) / len(tail)
                        w3 = sum(x[2] for x in tail) / len(tail)
                        w4 = sum(x[3] for x in tail) / len(tail)
                        icra_weight_summary[scen] = (w1, w2, w3, w4)

            # After running all protocols for this N, check consistency
            if len(protocols) == 3:   # only check if we ran all three
                cluster_dict = {p: results_this_n[p]["cluster_cost"] for p in protocols}
                role_dict   = {p: results_this_n[p]["role_changes"] for p in protocols}
                life_dict   = {p: results_this_n[p]["lifetime"] for p in protocols}
                iso_dict    = {p: results_this_n[p]["isolation"] for p in protocols}
                delay_dict  = {p: results_this_n[p]["delay"] for p in protocols}
                pdr_dict    = {p: results_this_n[p]["pdr"] for p in protocols}
                print_consistency_check(scen, n, cluster_dict, role_dict, life_dict,
                                        iso_dict, delay_dict, pdr_dict)

        plot_metric_vs_n(
            out_dir,
            f"Cluster protocol cost vs N ({scen})",
            "Number of nodes (N)",
            "Protocol cost (s)",
            series_cluster_cost,
            f"cluster_cost_{scen}.png",
        )
        plot_metric_vs_n(
            out_dir,
            f"Avg role changes vs N ({scen})",
            "Number of nodes (N)",
            "Avg role changes per node",
            series_role_changes,
            f"role_changes_{scen}.png",
        )
        plot_metric_vs_n(
            out_dir,
            f"Network lifetime vs N ({scen})",
            "Number of nodes (N)",
            "First-dead time (s)",
            series_lifetime,
            f"lifetime_{scen}.png",
        )
        plot_metric_vs_n(
            out_dir,
            f"Isolation clusters (avg) vs N ({scen})",
            "Number of nodes (N)",
            "Isolation clusters (avg)",
            series_iso,
            f"isolation_{scen}.png",
        )
        plot_metric_vs_n(
            out_dir,
            f"End-to-end delay vs N ({scen})",
            "Number of nodes (N)",
            "Avg delay (s)",
            series_delay,
            f"delay_{scen}.png",
        )
        plot_metric_vs_n(
            out_dir,
            f"Packet delivery ratio vs N ({scen})",
            "Number of nodes (N)",
            "PDR",
            series_pdr,
            f"pdr_{scen}.png",
        )

    save_csv(out_dir / "metrics.csv", metrics_rows)

    if icra_weight_summary:
        plot_weight_summary(out_dir, icra_weight_summary, "icra_weight_summary.png")


if __name__ == "__main__":
    main()