from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

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
        scen_cfg = ScenarioConfig.for_scenario(scen)

        series_cluster_cost: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_role_changes: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_lifetime: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_iso: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_delay: Dict[str, Dict[int, float]] = {p: {} for p in protocols}
        series_pdr: Dict[str, Dict[int, float]] = {p: {} for p in protocols}

        for n in Ns:
            for proto in protocols:
                print(f"Running: scenario={scen} N={n} protocol={proto}")
                res = run_simulation(protocol=proto, scenario=scen_cfg, n_nodes=n, cfg=cfg)
                m = res.metrics

                metrics_rows.append({
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
                })

                series_cluster_cost[proto][n] = m.cluster_creation_time_s
                series_role_changes[proto][n] = m.avg_role_changes
                series_lifetime[proto][n] = m.network_lifetime_s
                series_iso[proto][n] = m.isolation_clusters
                series_delay[proto][n] = m.avg_end_to_end_delay_s
                series_pdr[proto][n] = m.packet_delivery_ratio

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

        plot_metric_vs_n(
            out_dir,
            f"Cluster protocol cost vs N ({scen})",
            "Number of nodes (N)",
            "Protocol cost (s)",
            series_cluster_cost,
            f"cluster_cost_{scen}.png",
        )
        plot_metric_vs_n(out_dir, f"Avg role changes vs N ({scen})", "Number of nodes (N)", "Avg role changes per node", series_role_changes, f"role_changes_{scen}.png")
        plot_metric_vs_n(out_dir, f"Network lifetime vs N ({scen})", "Number of nodes (N)", "First-dead time (s)", series_lifetime, f"lifetime_{scen}.png")
        plot_metric_vs_n(out_dir, f"Isolation clusters (avg) vs N ({scen})", "Number of nodes (N)", "Isolation clusters (avg)", series_iso, f"isolation_{scen}.png")
        plot_metric_vs_n(out_dir, f"End-to-end delay vs N ({scen})", "Number of nodes (N)", "Avg delay (s)", series_delay, f"delay_{scen}.png")
        plot_metric_vs_n(out_dir, f"Packet delivery ratio vs N ({scen})", "Number of nodes (N)", "PDR", series_pdr, f"pdr_{scen}.png")

    save_csv(out_dir / "metrics.csv", metrics_rows)

    if icra_weight_summary:
        plot_weight_summary(out_dir, icra_weight_summary, "icra_weight_summary.png")


if __name__ == "__main__":
    main()