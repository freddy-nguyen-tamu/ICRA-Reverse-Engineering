from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional


EXPECTED_SCENARIOS = ["case1", "case2", "case3"]
EXPECTED_PROTOCOLS = ["icra", "wca", "dca"]
EXPECTED_NS = [10, 20, 50, 100]

REQUIRED_COLUMNS = [
    "scenario",
    "N",
    "protocol",
    "cluster_protocol_cost_s",
    "avg_role_changes",
    "network_lifetime_s",
    "dead_nodes",
    "isolation_clusters_avg",
    "avg_end_to_end_delay_s",
    "packet_delivery_ratio",
]

# Paper-inspired expectations used for qualitative reproducibility checks.
# These are not strict numeric oracles; they are qualitative reference expectations.
PAPER_EXPECTATIONS = {
    "cluster_time": {
        "description": "ICRA fastest, DCA next, WCA slowest",
    },
    "role_changes": {
        "description": "ICRA should generally have the fewest role changes",
    },
    "lifetime": {
        "description": "ICRA is expected to be best or near-best; if not, note discrepancy",
    },
    "isolation": {
        "description": "ICRA should generally be among the best and better than DCA",
        "case2_n100_icra_reference": 15.0,
    },
    "delay": {
        "description": "Delay should not collapse to an unrealistically tiny band; ICRA and DCA often close",
    },
    "pdr": {
        "description": "Case3 at larger N: ICRA and WCA should be close, DCA lower",
        "case3_reference_range_pct": (35.0, 70.0),
    },
    "weights": {
        "description": "ICRA weights should remain normalized and show scenario-dependent adaptation",
    },
}


@dataclass
class CheckMessage:
    level: str  # MATCH, PARTIAL_MATCH, MISMATCH, WARNING, ERROR, INFO
    category: str
    message: str


@dataclass
class CaseResult:
    scenario: str
    n: int
    messages: List[CheckMessage]


def parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception as exc:
        raise ValueError(f"Could not parse float for column '{key}' in row: {row}") from exc


def parse_int(row: Dict[str, str], key: str) -> int:
    try:
        return int(float(row[key]))
    except Exception as exc:
        raise ValueError(f"Could not parse int for column '{key}' in row: {row}") from exc


def relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def best_key(d: Dict[str, float], maximize: bool) -> str:
    return max(d, key=d.get) if maximize else min(d, key=d.get)


def rank_items(d: Dict[str, float], lower_better: bool) -> List[Tuple[str, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=not lower_better)


def rank_str(d: Dict[str, float], lower_better: bool, decimals: int = 6) -> str:
    items = rank_items(d, lower_better)
    sep = " < " if lower_better else " > "
    parts = [f"{k}:{v:.{decimals}f}" for k, v in items]
    return sep.join(parts)


def load_metrics(metrics_path: Path) -> List[Dict[str, str]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_path}")

    with metrics_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("metrics.csv has no header row.")

        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"metrics.csv is missing required columns: {missing}")

        rows = list(reader)
        if not rows:
            raise ValueError("metrics.csv is empty.")

    return rows


def validate_metrics_completeness(rows: List[Dict[str, str]]) -> List[CheckMessage]:
    messages: List[CheckMessage] = []

    seen = set()
    duplicate_keys = []
    for row in rows:
        key = (row["scenario"], parse_int(row, "N"), row["protocol"])
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)

    expected = {(s, n, p) for s in EXPECTED_SCENARIOS for n in EXPECTED_NS for p in EXPECTED_PROTOCOLS}
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)

    if missing:
        messages.append(CheckMessage(
            "ERROR",
            "coverage",
            f"metrics.csv is missing expected experiment rows: {missing}"
        ))
    else:
        messages.append(CheckMessage(
            "MATCH",
            "coverage",
            "metrics.csv contains all expected scenario/N/protocol combinations."
        ))

    if duplicate_keys:
        messages.append(CheckMessage(
            "ERROR",
            "coverage",
            f"metrics.csv contains duplicate scenario/N/protocol rows: {duplicate_keys}"
        ))

    if unexpected:
        messages.append(CheckMessage(
            "WARNING",
            "coverage",
            f"metrics.csv contains unexpected scenario/N/protocol rows: {unexpected}"
        ))

    return messages


def rows_to_nested(rows: List[Dict[str, str]]) -> Dict[Tuple[str, int], Dict[str, Dict[str, float]]]:
    data: Dict[Tuple[str, int], Dict[str, Dict[str, float]]] = {}

    for row in rows:
        scenario = row["scenario"]
        n = parse_int(row, "N")
        protocol = row["protocol"].lower()

        data.setdefault((scenario, n), {})[protocol] = {
            "cluster_cost": parse_float(row, "cluster_protocol_cost_s"),
            "role_changes": parse_float(row, "avg_role_changes"),
            "lifetime": parse_float(row, "network_lifetime_s"),
            "dead_nodes": parse_float(row, "dead_nodes"),
            "isolation": parse_float(row, "isolation_clusters_avg"),
            "delay": parse_float(row, "avg_end_to_end_delay_s"),
            "pdr": parse_float(row, "packet_delivery_ratio"),
        }

    return data


def check_cluster_time(case: CaseResult, cluster_cost: Dict[str, float]) -> None:
    cc_best = best_key(cluster_cost, maximize=False)
    wca_slowest = cluster_cost["wca"] >= cluster_cost["icra"] and cluster_cost["wca"] >= cluster_cost["dca"]

    if cc_best == "icra" and wca_slowest:
        case.messages.append(CheckMessage(
            "MATCH",
            "cluster_time",
            f"Cluster creation time matches the paper trend: {rank_str(cluster_cost, lower_better=True, decimals=3)}"
        ))
    elif cc_best == "icra":
        case.messages.append(CheckMessage(
            "PARTIAL_MATCH",
            "cluster_time",
            f"ICRA is fastest, but WCA is not clearly slowest: {rank_str(cluster_cost, lower_better=True, decimals=3)}"
        ))
    else:
        case.messages.append(CheckMessage(
            "MISMATCH",
            "cluster_time",
            f"Cluster creation time does not match expected ordering: {rank_str(cluster_cost, lower_better=True, decimals=3)}"
        ))


def check_role_changes(case: CaseResult, role_changes: Dict[str, float]) -> None:
    rc_best = best_key(role_changes, maximize=False)
    if rc_best == "icra":
        case.messages.append(CheckMessage(
            "MATCH",
            "role_changes",
            f"ICRA has the fewest role changes: {rank_str(role_changes, lower_better=True, decimals=2)}"
        ))
    else:
        case.messages.append(CheckMessage(
            "MISMATCH",
            "role_changes",
            f"ICRA does not have the fewest role changes: {rank_str(role_changes, lower_better=True, decimals=2)}"
        ))


def check_lifetime(case: CaseResult, lifetime: Dict[str, float]) -> None:
    life_best = best_key(lifetime, maximize=True)
    if life_best == "icra":
        case.messages.append(CheckMessage(
            "MATCH",
            "lifetime",
            f"ICRA has the best network lifetime: {rank_str(lifetime, lower_better=False, decimals=0)}"
        ))
    elif lifetime["icra"] >= lifetime["wca"]:
        case.messages.append(CheckMessage(
            "PARTIAL_MATCH",
            "lifetime",
            f"ICRA beats WCA but not the best overall: {rank_str(lifetime, lower_better=False, decimals=0)}"
        ))
    else:
        case.messages.append(CheckMessage(
            "MISMATCH",
            "lifetime",
            f"ICRA is not competitive on network lifetime: {rank_str(lifetime, lower_better=False, decimals=0)}"
        ))


def check_isolation(case: CaseResult, isolation: Dict[str, float]) -> None:
    iso_best = best_key(isolation, maximize=False)

    if iso_best == "icra":
        case.messages.append(CheckMessage(
            "MATCH",
            "isolation",
            f"ICRA has the fewest isolation clusters: {rank_str(isolation, lower_better=True, decimals=0)}"
        ))
    elif isolation["icra"] < isolation["dca"]:
        case.messages.append(CheckMessage(
            "PARTIAL_MATCH",
            "isolation",
            f"ICRA is not best, but still improves over DCA: {rank_str(isolation, lower_better=True, decimals=0)}"
        ))
    else:
        case.messages.append(CheckMessage(
            "MISMATCH",
            "isolation",
            f"ICRA does not improve isolation clusters in the expected direction: {rank_str(isolation, lower_better=True, decimals=0)}"
        ))

    if case.scenario == "case2" and case.n == 100:
        ref = PAPER_EXPECTATIONS["isolation"]["case2_n100_icra_reference"]
        observed = isolation["icra"]
        rd = relative_diff(observed, ref)
        if rd <= 0.20:
            case.messages.append(CheckMessage(
                "MATCH",
                "isolation_scale",
                f"ICRA case2 N=100 isolation is close to paper reference (~{ref:.0f}): observed {observed:.2f}"
            ))
        else:
            case.messages.append(CheckMessage(
                "WARNING",
                "isolation_scale",
                f"ICRA case2 N=100 isolation differs from paper reference (~{ref:.0f}): observed {observed:.2f}"
            ))


def check_delay(case: CaseResult, delay: Dict[str, float]) -> None:
    icra_vs_dca_close = relative_diff(delay["icra"], delay["dca"]) < 0.30
    if icra_vs_dca_close:
        case.messages.append(CheckMessage(
            "MATCH",
            "delay_order",
            f"ICRA and DCA delays are relatively close: {rank_str(delay, lower_better=True, decimals=6)}"
        ))
    else:
        case.messages.append(CheckMessage(
            "MISMATCH",
            "delay_order",
            f"ICRA and DCA delays are not close: {rank_str(delay, lower_better=True, decimals=6)}"
        ))

    max_delay = max(delay.values())
    min_delay = min(delay.values())

    if max_delay < 0.02:
        case.messages.append(CheckMessage(
            "WARNING",
            "delay_scale",
            f"All delays are extremely small (<20 ms), suggesting the analytic hop-delay model dominates: {rank_str(delay, lower_better=True, decimals=6)}"
        ))
    elif relative_diff(max_delay, min_delay) < 0.20:
        case.messages.append(CheckMessage(
            "WARNING",
            "delay_scale",
            f"Delay varies weakly across protocols, weaker than expected from the paper: {rank_str(delay, lower_better=True, decimals=6)}"
        ))


def check_pdr(case: CaseResult, pdr: Dict[str, float]) -> None:
    if case.scenario == "case3" and case.n >= 50:
        dca_lowest = best_key(pdr, maximize=False) == "dca"
        icra_wca_close = relative_diff(pdr["icra"], pdr["wca"]) < 0.20

        if dca_lowest and icra_wca_close:
            case.messages.append(CheckMessage(
                "MATCH",
                "pdr_order",
                f"Case3 PDR matches paper trend: ICRA and WCA close, DCA lower: {rank_str(pdr, lower_better=False, decimals=5)}"
            ))
        elif dca_lowest:
            case.messages.append(CheckMessage(
                "PARTIAL_MATCH",
                "pdr_order",
                f"DCA is lowest as expected, but ICRA and WCA are not especially close: {rank_str(pdr, lower_better=False, decimals=5)}"
            ))
        else:
            case.messages.append(CheckMessage(
                "MISMATCH",
                "pdr_order",
                f"Case3 PDR does not match expected ordering: {rank_str(pdr, lower_better=False, decimals=5)}"
            ))
    else:
        if pdr["icra"] >= pdr["dca"] and pdr["icra"] >= pdr["wca"]:
            case.messages.append(CheckMessage(
                "MATCH",
                "pdr_order",
                f"ICRA has the highest PDR in this setting: {rank_str(pdr, lower_better=False, decimals=5)}"
            ))
        elif pdr["icra"] >= pdr["dca"] or pdr["icra"] >= pdr["wca"]:
            case.messages.append(CheckMessage(
                "PARTIAL_MATCH",
                "pdr_order",
                f"ICRA is competitive on PDR but not dominant: {rank_str(pdr, lower_better=False, decimals=5)}"
            ))
        else:
            case.messages.append(CheckMessage(
                "MISMATCH",
                "pdr_order",
                f"ICRA underperforms both baselines on PDR: {rank_str(pdr, lower_better=False, decimals=5)}"
            ))

    max_pdr = max(pdr.values())
    lo_pct, hi_pct = PAPER_EXPECTATIONS["pdr"]["case3_reference_range_pct"]

    if max_pdr < 0.10:
        case.messages.append(CheckMessage(
            "WARNING",
            "pdr_scale",
            f"PDR values are far below the paper's scale ({lo_pct:.0f}%–{hi_pct:.0f}% range mentioned for case3 behavior); simulator/routing/connectivity likely differ."
        ))


def check_case(case_data: Dict[str, Dict[str, float]], scenario: str, n: int) -> CaseResult:
    case = CaseResult(scenario=scenario, n=n, messages=[])

    missing_protocols = [p for p in EXPECTED_PROTOCOLS if p not in case_data]
    if missing_protocols:
        case.messages.append(CheckMessage(
            "ERROR",
            "coverage",
            f"Missing protocols for {scenario}, N={n}: {missing_protocols}"
        ))
        return case

    cluster_cost = {p: case_data[p]["cluster_cost"] for p in EXPECTED_PROTOCOLS}
    role_changes = {p: case_data[p]["role_changes"] for p in EXPECTED_PROTOCOLS}
    lifetime = {p: case_data[p]["lifetime"] for p in EXPECTED_PROTOCOLS}
    isolation = {p: case_data[p]["isolation"] for p in EXPECTED_PROTOCOLS}
    delay = {p: case_data[p]["delay"] for p in EXPECTED_PROTOCOLS}
    pdr = {p: case_data[p]["pdr"] for p in EXPECTED_PROTOCOLS}

    check_cluster_time(case, cluster_cost)
    check_role_changes(case, role_changes)
    check_lifetime(case, lifetime)
    check_isolation(case, isolation)
    check_delay(case, delay)
    check_pdr(case, pdr)

    return case


def load_weight_history(path: Path) -> Optional[List[Tuple[float, float, float, float]]]:
    if not path.exists():
        return None

    rows: List[Tuple[float, float, float, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"round", "w1", "w2", "w3", "w4"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Weight file {path} is malformed.")

        for row in reader:
            rows.append((
                float(row["w1"]),
                float(row["w2"]),
                float(row["w3"]),
                float(row["w4"]),
            ))
    return rows


def weight_tail_mean(history: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    tail_len = max(5, len(history) // 10)
    tail = history[-tail_len:]
    return (
        mean(x[0] for x in tail),
        mean(x[1] for x in tail),
        mean(x[2] for x in tail),
        mean(x[3] for x in tail),
    )


def weight_tail_variation(history: List[Tuple[float, float, float, float]]) -> float:
    tail_len = max(5, len(history) // 10)
    tail = history[-tail_len:]
    # mean absolute deviation over all 4 weights
    center = weight_tail_mean(history)
    total = 0.0
    count = 0
    for row in tail:
        for i in range(4):
            total += abs(row[i] - center[i])
            count += 1
    return total / max(count, 1)


def check_weights(results_dir: Path) -> List[CheckMessage]:
    messages: List[CheckMessage] = []
    summary: Dict[str, Tuple[float, float, float, float]] = {}

    for scenario in EXPECTED_SCENARIOS:
        for n in EXPECTED_NS:
            path = results_dir / f"weights_icra_{scenario}_N{n}.csv"
            history = load_weight_history(path)
            if history is None:
                messages.append(CheckMessage(
                    "ERROR",
                    "weights",
                    f"Missing ICRA weight history file: {path.name}"
                ))
                continue

            if not history:
                messages.append(CheckMessage(
                    "ERROR",
                    "weights",
                    f"Empty ICRA weight history file: {path.name}"
                ))
                continue

            # normalization check
            bad_rows = []
            for idx, row in enumerate(history):
                s = sum(row)
                if abs(s - 1.0) > 1e-6:
                    bad_rows.append((idx, s))

            if bad_rows:
                messages.append(CheckMessage(
                    "MISMATCH",
                    "weights",
                    f"{path.name} contains rows whose weights do not sum to 1. First few: {bad_rows[:5]}"
                ))

            # simple stabilization check
            mad = weight_tail_variation(history)
            if mad <= 0.05:
                messages.append(CheckMessage(
                    "MATCH",
                    "weights",
                    f"{path.name} shows stable tail behavior (mean abs tail variation {mad:.4f})."
                ))
            else:
                messages.append(CheckMessage(
                    "WARNING",
                    "weights",
                    f"{path.name} tail is still noisy (mean abs tail variation {mad:.4f})."
                ))

            if n == 100:
                summary[scenario] = weight_tail_mean(history)

    if len(summary) == 3:
        # scenario-dependent adaptation sanity check
        c1 = summary["case1"]
        c2 = summary["case2"]
        c3 = summary["case3"]

        distinct = (
            relative_diff(c1[0], c2[0]) > 0.10 or
            relative_diff(c1[0], c3[0]) > 0.10 or
            relative_diff(c2[1], c1[1]) > 0.10 or
            relative_diff(c3[3], c1[3]) > 0.10
        )

        if distinct:
            messages.append(CheckMessage(
                "MATCH",
                "weights",
                f"Scenario-dependent weight adaptation is visible at N=100: case1={tuple(round(x, 3) for x in c1)}, case2={tuple(round(x, 3) for x in c2)}, case3={tuple(round(x, 3) for x in c3)}"
            ))
        else:
            messages.append(CheckMessage(
                "WARNING",
                "weights",
                f"Scenario-dependent adaptation is weak at N=100: case1={tuple(round(x, 3) for x in c1)}, case2={tuple(round(x, 3) for x in c2)}, case3={tuple(round(x, 3) for x in c3)}"
            ))

    return messages


def summarize_counts(messages: List[CheckMessage]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in messages:
        counts[m.level] = counts.get(m.level, 0) + 1
    return counts


def write_text_report(
    out_path: Path,
    coverage_messages: List[CheckMessage],
    weight_messages: List[CheckMessage],
    case_results: List[CaseResult],
) -> None:
    lines: List[str] = []
    lines.append("Reproducibility Consistency Report")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1. Dataset / output completeness")
    for m in coverage_messages:
        lines.append(f"[{m.level}] {m.category}: {m.message}")
    lines.append("")

    lines.append("2. ICRA weight-history checks")
    for m in weight_messages:
        lines.append(f"[{m.level}] {m.category}: {m.message}")
    lines.append("")

    lines.append("3. Per-case qualitative comparison")
    for cr in sorted(case_results, key=lambda x: (x.scenario, x.n)):
        lines.append(f"--- {cr.scenario}, N={cr.n} ---")
        for m in cr.messages:
            lines.append(f"[{m.level}] {m.category}: {m.message}")
        lines.append("")

    all_messages = coverage_messages + weight_messages + [m for cr in case_results for m in cr.messages]
    counts = summarize_counts(all_messages)

    lines.append("4. Overall summary")
    lines.append(f"Message counts: {counts}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- MATCH = agrees with the paper's expected qualitative behavior")
    lines.append("- PARTIAL_MATCH = some alignment, but not full agreement")
    lines.append("- MISMATCH = diverges from the paper's expected behavior")
    lines.append("- WARNING = important caveat or scale mismatch")
    lines.append("- ERROR = missing or invalid experiment output")
    lines.append("")
    lines.append("Important note:")
    lines.append("This checker does not claim exact equivalence to the original paper or OPNET implementation.")
    lines.append("It checks whether the reproduction matches the paper structurally, qualitatively, and, where possible, quantitatively.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing metrics.csv and weight history CSV files."
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    metrics_path = results_dir / "metrics.csv"

    rows = load_metrics(metrics_path)
    coverage_messages = validate_metrics_completeness(rows)
    nested = rows_to_nested(rows)

    case_results: List[CaseResult] = []
    for scenario in EXPECTED_SCENARIOS:
        for n in EXPECTED_NS:
            case_data = nested.get((scenario, n), {})
            case_results.append(check_case(case_data, scenario, n))

    weight_messages = check_weights(results_dir)

    # Terminal output
    print("\n=== Reproducibility Coverage Check ===")
    for m in coverage_messages:
        print(f"[{m.level}] {m.message}")

    print("\n=== ICRA Weight Checks ===")
    for m in weight_messages:
        print(f"[{m.level}] {m.message}")

    print("\n=== Per-Case Reproducibility Check ===")
    for cr in sorted(case_results, key=lambda x: (x.scenario, x.n)):
        print(f"\n--- {cr.scenario}, N={cr.n} ---")
        for m in cr.messages:
            print(f"[{m.level}] {m.category}: {m.message}")

    # Write text report
    report_path = results_dir / "reproducibility_report.txt"
    write_text_report(report_path, coverage_messages, weight_messages, case_results)

    # Write JSON report
    summary_json = {
        "coverage_messages": [asdict(m) for m in coverage_messages],
        "weight_messages": [asdict(m) for m in weight_messages],
        "case_results": [
            {
                "scenario": cr.scenario,
                "n": cr.n,
                "messages": [asdict(m) for m in cr.messages],
            }
            for cr in case_results
        ],
    }
    json_path = results_dir / "reproducibility_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"\nWrote text report: {report_path}")
    print(f"Wrote JSON summary: {json_path}")


if __name__ == "__main__":
    main()