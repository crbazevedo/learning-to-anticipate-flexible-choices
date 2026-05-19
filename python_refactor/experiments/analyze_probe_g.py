"""W22 Probe G — Pareto front weight stability + composition coherence.

NEW probe (added 2026-05-18, post-reflection):

Motivation: if the AMFC-chosen portfolios across consecutive periods have
WILDLY DIFFERENT compositions (different selected assets), then:
1. Transaction costs eat returns (thesis §7.2.2 transaction-cost model)
2. The "implemented portfolio" track u*_t is chaotic, not stable
3. KF can't learn cross-period dynamics if the underlying asset selection
   changes every period

If they're TOO STABLE (always picking the same 5 assets), there may be a
selection-pressure pathology (e.g., HV anchor lock-in).

This probe analyzes:
- L1 distance between consecutive AMFC weights (per scenario, per seed)
- Jaccard similarity of the active-asset SETS across periods
- Number of asset SWITCHES (assets that go from active to inactive)
- Cardinality stability across periods

Reads predictions.jsonl which contains per-portfolio weights and portfolio_idx.
Note: the AMFC portfolio_idx PER PERIOD is NOT directly logged in the
probe_a log, but we can use the first portfolio (portfolio_idx=0) as a
proxy for "implemented portfolio", or we can use the predicted-HV argmax.

Per the W22 Probe C analyzer, AMFC = argmax(pred_hv) within each period.

Usage:
    uv run python -m experiments.analyze_probe_g \\
        --log-path experiments/results/.../predictions.jsonl \\
        --output ../docs/W22-PROBE-G-WEIGHT-STABILITY.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_records(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def hv_proxy(roi: float, risk: float, R1: float = 0.0, R2: float = 0.2) -> float:
    return max(roi - R1, 0.0) * max(R2 - risk, 0.0)


def pick_amfc_weights(group: pd.DataFrame) -> np.ndarray:
    """Pick AMFC weights from a (scenario, seed, period) group."""
    pred_hv = group.apply(
        lambda r: hv_proxy(r["kf_predicted_ROI_t_plus_1"], r["kf_predicted_risk_t_plus_1"]),
        axis=1,
    ).values
    idx = int(np.argmax(pred_hv)) if np.any(pred_hv > 0) else 0
    return np.array(group.iloc[idx]["weights"])


def analyze(df: pd.DataFrame) -> dict:
    out = {
        "n_records": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_periods": int(df["period_t"].nunique()),
    }

    # For each (scenario, seed), get the time-series of AMFC weights
    amfc_trajectories: dict = {}
    for (scenario, seed), seed_grp in df.groupby(["scenario", "seed"]):
        weights_by_period: dict[int, np.ndarray] = {}
        for period, grp in seed_grp.groupby("period_t"):
            weights_by_period[int(period)] = pick_amfc_weights(grp)
        if weights_by_period:
            sorted_periods = sorted(weights_by_period.keys())
            traj = [weights_by_period[p] for p in sorted_periods]
            amfc_trajectories[(scenario, seed)] = traj

    # For each trajectory, compute consecutive-period metrics
    all_l1_distances = []
    all_jaccard = []
    all_n_switches = []
    all_card = []
    per_scenario = {scenario: {"l1": [], "jaccard": [], "switches": [], "card": []}
                    for scenario, _ in amfc_trajectories.keys()}

    for (scenario, seed), traj in amfc_trajectories.items():
        for w in traj:
            all_card.append(int(np.sum(np.abs(w) > 1e-12)))
            per_scenario[scenario]["card"].append(int(np.sum(np.abs(w) > 1e-12)))
        for i in range(1, len(traj)):
            w_prev = traj[i - 1]
            w_curr = traj[i]
            # L1 distance
            l1 = float(np.sum(np.abs(w_curr - w_prev)))
            all_l1_distances.append(l1)
            per_scenario[scenario]["l1"].append(l1)
            # Jaccard similarity of active sets
            active_prev = set(np.flatnonzero(np.abs(w_prev) > 1e-12).tolist())
            active_curr = set(np.flatnonzero(np.abs(w_curr) > 1e-12).tolist())
            union = active_prev | active_curr
            inter = active_prev & active_curr
            jaccard = float(len(inter) / len(union)) if union else 1.0
            all_jaccard.append(jaccard)
            per_scenario[scenario]["jaccard"].append(jaccard)
            # Number of asset switches (active → inactive OR inactive → active)
            switches = int(len(active_prev ^ active_curr))
            all_n_switches.append(switches)
            per_scenario[scenario]["switches"].append(switches)

    if all_l1_distances:
        out["l1_distance_mean"] = float(np.mean(all_l1_distances))
        out["l1_distance_median"] = float(np.median(all_l1_distances))
        out["l1_distance_max"] = float(np.max(all_l1_distances))
        out["jaccard_mean"] = float(np.mean(all_jaccard))
        out["jaccard_median"] = float(np.median(all_jaccard))
        out["n_switches_mean"] = float(np.mean(all_n_switches))
        out["n_switches_median"] = float(np.median(all_n_switches))
        out["n_switches_max"] = int(np.max(all_n_switches))
    if all_card:
        out["cardinality_mean"] = float(np.mean(all_card))
        out["cardinality_median"] = float(np.median(all_card))
        out["cardinality_min"] = int(np.min(all_card))
        out["cardinality_max"] = int(np.max(all_card))

    # Per-scenario aggregates
    per_scen_out = {}
    for scenario, stats in per_scenario.items():
        per_scen_out[scenario] = {
            "n_consecutive_pairs": int(len(stats["l1"])),
            "l1_mean": float(np.mean(stats["l1"])) if stats["l1"] else float("nan"),
            "jaccard_mean": float(np.mean(stats["jaccard"])) if stats["jaccard"] else float("nan"),
            "switches_mean": float(np.mean(stats["switches"])) if stats["switches"] else float("nan"),
            "cardinality_mean": float(np.mean(stats["card"])) if stats["card"] else float("nan"),
            "cardinality_min": int(np.min(stats["card"])) if stats["card"] else 0,
            "cardinality_max": int(np.max(stats["card"])) if stats["card"] else 0,
        }
    out["per_scenario"] = per_scen_out

    # Decision: portfolio stability thresholds
    # - jaccard > 0.5 = "moderately stable" (≥ 50% asset overlap)
    # - jaccard < 0.2 = "chaotic"
    if "jaccard_mean" in out:
        out["jaccard_stable"] = bool(out["jaccard_mean"] >= 0.5)
        out["jaccard_chaotic"] = bool(out["jaccard_mean"] < 0.2)

    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe G — Pareto front weight stability + composition coherence")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_g.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Scenarios: {summary['n_scenarios']}")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append("")

    if "jaccard_mean" not in summary:
        lines.append("Insufficient data for stability analysis.")
        return "\n".join(lines)

    j = summary["jaccard_mean"]
    if summary.get("jaccard_stable"):
        verdict = f"🟢 **AMFC weights are MODERATELY STABLE** across periods (mean Jaccard = {j:.3f} ≥ 0.5)"
    elif summary.get("jaccard_chaotic"):
        verdict = f"🔴 **AMFC weights are CHAOTIC** across periods (mean Jaccard = {j:.3f} < 0.2). Transaction costs may eat returns; KF can't learn cross-period dynamics."
    else:
        verdict = f"🟡 **AMFC weights have MODERATE STABILITY** (Jaccard = {j:.3f} in [0.2, 0.5])"
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")

    lines.append("## Cross-period weight stability metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| L1(w_t - w_{{t-1}}) mean | {summary['l1_distance_mean']:.4f} |")
    lines.append(f"| L1 median | {summary['l1_distance_median']:.4f} |")
    lines.append(f"| L1 max | {summary['l1_distance_max']:.4f} |")
    lines.append(f"| Jaccard(active_t, active_{{t-1}}) mean | {summary['jaccard_mean']:.4f} |")
    lines.append(f"| Jaccard median | {summary['jaccard_median']:.4f} |")
    lines.append(f"| # asset switches mean | {summary['n_switches_mean']:.2f} |")
    lines.append(f"| # asset switches median | {summary['n_switches_median']:.0f} |")
    lines.append(f"| # asset switches max | {summary['n_switches_max']} |")
    lines.append("")

    lines.append("## Cardinality (active assets)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| mean | {summary['cardinality_mean']:.2f} |")
    lines.append(f"| median | {summary['cardinality_median']:.0f} |")
    lines.append(f"| min | {summary['cardinality_min']} |")
    lines.append(f"| max | {summary['cardinality_max']} |")
    lines.append("")
    lines.append("(Thesis §7.2.3: c_l = 5, c_u = 15)")
    lines.append("")

    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append("| scenario | n_pairs | L1 mean | Jaccard mean | Switches mean | Card mean | Card range |")
    lines.append("|---|---|---|---|---|---|---|")
    for scenario, stats in summary["per_scenario"].items():
        lines.append(
            f"| {scenario} | {stats['n_consecutive_pairs']} "
            f"| {stats['l1_mean']:.4f} "
            f"| {stats['jaccard_mean']:.4f} "
            f"| {stats['switches_mean']:.2f} "
            f"| {stats['cardinality_mean']:.2f} "
            f"| [{stats['cardinality_min']}, {stats['cardinality_max']}] |"
        )
    lines.append("")

    lines.append("## Decision per backlog spec")
    lines.append("")
    if summary.get("jaccard_chaotic"):
        lines.append("- 🔴 **Chaotic weight changes** → high transaction-cost burden; suggests:")
        lines.append("  - Enable transaction-cost-aware selection (already wired via previous_weights in walk_forward)")
        lines.append("  - Consider portfolio-level coherence regularization")
        lines.append("  - KF cross-period dynamics learning is fundamentally degraded (NC8c-v2 carry meaningless if AMFC portfolios are unrelated)")
    elif summary.get("jaccard_stable"):
        lines.append("- ✅ **Stable AMFC compositions** → low transaction-cost burden; KF can learn cross-period dynamics on a coherent portfolio trajectory")
    else:
        lines.append("- 🟡 **Moderate stability** → some transaction cost expected; KF cross-period dynamics partially viable")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.log_path.exists():
        print(f"ERROR: log file not found at {args.log_path}", file=sys.stderr)
        return 1

    df = load_records(args.log_path)
    if df.empty:
        print(f"ERROR: log file is empty at {args.log_path}", file=sys.stderr)
        return 1

    summary = analyze(df)
    report = format_report(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"[probe-g] wrote report to {args.output}")

    summary_json = args.output.parent / (args.output.stem + "_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[probe-g] wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
