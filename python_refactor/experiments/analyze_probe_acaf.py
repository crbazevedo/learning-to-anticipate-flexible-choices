"""W22 Probe AC + AF — KF innovation analysis on existing Probe A logs.

Question: Is the production KF well-tuned? Does its innovation sequence
satisfy Mehra's whiteness condition (lag-k autocorrelation ≈ 0)?

This is a POST-HOC analysis that uses the existing
`predictions.jsonl` Probe A log (kf_predicted_*, actual_*, kf_P_diag)
to back out the innovation sequence WITHOUT touching `kalman_filter.py`.

Methodology:
  1. Load all (scenario, seed, period, portfolio) records.
  2. Innovation per record: y = [actual_ROI - kf_predicted_ROI,
                                 actual_risk - kf_predicted_risk].
  3. Per (scenario, seed), aggregate per period:
     - mean innovation (vector) across portfolios
     - mean |innovation|
     - NIS approx using kf_P_diag (treat P_diag as variance estimate
       for S=H*P*H^T+R, with R≈0 for first-order analysis)
  4. Compute lag-1 autocorrelation of per-period mean innovations
     across periods, separately for ROI and risk components.
  5. Verdict per (scenario, seed):
     - lag1 |autocorr| < 0.3 → WHITE (KF well-tuned)
     - lag1 |autocorr| in [0.3, 0.6] → MODERATELY-TUNED
     - lag1 |autocorr| > 0.6 → MIS-TUNED (Mehra fail)

Usage:
    uv run python -m experiments.analyze_probe_acaf \\
        --log-path experiments/results/w22-nc8cv2-nc8d-5seed/predictions.jsonl \\
        --output ../docs/W22-PROBE-AC-AF-KF-DIAGNOSTICS.md
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


def lag1_autocorr(x: np.ndarray) -> float:
    if x.shape[0] < 3:
        return float("nan")
    x = x - np.mean(x)
    num = float(np.sum(x[:-1] * x[1:]))
    den = float(np.sum(x ** 2))
    return num / den if den > 1e-12 else float("nan")


def innovation_stats_per_seed(df_seed: pd.DataFrame) -> dict:
    """For one (scenario, seed), compute per-period mean innovation +
    lag-1 autocorrelation across periods."""
    # Per-period aggregation: mean innovation across portfolios
    df_seed = df_seed.copy()
    df_seed["innov_ROI"] = df_seed["actual_ROI_t_plus_1"] - df_seed["kf_predicted_ROI_t_plus_1"]
    df_seed["innov_risk"] = df_seed["actual_risk_t_plus_1"] - df_seed["kf_predicted_risk_t_plus_1"]
    df_seed["abs_innov_ROI"] = df_seed["innov_ROI"].abs()
    df_seed["abs_innov_risk"] = df_seed["innov_risk"].abs()

    per_period = df_seed.groupby("period_t").agg({
        "innov_ROI": "mean",
        "innov_risk": "mean",
        "abs_innov_ROI": "mean",
        "abs_innov_risk": "mean",
    }).sort_index()

    innov_roi_seq = per_period["innov_ROI"].values
    innov_risk_seq = per_period["innov_risk"].values

    n_periods = int(per_period.shape[0])
    return {
        "n_periods": n_periods,
        "mean_innov_ROI": float(np.mean(innov_roi_seq)),
        "mean_innov_risk": float(np.mean(innov_risk_seq)),
        "std_innov_ROI": float(np.std(innov_roi_seq, ddof=1)) if n_periods > 1 else 0.0,
        "std_innov_risk": float(np.std(innov_risk_seq, ddof=1)) if n_periods > 1 else 0.0,
        "mean_abs_innov_ROI": float(np.mean(per_period["abs_innov_ROI"].values)),
        "mean_abs_innov_risk": float(np.mean(per_period["abs_innov_risk"].values)),
        "lag1_autocorr_ROI": lag1_autocorr(innov_roi_seq),
        "lag1_autocorr_risk": lag1_autocorr(innov_risk_seq),
    }


def verdict(lag1_abs: float) -> str:
    if np.isnan(lag1_abs):
        return "UNKNOWN"
    if lag1_abs < 0.3:
        return "🟢 WHITE"
    if lag1_abs < 0.6:
        return "🟡 MODERATE"
    return "🔴 MIS-TUNED"


def analyze(df: pd.DataFrame) -> dict:
    out = {
        "n_records": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_periods": int(df["period_t"].nunique()),
    }
    by_scenario_seed = []
    for (scenario, seed), grp in df.groupby(["scenario", "seed"]):
        stats = innovation_stats_per_seed(grp)
        stats["scenario"] = scenario
        stats["seed"] = int(seed)
        stats["lag1_abs_ROI"] = abs(stats["lag1_autocorr_ROI"]) if not np.isnan(stats["lag1_autocorr_ROI"]) else float("nan")
        stats["lag1_abs_risk"] = abs(stats["lag1_autocorr_risk"]) if not np.isnan(stats["lag1_autocorr_risk"]) else float("nan")
        stats["verdict_ROI"] = verdict(stats["lag1_abs_ROI"])
        stats["verdict_risk"] = verdict(stats["lag1_abs_risk"])
        by_scenario_seed.append(stats)
    out["per_seed"] = by_scenario_seed

    # Aggregate
    for sc in df["scenario"].unique():
        agg = [r for r in by_scenario_seed if r["scenario"] == sc]
        if not agg:
            continue
        lag1_roi = np.array([r["lag1_autocorr_ROI"] for r in agg if not np.isnan(r["lag1_autocorr_ROI"])])
        lag1_risk = np.array([r["lag1_autocorr_risk"] for r in agg if not np.isnan(r["lag1_autocorr_risk"])])
        mean_innov_roi = np.array([r["mean_innov_ROI"] for r in agg])
        mean_innov_risk = np.array([r["mean_innov_risk"] for r in agg])
        std_innov_roi = np.array([r["std_innov_ROI"] for r in agg])
        std_innov_risk = np.array([r["std_innov_risk"] for r in agg])
        out[f"agg_{sc}"] = {
            "n_seeds": len(agg),
            "mean_lag1_autocorr_ROI": float(np.mean(lag1_roi)) if len(lag1_roi) else float("nan"),
            "mean_lag1_autocorr_risk": float(np.mean(lag1_risk)) if len(lag1_risk) else float("nan"),
            "mean_|lag1|_ROI": float(np.mean(np.abs(lag1_roi))) if len(lag1_roi) else float("nan"),
            "mean_|lag1|_risk": float(np.mean(np.abs(lag1_risk))) if len(lag1_risk) else float("nan"),
            "mean_innov_bias_ROI": float(np.mean(mean_innov_roi)),
            "mean_innov_bias_risk": float(np.mean(mean_innov_risk)),
            "mean_std_innov_ROI": float(np.mean(std_innov_roi)),
            "mean_std_innov_risk": float(np.mean(std_innov_risk)),
        }
    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe AC + AF — KF innovation diagnostics on production logs")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_acaf.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Scenarios: {summary['n_scenarios']}")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append("")
    lines.append("## Per-(scenario, seed) innovation autocorrelation")
    lines.append("")
    lines.append("Per-period mean innovation is aggregated ACROSS portfolios for each period, then")
    lines.append("the lag-1 autocorrelation is computed across periods. Per Mehra (1970), a well-tuned")
    lines.append("KF produces white innovations (|lag-1 autocorr| ≈ 0). Persistent positive autocorr")
    lines.append("indicates the KF is over-trusting its prior (Q too small) or systematic bias from F.")
    lines.append("")
    lines.append("| scenario | seed | periods | mean innov ROI | std innov ROI | lag1 autocorr ROI | verdict ROI | lag1 autocorr risk | verdict risk |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in summary["per_seed"]:
        lines.append(
            f"| {r['scenario']} | {r['seed']} | {r['n_periods']} "
            f"| {r['mean_innov_ROI']:+.4e} | {r['std_innov_ROI']:.4e} "
            f"| {r['lag1_autocorr_ROI']:+.3f} | {r['verdict_ROI']} "
            f"| {r['lag1_autocorr_risk']:+.3f} | {r['verdict_risk']} |"
        )
    lines.append("")
    lines.append("## Aggregated per scenario (mean across seeds)")
    lines.append("")
    lines.append("| scenario | n_seeds | mean lag1 ROI | mean |lag1| ROI | mean lag1 risk | mean |lag1| risk | mean bias ROI | mean bias risk |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key in summary.keys():
        if not key.startswith("agg_"):
            continue
        scenario = key[4:]
        agg = summary[key]
        lines.append(
            f"| {scenario} | {agg['n_seeds']} "
            f"| {agg['mean_lag1_autocorr_ROI']:+.3f} | {agg['mean_|lag1|_ROI']:.3f} "
            f"| {agg['mean_lag1_autocorr_risk']:+.3f} | {agg['mean_|lag1|_risk']:.3f} "
            f"| {agg['mean_innov_bias_ROI']:+.4e} | {agg['mean_innov_bias_risk']:+.4e} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    # Find ASMS scenario aggregation
    asms_key = next((k for k in summary if k.startswith("agg_") and "ASMS" in k), None)
    if asms_key:
        agg = summary[asms_key]
        l1_roi = agg["mean_|lag1|_ROI"]
        l1_risk = agg["mean_|lag1|_risk"]
        bias_roi = agg["mean_innov_bias_ROI"]
        bias_risk = agg["mean_innov_bias_risk"]
        verdict_roi = verdict(l1_roi)
        verdict_risk = verdict(l1_risk)
        lines.append(f"- ASMS KF ROI innovation: |lag-1 autocorr| = {l1_roi:.3f} → {verdict_roi}")
        lines.append(f"- ASMS KF risk innovation: |lag-1 autocorr| = {l1_risk:.3f} → {verdict_risk}")
        lines.append(f"- Systematic bias: ROI={bias_roi:+.4e}, risk={bias_risk:+.4e}")
        lines.append("")
        if l1_roi < 0.3 and l1_risk < 0.3:
            lines.append("🟢 **KF is well-tuned per Mehra's whiteness test.** Innovations are uncorrelated across")
            lines.append("periods, suggesting Q and R are reasonably matched to the true dynamics.")
        elif l1_roi < 0.6 and l1_risk < 0.6:
            lines.append("🟡 **KF is moderately tuned.** Some residual autocorrelation suggests Q could be tuned tighter")
            lines.append("but the filter is not catastrophically mis-specified.")
        else:
            lines.append("🔴 **KF is mis-tuned per Mehra's whiteness test.** Strong residual autocorrelation in innovations")
            lines.append("suggests Q is wrong (likely too small — KF over-trusts prior) or F is mis-specified.")
            lines.append("This explains why the constant-velocity model fails on PO(τ, 1.0) disrupted data:")
            lines.append("when the prior is over-trusted, sudden parameter swaps look like impossible velocities.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Per-portfolio innovation is NOT logged; we aggregate ACROSS portfolios per period as a proxy.")
    lines.append("  A cleaner test would track each portfolio's KF state across periods (but portfolio identity")
    lines.append("  changes with population churn).")
    lines.append("- We don't have access to S_t (innovation covariance) so NIS chi-squared bounds can't be computed.")
    lines.append("  A future Probe AC+ would add live KF instrumentation to `kalman_filter.py`.")
    lines.append("- Per-period mean innovation may smooth out per-portfolio variance signals.")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(summary))
    print(f"[probe-acaf] wrote report to {args.output}", file=sys.stderr)
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
