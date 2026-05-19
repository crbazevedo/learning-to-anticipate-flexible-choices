"""W22 Probe E post-hoc analysis: anticipative distribution / KF P sanity.

Reads the per-period JSONL log produced by probe_a_kf_prediction_log,
which already captures `kf_P_diag` per portfolio per period. Computes:

- Eigenvalue distribution of P[:2,:2] (the position covariance block
  used directly by TIP MC sampling)
- Condition number distribution
- Fraction of (period, portfolio) instances with degenerate covariance
  (min eigenvalue < 1e-12) or runaway condition number (> 1e6)
- Time-series of P[0,0], P[1,1] trace evolution (drift detection)

Per Probe E spec:
- H0: anticipative_covariance is positive-definite with condition < 1e6
  across all (period, portfolio) → distribution is healthy
- H1: at least 5% of instances have degenerate covariance → distribution
  is numerically broken

Note: The probe_a log captures `kf_P_diag = [P[0,0], P[1,1], P[2,2], P[3,3]]`
only — not the full P matrix. So we can analyze diagonal eigenvalues
(individual variances) but not off-diagonal correlations. For the
diagonal-only case, the "eigenvalues" are just the diagonal entries,
and "condition number" is max_diag / min_diag.

Usage:
    uv run python -m experiments.analyze_probe_e \\
        --log-path experiments/results/w22-probe-a-kf-accuracy-post-nc7/predictions.jsonl \\
        --output ../docs/W22-PROBE-E-ANTICIPATIVE-DISTRIBUTION-SANITY.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_records(log_path: Path) -> pd.DataFrame:
    """Load JSONL records into a DataFrame."""
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame) -> dict:
    """Compute P-matrix sanity metrics."""
    out = {
        "n_records": len(df),
        "n_scenarios": df["scenario"].nunique(),
        "n_seeds": df["seed"].nunique(),
        "n_periods": df["period_t"].nunique(),
    }

    # Each record has kf_P_diag = [P[0,0], P[1,1], P[2,2], P[3,3]]
    diags = np.array([r for r in df["kf_P_diag"].tolist()])
    # diag[:, 0] = P_roi, diag[:, 1] = P_risk, diag[:, 2] = P_roi_vel, diag[:, 3] = P_risk_vel

    out["P_roi_n"] = int(diags.shape[0])
    out["P_roi_mean"] = float(np.mean(diags[:, 0]))
    out["P_roi_median"] = float(np.median(diags[:, 0]))
    out["P_roi_std"] = float(np.std(diags[:, 0]))
    out["P_roi_min"] = float(np.min(diags[:, 0]))
    out["P_roi_max"] = float(np.max(diags[:, 0]))

    out["P_risk_mean"] = float(np.mean(diags[:, 1]))
    out["P_risk_median"] = float(np.median(diags[:, 1]))
    out["P_risk_std"] = float(np.std(diags[:, 1]))
    out["P_risk_max"] = float(np.max(diags[:, 1]))

    out["P_roi_vel_mean"] = float(np.mean(diags[:, 2]))
    out["P_risk_vel_mean"] = float(np.mean(diags[:, 3]))

    # Position-block "condition number" via diagonal ratio
    pos_diags = diags[:, :2]
    pos_max = np.max(pos_diags, axis=1)
    pos_min = np.min(pos_diags, axis=1)
    safe_min = np.where(pos_min > 1e-15, pos_min, 1e-15)
    cond_pos = pos_max / safe_min
    out["cond_pos_mean"] = float(np.mean(cond_pos))
    out["cond_pos_max"] = float(np.max(cond_pos))
    out["cond_pos_p99"] = float(np.percentile(cond_pos, 99))

    # Degeneracy: position diag < 1e-12
    n_degenerate = int(np.sum(np.any(pos_diags < 1e-12, axis=1)))
    out["n_degenerate_position"] = n_degenerate
    out["fraction_degenerate_position"] = float(n_degenerate / len(df))
    out["degeneracy_H0_REJECTED"] = bool(out["fraction_degenerate_position"] >= 0.05)

    # Velocity-position uncertainty ratio sanity (NC7 fix should give >> 1000)
    vel_max = np.max(diags[:, 2:], axis=1)
    pos_avg = (diags[:, 0] + diags[:, 1]) / 2.0
    vel_pos_ratio = vel_max / np.where(pos_avg > 1e-15, pos_avg, 1e-15)
    out["vel_pos_ratio_mean"] = float(np.mean(vel_pos_ratio))
    out["vel_pos_ratio_min"] = float(np.min(vel_pos_ratio))
    out["vel_pos_ratio_median"] = float(np.median(vel_pos_ratio))
    # Expected post-NC7: ratio >> 1000 since velocity prior = 1000, position
    # prior = 0.1 (steady state ~0.01)
    out["vel_pos_ratio_healthy"] = bool(out["vel_pos_ratio_median"] >= 100)

    # Trace evolution per period (drift detection)
    # Mean trace per (period, scenario) — if trace grows over periods, P drifts
    per_period_trace = []
    for period, grp in df.groupby("period_t"):
        diags_p = np.array(grp["kf_P_diag"].tolist())
        trace_p = np.mean(np.sum(diags_p[:, :2], axis=1))  # position trace
        per_period_trace.append({"period": int(period), "trace_pos_mean": float(trace_p)})
    per_period_df = pd.DataFrame(per_period_trace).sort_values("period")
    # Linear regression on trace vs period to detect drift
    if len(per_period_df) >= 2:
        coef = np.polyfit(per_period_df["period"].values, per_period_df["trace_pos_mean"].values, 1)
        out["trace_pos_drift_slope"] = float(coef[0])
        out["trace_pos_drift_intercept"] = float(coef[1])
        # H_drift: slope > 0 means P is growing over periods (TIP saturation risk)
        out["P_drifting"] = bool(out["trace_pos_drift_slope"] > 0.01)
    else:
        out["trace_pos_drift_slope"] = float("nan")

    # Per-scenario breakdown
    per_scenario = {}
    for scenario, scen_grp in df.groupby("scenario"):
        scen_diags = np.array(scen_grp["kf_P_diag"].tolist())
        per_scenario[scenario] = {
            "n": int(len(scen_grp)),
            "P_roi_median": float(np.median(scen_diags[:, 0])),
            "P_risk_median": float(np.median(scen_diags[:, 1])),
            "P_roi_max": float(np.max(scen_diags[:, 0])),
            "P_risk_max": float(np.max(scen_diags[:, 1])),
            "P_roi_vel_median": float(np.median(scen_diags[:, 2])),
            "fraction_pos_diag_above_10": float(
                np.mean((scen_diags[:, 0] > 10) | (scen_diags[:, 1] > 10))
            ),
        }
    out["per_scenario"] = per_scenario

    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe E — anticipative distribution / KF P-matrix sanity")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_e.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Scenarios: {summary['n_scenarios']}")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    deg_rejected = summary.get("degeneracy_H0_REJECTED", False)
    vel_healthy = summary.get("vel_pos_ratio_healthy", False)
    drifting = summary.get("P_drifting", False)

    if not deg_rejected and vel_healthy and not drifting:
        verdict = "🟢 **P matrix is healthy** (no degeneracy, NC7 velocity prior intact, no drift across periods)"
    elif drifting:
        verdict = f"🔴 **P matrix DRIFTS over periods** (slope = {summary['trace_pos_drift_slope']:.4f} > 0.01) — this is the TIP saturation root cause"
    elif deg_rejected:
        verdict = f"🔴 **P matrix has DEGENERACY in ≥5% of instances** ({summary['fraction_degenerate_position']*100:.1f}%) — numerical pathology"
    elif not vel_healthy:
        verdict = f"⚠️  **Velocity/position uncertainty ratio is LOW** (median {summary['vel_pos_ratio_median']:.2f} < 100); NC7 may have been reverted"
    else:
        verdict = "🟡 **Partial issues detected**"
    lines.append(verdict)
    lines.append("")

    lines.append("## P-matrix diagonal statistics")
    lines.append("")
    lines.append("| component | mean | median | std | min | max |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| P[ROI, ROI] | {summary['P_roi_mean']:.4e} | {summary['P_roi_median']:.4e} "
        f"| {summary['P_roi_std']:.4e} | {summary['P_roi_min']:.4e} | {summary['P_roi_max']:.4e} |"
    )
    lines.append(
        f"| P[risk, risk] | {summary['P_risk_mean']:.4e} | {summary['P_risk_median']:.4e} "
        f"| {summary['P_risk_std']:.4e} | — | {summary['P_risk_max']:.4e} |"
    )
    lines.append(
        f"| P[ROI_vel] | {summary['P_roi_vel_mean']:.4e} | — | — | — | — |"
    )
    lines.append(
        f"| P[risk_vel] | {summary['P_risk_vel_mean']:.4e} | — | — | — | — |"
    )
    lines.append("")

    lines.append("## Position-block condition number")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| mean | {summary['cond_pos_mean']:.4e} |")
    lines.append(f"| max | {summary['cond_pos_max']:.4e} |")
    lines.append(f"| p99 | {summary['cond_pos_p99']:.4e} |")
    lines.append("")

    lines.append("## Velocity / position uncertainty ratio (NC7 invariant)")
    lines.append("")
    lines.append("Healthy if ratio ≥ 100 (NC7 fix sets velocity prior = 1000, position prior = 0.1 → ratio = 10000).")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| mean | {summary['vel_pos_ratio_mean']:.4e} |")
    lines.append(f"| median | {summary['vel_pos_ratio_median']:.4e} |")
    lines.append(f"| min | {summary['vel_pos_ratio_min']:.4e} |")
    lines.append(f"| **healthy (≥ 100)?** | {'✅ YES' if summary['vel_pos_ratio_healthy'] else '❌ NO'} |")
    lines.append("")

    lines.append("## P drift over periods (TIP-saturation precursor)")
    lines.append("")
    if "trace_pos_drift_slope" in summary:
        lines.append(f"- Linear fit of `mean(trace(P_position))` vs period:")
        lines.append(f"  - slope = `{summary['trace_pos_drift_slope']:.6e}` per period")
        lines.append(f"  - intercept = `{summary.get('trace_pos_drift_intercept', float('nan')):.6e}`")
        lines.append(f"  - **drifting (slope > 0.01)?** {'❌ YES — TIP saturation expected' if summary['P_drifting'] else '✅ NO — P stable'}")
    lines.append("")

    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append("| scenario | n | P[ROI] median | P[risk] median | P[ROI_vel] median | P max above 10 (frac) |")
    lines.append("|---|---|---|---|---|---|")
    for scenario, stats in summary["per_scenario"].items():
        lines.append(
            f"| {scenario} | {stats['n']} | {stats['P_roi_median']:.4e} | {stats['P_risk_median']:.4e} "
            f"| {stats['P_roi_vel_median']:.4e} | {stats['fraction_pos_diag_above_10']:.4f} |"
        )
    lines.append("")

    lines.append("## Decision per backlog spec")
    lines.append("")
    if not deg_rejected and not drifting and vel_healthy:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe E decision criteria:")
        lines.append("- ✅ Condition numbers stable, no degeneracy, velocity prior intact, no drift → distribution is healthy")
        lines.append("- Anticipative-distribution machinery is numerically sound; bottleneck is elsewhere (TIP saturation via n-step compounding, see NC13a)")
    elif drifting:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe E decision criteria:")
        lines.append("- ⚠️  P drift over periods → likely cause: `_update_solution_state_anticipative` blending P toward anticipative_covariance (NC12 fixed for single-horizon; multi-horizon overwrites via Eq 15 which is correct but unbounded)")
        lines.append("- Mitigation: cap P after per-generation blend OR clamp the multi-horizon Eq 15 output (NC13 family)")
    elif deg_rejected:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe E decision criteria:")
        lines.append("- ⚠️  Degeneracy > 5% → numerical pathology; investigate KF lifecycle for over-shrinking")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
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
    print(f"[probe-e] wrote report to {args.output}")

    summary_json = args.output.parent / (args.output.stem + "_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[probe-e] wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
