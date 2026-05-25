"""W22 Probe A post-hoc analysis: KF predictive accuracy vs persistence.

Reads the per-period JSONL log produced by probe_a_kf_prediction_log
(see docs/W22-INSPECTION-BACKLOG.md probe A spec). Computes:
- MAE/RMSE for KF prediction vs persistence baseline, per ROI/risk
- Paired Wilcoxon signed-rank test (one-sided): is KF better?
- Bias: mean(KF_pred - actual) per component (should be ~0)
- Correlation: pearsonr(KF_pred, actual) per component
- Per-period trajectory dumps for diagnostic plots

Output: writes verdict + summary to a markdown file and saves
matplotlib figures.

Usage:
    uv run python -m experiments.analyze_probe_a \\
        --log-path experiments/results/w22-probe-a-kf-accuracy/predictions.jsonl \\
        --output ../docs/W22-PROBE-A-KF-PREDICTIVE-ACCURACY.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:
    print("ERROR: scipy required for probe A analysis", file=sys.stderr)
    sys.exit(1)


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
    """Compute KF vs persistence comparison metrics + statistical tests."""
    # Both objectives
    out = {"n_records": len(df), "n_scenarios": df["scenario"].nunique(),
           "n_seeds": df["seed"].nunique(), "n_periods": df["period_t"].nunique()}

    for obj in ("ROI", "risk"):
        kf_pred = df[f"kf_predicted_{obj}_t_plus_1"].values
        persistence = df[f"persistence_{obj}_t"].values
        actual = df[f"actual_{obj}_t_plus_1"].values

        kf_err = np.abs(kf_pred - actual)
        pers_err = np.abs(persistence - actual)

        # MAE, RMSE
        out[f"{obj}_mae_kf"] = float(np.mean(kf_err))
        out[f"{obj}_mae_persistence"] = float(np.mean(pers_err))
        out[f"{obj}_rmse_kf"] = float(np.sqrt(np.mean((kf_pred - actual) ** 2)))
        out[f"{obj}_rmse_persistence"] = float(np.sqrt(np.mean((persistence - actual) ** 2)))

        # Bias
        out[f"{obj}_bias_kf"] = float(np.mean(kf_pred - actual))
        out[f"{obj}_bias_persistence"] = float(np.mean(persistence - actual))

        # Correlation
        if len(actual) > 1 and np.std(actual) > 0:
            out[f"{obj}_corr_kf_actual"] = float(np.corrcoef(kf_pred, actual)[0, 1])
            out[f"{obj}_corr_persistence_actual"] = float(np.corrcoef(persistence, actual)[0, 1])
        else:
            out[f"{obj}_corr_kf_actual"] = float("nan")
            out[f"{obj}_corr_persistence_actual"] = float("nan")

        # Paired Wilcoxon signed-rank test: is KF error LESS than persistence error?
        # H0: median(kf_err - pers_err) >= 0  (KF no better than persistence)
        # H1: median(kf_err - pers_err) < 0   (KF is better)
        try:
            stat, pvalue = scipy_stats.wilcoxon(
                kf_err, pers_err,
                alternative="less",  # KF error < persistence error
                zero_method="zsplit",
            )
            out[f"{obj}_wilcoxon_stat"] = float(stat)
            out[f"{obj}_wilcoxon_pvalue"] = float(pvalue)
            out[f"{obj}_kf_better"] = bool(pvalue < 0.025)  # Bonferroni for 2 tests
        except ValueError as e:
            out[f"{obj}_wilcoxon_stat"] = float("nan")
            out[f"{obj}_wilcoxon_pvalue"] = float("nan")
            out[f"{obj}_kf_better"] = False
            out[f"{obj}_wilcoxon_error"] = str(e)

        # Effect size: relative MAE reduction
        if out[f"{obj}_mae_persistence"] > 0:
            out[f"{obj}_mae_relative_reduction"] = float(
                (out[f"{obj}_mae_persistence"] - out[f"{obj}_mae_kf"])
                / out[f"{obj}_mae_persistence"]
            )
        else:
            out[f"{obj}_mae_relative_reduction"] = 0.0

    return out


def per_seed_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Compute KF vs persistence metrics broken down by seed (for variability check)."""
    rows = []
    for (scenario, seed), grp in df.groupby(["scenario", "seed"]):
        for obj in ("ROI", "risk"):
            kf = grp[f"kf_predicted_{obj}_t_plus_1"].values
            pers = grp[f"persistence_{obj}_t"].values
            actual = grp[f"actual_{obj}_t_plus_1"].values
            rows.append({
                "scenario": scenario, "seed": seed, "objective": obj,
                "n_records": len(grp),
                "mae_kf": float(np.mean(np.abs(kf - actual))),
                "mae_persistence": float(np.mean(np.abs(pers - actual))),
                "ratio_kf_over_persistence": float(
                    np.mean(np.abs(kf - actual)) / np.mean(np.abs(pers - actual))
                    if np.mean(np.abs(pers - actual)) > 0 else float("nan")
                ),
            })
    return pd.DataFrame(rows)


def format_report(summary: dict, per_seed: pd.DataFrame) -> str:
    lines = []
    lines.append("# W22 Probe A — KF predictive accuracy vs naive persistence")
    lines.append("")
    lines.append(f"*Auto-generated by `experiments/analyze_probe_a.py`.*")
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
    roi_better = summary.get("ROI_kf_better", False)
    risk_better = summary.get("risk_kf_better", False)
    if roi_better and risk_better:
        verdict = "🟢 **KF beats persistence on BOTH ROI and risk** (Wilcoxon p < 0.025 each)"
    elif roi_better:
        verdict = "🟡 **KF beats persistence on ROI ONLY** (risk: KF not significantly better)"
    elif risk_better:
        verdict = "🟡 **KF beats persistence on risk ONLY** (ROI: KF not significantly better)"
    else:
        verdict = "🔴 **KF does NOT beat persistence on either ROI or risk** (anticipation arm has no signal)"
    lines.append(verdict)
    lines.append("")

    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Component | MAE (KF) | MAE (Persistence) | Rel. MAE reduction | Wilcoxon p | Corr(KF, actual) | Bias (KF) |")
    lines.append("|---|---|---|---|---|---|---|")
    for obj in ("ROI", "risk"):
        lines.append(
            f"| {obj} "
            f"| {summary[f'{obj}_mae_kf']:.6e} "
            f"| {summary[f'{obj}_mae_persistence']:.6e} "
            f"| {summary[f'{obj}_mae_relative_reduction']*100:+.2f}% "
            f"| {summary[f'{obj}_wilcoxon_pvalue']:.4g} "
            f"| {summary[f'{obj}_corr_kf_actual']:.4f} "
            f"| {summary[f'{obj}_bias_kf']:+.4e} |"
        )
    lines.append("")

    lines.append("## Per-seed breakdown")
    lines.append("")
    lines.append("```")
    lines.append(per_seed.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## Decision per backlog spec")
    lines.append("")
    if roi_better and risk_better:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe A decision criteria:")
        lines.append("- KF is functional → anticipation has signal → focus on downstream conditions (Probes B-F)")
    elif not roi_better and not risk_better:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe A decision criteria:")
        lines.append("- **KF model is WRONG** → either replace constant-velocity F matrix or remove KF entirely")
        lines.append("- ASMS cannot beat SMS until KF is fixed")
        lines.append("- Recommended next: investigate alternative state-space models (random walk, AR(1), regime-switching)")
    else:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe A decision criteria:")
        lines.append("- KF helps for one objective but not the other → consider asymmetric anticipation")

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
    per_seed = per_seed_breakdown(df)

    report = format_report(summary, per_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"[probe-a] wrote report to {args.output}")

    summary_json = args.output.parent / (args.output.stem + "_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[probe-a] wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
