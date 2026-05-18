"""W22 Probe B post-hoc analysis: TIP + λ + anticipation_rate signal distributions.

Reads the λ trace CSV produced by AnticipatoryLearning.flush_lambda_trace_csv
(W16-4 infrastructure, re-purposed for Probe B per W22-INSPECTION-BACKLOG.md).

Per Probe B spec:
- H0 (TIP saturation): TIP distribution across periods × portfolios has
  > 50% of mass within (0.45, 0.55) → TIP is functionally constant
- H1 (TIP varies): TIP distribution has Shannon entropy > 1.5 nats
- For λ: H0: λ_combined CoV < 0.2 across portfolios within a period; H1: ≥ 0.2

Tests:
- Kolmogorov-Smirnov test of TIP histogram vs Uniform[0,1]
- Coefficient of variation of λ within each period
- Shannon entropy of TIP marginal

Usage:
    uv run python -m experiments.analyze_probe_b \\
        --csv-path experiments/results/w22-probe-b-signals/lambda_trace.csv \\
        --output ../docs/W22-PROBE-B-SIGNAL-DISTRIBUTIONS.md
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
    print("ERROR: scipy required for probe B analysis", file=sys.stderr)
    sys.exit(1)


def load_trace(csv_path: Path) -> pd.DataFrame:
    """Load λ trace CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    # Coerce numeric columns (CSV may have NaN as empty string)
    for col in ("lambda_h", "lambda_k", "lambda", "tip"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def shannon_entropy(values: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of a 1D distribution (nats)."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    hist, _ = np.histogram(finite, bins=bins, density=False)
    probs = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)
    return float(scipy_stats.entropy(probs[probs > 0]))


def coefficient_of_variation(values: np.ndarray) -> float:
    """CoV = std/mean; returns NaN if mean == 0."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    mean = np.mean(finite)
    if abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(finite) / abs(mean))


def analyze(df: pd.DataFrame) -> dict:
    """Compute Probe B metrics + hypothesis tests."""
    out = {
        "n_records": len(df),
        "n_periods": df["period"].nunique() if "period" in df.columns else 0,
        "lambda_k_branches": (df["lambda_k_branch"].value_counts().to_dict()
                              if "lambda_k_branch" in df.columns else {}),
    }

    # ── TIP distribution analysis ────────────────────────────────────
    tip = df["tip"].dropna().values
    out["tip_n"] = len(tip)
    if len(tip) > 0:
        out["tip_mean"] = float(np.mean(tip))
        out["tip_std"] = float(np.std(tip))
        out["tip_min"] = float(np.min(tip))
        out["tip_max"] = float(np.max(tip))
        out["tip_median"] = float(np.median(tip))

        # Saturation: fraction of TIP samples in (0.45, 0.55)
        saturated = ((tip > 0.45) & (tip < 0.55)).sum()
        out["tip_fraction_saturated_45_55"] = float(saturated / len(tip))
        # Stronger saturation: (0.48, 0.52)
        out["tip_fraction_saturated_48_52"] = float(
            ((tip > 0.48) & (tip < 0.52)).sum() / len(tip)
        )

        # Shannon entropy (20 bins)
        out["tip_entropy_nats"] = shannon_entropy(tip, bins=20)

        # KS test vs Uniform[0,1]
        ks_stat, ks_p = scipy_stats.kstest(tip, "uniform", args=(0, 1))
        out["tip_ks_vs_uniform_stat"] = float(ks_stat)
        out["tip_ks_vs_uniform_pvalue"] = float(ks_p)

        # H0 decision: > 50% saturation in (0.45, 0.55) → TIP functionally constant
        out["tip_saturation_H0_REJECTED"] = bool(
            out["tip_fraction_saturated_45_55"] < 0.5
        )
        # Spec target: entropy ≥ 1.5 nats for non-saturated
        out["tip_entropy_above_1p5"] = bool(out["tip_entropy_nats"] >= 1.5)

    # ── λ distribution analysis ──────────────────────────────────────
    for col in ("lambda_h", "lambda_k", "lambda"):
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        out[f"{col}_n"] = len(values)
        out[f"{col}_mean"] = float(np.mean(values))
        out[f"{col}_std"] = float(np.std(values))
        out[f"{col}_min"] = float(np.min(values))
        out[f"{col}_max"] = float(np.max(values))
        out[f"{col}_median"] = float(np.median(values))

    # ── Per-period CoV for λ_combined ────────────────────────────────
    if "lambda" in df.columns and "period" in df.columns:
        per_period_cov = []
        for period, grp in df.groupby("period"):
            cov = coefficient_of_variation(grp["lambda"].dropna().values)
            if np.isfinite(cov):
                per_period_cov.append(cov)
        if per_period_cov:
            out["lambda_combined_per_period_cov_mean"] = float(np.mean(per_period_cov))
            out["lambda_combined_per_period_cov_min"] = float(np.min(per_period_cov))
            out["lambda_combined_per_period_cov_max"] = float(np.max(per_period_cov))
            # Decision: CoV ≥ 0.2 within periods → meaningful per-portfolio differentiation
            out["lambda_cov_above_0p2_fraction_periods"] = float(
                np.mean([c >= 0.2 for c in per_period_cov])
            )
            out["lambda_cov_H0_REJECTED"] = bool(
                out["lambda_combined_per_period_cov_mean"] >= 0.2
            )

    return out


def per_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-period TIP + λ summary table."""
    rows = []
    for period, grp in df.groupby("period"):
        tip_vals = grp["tip"].dropna().values
        lam_vals = grp["lambda"].dropna().values
        rows.append({
            "period": period,
            "n": len(grp),
            "tip_mean": float(np.mean(tip_vals)) if len(tip_vals) else float("nan"),
            "tip_std": float(np.std(tip_vals)) if len(tip_vals) else float("nan"),
            "tip_min": float(np.min(tip_vals)) if len(tip_vals) else float("nan"),
            "tip_max": float(np.max(tip_vals)) if len(tip_vals) else float("nan"),
            "lambda_mean": float(np.mean(lam_vals)) if len(lam_vals) else float("nan"),
            "lambda_cov": coefficient_of_variation(lam_vals),
        })
    return pd.DataFrame(rows)


def format_report(summary: dict, per_period: pd.DataFrame) -> str:
    lines = []
    lines.append("# W22 Probe B — TIP + λ + anticipation_rate signal distributions")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_b.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append(f"- λ^K branches observed: {summary.get('lambda_k_branches', {})}")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    tip_saturated_h0_rejected = summary.get("tip_saturation_H0_REJECTED", None)
    lambda_cov_h0_rejected = summary.get("lambda_cov_H0_REJECTED", None)

    if tip_saturated_h0_rejected and lambda_cov_h0_rejected:
        verdict = ("🟢 **Both H0 rejected — TIP varies meaningfully AND λ has "
                   "per-portfolio differentiation**")
    elif not tip_saturated_h0_rejected and not lambda_cov_h0_rejected:
        verdict = ("🔴 **Both H0 stand — TIP saturated AND λ uniform** "
                   "(anticipation has no differentiating signal)")
    elif not tip_saturated_h0_rejected:
        verdict = "🟡 **TIP saturated (H0 stands) but λ varies — replace uncertainty measure**"
    else:
        verdict = "🟡 **λ uniform (H0 stands) but TIP varies — fix per-portfolio differentiation (EQ2/Eq 6.9)**"
    lines.append(verdict)
    lines.append("")

    lines.append("## TIP distribution")
    lines.append("")
    if summary.get("tip_n", 0) > 0:
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| n | {summary['tip_n']} |")
        lines.append(f"| mean | {summary['tip_mean']:.4f} |")
        lines.append(f"| std | {summary['tip_std']:.4f} |")
        lines.append(f"| min | {summary['tip_min']:.4f} |")
        lines.append(f"| max | {summary['tip_max']:.4f} |")
        lines.append(f"| median | {summary['tip_median']:.4f} |")
        lines.append(f"| fraction in (0.45, 0.55) [saturation] | {summary['tip_fraction_saturated_45_55']:.4f} |")
        lines.append(f"| fraction in (0.48, 0.52) [strong saturation] | {summary['tip_fraction_saturated_48_52']:.4f} |")
        lines.append(f"| Shannon entropy (nats, 20 bins) | {summary['tip_entropy_nats']:.4f} |")
        lines.append(f"| KS test vs Uniform[0,1] stat | {summary['tip_ks_vs_uniform_stat']:.4f} |")
        lines.append(f"| KS test vs Uniform[0,1] p-value | {summary['tip_ks_vs_uniform_pvalue']:.4g} |")
        lines.append("")
        lines.append("**Decision (TIP saturation):**")
        if summary["tip_saturation_H0_REJECTED"]:
            lines.append(f"- ✅ H0 (TIP saturated) REJECTED: only {summary['tip_fraction_saturated_45_55']:.1%} of samples in (0.45, 0.55) [< 50% threshold]")
        else:
            lines.append(f"- ❌ H0 (TIP saturated) STANDS: {summary['tip_fraction_saturated_45_55']:.1%} of samples in (0.45, 0.55) [≥ 50% threshold]")
        if summary["tip_entropy_above_1p5"]:
            lines.append(f"- ✅ Entropy above 1.5 nats target: {summary['tip_entropy_nats']:.4f}")
        else:
            lines.append(f"- ⚠️  Entropy below 1.5 nats target: {summary['tip_entropy_nats']:.4f}")
        lines.append("")
    else:
        lines.append("No TIP records found.")
        lines.append("")

    lines.append("## λ distributions")
    lines.append("")
    lines.append("| signal | n | mean | std | min | max | median |")
    lines.append("|---|---|---|---|---|---|---|")
    for col in ("lambda_h", "lambda_k", "lambda"):
        if f"{col}_n" in summary:
            lines.append(
                f"| {col} | {summary[f'{col}_n']} "
                f"| {summary[f'{col}_mean']:.4e} "
                f"| {summary[f'{col}_std']:.4e} "
                f"| {summary[f'{col}_min']:.4e} "
                f"| {summary[f'{col}_max']:.4e} "
                f"| {summary[f'{col}_median']:.4e} |"
            )
    lines.append("")

    if "lambda_combined_per_period_cov_mean" in summary:
        lines.append("### λ_combined per-period CoV (differentiation across portfolios)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(f"| mean CoV across periods | {summary['lambda_combined_per_period_cov_mean']:.4f} |")
        lines.append(f"| min CoV | {summary['lambda_combined_per_period_cov_min']:.4f} |")
        lines.append(f"| max CoV | {summary['lambda_combined_per_period_cov_max']:.4f} |")
        lines.append(f"| fraction of periods with CoV ≥ 0.2 | {summary['lambda_cov_above_0p2_fraction_periods']:.4f} |")
        lines.append("")
        lines.append("**Decision (λ uniformity):**")
        if summary["lambda_cov_H0_REJECTED"]:
            lines.append(f"- ✅ H0 (λ uniform) REJECTED: mean per-period CoV {summary['lambda_combined_per_period_cov_mean']:.4f} ≥ 0.2 threshold")
        else:
            lines.append(f"- ❌ H0 (λ uniform) STANDS: mean per-period CoV {summary['lambda_combined_per_period_cov_mean']:.4f} < 0.2 threshold")
        lines.append("")

    lines.append("## Per-period summary")
    lines.append("")
    lines.append("```")
    lines.append(per_period.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("```")
    lines.append("")

    lines.append("## Decision per backlog spec")
    lines.append("")
    if tip_saturated_h0_rejected and lambda_cov_h0_rejected:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe B decision criteria:")
        lines.append("- ✅ Both signals healthy → bottleneck is downstream (Probes A/D/E/C)")
    elif not tip_saturated_h0_rejected:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe B decision criteria:")
        lines.append("- ⚠️  TIP saturated → Reading C confirmed at code level → need different uncertainty measure (e.g., raw variance ratio, not symmetric TIP)")
    elif not lambda_cov_h0_rejected:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe B decision criteria:")
        lines.append("- ⚠️  λ has low CoV → anticipation is uniform → selection sees no signal → fix per-portfolio differentiation (probably λ^K min-max per Eq 6.9, the EQ2 finding)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        print(f"ERROR: λ trace CSV not found at {args.csv_path}", file=sys.stderr)
        return 1

    df = load_trace(args.csv_path)
    if df.empty:
        print(f"ERROR: λ trace CSV is empty at {args.csv_path}", file=sys.stderr)
        return 1

    summary = analyze(df)
    per_period = per_period_summary(df)

    report = format_report(summary, per_period)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"[probe-b] wrote report to {args.output}")

    summary_json = args.output.parent / (args.output.stem + "_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[probe-b] wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
