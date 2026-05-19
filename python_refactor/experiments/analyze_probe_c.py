"""W22 Probe C post-hoc analysis: AMFC vs alternative DMs.

Reads probe_a_kf_prediction_log JSONL (which records per-portfolio
weights + KF prediction + persistence + actual t+1 ROI/risk). Computes
ranking of each Pareto-front portfolio under multiple decision-maker
(DM) rules, then evaluates each DM's expected OOS performance.

Per Probe C spec:
- AMFC (current default): argmax E[single-point HV via per-portfolio EFHV]
- HighROI: argmax mean predicted ROI
- LowRisk: argmin mean predicted risk
- Sharpe: argmax (predicted_ROI − R1) / sqrt(predicted_risk − R2 + ε)
- Median: 50th percentile by predicted ROI
- First: weights[0] (the W17-4 fallback)
- Oracle: argmax of *actual realized* HV-proxy (post-hoc; upper bound)

The "actual HV-proxy" for each portfolio: (actual_ROI - R1) * (R2 - actual_risk)
which is the single-point hypervolume contribution to z_ref = (R1, R2)
= (0.0, 0.2).

Comparison metrics:
- Per-period: each DM's pick's actual realized OOS HV-proxy
- Per-scenario aggregate: mean realized HV-proxy per DM
- Rank correlation: AMFC's predicted-EFHV ordering vs realized HV
  ordering (Spearman ρ) — measures AMFC's informativeness
- Gap-to-Oracle: (Oracle - AMFC) / Oracle

Hypothesis:
- H0: AMFC realized HV ≤ Random pick realized HV (AMFC uninformative)
- H1: AMFC > Random; AMFC ≈ Oracle within 20%

Usage:
    uv run python -m experiments.analyze_probe_c \\
        --log-path experiments/results/.../predictions.jsonl \\
        --output ../docs/W22-PROBE-C-AMFC-VS-ALTERNATIVES.md
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
    print("ERROR: scipy required", file=sys.stderr)
    sys.exit(1)


# Reference point for HV-proxy computation (matches walk_forward_report default).
R1 = 0.0   # ROI lower bound
R2 = 0.2   # risk upper bound (we WANT risk < R2)


def hv_proxy(roi: float, risk: float) -> float:
    """Single-point HV contribution to z_ref = (R1, R2)."""
    return max(roi - R1, 0.0) * max(R2 - risk, 0.0)


def load_records(log_path: Path) -> pd.DataFrame:
    """Load JSONL records."""
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def rank_portfolios(group: pd.DataFrame) -> dict:
    """Compute per-DM pick index within a (scenario, seed, period) group."""
    g = group.reset_index(drop=True)
    n = len(g)
    if n == 0:
        return {}

    # Predicted values (from KF):
    pred_roi = g["kf_predicted_ROI_t_plus_1"].values
    pred_risk = g["kf_predicted_risk_t_plus_1"].values
    # Persistence values (current-period observation):
    pers_roi = g["persistence_ROI_t"].values
    pers_risk = g["persistence_risk_t"].values
    # Actual realized t+1:
    actual_roi = g["actual_ROI_t_plus_1"].values
    actual_risk = g["actual_risk_t_plus_1"].values

    # AMFC: argmax expected HV under KF prediction
    pred_hv = np.array([hv_proxy(r, k) for r, k in zip(pred_roi, pred_risk)])
    # If all-zeros (degenerate case), AMFC defaults to index 0
    amfc_idx = int(np.argmax(pred_hv)) if np.any(pred_hv > 0) else 0

    # HighROI: argmax predicted ROI
    highroi_idx = int(np.argmax(pred_roi))

    # LowRisk: argmin predicted risk
    lowrisk_idx = int(np.argmin(pred_risk))

    # Sharpe-like: argmax (pred_roi - R1) / sqrt(pred_risk - R2 + eps).
    # Risk should be < R2 (we're below upper bound). To make a positive
    # denominator we use sqrt(max(pred_risk, eps)).
    eps = 1e-12
    sharpe = pred_roi / np.sqrt(np.maximum(pred_risk, eps))
    sharpe_idx = int(np.argmax(sharpe))

    # Median by predicted ROI: pick median-ranked portfolio
    median_rank_target = (n - 1) // 2
    sorted_idx = np.argsort(pred_roi)
    median_idx = int(sorted_idx[median_rank_target])

    # First: index 0 (W17-4 degenerate fallback)
    first_idx = 0

    # Random: averaged over `n` portfolios — just use uniform pick mean
    # (we'll compute mean below). Track as None.

    # Oracle: argmax realized HV-proxy
    actual_hv = np.array([hv_proxy(r, k) for r, k in zip(actual_roi, actual_risk)])
    oracle_idx = int(np.argmax(actual_hv)) if np.any(actual_hv > 0) else 0

    # Compute realized HV per DM pick
    return {
        "n_pareto": n,
        "amfc_realized_hv": float(actual_hv[amfc_idx]),
        "amfc_idx": amfc_idx,
        "highroi_realized_hv": float(actual_hv[highroi_idx]),
        "lowrisk_realized_hv": float(actual_hv[lowrisk_idx]),
        "sharpe_realized_hv": float(actual_hv[sharpe_idx]),
        "median_realized_hv": float(actual_hv[median_idx]),
        "first_realized_hv": float(actual_hv[first_idx]),
        "oracle_realized_hv": float(actual_hv[oracle_idx]),
        "random_realized_hv_mean": float(np.mean(actual_hv)),
        # Rank correlation: AMFC's predicted-HV ranking vs realized-HV ranking
        # (Spearman ρ; -1 to +1; +1 = AMFC ranks perfectly)
        "spearman_pred_vs_actual_hv": (
            float(scipy_stats.spearmanr(pred_hv, actual_hv).statistic)
            if n >= 2 and np.std(pred_hv) > 0 and np.std(actual_hv) > 0
            else float("nan")
        ),
        # Persistence-based "DM" (argmax HV using persistence as prediction)
        "persistence_amfc_realized_hv": float(actual_hv[
            int(np.argmax(np.array([hv_proxy(r, k) for r, k in zip(pers_roi, pers_risk)])))
            if np.any(np.array([hv_proxy(r, k) for r, k in zip(pers_roi, pers_risk)]) > 0)
            else 0
        ]),
    }


def analyze(df: pd.DataFrame) -> dict:
    """Aggregate per-DM realized HV across periods."""
    out = {
        "n_records": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_periods": int(df["period_t"].nunique()),
        "R1": R1,
        "R2": R2,
    }

    per_period_rows = []
    for (scenario, seed, period), grp in df.groupby(["scenario", "seed", "period_t"]):
        rec = rank_portfolios(grp)
        rec.update({"scenario": scenario, "seed": seed, "period": period})
        per_period_rows.append(rec)

    if not per_period_rows:
        return out

    pp = pd.DataFrame(per_period_rows)

    dm_metrics = [
        "amfc_realized_hv",
        "persistence_amfc_realized_hv",
        "highroi_realized_hv",
        "lowrisk_realized_hv",
        "sharpe_realized_hv",
        "median_realized_hv",
        "first_realized_hv",
        "random_realized_hv_mean",
        "oracle_realized_hv",
    ]
    dm_aggregate = {}
    for dm in dm_metrics:
        vals = pp[dm].values
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        dm_aggregate[dm] = {
            "n_periods": int(len(finite)),
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "std": float(np.std(finite)),
            "max": float(np.max(finite)),
        }
    out["dm_aggregate"] = dm_aggregate

    # Per-scenario breakdowns
    per_scenario = {}
    for scenario, scen_grp in pp.groupby("scenario"):
        per_scenario[scenario] = {
            "n_periods": int(len(scen_grp)),
            "amfc_mean": float(np.mean(scen_grp["amfc_realized_hv"])),
            "oracle_mean": float(np.mean(scen_grp["oracle_realized_hv"])),
            "highroi_mean": float(np.mean(scen_grp["highroi_realized_hv"])),
            "random_mean": float(np.mean(scen_grp["random_realized_hv_mean"])),
            "spearman_mean": float(np.mean(scen_grp["spearman_pred_vs_actual_hv"].dropna())),
        }
    out["per_scenario"] = per_scenario

    # AMFC vs Random (KEY hypothesis)
    amfc_vals = pp["amfc_realized_hv"].dropna().values
    random_vals = pp["random_realized_hv_mean"].dropna().values
    if len(amfc_vals) >= 2 and len(random_vals) >= 2:
        # Paired Wilcoxon: is AMFC better than Random (one-sided)?
        try:
            stat, pvalue = scipy_stats.wilcoxon(
                amfc_vals, random_vals,
                alternative="greater", zero_method="zsplit",
            )
            out["amfc_vs_random_wilcoxon_stat"] = float(stat)
            out["amfc_vs_random_wilcoxon_pvalue"] = float(pvalue)
            out["amfc_beats_random"] = bool(pvalue < 0.05)
        except ValueError as e:
            out["amfc_vs_random_wilcoxon_error"] = str(e)
            out["amfc_beats_random"] = False

    # Gap-to-Oracle
    oracle_vals = pp["oracle_realized_hv"].dropna().values
    if np.mean(oracle_vals) > 0:
        out["gap_to_oracle_amfc"] = float(
            (np.mean(oracle_vals) - np.mean(amfc_vals)) / np.mean(oracle_vals)
        )

    # Spearman rank correlation aggregate
    spearman_vals = pp["spearman_pred_vs_actual_hv"].dropna().values
    if len(spearman_vals) > 0:
        out["spearman_mean"] = float(np.mean(spearman_vals))
        out["spearman_median"] = float(np.median(spearman_vals))
        out["fraction_positive_spearman"] = float(np.mean(spearman_vals > 0))

    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe C — AMFC vs alternative DMs")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_c.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Scenarios: {summary['n_scenarios']}")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append(f"- Reference point (R1, R2) = ({summary['R1']}, {summary['R2']})")
    lines.append("")

    amfc_beats_random = summary.get("amfc_beats_random", None)
    spearman_mean = summary.get("spearman_mean", float("nan"))

    lines.append("## Verdict")
    lines.append("")
    if amfc_beats_random is True:
        verdict = f"🟢 **AMFC beats Random pick** (Wilcoxon p = {summary.get('amfc_vs_random_wilcoxon_pvalue', 'NA'):.4g})"
    elif amfc_beats_random is False:
        verdict = f"🔴 **AMFC does NOT beat Random pick** (Wilcoxon p = {summary.get('amfc_vs_random_wilcoxon_pvalue', 'NA'):.4g}) — DM may be uninformative"
    else:
        verdict = "⚠️ Insufficient data for AMFC vs Random test"
    lines.append(verdict)
    if "gap_to_oracle_amfc" in summary:
        gap = summary["gap_to_oracle_amfc"]
        if gap > 0.5:
            lines.append(f"  - Gap-to-Oracle = {gap*100:.1f}% (LARGE: AMFC leaves substantial value on the table)")
        elif gap > 0.2:
            lines.append(f"  - Gap-to-Oracle = {gap*100:.1f}% (moderate)")
        else:
            lines.append(f"  - Gap-to-Oracle = {gap*100:.1f}% (small: AMFC close to oracle)")
    lines.append("")

    lines.append("## DM aggregate metrics")
    lines.append("")
    lines.append("| DM | n periods | mean realized HV | std | max |")
    lines.append("|---|---|---|---|---|")
    for dm, stats in summary.get("dm_aggregate", {}).items():
        label = dm.replace("_realized_hv", "").replace("_mean", "")
        lines.append(
            f"| {label} | {stats['n_periods']} "
            f"| {stats['mean']:.4e} "
            f"| {stats['std']:.4e} "
            f"| {stats['max']:.4e} |"
        )
    lines.append("")

    lines.append("## AMFC informativeness")
    lines.append("")
    if "spearman_mean" in summary:
        lines.append(f"- Mean Spearman ρ(AMFC predicted HV, realized HV) per period: **{spearman_mean:+.4f}**")
        lines.append(f"- Median: {summary.get('spearman_median', float('nan')):+.4f}")
        lines.append(f"- Fraction of periods with positive ρ: {summary.get('fraction_positive_spearman', float('nan')):.4f}")
        lines.append("")
        if spearman_mean > 0.3:
            lines.append("✅ **AMFC's predicted-HV ranking correlates POSITIVELY with realized HV**: ranking is informative.")
        elif spearman_mean > 0:
            lines.append("🟡 **Mildly positive correlation**: AMFC ranking is weakly informative.")
        else:
            lines.append("🔴 **AMFC's predicted-HV ranking correlates NEGATIVELY or NEUTRALLY with realized HV**: ranking is uninformative or counter-informative.")
    lines.append("")

    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append("| scenario | n periods | AMFC mean | Oracle mean | HighROI mean | Random mean | ρ(pred, realized) |")
    lines.append("|---|---|---|---|---|---|---|")
    for scenario, stats in summary.get("per_scenario", {}).items():
        lines.append(
            f"| {scenario} | {stats['n_periods']} "
            f"| {stats['amfc_mean']:.4e} "
            f"| {stats['oracle_mean']:.4e} "
            f"| {stats['highroi_mean']:.4e} "
            f"| {stats['random_mean']:.4e} "
            f"| {stats['spearman_mean']:+.4f} |"
        )
    lines.append("")

    lines.append("## Decision per backlog spec")
    lines.append("")
    if amfc_beats_random:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe C decision criteria:")
        lines.append("- ✅ AMFC > Random → DM is justified; problem is upstream (KF / TIP / λ machinery)")
        if "gap_to_oracle_amfc" in summary and summary["gap_to_oracle_amfc"] > 0.5:
            lines.append("- ⚠️  Large gap-to-Oracle suggests ensemble (median of DMs) could close more of it")
    else:
        lines.append("Per W22-INSPECTION-BACKLOG.md Probe C decision criteria:")
        lines.append("- 🔴 AMFC ≈ Random → DM is uninformative; **consider replacing DM** (HighROI or Sharpe candidates)")
        # Recommend best alternative
        agg = summary.get("dm_aggregate", {})
        alternatives = [(k, v["mean"]) for k, v in agg.items()
                         if k not in ("amfc_realized_hv", "oracle_realized_hv", "random_realized_hv_mean", "persistence_amfc_realized_hv")]
        if alternatives:
            best_alt = max(alternatives, key=lambda x: x[1])
            lines.append(f"- Best alternative DM (by mean realized HV): **{best_alt[0]}** = {best_alt[1]:.4e}")
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
    print(f"[probe-c] wrote report to {args.output}")

    summary_json = args.output.parent / (args.output.stem + "_summary.json")
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[probe-c] wrote summary JSON to {summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
