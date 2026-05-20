"""W22 Probe P — Mutual information between KF state and OOS Ŝ.

Resolves the KF PARADOX: KF predictions are WORSE than persistence
(MAE -61%/-78%) yet ASMS beats SMS via per-portfolio differentiation.

Per W22-DEEP-PROBES-K-S.md Probe P spec.

Theory (Cover & Thomas §8; Kraskov et al. 2004 k-NN estimator):
- MAE penalizes magnitude error symmetrically
- MI captures whether the RANKING/ORDERING is preserved
- If I(KF_pred ; actual) > I(persistence ; actual), KF carries
  decision-relevant info even when pointwise prediction is poor

Hypotheses:
- H0: I(KF[ROI] ; actual_ROI) ≤ I(persistence_ROI ; actual_ROI)
- H1: I_KF > I_persistence despite worse MAE → KF is a DECISION
  COMPRESSOR, not a predictor

Methodology:
- For each (dataset, period) bucket, build samples:
  - {(kf_predicted_ROI_i, actual_ROI_i)}
  - {(persistence_ROI_i, actual_ROI_i)}
- Estimate MI via k-NN (Kraskov estimator, k=3 default)
- Compare paired across periods, Wilcoxon signed-rank test

Usage:
    uv run python -m experiments.analyze_probe_p \\
        --log-path experiments/results/w22-nc8cv2-nc8d-5seed/predictions.jsonl \\
        --output ../docs/W22-PROBE-P-MUTUAL-INFO.md
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
    from scipy.special import digamma
except ImportError:
    print("ERROR: scipy required", file=sys.stderr)
    sys.exit(1)


def kraskov_mi(x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    """Kraskov et al. (2004) k-NN mutual information estimator.

    MI(X, Y) = ψ(k) - <ψ(n_x + 1) + ψ(n_y + 1)> + ψ(N)

    where n_x, n_y are counts within Chebyshev ε-radius of the k-th nearest
    neighbor in (X, Y) joint space, projected to X and Y marginals.

    Simple, non-parametric, robust to non-Gaussianity. Returns MI in nats.
    """
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    n = len(x)
    if n < 2 * k + 1:
        return float("nan")
    # Joint space: 2D
    z = np.hstack([x, y])
    # k-th nearest neighbor distance per point (Chebyshev = L∞)
    # Compute pairwise Chebyshev distance matrix
    # For each i, find the k-th smallest non-self distance
    eps = np.empty(n)
    for i in range(n):
        d = np.maximum(np.abs(z[i, 0] - z[:, 0]), np.abs(z[i, 1] - z[:, 1]))
        d[i] = np.inf  # exclude self
        d_sorted = np.partition(d, k - 1)[:k]
        eps[i] = np.max(d_sorted)  # k-th nearest neighbor distance
    # Count points within ε in marginal spaces
    n_x = np.empty(n, dtype=int)
    n_y = np.empty(n, dtype=int)
    for i in range(n):
        n_x[i] = int(np.sum(np.abs(x[:, 0] - x[i, 0]) < eps[i]) - 1)  # exclude self
        n_y[i] = int(np.sum(np.abs(y[:, 0] - y[i, 0]) < eps[i]) - 1)
    # Kraskov MI estimator
    mi = float(digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n))
    return mi


def load_records(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame, k: int = 3) -> dict:
    """Compute MI(KF, actual) vs MI(persistence, actual) per period."""
    out = {
        "n_records": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_periods": int(df["period_t"].nunique()),
        "kraskov_k": k,
    }

    # Per (scenario, seed, period) bucket, need ≥ 2k+1 portfolios
    # for KSG estimator. Pareto front size 5-15 → marginal but workable.
    rows = []
    for (scenario, seed, period), grp in df.groupby(["scenario", "seed", "period_t"]):
        if len(grp) < 2 * k + 1:
            continue
        kf_roi = grp["kf_predicted_ROI_t_plus_1"].values
        pers_roi = grp["persistence_ROI_t"].values
        actual_roi = grp["actual_ROI_t_plus_1"].values
        kf_risk = grp["kf_predicted_risk_t_plus_1"].values
        pers_risk = grp["persistence_risk_t"].values
        actual_risk = grp["actual_risk_t_plus_1"].values

        mi_kf_roi = kraskov_mi(kf_roi, actual_roi, k=k)
        mi_pers_roi = kraskov_mi(pers_roi, actual_roi, k=k)
        mi_kf_risk = kraskov_mi(kf_risk, actual_risk, k=k)
        mi_pers_risk = kraskov_mi(pers_risk, actual_risk, k=k)

        rows.append({
            "scenario": scenario,
            "seed": seed,
            "period": period,
            "n_portfolios": len(grp),
            "mi_kf_roi": mi_kf_roi,
            "mi_persistence_roi": mi_pers_roi,
            "mi_kf_risk": mi_kf_risk,
            "mi_persistence_risk": mi_pers_risk,
            "diff_roi": mi_kf_roi - mi_pers_roi,
            "diff_risk": mi_kf_risk - mi_pers_risk,
        })

    if not rows:
        return {**out, "error": "no per-period buckets met n ≥ 2k+1"}

    pp = pd.DataFrame(rows)
    out["n_period_buckets"] = int(len(pp))

    # Aggregate per scenario
    per_scen = {}
    for scenario, scen_grp in pp.groupby("scenario"):
        # Drop NaN
        valid_roi = scen_grp.dropna(subset=["mi_kf_roi", "mi_persistence_roi"])
        valid_risk = scen_grp.dropna(subset=["mi_kf_risk", "mi_persistence_risk"])
        n_roi = len(valid_roi)
        n_risk = len(valid_risk)

        # Paired Wilcoxon: is KF MI > persistence MI?
        roi_p = float("nan")
        roi_better = None
        if n_roi >= 5:
            try:
                stat, p = scipy_stats.wilcoxon(
                    valid_roi["mi_kf_roi"], valid_roi["mi_persistence_roi"],
                    alternative="greater",
                )
                roi_p = float(p)
                roi_better = bool(p < 0.05)
            except ValueError:
                pass
        risk_p = float("nan")
        risk_better = None
        if n_risk >= 5:
            try:
                stat, p = scipy_stats.wilcoxon(
                    valid_risk["mi_kf_risk"], valid_risk["mi_persistence_risk"],
                    alternative="greater",
                )
                risk_p = float(p)
                risk_better = bool(p < 0.05)
            except ValueError:
                pass

        per_scen[scenario] = {
            "n_periods": n_roi,
            "mi_kf_roi_mean": float(np.nanmean(valid_roi["mi_kf_roi"])) if n_roi else float("nan"),
            "mi_persistence_roi_mean": float(np.nanmean(valid_roi["mi_persistence_roi"])) if n_roi else float("nan"),
            "diff_roi_mean": float(np.nanmean(valid_roi["diff_roi"])) if n_roi else float("nan"),
            "wilcoxon_roi_p": roi_p,
            "kf_beats_persistence_roi": roi_better,
            "mi_kf_risk_mean": float(np.nanmean(valid_risk["mi_kf_risk"])) if n_risk else float("nan"),
            "mi_persistence_risk_mean": float(np.nanmean(valid_risk["mi_persistence_risk"])) if n_risk else float("nan"),
            "diff_risk_mean": float(np.nanmean(valid_risk["diff_risk"])) if n_risk else float("nan"),
            "wilcoxon_risk_p": risk_p,
            "kf_beats_persistence_risk": risk_better,
        }
    out["per_scenario"] = per_scen
    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe P — Mutual information KF state ↔ OOS Ŝ")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_p.py`. Kraskov et al. 2004 k-NN estimator.*")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Period buckets (n ≥ 2k+1 = 7): {summary.get('n_period_buckets', 0)}")
    lines.append(f"- k-NN parameter: k = {summary['kraskov_k']}")
    lines.append("")

    if "per_scenario" not in summary or not summary["per_scenario"]:
        return "\n".join(lines + ["Insufficient data."])

    lines.append("## The KF Paradox — does MI explain it?")
    lines.append("")
    lines.append("If MI(KF, actual) > MI(persistence, actual) despite worse MAE,")
    lines.append("then KF is a DECISION COMPRESSOR (preserves ranking) — not a predictor.")
    lines.append("")

    lines.append("## Verdict (per scenario)")
    lines.append("")
    for scenario, stats in summary["per_scenario"].items():
        roi_sig = "✅" if stats.get("kf_beats_persistence_roi") else "🔴" if stats.get("kf_beats_persistence_roi") is False else "—"
        risk_sig = "✅" if stats.get("kf_beats_persistence_risk") else "🔴" if stats.get("kf_beats_persistence_risk") is False else "—"
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append(f"- n period buckets: {stats['n_periods']}")
        lines.append(f"- {roi_sig} ROI: MI(KF) = {stats['mi_kf_roi_mean']:.4f} vs MI(pers) = {stats['mi_persistence_roi_mean']:.4f}; "
                       f"Δ MI = {stats['diff_roi_mean']:+.4f}; Wilcoxon p = {stats['wilcoxon_roi_p']:.4f}")
        lines.append(f"- {risk_sig} risk: MI(KF) = {stats['mi_kf_risk_mean']:.4f} vs MI(pers) = {stats['mi_persistence_risk_mean']:.4f}; "
                       f"Δ MI = {stats['diff_risk_mean']:+.4f}; Wilcoxon p = {stats['wilcoxon_risk_p']:.4f}")
        lines.append("")

    lines.append("## Decision per probe spec")
    lines.append("")
    for scenario, stats in summary["per_scenario"].items():
        roi_b = stats.get("kf_beats_persistence_roi")
        if roi_b:
            lines.append(f"- ✅ **{scenario} (ROI)**: KF MI > persistence MI → KF preserves decision-relevant ORDERING even though MAE is worse. Reframe KF as DECISION COMPRESSOR.")
        elif roi_b is False:
            lines.append(f"- 🔴 **{scenario} (ROI)**: KF MI ≤ persistence MI → KF genuinely uninformative for ROI prediction. Win mechanism must come from elsewhere (NC8b selection-quality?).")
        else:
            lines.append(f"- ⚠️ **{scenario} (ROI)**: insufficient data for Wilcoxon (n < 5 valid buckets).")
    lines.append("")

    lines.append("## Note on caveats")
    lines.append("")
    lines.append("- KSG estimator is non-parametric but has finite-sample bias for small n.")
    lines.append("- Each (scenario, seed, period) bucket has ~5-15 portfolios → small per-period sample.")
    lines.append("- We aggregate WILCOXON over periods (each period contributes one (MI_kf, MI_pers) pair).")
    lines.append("- Negative MI estimates can occur due to estimator bias; absolute values aren't directly interpretable.")
    lines.append("- The COMPARISON (KF vs persistence) is what matters.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.log_path.exists():
        print(f"ERROR: log file not found: {args.log_path}", file=sys.stderr)
        return 1
    df = load_records(args.log_path)
    if df.empty:
        print(f"ERROR: log file empty", file=sys.stderr)
        return 1
    summary = analyze(df, k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(summary))
    print(f"[probe-p] wrote report to {args.output}")
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[probe-p] wrote summary JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
