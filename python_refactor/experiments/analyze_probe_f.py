"""W22 Probe F — Dirichlet predictor informativeness (decision-space sibling of Probe A).

Per W22-INSPECTION-BACKLOG.md Probe F spec:
Tests whether the Dirichlet predictor (decision-space) adds information
beyond simple persistence (re-use previous period's AMFC weights).

Methodology:
- For each (scenario, seed) trajectory, pick AMFC weights per period
  (argmax of predicted EFHV, same convention as Probe G/J)
- For period t+1, compute Dirichlet prediction = blend(prev, current, λ)
  via DirichletPredictor.dirichlet_mean_prediction_vec
- Compare to actual t+1 AMFC weights (closest match approximation)
- Compute L1 distance + Jaccard similarity vs:
  (a) Dirichlet prediction
  (b) Persistence baseline (use period-(t-1)'s AMFC as predictor)

Hypothesis:
- H0: L1(Dirichlet, actual) ≥ L1(persistence, actual) — Dirichlet no better
- H1: Dirichlet < persistence (Dirichlet adds info)

Limitations:
- "Actual" is approximated by the closest-matching portfolio in period
  t+1's logged Pareto front (no asset-level OOS returns logged)
- Persistence ≡ period-(t-1)'s AMFC weights

Usage:
    uv run python -m experiments.analyze_probe_f \\
        --log-path experiments/results/.../predictions.jsonl \\
        --output ../docs/W22-PROBE-F-DIRICHLET-INFORMATIVENESS.md
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

# Import the predictor under test
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from src.algorithms.anticipatory_learning import DirichletPredictor
except ImportError:
    print("ERROR: cannot import DirichletPredictor", file=sys.stderr)
    sys.exit(1)


R1, R2 = 0.0, 0.2


def hv_proxy(roi: float, risk: float) -> float:
    return max(roi - R1, 0.0) * max(R2 - risk, 0.0)


def load_records(log_path: Path) -> pd.DataFrame:
    rows = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def pick_amfc_weights(group: pd.DataFrame) -> np.ndarray:
    """Pick AMFC weights from a (scenario, seed, period) group."""
    pred_hv = group.apply(
        lambda r: hv_proxy(r["kf_predicted_ROI_t_plus_1"], r["kf_predicted_risk_t_plus_1"]),
        axis=1,
    ).values
    idx = int(np.argmax(pred_hv)) if np.any(pred_hv > 0) else 0
    return np.array(group.iloc[idx]["weights"])


def jaccard(w1: np.ndarray, w2: np.ndarray, threshold: float = 1e-12) -> float:
    a1 = set(np.flatnonzero(np.abs(w1) > threshold).tolist())
    a2 = set(np.flatnonzero(np.abs(w2) > threshold).tolist())
    union = a1 | a2
    inter = a1 & a2
    return float(len(inter) / len(union)) if union else 1.0


def analyze(df: pd.DataFrame, anticipative_rate: float = 0.5) -> dict:
    """Compute Probe F metrics on existing predictions.jsonl."""
    out = {
        "n_records": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_periods": int(df["period_t"].nunique()),
        "anticipative_rate_default": float(anticipative_rate),
    }

    rows = []
    for (scenario, seed), seed_grp in df.groupby(["scenario", "seed"]):
        amfc_by_period = {}
        for period, period_grp in seed_grp.groupby("period_t"):
            amfc_by_period[int(period)] = pick_amfc_weights(period_grp)

        sorted_periods = sorted(amfc_by_period.keys())
        # For each consecutive triple (t-1, t, t+1):
        # - persistence prediction for t+1 = AMFC at t-1
        # - Dirichlet prediction for t+1 = mix(AMFC at t-1, AMFC at t)
        # - actual t+1 = closest-match portfolio in t+1's logged front
        for i in range(2, len(sorted_periods)):
            t_minus_2 = sorted_periods[i - 2]
            t_minus_1 = sorted_periods[i - 1]
            t_actual = sorted_periods[i]
            w_prev = amfc_by_period[t_minus_2]
            w_curr = amfc_by_period[t_minus_1]
            w_actual = amfc_by_period[t_actual]  # treat AMFC at t as "actual"

            # Dirichlet prediction = mix(w_prev, w_curr, anticipative_rate)
            w_dirichlet = DirichletPredictor.dirichlet_mean_prediction_vec(
                w_prev, w_curr, anticipative_rate
            )
            # Persistence prediction = w_curr (re-use current AMFC)
            w_persistence = w_curr.copy()

            l1_dirichlet = float(np.sum(np.abs(w_dirichlet - w_actual)))
            l1_persistence = float(np.sum(np.abs(w_persistence - w_actual)))
            jaccard_dirichlet = jaccard(w_dirichlet, w_actual)
            jaccard_persistence = jaccard(w_persistence, w_actual)
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "period": t_actual,
                "l1_dirichlet": l1_dirichlet,
                "l1_persistence": l1_persistence,
                "jaccard_dirichlet": jaccard_dirichlet,
                "jaccard_persistence": jaccard_persistence,
                "l1_improvement": l1_persistence - l1_dirichlet,  # > 0 = Dirichlet better
                "jaccard_improvement": jaccard_dirichlet - jaccard_persistence,
            })

    if not rows:
        return out

    pp = pd.DataFrame(rows)

    out["l1_dirichlet_mean"] = float(np.mean(pp["l1_dirichlet"]))
    out["l1_persistence_mean"] = float(np.mean(pp["l1_persistence"]))
    out["l1_dirichlet_median"] = float(np.median(pp["l1_dirichlet"]))
    out["l1_persistence_median"] = float(np.median(pp["l1_persistence"]))
    out["jaccard_dirichlet_mean"] = float(np.mean(pp["jaccard_dirichlet"]))
    out["jaccard_persistence_mean"] = float(np.mean(pp["jaccard_persistence"]))

    # Paired Wilcoxon: is Dirichlet's L1 LESS than persistence's L1?
    try:
        stat, pvalue = scipy_stats.wilcoxon(
            pp["l1_dirichlet"].values, pp["l1_persistence"].values,
            alternative="less",
        )
        out["wilcoxon_l1_dirichlet_less_than_persistence_stat"] = float(stat)
        out["wilcoxon_l1_dirichlet_less_than_persistence_pvalue"] = float(pvalue)
        out["dirichlet_beats_persistence"] = bool(pvalue < 0.05)
    except ValueError as e:
        out["wilcoxon_error"] = str(e)
        out["dirichlet_beats_persistence"] = False

    # Per-scenario
    per_scen = {}
    for scenario, scen_grp in pp.groupby("scenario"):
        per_scen[scenario] = {
            "n": int(len(scen_grp)),
            "l1_dirichlet_mean": float(np.mean(scen_grp["l1_dirichlet"])),
            "l1_persistence_mean": float(np.mean(scen_grp["l1_persistence"])),
            "l1_improvement_mean": float(np.mean(scen_grp["l1_improvement"])),
            "fraction_dirichlet_better": float(np.mean(scen_grp["l1_improvement"] > 0)),
        }
    out["per_scenario"] = per_scen

    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe F — Dirichlet predictor informativeness")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_f.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Scenarios: {summary['n_scenarios']}")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append(f"- Anticipative rate (default): {summary.get('anticipative_rate_default', 'NA')}")
    lines.append("")

    if "l1_dirichlet_mean" not in summary:
        return "\n".join(lines + ["Insufficient data for Probe F analysis."])

    dbp = summary.get("dirichlet_beats_persistence", None)
    lines.append("## Verdict")
    lines.append("")
    if dbp:
        verdict = f"🟢 **Dirichlet predictor beats persistence** (Wilcoxon p = {summary['wilcoxon_l1_dirichlet_less_than_persistence_pvalue']:.4g}) — decision-space anticipation has signal"
    elif dbp is False:
        verdict = f"🔴 **Dirichlet predictor does NOT beat persistence** (Wilcoxon p = {summary['wilcoxon_l1_dirichlet_less_than_persistence_pvalue']:.4g}) — simplify (drop Dirichlet or use persistence)"
    else:
        verdict = "⚠️ Insufficient data for paired test"
    lines.append(verdict)
    lines.append("")

    lines.append("## L1 distance comparison (lower is better — closer to actual)")
    lines.append("")
    lines.append("| metric | Dirichlet | Persistence | Δ (Pers − Dir) |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| mean L1 | {summary['l1_dirichlet_mean']:.4f} "
        f"| {summary['l1_persistence_mean']:.4f} "
        f"| {summary['l1_persistence_mean'] - summary['l1_dirichlet_mean']:+.4f} |"
    )
    lines.append(
        f"| median L1 | {summary['l1_dirichlet_median']:.4f} "
        f"| {summary['l1_persistence_median']:.4f} "
        f"| {summary['l1_persistence_median'] - summary['l1_dirichlet_median']:+.4f} |"
    )
    lines.append("")

    lines.append("## Jaccard similarity (higher is better — more asset overlap with actual)")
    lines.append("")
    lines.append("| metric | Dirichlet | Persistence | Δ (Dir − Pers) |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| mean Jaccard | {summary['jaccard_dirichlet_mean']:.4f} "
        f"| {summary['jaccard_persistence_mean']:.4f} "
        f"| {summary['jaccard_dirichlet_mean'] - summary['jaccard_persistence_mean']:+.4f} |"
    )
    lines.append("")

    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append("| scenario | n | L1 Dirichlet | L1 Persistence | L1 improvement | Fraction Dir < Pers |")
    lines.append("|---|---|---|---|---|---|")
    for scenario, stats in summary["per_scenario"].items():
        lines.append(
            f"| {scenario} | {stats['n']} "
            f"| {stats['l1_dirichlet_mean']:.4f} "
            f"| {stats['l1_persistence_mean']:.4f} "
            f"| {stats['l1_improvement_mean']:+.4f} "
            f"| {stats['fraction_dirichlet_better']:.4f} |"
        )
    lines.append("")

    lines.append("## Decision per backlog spec")
    lines.append("")
    if dbp:
        lines.append("- ✅ Dirichlet > persistence → decision-space anticipation has signal → focus on integrating with objective-space")
    else:
        lines.append("- 🔴 Dirichlet ≈ persistence → simplify (drop Dirichlet, just use prev AMFC weights as predictor)")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("- \"Actual\" t+1 weights are approximated by AMFC's chosen weights at period t+1.")
    lines.append("- This treats AMFC's choice as the ground-truth target. True actual would require a different oracle (e.g., the OOS-optimal portfolio).")
    lines.append("- Dirichlet prediction uses default anticipative_rate = 0.5.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anticipative-rate", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not args.log_path.exists():
        print(f"ERROR: log file not found at {args.log_path}", file=sys.stderr)
        return 1
    df = load_records(args.log_path)
    if df.empty:
        print(f"ERROR: log file is empty at {args.log_path}", file=sys.stderr)
        return 1
    summary = analyze(df, anticipative_rate=args.anticipative_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(summary))
    print(f"[probe-f] wrote report to {args.output}")
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[probe-f] wrote summary JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
