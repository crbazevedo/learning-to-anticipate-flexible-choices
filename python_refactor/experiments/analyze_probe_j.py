"""W22 Probe J — "do nothing" baseline (skip optimization, reuse prev AMFC).

Motivation: with Probe G showing chaotic AMFC trajectory (Jaccard 0.169),
maybe the optimizer is HURTING by re-shuffling portfolios each period.
A simple "buy and hold the previous AMFC" baseline would:
- Incur zero transaction cost
- Eliminate chaos
- Be the trivial-but-strong benchmark

This probe computes: for each period t > 0, what would the realized OOS
HV be if we simply held the AMFC weights from period t-1 (instead of
re-running optimization for period t)?

Compare to actual ASMS/SMS realized HV per period. If "do nothing"
matches or beats, the optimization is mostly noise.

Reads predictions.jsonl which contains per-portfolio per-period:
- weights, actual_ROI_t_plus_1, actual_risk_t_plus_1
We can pick AMFC weights per period (via predicted-HV argmax), then
for each consecutive pair (t, t+1) compute what holding period-t's AMFC
weights would have yielded at period t+1.

For this to work we need to also evaluate the period-t AMFC weights
using period-t+1's OOS data — which requires the actual returns. We
approximate using period-t+1's actual_ROI/risk of the closest-matching
portfolio in period-t+1's logged front (since we don't have raw returns).

Simpler approach: compute baseline as the period-t+1 actual ROI/risk
of the period-t AMFC weights (via dot product with the asset-level
period-t+1 stats which we don't have either).

Cleanest approach: this probe REQUIRES adding asset-level OOS data
logging OR re-running with explicit "do nothing" wiring. Defer the
full Probe J to a follow-up.

For now, this analyzer outputs a SIMPLIFIED Probe J: per-period AMFC
"persistence" (just reusing the actual ROI/risk of the prev AMFC
portfolio assuming the asset universe is identical). This isn't quite
"do nothing" but gives a rough lower-bound on what the no-optimization
strategy would yield.

Usage:
    uv run python -m experiments.analyze_probe_j \\
        --log-path experiments/results/.../predictions.jsonl \\
        --output ../docs/W22-PROBE-J-DO-NOTHING-BASELINE.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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


def pick_amfc(group: pd.DataFrame) -> dict:
    """Pick AMFC portfolio from a (scenario, seed, period) group."""
    pred_hv = group.apply(
        lambda r: hv_proxy(r["kf_predicted_ROI_t_plus_1"], r["kf_predicted_risk_t_plus_1"]),
        axis=1,
    ).values
    idx = int(np.argmax(pred_hv)) if np.any(pred_hv > 0) else 0
    row = group.iloc[idx]
    return {
        "weights": np.array(row["weights"]),
        "actual_roi": float(row["actual_ROI_t_plus_1"]),
        "actual_risk": float(row["actual_risk_t_plus_1"]),
        "realized_hv": hv_proxy(row["actual_ROI_t_plus_1"], row["actual_risk_t_plus_1"]),
    }


def analyze(df: pd.DataFrame) -> dict:
    """Compute Probe J: do-nothing baseline vs actual AMFC trajectory."""
    out = {
        "n_records": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "n_periods": int(df["period_t"].nunique()),
    }

    # For each (scenario, seed) trajectory, compute AMFC realized HV per period
    # PLUS "do nothing" baseline (period-t result of holding period-(t-1) weights
    # — approximated by the closest-matching portfolio in period-t's logged front)
    rows = []
    for (scenario, seed), seed_grp in df.groupby(["scenario", "seed"]):
        amfc_by_period: dict = {}
        for period, period_grp in seed_grp.groupby("period_t"):
            amfc_by_period[int(period)] = pick_amfc(period_grp)

        sorted_periods = sorted(amfc_by_period.keys())
        # For each consecutive pair (t-1, t): "do nothing" baseline =
        # find the portfolio in period-t's logged front with weights closest
        # to period-(t-1)'s AMFC weights, take its realized HV.
        for i in range(1, len(sorted_periods)):
            t_prev = sorted_periods[i - 1]
            t_curr = sorted_periods[i]
            prev_w = amfc_by_period[t_prev]["weights"]
            curr_grp = seed_grp[seed_grp["period_t"] == t_curr]

            # Find closest-matching portfolio (L1 distance) in period-t's front
            l1_dists = curr_grp.apply(
                lambda r: float(np.sum(np.abs(np.array(r["weights"]) - prev_w))), axis=1
            ).values
            closest_idx = int(np.argmin(l1_dists))
            closest_row = curr_grp.iloc[closest_idx]

            do_nothing_realized = hv_proxy(
                float(closest_row["actual_ROI_t_plus_1"]),
                float(closest_row["actual_risk_t_plus_1"]),
            )
            actual_amfc_realized = amfc_by_period[t_curr]["realized_hv"]
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "period_t_curr": t_curr,
                "amfc_realized_hv": actual_amfc_realized,
                "do_nothing_realized_hv": do_nothing_realized,
                "l1_closest_dist": float(l1_dists[closest_idx]),
            })

    if not rows:
        return out

    pp = pd.DataFrame(rows)
    out["dm_aggregate"] = {
        "amfc": {
            "n_periods": int(len(pp)),
            "mean": float(np.mean(pp["amfc_realized_hv"])),
            "median": float(np.median(pp["amfc_realized_hv"])),
            "std": float(np.std(pp["amfc_realized_hv"])),
        },
        "do_nothing": {
            "n_periods": int(len(pp)),
            "mean": float(np.mean(pp["do_nothing_realized_hv"])),
            "median": float(np.median(pp["do_nothing_realized_hv"])),
            "std": float(np.std(pp["do_nothing_realized_hv"])),
        },
    }
    out["fraction_do_nothing_better"] = float(
        np.mean(pp["do_nothing_realized_hv"] > pp["amfc_realized_hv"])
    )
    out["mean_l1_closest_match"] = float(np.mean(pp["l1_closest_dist"]))

    # Per-scenario breakdown
    per_scen = {}
    for scenario, scen_grp in pp.groupby("scenario"):
        per_scen[scenario] = {
            "n": int(len(scen_grp)),
            "amfc_mean": float(np.mean(scen_grp["amfc_realized_hv"])),
            "do_nothing_mean": float(np.mean(scen_grp["do_nothing_realized_hv"])),
            "fraction_dn_better": float(np.mean(
                scen_grp["do_nothing_realized_hv"] > scen_grp["amfc_realized_hv"]
            )),
        }
    out["per_scenario"] = per_scen

    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe J — \"do nothing\" baseline (reuse prev AMFC)")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_j.py`.*")
    lines.append("")
    lines.append("## Sample")
    lines.append("")
    lines.append(f"- Records: {summary['n_records']}")
    lines.append(f"- Scenarios: {summary['n_scenarios']}")
    lines.append(f"- Seeds: {summary['n_seeds']}")
    lines.append(f"- Periods: {summary['n_periods']}")
    lines.append(f"- Mean L1 dist to closest-match portfolio: {summary.get('mean_l1_closest_match', 'NA'):.4f}")
    lines.append("")

    if "dm_aggregate" not in summary:
        return "\n".join(lines + ["Insufficient data."])

    frac_dn = summary["fraction_do_nothing_better"]
    if frac_dn > 0.5:
        verdict = f"🔴 **\"Do nothing\" beats actual AMFC** in {frac_dn*100:.1f}% of period transitions — optimizer may be HURTING"
    elif frac_dn > 0.3:
        verdict = f"🟡 **\"Do nothing\" competitive** ({frac_dn*100:.1f}% of transitions) — moderate optimization value"
    else:
        verdict = f"🟢 **AMFC beats \"do nothing\"** in {(1-frac_dn)*100:.1f}% of transitions — optimization is justified"
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")

    lines.append("## Realized HV comparison")
    lines.append("")
    lines.append("| DM | n periods | mean realized HV | std |")
    lines.append("|---|---|---|---|")
    for dm, stats in summary["dm_aggregate"].items():
        lines.append(f"| {dm} | {stats['n_periods']} | {stats['mean']:.4e} | {stats['std']:.4e} |")
    lines.append("")
    a = summary["dm_aggregate"]["amfc"]["mean"]
    d = summary["dm_aggregate"]["do_nothing"]["mean"]
    if d > 0:
        lines.append(f"- Δ = AMFC − DoNothing = {a - d:+.4e} ({(a-d)/d*100:+.2f}%)")
    lines.append("")

    lines.append("## Per-scenario breakdown")
    lines.append("")
    lines.append("| scenario | n | AMFC mean | DoNothing mean | Fraction DN better |")
    lines.append("|---|---|---|---|---|")
    for scenario, stats in summary["per_scenario"].items():
        lines.append(
            f"| {scenario} | {stats['n']} "
            f"| {stats['amfc_mean']:.4e} "
            f"| {stats['do_nothing_mean']:.4e} "
            f"| {stats['fraction_dn_better']:.4f} |"
        )
    lines.append("")

    lines.append("## Caveat")
    lines.append("")
    lines.append(f"- \"Do nothing\" baseline is APPROXIMATE: it uses the period-t Pareto-front portfolio closest to period-(t-1)'s AMFC (mean L1 dist = {summary.get('mean_l1_closest_match', 'NA'):.4f}, max possible = 2.0).")
    lines.append("- True \"buy and hold\" would require re-evaluating period-(t-1)'s AMFC weights with period-t's actual asset returns — not directly possible from the probe log (no asset-level OOS returns logged). The full Probe J would require adding asset-level OOS data logging to walk_forward.py.")
    lines.append("- This approximation BIASES the do-nothing baseline downward when the closest match has noticeable L1 distance, so the true do-nothing value may be HIGHER (more competitive against AMFC).")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if not args.log_path.exists():
        print(f"ERROR: {args.log_path} not found", file=sys.stderr)
        return 1
    df = load_records(args.log_path)
    if df.empty:
        print(f"ERROR: {args.log_path} empty", file=sys.stderr)
        return 1
    summary = analyze(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(summary))
    print(f"[probe-j] wrote report to {args.output}")
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[probe-j] wrote summary JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
