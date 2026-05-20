"""W22 Probe AD — empirical Δ_S rectangle comparison.

Question: Does the rectangle mismatch between Python deterministic
`_compute_hypervolume_contributions_class` (sms_emoa.py:723:
`(ROI_i − ROI_{i+1})(risk_{i−1} − risk_i)`) and the canonical Eq 6.36
rectangle (`(ROI_i − ROI_{i−1})(risk_{i+1} − risk_i)`) cause meaningful
argmax disagreement on realistic Pareto fronts?

Test: synthesize concave Pareto fronts of varying densities and
non-uniformities, compute 4 estimators of Δ_S per solution:

  1. `deterministic_python` (current buggy rectangle)
  2. `deterministic_eq636` (corrected rectangle = true 2D HV contribution)
  3. `stochastic_eq641` (Eq 6.41 = eq636 minus self-cov)
  4. `mc_ground_truth_eq636` (sampled-front avg of eq636 rectangle)

Report per-front:
  - L1 |det_python − mc_eq636|
  - L1 |det_eq636 − mc_eq636|  (should be ≈ 0 in expectation)
  - argmax-agreement: does det_python pick the SAME worst solution as
    det_eq636 / stoch / mc? (the argmax-min of Δ_S is what
    `_remove_worst_solution` uses.)

Conclusion test: if det_python disagrees with eq636 / stoch / mc on the
worst-solution choice in >X% of trials, the bug is materially affecting
SMS-EMOA's removal decisions and a fix is justified.

Usage:
    uv run python -m experiments.analyze_probe_ad \
        --output ../docs/W22-PROBE-AD-EMPIRICAL.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.probes.probe_ad_delta_s_comparison import (
    compare_methods_with_eq636,
    deterministic_delta_s,
    deterministic_eq636_delta_s,
    stochastic_delta_s,
)


R1, R2 = 0.0, 0.2


def synthesize_concave_front(n: int, curvature: float,
                              ranges: tuple[float, float, float, float],
                              jitter: float,
                              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a concave (maximize ROI, minimize risk) Pareto front.

    ROI[i] = roi_min + (roi_max - roi_min) * t[i]^curvature
    risk[i] = risk_min + (risk_max - risk_min) * t[i]
    where t = linspace(0, 1, n).

    `curvature > 1` makes the front bowed toward (high ROI, low risk);
    `curvature < 1` bows it toward (low ROI, high risk).

    `jitter` adds small per-point Gaussian noise to ROI to make the
    front non-uniform (the case where the rectangle bug matters most).
    """
    roi_min, roi_max, risk_min, risk_max = ranges
    t = np.linspace(0, 1, n)
    rois = roi_min + (roi_max - roi_min) * np.power(t, curvature)
    risks = risk_min + (risk_max - risk_min) * t
    rois += rng.normal(0, jitter, size=n)
    # Keep ROI sorted (jitter could break order, re-enforce):
    rois.sort()
    # Generate small positive per-solution Cov(ROI, risk).
    # Make ~30% positive (correlated objectives), 30% zero, 30% negative,
    # to exercise the full stochastic - cov correction.
    covs = rng.uniform(-0.001, 0.001, size=n)
    return rois, risks, covs


def run_one_trial(seed: int, n: int, curvature: float, jitter: float,
                   n_mc: int) -> dict:
    rng = np.random.default_rng(seed)
    rois, risks, covs = synthesize_concave_front(
        n, curvature, (0.0001, 0.0008, 0.005, 0.18), jitter, rng,
    )
    result = compare_methods_with_eq636(rois, risks, covs, R1, R2,
                                          n_mc=n_mc, rng=rng)
    det_py = result["deterministic"]
    det_eq636 = result["deterministic_eq636"]
    stoch = result["stochastic"]
    mc_eq636 = result["mc_eq636"]

    argmin_det_py = int(np.argmin(det_py))
    argmin_eq636 = int(np.argmin(det_eq636))
    argmin_stoch = int(np.argmin(stoch))
    argmin_mc_eq636 = int(np.argmin(mc_eq636))

    argmax_det_py = int(np.argmax(det_py))
    argmax_eq636 = int(np.argmax(det_eq636))
    argmax_stoch = int(np.argmax(stoch))
    argmax_mc_eq636 = int(np.argmax(mc_eq636))

    return {
        "seed": seed,
        "n": n,
        "curvature": curvature,
        "jitter": jitter,
        "l1_detpy_vs_mc_eq636": result["l1_det_vs_mc_eq636"],
        "l1_eq636_vs_mc_eq636": result["l1_eq636_vs_mc_eq636"],
        "l1_stoch_vs_mc_eq636": result["l1_stoch_vs_mc_eq636"],
        "argmin_detpy": argmin_det_py,
        "argmin_eq636": argmin_eq636,
        "argmin_stoch": argmin_stoch,
        "argmin_mc": argmin_mc_eq636,
        "argmax_detpy": argmax_det_py,
        "argmax_eq636": argmax_eq636,
        "argmax_stoch": argmax_stoch,
        "argmax_mc": argmax_mc_eq636,
        # Worst-removal disagreement: bool
        "argmin_detpy_vs_mc_disagree": argmin_det_py != argmin_mc_eq636,
        "argmin_eq636_vs_mc_disagree": argmin_eq636 != argmin_mc_eq636,
        "argmin_stoch_vs_mc_disagree": argmin_stoch != argmin_mc_eq636,
        # Tournament-pick disagreement: bool
        "argmax_detpy_vs_mc_disagree": argmax_det_py != argmax_mc_eq636,
        "argmax_eq636_vs_mc_disagree": argmax_eq636 != argmax_mc_eq636,
        "argmax_stoch_vs_mc_disagree": argmax_stoch != argmax_mc_eq636,
    }


def summarize(trials: list[dict]) -> dict:
    n_trials = len(trials)
    out = {"n_trials": n_trials}
    for key in ["l1_detpy_vs_mc_eq636", "l1_eq636_vs_mc_eq636",
                  "l1_stoch_vs_mc_eq636"]:
        vals = [t[key] for t in trials]
        out[key + "_mean"] = float(np.mean(vals))
        out[key + "_median"] = float(np.median(vals))
        out[key + "_max"] = float(np.max(vals))
    for key in ["argmin_detpy_vs_mc_disagree",
                  "argmin_eq636_vs_mc_disagree",
                  "argmin_stoch_vs_mc_disagree",
                  "argmax_detpy_vs_mc_disagree",
                  "argmax_eq636_vs_mc_disagree",
                  "argmax_stoch_vs_mc_disagree"]:
        n_disagree = sum(int(t[key]) for t in trials)
        out[key + "_pct"] = 100.0 * n_disagree / n_trials
        out[key + "_count"] = n_disagree
    return out


def format_report(grid: list[dict], grid_summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe AD — Δ_S rectangle empirical comparison")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_ad.py`.*")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("Does the rectangle mismatch in `sms_emoa.py:_compute_hypervolume_contributions_class` middle branch")
    lines.append("(uses `(ROI_i − ROI_{i+1})(risk_{i−1} − risk_i)`) vs. Eq 6.36 canonical rectangle")
    lines.append("(`(ROI_i − ROI_{i−1})(risk_{i+1} − risk_i)`) cause meaningful disagreement on synthetic Pareto fronts?")
    lines.append("")
    lines.append("Both rectangles are positive on a Pareto front, but they're DIFFERENT rectangles. The Eq 6.36 form is")
    lines.append("the TRUE unique 2D hypervolume contribution of each solution.")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- Synthesize concave Pareto fronts of varying size (n), curvature, and jitter.")
    lines.append(f"- For each trial, compute 4 estimators and the L1 error vs. MC (n_mc=5000) ground truth using Eq 6.36 rectangle on sampled fronts.")
    lines.append("- Report: argmin agreement (worst-solution removal — used by `_remove_worst_solution`) and argmax agreement (tournament-selection winner).")
    lines.append("")
    lines.append("## Per-configuration summary")
    lines.append("")
    lines.append("| n | curvature | jitter | trials | L1 det_py | L1 eq636 | L1 stoch | argmin det_py disagree % | argmin eq636 disagree % | argmin stoch disagree % | argmax det_py disagree % |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in grid:
        s = cfg["summary"]
        lines.append(
            f"| {cfg['n']} | {cfg['curvature']:.2f} | {cfg['jitter']:.4f} | {s['n_trials']} "
            f"| {s['l1_detpy_vs_mc_eq636_mean']:.4e} "
            f"| {s['l1_eq636_vs_mc_eq636_mean']:.4e} "
            f"| {s['l1_stoch_vs_mc_eq636_mean']:.4e} "
            f"| {s['argmin_detpy_vs_mc_disagree_pct']:.1f}% "
            f"| {s['argmin_eq636_vs_mc_disagree_pct']:.1f}% "
            f"| {s['argmin_stoch_vs_mc_disagree_pct']:.1f}% "
            f"| {s['argmax_detpy_vs_mc_disagree_pct']:.1f}% |"
        )
    lines.append("")
    lines.append("## Grand summary across all configurations")
    lines.append("")
    lines.append(f"- Trials: {grid_summary['n_trials']}")
    lines.append(f"- L1 |det_python − MC_eq636| mean: {grid_summary['l1_detpy_vs_mc_eq636_mean']:.4e}")
    lines.append(f"- L1 |det_eq636 − MC_eq636| mean: {grid_summary['l1_eq636_vs_mc_eq636_mean']:.4e}")
    lines.append(f"- L1 |stoch_eq641 − MC_eq636| mean: {grid_summary['l1_stoch_vs_mc_eq636_mean']:.4e}")
    lines.append(f"- **argmin (worst-removal) det_py disagrees with MC_eq636: {grid_summary['argmin_detpy_vs_mc_disagree_pct']:.1f}% of trials ({grid_summary['argmin_detpy_vs_mc_disagree_count']}/{grid_summary['n_trials']})**")
    lines.append(f"- argmin det_eq636 disagrees with MC_eq636: {grid_summary['argmin_eq636_vs_mc_disagree_pct']:.1f}%")
    lines.append(f"- argmin stoch disagrees with MC_eq636: {grid_summary['argmin_stoch_vs_mc_disagree_pct']:.1f}%")
    lines.append(f"- argmax (tournament-winner) det_py disagrees with MC_eq636: {grid_summary['argmax_detpy_vs_mc_disagree_pct']:.1f}%")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if grid_summary['argmin_detpy_vs_mc_disagree_pct'] > 20:
        lines.append("🔴 **Material impact**: the Python deterministic rectangle disagrees with MC ground truth on")
        lines.append(f"worst-solution removal in **{grid_summary['argmin_detpy_vs_mc_disagree_pct']:.0f}% of trials**. SMS-EMOA's")
        lines.append("removal decisions are systematically biased by the rectangle bug. A fix to align")
        lines.append("`_compute_hypervolume_contributions_class` with Eq 6.36 is warranted.")
    elif grid_summary['argmin_detpy_vs_mc_disagree_pct'] > 5:
        lines.append(f"🟡 **Moderate impact**: {grid_summary['argmin_detpy_vs_mc_disagree_pct']:.1f}% disagreement on worst-removal.")
        lines.append("The rectangle bug occasionally biases SMS-EMOA's decisions. Worth fixing in a follow-up wave.")
    else:
        lines.append(f"🟢 **Negligible impact**: only {grid_summary['argmin_detpy_vs_mc_disagree_pct']:.1f}% of trials disagree.")
        lines.append("The rectangle mismatch is mathematically real but doesn't change SMS-EMOA's behavior in practice.")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("Synthetic concave Pareto fronts may differ from those produced by real FTSE / PO data. A")
    lines.append("genuine follow-up would log per-generation actual Pareto fronts from a real run and rerun")
    lines.append("the analysis on those.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-mc", type=int, default=5000)
    parser.add_argument("--seeds-per-config", type=int, default=50)
    args = parser.parse_args(argv)

    # Sweep grid
    configs = []
    for n in [5, 10, 20]:
        for curvature in [0.5, 1.0, 2.0]:
            for jitter in [0.00001, 0.00005, 0.0002]:
                configs.append({"n": n, "curvature": curvature, "jitter": jitter})

    grid = []
    all_trials = []
    for cfg in configs:
        trials = [
            run_one_trial(seed, cfg["n"], cfg["curvature"], cfg["jitter"],
                            args.n_mc)
            for seed in range(args.seeds_per_config)
        ]
        all_trials.extend(trials)
        summary = summarize(trials)
        grid.append({**cfg, "summary": summary})
        print(f"[probe-ad] n={cfg['n']} curv={cfg['curvature']:.2f} "
              f"jitter={cfg['jitter']:.4f}  "
              f"argmin_detpy_disagree={summary['argmin_detpy_vs_mc_disagree_pct']:.1f}%  "
              f"argmax_detpy_disagree={summary['argmax_detpy_vs_mc_disagree_pct']:.1f}%",
              file=sys.stderr)

    grand_summary = summarize(all_trials)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(grid, grand_summary))
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps({"grid": grid, "grand": grand_summary}, indent=2)
    )
    print(f"[probe-ad] wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
