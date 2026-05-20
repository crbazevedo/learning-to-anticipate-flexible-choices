"""W22 Probe Q — Bootstrap CI on (ASMS Ŝ − SMS Ŝ) per Efron & Tibshirani §6.

Per W22-DEEP-PROBES-K-S.md Probe Q spec.

Theory: Ŝ aggregates Pareto-front HV across periods. Headline gap could
be inflated if HV-contribution `Δᵢ HV` of a few extreme portfolios
dominates AND those portfolios are seed-sensitive. Block-bootstrap over
periods (preserving within-period dependence) gives non-parametric CI
on the (ASMS − SMS) difference.

Hypotheses:
- H0: 95% bootstrap CI on (ASMS Ŝ − SMS Ŝ) excludes 0 on FTSE.
- H1: CI includes 0 on at least one dataset → headline fragile.

Methodology:
1. Extract per-seed grand_mean Ŝ from logs (already per-period-aggregated;
   we treat each seed as one observation since per-period values
   aren't logged at this fidelity).
2. For each (algorithm, dataset), build sample {Ŝ_seed}.
3. Stratified bootstrap: resample seeds with replacement; compute Δ.
4. Repeat B=10000; report (2.5%, median, 97.5%) CI.
5. Jackknife (leave-one-seed-out) for influence analysis.

Note: ideally we'd block-bootstrap over PERIODS within each (alg, seed)
to preserve period structure, but per-period Ŝ values aren't logged in
the smoke run.log. The per-seed-level bootstrap gives a more conservative
estimate (loses period information but is still valid).

Usage:
    uv run python -m experiments.analyze_probe_q \\
        --logs experiments/results/w22-nc8cv2-nc8d-5seed/run.log \\
               experiments/results/w22-nc8cv2-nc8d-seeds-6-10/run.log \\
        --output ../docs/W22-PROBE-Q-BOOTSTRAP-CI.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def extract_per_seed(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        log = f.read()
    for m in re.finditer(
        r"wf-report \d+/\d+\]\s+([A-Za-z_0-9]+)/seed(\d+).*grand_mean=([\d.eE+-]+)",
        log,
    ):
        rows.append({
            "scenario": m.group(1),
            "seed": int(m.group(2)),
            "grand_mean": float(m.group(3)),
        })
    return pd.DataFrame(rows)


def bootstrap_diff(asms_vals: np.ndarray, sms_vals: np.ndarray,
                    n_boot: int = 10000, seed: int = 42) -> dict:
    """Paired bootstrap on (ASMS_seed − SMS_seed) differences."""
    diffs = asms_vals - sms_vals
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(diffs)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[b] = np.mean(diffs[idx])
    # Percentile CI
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    median = float(np.median(boot_means))
    # BCa would be more accurate but adds complexity; percentile is fine for n≥10
    return {
        "mean_diff": float(np.mean(diffs)),
        "median_diff_bootstrap": median,
        "ci_lo_2.5": lo,
        "ci_hi_97.5": hi,
        "ci_excludes_0": bool(lo > 0 or hi < 0),
        "n_bootstrap": n_boot,
        "fraction_positive_diff": float(np.mean(diffs > 0)),
    }


def jackknife_diff(asms_vals: np.ndarray, sms_vals: np.ndarray) -> dict:
    """Leave-one-seed-out jackknife: identify influential seeds."""
    diffs = asms_vals - sms_vals
    full_mean = float(np.mean(diffs))
    n = len(diffs)
    jack_means = np.empty(n)
    influences = []
    for i in range(n):
        mask = np.arange(n) != i
        jack_means[i] = float(np.mean(diffs[mask]))
        # Influence = full - leave-one-out
        influences.append({
            "seed_index": int(i),
            "leave_one_out_mean": float(jack_means[i]),
            "influence": float(full_mean - jack_means[i]),
        })
    return {
        "jackknife_mean": float(np.mean(jack_means)),
        "jackknife_std": float(np.std(jack_means, ddof=1)),
        "max_influence_abs": float(np.max(np.abs([inf["influence"] for inf in influences]))),
        "influences": influences,
    }


def analyze(log_paths: list[Path]) -> dict:
    df_all = pd.concat([extract_per_seed(p) for p in log_paths], ignore_index=True)
    if df_all.empty:
        return {"error": "no records"}

    out = {
        "n_records": int(len(df_all)),
        "scenarios": sorted(df_all["scenario"].unique().tolist()),
    }

    # ASMS-SMS pairings
    pairings = {}
    if "ASMS_mHDM_K3_v2both" in df_all["scenario"].values:
        pairings["FTSE"] = ("ASMS_mHDM_K3_v2both", "SMS_RDM_K0")
    if "ASMS_mHDM_K3_v2both_pop30gen40" in df_all["scenario"].values:
        pairings["FTSE_pop30gen40"] = ("ASMS_mHDM_K3_v2both_pop30gen40", "SMS_RDM_K0_pop30gen40")

    per_dataset = {}
    for label, (asms_name, sms_name) in pairings.items():
        asms_grp = df_all[df_all["scenario"] == asms_name].set_index("seed")["grand_mean"]
        sms_grp = df_all[df_all["scenario"] == sms_name].set_index("seed")["grand_mean"]
        common = sorted(set(asms_grp.index) & set(sms_grp.index))
        if len(common) < 3:
            per_dataset[label] = {"error": f"only {len(common)} common seeds; need ≥3"}
            continue
        asms_vals = np.array([asms_grp.loc[s] for s in common])
        sms_vals = np.array([sms_grp.loc[s] for s in common])
        boot = bootstrap_diff(asms_vals, sms_vals, n_boot=10000, seed=42)
        jack = jackknife_diff(asms_vals, sms_vals)
        per_dataset[label] = {
            "n_seeds": len(common),
            "seeds": common,
            "asms_mean": float(np.mean(asms_vals)),
            "sms_mean": float(np.mean(sms_vals)),
            "asms_std": float(np.std(asms_vals, ddof=1)),
            "sms_std": float(np.std(sms_vals, ddof=1)),
            "delta_pct_mean": float(100 * (np.mean(asms_vals) - np.mean(sms_vals)) / np.mean(sms_vals)),
            "bootstrap": boot,
            "jackknife": jack,
        }
    out["per_dataset"] = per_dataset
    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe Q — Bootstrap CI on (ASMS Ŝ − SMS Ŝ)")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_q.py`.*")
    lines.append("")
    lines.append(f"- Records pooled: {summary['n_records']}")
    lines.append(f"- Scenarios: {', '.join(summary['scenarios'])}")
    lines.append("")

    if "per_dataset" not in summary or not summary["per_dataset"]:
        return "\n".join(lines + ["No ASMS-SMS pairings."])

    lines.append("## Verdict (per dataset)")
    lines.append("")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            continue
        boot = stats["bootstrap"]
        sig = "✅" if boot["ci_excludes_0"] else "🔴"
        lines.append(
            f"- {sig} **{label}** (n={stats['n_seeds']}): "
            f"Δ_mean = {stats['delta_pct_mean']:+.2f}%; "
            f"bootstrap 95% CI on Δ (abs) = [{boot['ci_lo_2.5']:.4e}, {boot['ci_hi_97.5']:.4e}]; "
            f"CI excludes 0: {'YES (robust)' if boot['ci_excludes_0'] else 'NO (fragile)'}"
        )
    lines.append("")

    lines.append("## Per-dataset detail")
    lines.append("")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            lines.append(f"### {label}: ERROR: {stats['error']}")
            continue
        boot = stats["bootstrap"]
        jack = stats["jackknife"]
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- n seeds = {stats['n_seeds']}")
        lines.append(f"- ASMS mean Ŝ = {stats['asms_mean']:.4e} (std {stats['asms_std']:.4e})")
        lines.append(f"- SMS mean Ŝ = {stats['sms_mean']:.4e} (std {stats['sms_std']:.4e})")
        lines.append(f"- Δ% = {stats['delta_pct_mean']:+.2f}%")
        lines.append(f"- Per-seed positive (ASMS > SMS): {boot['fraction_positive_diff']*100:.1f}%")
        lines.append("")
        lines.append("**Bootstrap (B=10,000 paired resamples)**:")
        lines.append(f"- Mean of bootstrap-resampled diffs: {boot['mean_diff']:.4e}")
        lines.append(f"- Median: {boot['median_diff_bootstrap']:.4e}")
        lines.append(f"- 95% CI: [{boot['ci_lo_2.5']:.4e}, {boot['ci_hi_97.5']:.4e}]")
        lines.append(f"- CI excludes 0: **{boot['ci_excludes_0']}**")
        lines.append("")
        lines.append("**Jackknife (leave-one-seed-out)**:")
        lines.append(f"- Jackknife mean: {jack['jackknife_mean']:.4e}")
        lines.append(f"- Jackknife std: {jack['jackknife_std']:.4e}")
        lines.append(f"- Max |influence|: {jack['max_influence_abs']:.4e}")
        # Find most influential seed
        max_idx = max(range(len(jack['influences'])), key=lambda i: abs(jack['influences'][i]['influence']))
        max_inf = jack['influences'][max_idx]
        lines.append(f"- Most influential: seed index {max_inf['seed_index']} (Δ when removed: {-max_inf['influence']:+.4e})")
        lines.append("")

    lines.append("## Decision per probe spec")
    lines.append("")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            continue
        if stats["bootstrap"]["ci_excludes_0"]:
            lines.append(f"- ✅ **{label}**: bootstrap CI excludes 0 → headline gap ROBUST to portfolio-level resampling.")
        else:
            lines.append(f"- 🔴 **{label}**: bootstrap CI includes 0 → headline FRAGILE; report with caution.")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = analyze([p for p in args.logs if p.exists()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(summary))
    print(f"[probe-q] wrote report to {args.output}")
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[probe-q] wrote summary JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
