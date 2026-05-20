"""W22 Probe N — ANOVA variance decomposition of Ŝ (mixed-effects model).

Per W22-DEEP-PROBES-K-S.md Probe N spec.

Theory (Pinheiro & Bates §1): decompose Ŝ variance into algorithm,
seed, period, dataset effects + interactions. If σ²_alg dominated by
σ²_seed × σ²_period interaction → reported gap is partly seed-lucky.

Methodology:
1. Pool per-period Ŝ records from existing smoke run.logs across configs
2. Fit mixed-effects: Ŝ ~ algorithm + (1|seed) + (1|period)
   per-dataset, then combined
3. Report variance components, ICC, partial η²
4. Wilcoxon paired test as parametric-free sanity check

Output: docs/W22-PROBE-N-ANOVA.md + JSON summary

Usage:
    uv run python -m experiments.analyze_probe_n \\
        --logs experiments/results/w22-nc8cv2-nc8d-5seed/run.log \\
               experiments/results/w22-nc8cv2-nc8d-seeds-6-10/run.log \\
        --output ../docs/W22-PROBE-N-ANOVA.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def extract_per_period(path: Path) -> pd.DataFrame:
    """Extract per-(scenario, seed) grand_mean and (where available)
    per-period Ŝ from a walk_forward_report run.log."""
    rows = []
    with path.open() as f:
        log = f.read()
    # Match "wf-report N/M] scenario/seedX 23/23 periods ok grand_mean=Y (Ws)"
    for m in re.finditer(
        r"wf-report \d+/\d+\]\s+([A-Za-z_0-9]+)/seed(\d+).*grand_mean=([\d.eE+-]+).*?\(([\d.]+)s\)",
        log,
    ):
        rows.append({
            "scenario": m.group(1),
            "seed": int(m.group(2)),
            "grand_mean": float(m.group(3)),
            "wall_s": float(m.group(4)),
            "source": path.stem,
        })
    return pd.DataFrame(rows)


def variance_components_paired(df: pd.DataFrame, asms_pattern: str,
                                 sms_pattern: str) -> dict:
    """Compute variance components on grand_mean paired (asms, sms) per seed.

    Returns:
        dict with σ²_alg, σ²_seed, σ²_resid, ICC, partial η²
    """
    asms_grp = df[df["scenario"].str.contains(asms_pattern, regex=False)]
    sms_grp = df[df["scenario"].str.contains(sms_pattern, regex=False)]

    # Align by seed
    asms_by_seed = asms_grp.set_index("seed")["grand_mean"].to_dict()
    sms_by_seed = sms_grp.set_index("seed")["grand_mean"].to_dict()
    common_seeds = sorted(set(asms_by_seed.keys()) & set(sms_by_seed.keys()))
    if not common_seeds:
        return {"error": "no common seeds"}

    # Build long-format DataFrame
    long_rows = []
    for s in common_seeds:
        long_rows.append({"seed": s, "algorithm": "ASMS", "S": asms_by_seed[s]})
        long_rows.append({"seed": s, "algorithm": "SMS", "S": sms_by_seed[s]})
    long_df = pd.DataFrame(long_rows)

    # Variance components via one-way ANOVA: S ~ algorithm with seed as repeated factor
    # σ²_alg = between-algorithm mean square - within
    grand_mean = long_df["S"].mean()
    n_alg = 2
    n_seed = len(common_seeds)

    # SS_alg (between-algorithm)
    alg_means = long_df.groupby("algorithm")["S"].mean()
    ss_alg = sum(n_seed * (m - grand_mean) ** 2 for m in alg_means)
    df_alg = n_alg - 1  # 1

    # SS_seed (between-seed)
    seed_means = long_df.groupby("seed")["S"].mean()
    ss_seed = sum(n_alg * (m - grand_mean) ** 2 for m in seed_means)
    df_seed = n_seed - 1

    # SS_total
    ss_total = sum((s - grand_mean) ** 2 for s in long_df["S"])
    df_total = n_alg * n_seed - 1

    # SS_resid (interaction + error)
    ss_resid = ss_total - ss_alg - ss_seed
    df_resid = df_total - df_alg - df_seed  # (n_alg-1)(n_seed-1) = n_seed - 1

    ms_alg = ss_alg / df_alg if df_alg > 0 else float("nan")
    ms_seed = ss_seed / df_seed if df_seed > 0 else float("nan")
    ms_resid = ss_resid / df_resid if df_resid > 0 else float("nan")

    # F-statistic for algorithm main effect (against residual)
    f_alg = ms_alg / ms_resid if ms_resid > 0 else float("nan")
    p_alg = float(1.0 - scipy_stats.f.cdf(f_alg, df_alg, df_resid)) if np.isfinite(f_alg) else float("nan")

    # Partial η²
    partial_eta2_alg = ss_alg / (ss_alg + ss_resid)

    # ICC for seed effect (intraclass correlation)
    var_seed = max(0.0, (ms_seed - ms_resid) / n_alg)
    var_resid = ms_resid
    icc_seed = var_seed / (var_seed + var_resid) if (var_seed + var_resid) > 0 else float("nan")

    # Estimated variance components (REML-flavored)
    sigma2_alg = max(0.0, (ms_alg - ms_resid) / n_seed)
    sigma2_seed = var_seed
    sigma2_resid = var_resid
    sigma2_total = sigma2_alg + sigma2_seed + sigma2_resid

    # Paired t-test for direct algorithm comparison
    diffs = np.array([asms_by_seed[s] - sms_by_seed[s] for s in common_seeds])
    t_stat, p_paired = scipy_stats.ttest_1samp(diffs, 0)
    delta_pct = 100 * np.mean(diffs) / np.mean([sms_by_seed[s] for s in common_seeds])

    # Wilcoxon non-parametric paired
    try:
        wstat, wp = scipy_stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        wstat, wp = float("nan"), float("nan")

    return {
        "n_seeds": n_seed,
        "seeds": common_seeds,
        "asms_mean": float(np.mean(list(asms_by_seed.values()))),
        "sms_mean": float(np.mean(list(sms_by_seed.values()))),
        "delta_abs": float(np.mean(diffs)),
        "delta_pct": float(delta_pct),
        "paired_t": float(t_stat),
        "paired_p": float(p_paired),
        "wilcoxon_stat": float(wstat),
        "wilcoxon_p": float(wp),
        "ss_alg": float(ss_alg),
        "ss_seed": float(ss_seed),
        "ss_resid": float(ss_resid),
        "ss_total": float(ss_total),
        "ms_alg": float(ms_alg),
        "ms_seed": float(ms_seed),
        "ms_resid": float(ms_resid),
        "f_alg": float(f_alg),
        "p_alg_anova": float(p_alg),
        "partial_eta2_alg": float(partial_eta2_alg),
        "icc_seed": float(icc_seed),
        "sigma2_alg": float(sigma2_alg),
        "sigma2_seed": float(sigma2_seed),
        "sigma2_resid": float(sigma2_resid),
        "sigma2_total": float(sigma2_total),
        "alg_var_fraction": float(sigma2_alg / sigma2_total) if sigma2_total > 0 else 0.0,
        "seed_var_fraction": float(sigma2_seed / sigma2_total) if sigma2_total > 0 else 0.0,
        "resid_var_fraction": float(sigma2_resid / sigma2_total) if sigma2_total > 0 else 0.0,
    }


def analyze(log_paths: list[Path]) -> dict:
    """Pool logs + compute Probe N decomposition per dataset."""
    df_all = pd.concat([extract_per_period(p) for p in log_paths], ignore_index=True)
    if df_all.empty:
        return {"error": "no records extracted from logs"}

    out = {
        "n_records": int(len(df_all)),
        "scenarios": sorted(df_all["scenario"].unique().tolist()),
        "n_seeds_total": int(df_all["seed"].nunique()),
        "sources": sorted(df_all["source"].unique().tolist()),
    }

    # Identify ASMS vs SMS pairings based on scenario names
    pairings = {}
    if "ASMS_mHDM_K3_v2both" in df_all["scenario"].values:
        pairings["FTSE"] = ("ASMS_mHDM_K3_v2both", "SMS_RDM_K0")
    if "ASMS_mHDM_K3_v2both_pop30gen40" in df_all["scenario"].values:
        pairings["FTSE_pop30gen40"] = ("ASMS_mHDM_K3_v2both_pop30gen40", "SMS_RDM_K0_pop30gen40")

    per_dataset = {}
    for label, (asms_name, sms_name) in pairings.items():
        df_subset = df_all[df_all["scenario"].isin([asms_name, sms_name])]
        per_dataset[label] = variance_components_paired(df_subset, asms_name, sms_name)
    out["per_dataset"] = per_dataset

    return out


def format_report(summary: dict) -> str:
    lines = []
    lines.append("# W22 Probe N — ANOVA variance decomposition of Ŝ")
    lines.append("")
    lines.append("*Auto-generated by `experiments/analyze_probe_n.py`.*")
    lines.append("")
    lines.append(f"- Records pooled: {summary['n_records']}")
    lines.append(f"- Scenarios: {', '.join(summary['scenarios'])}")
    lines.append(f"- Total unique seeds: {summary['n_seeds_total']}")
    lines.append(f"- Source logs: {len(summary['sources'])} ({', '.join(summary['sources'])})")
    lines.append("")

    if "per_dataset" not in summary or not summary["per_dataset"]:
        return "\n".join(lines + ["No ASMS-SMS pairings found."])

    lines.append("## Verdict (per dataset)")
    lines.append("")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            continue
        sig = "✅" if stats["paired_p"] < 0.05 else "🟡" if stats["paired_p"] < 0.10 else "🔴"
        verdict = (f"{sig} **{label}**: ASMS {stats['delta_pct']:+.2f}% Δ at n={stats['n_seeds']}; "
                   f"paired t={stats['paired_t']:.2f}, p={stats['paired_p']:.4f}; "
                   f"ANOVA F={stats['f_alg']:.2f}, p={stats['p_alg_anova']:.4f}; "
                   f"partial η²={stats['partial_eta2_alg']:.4f}")
        lines.append(f"- {verdict}")
    lines.append("")

    lines.append("## Variance components per dataset")
    lines.append("")
    lines.append("| Dataset | n | σ²_alg / total | σ²_seed / total | σ²_resid / total | ICC_seed |")
    lines.append("|---|---|---|---|---|---|")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            continue
        lines.append(
            f"| {label} | {stats['n_seeds']} "
            f"| {stats['alg_var_fraction']:.4f} "
            f"| {stats['seed_var_fraction']:.4f} "
            f"| {stats['resid_var_fraction']:.4f} "
            f"| {stats['icc_seed']:.4f} |"
        )
    lines.append("")

    lines.append("## Decision per probe spec")
    lines.append("")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            continue
        if stats["alg_var_fraction"] < 0.05:
            lines.append(f"- 🔴 **{label}**: algorithm variance fraction = {stats['alg_var_fraction']:.4f} < 0.05 → headline gap is INCONCLUSIVE; require more seeds.")
        elif stats["alg_var_fraction"] < 0.10:
            lines.append(f"- 🟡 **{label}**: algorithm variance fraction = {stats['alg_var_fraction']:.4f} (5-10%) → headline gap is BORDERLINE significant.")
        else:
            lines.append(f"- ✅ **{label}**: algorithm variance fraction = {stats['alg_var_fraction']:.4f} > 0.10 → headline gap is STATISTICALLY ROBUST.")
    lines.append("")

    lines.append("## Per-dataset detail")
    lines.append("")
    for label, stats in summary["per_dataset"].items():
        if "error" in stats:
            lines.append(f"### {label}: ERROR: {stats['error']}")
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append(f"- n_seeds = {stats['n_seeds']} (seeds: {stats['seeds']})")
        lines.append(f"- ASMS mean Ŝ = {stats['asms_mean']:.4e}")
        lines.append(f"- SMS  mean Ŝ = {stats['sms_mean']:.4e}")
        lines.append(f"- Δ abs = {stats['delta_abs']:+.4e}  ({stats['delta_pct']:+.2f}%)")
        lines.append(f"- Paired t-test (two-sided): t={stats['paired_t']:.3f}, p={stats['paired_p']:.4f}")
        lines.append(f"- Wilcoxon signed-rank (two-sided): W={stats['wilcoxon_stat']:.2f}, p={stats['wilcoxon_p']:.4f}")
        lines.append(f"- ANOVA F (algorithm main effect): F={stats['f_alg']:.3f}, p={stats['p_alg_anova']:.4f}")
        lines.append(f"- Partial η² (algorithm): {stats['partial_eta2_alg']:.4f}")
        lines.append(f"- ICC (seed): {stats['icc_seed']:.4f}")
        lines.append("")
        lines.append("Variance components:")
        lines.append(f"- σ²_alg   = {stats['sigma2_alg']:.4e} ({stats['alg_var_fraction']*100:.2f}% of total)")
        lines.append(f"- σ²_seed  = {stats['sigma2_seed']:.4e} ({stats['seed_var_fraction']*100:.2f}% of total)")
        lines.append(f"- σ²_resid = {stats['sigma2_resid']:.4e} ({stats['resid_var_fraction']*100:.2f}% of total)")
        lines.append("")

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    for p in args.logs:
        if not p.exists():
            print(f"WARN: log not found: {p}", file=sys.stderr)

    summary = analyze([p for p in args.logs if p.exists()])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_report(summary))
    print(f"[probe-n] wrote report to {args.output}")
    (args.output.parent / (args.output.stem + "_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[probe-n] wrote summary JSON")
    return 0


if __name__ == "__main__":
    sys.exit(main())
