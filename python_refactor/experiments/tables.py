"""
W8-3 table generators (A-H) from aggregated validation results.

Pure functions. Input shapes vary per table (summary CSV row-list vs
per-run metric dicts vs pairwise-test rows). Output: Markdown-formatted
string suitable for embedding in VALIDATION-RESULTS.md.

Table catalog (per ANALYTICS-PLAN.md §7):
    A: descriptive_stats   — scenario × window × metric → (mean, std, ...)
    B: pairwise_tests      — (scenario_i, scenario_j) × window × metric → (U, p, d, δ)
    C: anova               — window × metric × source → (SS, df, MS, F, p)
    D: synergy_contrast    — window × metric → (Δ_interaction, 95% CI)
    E: horizon_ablation    — comparison × window × metric → (U, p, d)
    F: regime_segmentation — regime × scenario × metric → mean ± std
    G: sub_window_stability — sub_window × scenario × metric → mean ± std
    H: paper_comparison    — metric → (paper value, this impl, diff, agree?)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def fmt_sigfig(value: float, sig: int = 4) -> str:
    """Format `value` to `sig` significant figures. NaN-safe."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:  # NaN
        return "nan"
    if v == 0:
        return "0"
    import math
    digits = sig - int(math.floor(math.log10(abs(v)))) - 1
    return f"{v:.{max(0, digits)}f}"


def fmt_pvalue(p: float) -> str:
    """Render p-value: '<0.001' for very small, else 3 sig figs."""
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return str(p)
    if pv != pv:
        return "nan"
    if pv < 0.001:
        return "<0.001"
    return fmt_sigfig(pv, sig=3)


def fmt_effect(d: float) -> str:
    """Render effect size to 3 decimals (Cohen's d / Cliff's δ scale)."""
    try:
        return f"{float(d):.3f}"
    except (TypeError, ValueError):
        return str(d)


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_table_a(summary_rows: list[dict[str, Any]]) -> str:
    """Table A — descriptive statistics.

    Input: rows from aggregate_validation.py summary CSV with columns
    scenario, window, metric, mean, std, median, min, max, n_seeds.
    """
    if not summary_rows:
        return "_(Table A: no data)_"
    header = "| scenario | window | metric | mean | std | median | min | max | n |\n"
    sep    = "|---|---|---|---|---|---|---|---|---|\n"
    body_rows = []
    for r in sorted(summary_rows, key=lambda x: (x.get("scenario", ""),
                                                    x.get("window", ""),
                                                    x.get("metric", ""))):
        body_rows.append(
            "| {scenario} | {window} | {metric} | {mean} | {std} | {median} | {minv} | {maxv} | {n} |".format(
                scenario=r.get("scenario", ""),
                window=r.get("window", ""),
                metric=r.get("metric", ""),
                mean=fmt_sigfig(r.get("mean", float("nan"))),
                std=fmt_sigfig(r.get("std", float("nan"))),
                median=fmt_sigfig(r.get("median", float("nan"))),
                minv=fmt_sigfig(r.get("min", float("nan"))),
                maxv=fmt_sigfig(r.get("max", float("nan"))),
                n=r.get("n_seeds", "?"),
            )
        )
    return header + sep + "\n".join(body_rows)


def generate_table_b(pairwise_rows: list[dict[str, Any]]) -> str:
    """Table B — pairwise scenario comparison.

    Input rows: pair, window, metric, U, p_raw, p_holm, cohens_d, cliffs_delta, cles.
    """
    if not pairwise_rows:
        return "_(Table B: no data)_"
    header = "| pair | window | metric | U | p (raw) | p (Holm) | Cohen's d | Cliff's δ | CLES |\n"
    sep    = "|---|---|---|---|---|---|---|---|---|\n"
    body_rows = []
    for r in pairwise_rows:
        body_rows.append(
            "| {pair} | {window} | {metric} | {U} | {p_raw} | {p_holm} | {d} | {delta} | {cles} |".format(
                pair=r.get("pair", ""),
                window=r.get("window", ""),
                metric=r.get("metric", ""),
                U=fmt_sigfig(r.get("U", float("nan"))),
                p_raw=fmt_pvalue(r.get("p_raw", float("nan"))),
                p_holm=fmt_pvalue(r.get("p_holm", float("nan"))),
                d=fmt_effect(r.get("cohens_d", float("nan"))),
                delta=fmt_effect(r.get("cliffs_delta", float("nan"))),
                cles=fmt_effect(r.get("cles", float("nan"))),
            )
        )
    return header + sep + "\n".join(body_rows)


def generate_table_c(anova_rows: list[dict[str, Any]]) -> str:
    """Table C — one-way ANOVA over Multi-Horizon levels.

    Input rows: window, metric, source, SS, df, MS, F, p.
    """
    if not anova_rows:
        return "_(Table C: no data)_"
    header = "| window | metric | source | SS | df | MS | F | p |\n"
    sep    = "|---|---|---|---|---|---|---|---|\n"
    body_rows = []
    for r in anova_rows:
        body_rows.append(
            "| {window} | {metric} | {source} | {SS} | {df} | {MS} | {F} | {p} |".format(
                window=r.get("window", ""),
                metric=r.get("metric", ""),
                source=r.get("source", ""),
                SS=fmt_sigfig(r.get("SS", float("nan"))),
                df=r.get("df", "?"),
                MS=fmt_sigfig(r.get("MS", float("nan"))),
                F=fmt_sigfig(r.get("F", float("nan"))),
                p=fmt_pvalue(r.get("p", float("nan"))),
            )
        )
    return header + sep + "\n".join(body_rows)


def generate_table_d(synergy_rows: list[dict[str, Any]]) -> str:
    """Table D — synergy contrast Δ_interaction with bootstrap 95% CI.

    Input rows: window, metric, delta_interaction, ci_lo, ci_hi, interpretation.
    """
    if not synergy_rows:
        return "_(Table D: no data)_"
    header = "| window | metric | Δ_interaction | 95% CI | interpretation |\n"
    sep    = "|---|---|---|---|---|\n"
    body_rows = []
    for r in synergy_rows:
        ci = "[{lo}, {hi}]".format(
            lo=fmt_sigfig(r.get("ci_lo", float("nan"))),
            hi=fmt_sigfig(r.get("ci_hi", float("nan"))),
        )
        body_rows.append(
            "| {window} | {metric} | {delta} | {ci} | {interp} |".format(
                window=r.get("window", ""),
                metric=r.get("metric", ""),
                delta=fmt_sigfig(r.get("delta_interaction", float("nan"))),
                ci=ci,
                interp=r.get("interpretation", ""),
            )
        )
    return header + sep + "\n".join(body_rows)


def generate_table_e(ablation_rows: list[dict[str, Any]]) -> str:
    """Table E — per-horizon ablation.

    Input rows: comparison, window, metric, U, p, cohens_d, interpretation.
    """
    if not ablation_rows:
        return "_(Table E: no data)_"
    header = "| comparison | window | metric | U | p | Cohen's d | interpretation |\n"
    sep    = "|---|---|---|---|---|---|---|\n"
    body_rows = []
    for r in ablation_rows:
        body_rows.append(
            "| {cmp} | {window} | {metric} | {U} | {p} | {d} | {interp} |".format(
                cmp=r.get("comparison", ""),
                window=r.get("window", ""),
                metric=r.get("metric", ""),
                U=fmt_sigfig(r.get("U", float("nan"))),
                p=fmt_pvalue(r.get("p", float("nan"))),
                d=fmt_effect(r.get("cohens_d", float("nan"))),
                interp=r.get("interpretation", ""),
            )
        )
    return header + sep + "\n".join(body_rows)


def generate_table_f(regime_rows: list[dict[str, Any]]) -> str:
    """Table F — volatility regime segmentation.

    Input rows: regime, scenario, metric, mean, std.
    """
    if not regime_rows:
        return "_(Table F: no data)_"
    # Group metrics into columns for compact rendering.
    metrics_seen: list[str] = []
    for r in regime_rows:
        m = r.get("metric")
        if m and m not in metrics_seen:
            metrics_seen.append(m)
    metric_cols = " | ".join(f"{m} (mean ± std)" for m in metrics_seen)
    header = f"| regime | scenario | {metric_cols} |\n"
    sep_cells = ["---"] * (2 + len(metrics_seen))
    sep = "| " + " | ".join(sep_cells) + " |\n"

    # Build per-(regime, scenario) row.
    grouped: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for r in regime_rows:
        key = (r.get("regime", ""), r.get("scenario", ""))
        grouped.setdefault(key, {})[r.get("metric", "")] = (
            r.get("mean", float("nan")), r.get("std", float("nan")),
        )

    body_rows = []
    for (regime, scenario), metric_map in sorted(grouped.items()):
        cells = []
        for m in metrics_seen:
            mean, std = metric_map.get(m, (float("nan"), float("nan")))
            cells.append(f"{fmt_sigfig(mean)} ± {fmt_sigfig(std)}")
        body_rows.append(f"| {regime} | {scenario} | " + " | ".join(cells) + " |")
    return header + sep + "\n".join(body_rows)


def generate_table_g(sub_window_rows: list[dict[str, Any]]) -> str:
    """Table G — sub-window stability.

    Input rows: sub_window, scenario, metric, mean, std.
    """
    if not sub_window_rows:
        return "_(Table G: no data)_"
    header = "| sub-window | scenario | metric | mean | std |\n"
    sep    = "|---|---|---|---|---|\n"
    body_rows = []
    for r in sub_window_rows:
        body_rows.append(
            "| {sw} | {scenario} | {metric} | {mean} | {std} |".format(
                sw=r.get("sub_window", ""),
                scenario=r.get("scenario", ""),
                metric=r.get("metric", ""),
                mean=fmt_sigfig(r.get("mean", float("nan"))),
                std=fmt_sigfig(r.get("std", float("nan"))),
            )
        )
    return header + sep + "\n".join(body_rows)


def generate_table_h(paper_rows: list[dict[str, Any]]) -> str:
    """Table H — paper comparison.

    Input rows: metric, paper_value, our_value, abs_diff, rel_diff, agree.
    """
    if not paper_rows:
        return "_(Table H: no data)_"
    header = "| metric | paper value | this impl (S2, paper-window) | abs diff | rel diff | agree? |\n"
    sep    = "|---|---|---|---|---|---|\n"
    body_rows = []
    for r in paper_rows:
        body_rows.append(
            "| {metric} | {paper} | {ours} | {abs_d} | {rel_d} | {agree} |".format(
                metric=r.get("metric", ""),
                paper=fmt_sigfig(r.get("paper_value", float("nan"))),
                ours=fmt_sigfig(r.get("our_value", float("nan"))),
                abs_d=fmt_sigfig(r.get("abs_diff", float("nan"))),
                rel_d=fmt_sigfig(r.get("rel_diff", float("nan"))),
                agree=r.get("agree", "?"),
            )
        )
    return header + sep + "\n".join(body_rows)
