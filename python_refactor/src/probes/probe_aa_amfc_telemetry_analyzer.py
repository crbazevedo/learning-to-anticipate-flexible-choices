"""W22 Probe AA — AMFC-vs-Hv-DM telemetry analysis harness.

Standalone analysis module that consumes the telemetry produced by
``src.algorithms.amfc_selector.select_amfc(..., collect_telemetry=True)``
and renders reportable comparisons between the AMFC and Hv-DM pickers.

This module makes NO modifications to shared code paths. It only reads
the per-period telemetry records flushed via ``get_amfc_telemetry()``
and produces summary statistics, markdown reports, and JSON IO.

Per-record schema (from amfc_selector.py):
    amfc_idx                          int
    hv_dm_idx                         int
    amfc_agrees_with_hv_dm            bool
    amfc_pick_roi                     float
    amfc_pick_risk                    float
    hv_dm_pick_roi                    float
    hv_dm_pick_risk                   float
    n_candidates                      int
    horizon                           int
    n_mc                              int
    top1_expected_contrib             float
    mean_forecast_variance            float
    amfc_pick_forecast_variance       float
    tie_break_fired                   bool
    R1                                float
    R2                                float

Hypothesis (see docs/W22-PROBE-AA-AMFC-TELEMETRY.md):
    On real walk-forward data, AMFC disagrees with Hv-DM on ≥30% of
    periods AND consistently picks lower-variance forecast candidates
    than the average candidate (forecast-uncertainty discrimination).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

# --- canonical summary keys (closed-enum; new keys require a paired-diff
#     update to the markdown renderer + comparison renderer below) ---------
SUMMARY_KEYS = (
    "n_periods",
    "agreement_rate",
    "mean_amfc_pick_roi",
    "mean_hv_dm_pick_roi",
    "roi_delta_mean",
    "roi_delta_std",
    "tie_break_fire_rate",
    "mean_forecast_variance",
    "mean_amfc_pick_forecast_variance",
    "amfc_picks_more_certain_than_average",
)


def _empty_summary() -> dict[str, Any]:
    """Return a deterministic empty-input summary.

    The boolean ``amfc_picks_more_certain_than_average`` defaults to
    False on empty input — there is no AMFC pick to compare.
    """
    return {
        "n_periods": 0,
        "agreement_rate": 0.0,
        "mean_amfc_pick_roi": 0.0,
        "mean_hv_dm_pick_roi": 0.0,
        "roi_delta_mean": 0.0,
        "roi_delta_std": 0.0,
        "tie_break_fire_rate": 0.0,
        "mean_forecast_variance": 0.0,
        "mean_amfc_pick_forecast_variance": 0.0,
        "amfc_picks_more_certain_than_average": False,
    }


def summarize_telemetry(telemetry: Iterable[dict]) -> dict[str, Any]:
    """Compute summary statistics from a list of AMFC telemetry records.

    Args:
        telemetry: iterable of per-period telemetry dicts as returned by
            ``amfc_selector.get_amfc_telemetry()``. May be empty.

    Returns:
        Dict with the canonical SUMMARY_KEYS:
            n_periods                              count of records
            agreement_rate                         frac where amfc==hv_dm
            mean_amfc_pick_roi                     mean of amfc_pick_roi
            mean_hv_dm_pick_roi                    mean of hv_dm_pick_roi
            roi_delta_mean                         mean(amfc-hv_dm)
            roi_delta_std                          stdev(amfc-hv_dm), 0 if n<2
            tie_break_fire_rate                    frac where tie_break_fired
            mean_forecast_variance                 mean of mean_forecast_variance
            mean_amfc_pick_forecast_variance       mean of amfc_pick_forecast_variance
            amfc_picks_more_certain_than_average   bool: AMFC pick variance < pop mean
    """
    records = list(telemetry)
    n = len(records)
    if n == 0:
        return _empty_summary()

    agreements = sum(1 for r in records if r.get("amfc_agrees_with_hv_dm"))
    tie_breaks = sum(1 for r in records if r.get("tie_break_fired"))
    amfc_rois = [float(r.get("amfc_pick_roi", 0.0)) for r in records]
    hv_dm_rois = [float(r.get("hv_dm_pick_roi", 0.0)) for r in records]
    deltas = [a - h for a, h in zip(amfc_rois, hv_dm_rois)]
    pop_vars = [float(r.get("mean_forecast_variance", 0.0)) for r in records]
    amfc_pick_vars = [float(r.get("amfc_pick_forecast_variance", 0.0)) for r in records]

    mean_amfc_roi = statistics.fmean(amfc_rois)
    mean_hv_dm_roi = statistics.fmean(hv_dm_rois)
    delta_mean = statistics.fmean(deltas)
    delta_std = statistics.pstdev(deltas) if n >= 2 else 0.0
    mean_pop_var = statistics.fmean(pop_vars)
    mean_amfc_pick_var = statistics.fmean(amfc_pick_vars)

    return {
        "n_periods": n,
        "agreement_rate": agreements / n,
        "mean_amfc_pick_roi": mean_amfc_roi,
        "mean_hv_dm_pick_roi": mean_hv_dm_roi,
        "roi_delta_mean": delta_mean,
        "roi_delta_std": delta_std,
        "tie_break_fire_rate": tie_breaks / n,
        "mean_forecast_variance": mean_pop_var,
        "mean_amfc_pick_forecast_variance": mean_amfc_pick_var,
        "amfc_picks_more_certain_than_average": mean_amfc_pick_var < mean_pop_var,
    }


def format_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a summary dict as a markdown table.

    Args:
        summary: dict produced by ``summarize_telemetry``.

    Returns:
        Markdown-formatted string suitable for embedding in a probe report.
    """
    n = int(summary.get("n_periods", 0))
    if n == 0:
        return (
            "## AMFC-vs-Hv-DM Telemetry Summary\n\n"
            "_No telemetry records — empty input._\n"
        )

    rows = [
        ("n_periods", f"{n}"),
        ("agreement_rate", f"{summary['agreement_rate']:.4f}"),
        ("mean_amfc_pick_roi", f"{summary['mean_amfc_pick_roi']:.6f}"),
        ("mean_hv_dm_pick_roi", f"{summary['mean_hv_dm_pick_roi']:.6f}"),
        ("roi_delta_mean (amfc − hv_dm)", f"{summary['roi_delta_mean']:+.6f}"),
        ("roi_delta_std", f"{summary['roi_delta_std']:.6f}"),
        ("tie_break_fire_rate", f"{summary['tie_break_fire_rate']:.4f}"),
        ("mean_forecast_variance (pop)", f"{summary['mean_forecast_variance']:.6e}"),
        (
            "mean_amfc_pick_forecast_variance",
            f"{summary['mean_amfc_pick_forecast_variance']:.6e}",
        ),
        (
            "amfc_picks_more_certain_than_average",
            "YES" if summary["amfc_picks_more_certain_than_average"] else "NO",
        ),
    ]
    lines = [
        "## AMFC-vs-Hv-DM Telemetry Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return "\n".join(lines)


def load_telemetry_from_json(path: str | Path) -> list[dict]:
    """Load a list of telemetry records from a JSON file.

    Args:
        path: filesystem path to a JSON file containing a list of records.

    Returns:
        List of dicts.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"expected a JSON list at {path}, got {type(data).__name__}"
        )
    return data


def save_telemetry_to_json(telemetry: Iterable[dict], path: str | Path) -> None:
    """Persist telemetry records to a JSON file.

    Args:
        telemetry: iterable of per-period dicts.
        path: filesystem path to write.
    """
    records = list(telemetry)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def compare_telemetry_summaries(
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> str:
    """Side-by-side comparison markdown for two summaries.

    Args:
        summary_a, summary_b: summary dicts (each from ``summarize_telemetry``).
        label_a, label_b: short labels for the two columns.

    Returns:
        Markdown string with a 3-column table (Metric | A | B | Δ when numeric).
    """
    numeric_keys = (
        "n_periods",
        "agreement_rate",
        "mean_amfc_pick_roi",
        "mean_hv_dm_pick_roi",
        "roi_delta_mean",
        "roi_delta_std",
        "tie_break_fire_rate",
        "mean_forecast_variance",
        "mean_amfc_pick_forecast_variance",
    )
    bool_keys = ("amfc_picks_more_certain_than_average",)

    lines = [
        f"## Comparison: {label_a} vs {label_b}",
        "",
        f"| Metric | {label_a} | {label_b} | Δ (B − A) |",
        "|---|---|---|---|",
    ]
    for k in numeric_keys:
        a = float(summary_a.get(k, 0.0))
        b = float(summary_b.get(k, 0.0))
        delta = b - a
        if k == "n_periods":
            lines.append(f"| {k} | {int(a)} | {int(b)} | {int(delta):+d} |")
        elif "rate" in k:
            lines.append(f"| {k} | {a:.4f} | {b:.4f} | {delta:+.4f} |")
        elif "variance" in k:
            lines.append(f"| {k} | {a:.6e} | {b:.6e} | {delta:+.6e} |")
        else:
            lines.append(f"| {k} | {a:.6f} | {b:.6f} | {delta:+.6f} |")
    for k in bool_keys:
        a_v = "YES" if summary_a.get(k) else "NO"
        b_v = "YES" if summary_b.get(k) else "NO"
        same = "same" if summary_a.get(k) == summary_b.get(k) else "DIFFERS"
        lines.append(f"| {k} | {a_v} | {b_v} | {same} |")
    lines.append("")
    return "\n".join(lines)
