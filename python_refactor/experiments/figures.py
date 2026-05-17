"""
W7-1 figure generation.

Generates per-run PNG figures from a single metrics.csv (long format,
output by validation_matrix.py) OR from an aggregate summary CSV
(output by aggregate_validation.py).

CLI:
    # Per-run figures (one Pareto-evolution PNG, one wealth-trajectory PNG, …)
    python -m experiments.figures \\
        --input results/S2_paper_seed42/metrics.csv \\
        --output results/S2_paper_seed42/figures/

    # Comparison figures (mean ± std across scenarios from summary CSV)
    python -m experiments.figures \\
        --input docs/VALIDATION-SUMMARY.csv \\
        --output docs/figures/ \\
        --mode comparison

Outputs (per run mode):
    wealth_trajectory.png
    lambda_trajectory.png         (when present in metrics)
    belief_trajectory.png         (when present)
    pareto_frontier_evolution.png (when frontier snapshots are recorded)

Outputs (comparison mode):
    wealth_by_scenario.png
    hypv_by_scenario.png
    pocid_by_scenario.png
    forest_paper_window.png

Headless matplotlib (Agg backend) — runs on CI without a display server.

Honest scar: this scaffold targets the metric schema that W8's runs will
produce. The exact column names + presence (e.g. lambda trajectory)
depend on what ExperimentManager.final_metrics surfaces. The plot
functions degrade gracefully (skip with a notice) when an expected
metric is absent.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# Per-run figures (from a single metrics.csv long-format file)
# --------------------------------------------------------------------------

def read_long_metrics(path: Path) -> dict[str, list[float]]:
    """Read long-format metrics CSV → {metric: [values]}.

    Long format (per validation_matrix.py): scenario, window, seed, metric, value.
    """
    out: dict[str, list[float]] = defaultdict(list)
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["metric"]].append(float(row["value"]))
            except (TypeError, ValueError, KeyError):
                continue
    return out


def plot_wealth_trajectory(metrics: dict[str, list[float]], output: Path) -> bool:
    """Plot wealth trajectory if 'wealth_trajectory' present.

    Returns True when written; False when skipped.
    """
    series = metrics.get("wealth_trajectory")
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(series, label="W_t")
    ax.set_xlabel("Investment period t")
    ax.set_ylabel("Accumulated wealth W_t")
    ax.set_title("Wealth trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


def plot_trajectory(metrics: dict[str, list[float]], key: str,
                     ylabel: str, title: str, output: Path) -> bool:
    series = metrics.get(key)
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(series)
    ax.set_xlabel("Investment period t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# Comparison figures (from aggregate summary CSV)
# --------------------------------------------------------------------------

def read_summary(path: Path) -> list[dict]:
    """Read the aggregate summary CSV (one row per scenario × window × metric)."""
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_metric_by_scenario(rows: list[dict], metric: str, window: str,
                              output: Path, ylabel: str | None = None) -> bool:
    """Bar chart: mean ± std for the given metric across scenarios in one window."""
    relevant = [r for r in rows if r["metric"] == metric and r["window"] == window]
    if not relevant:
        return False
    scenarios = [r["scenario"] for r in relevant]
    means = [float(r["mean"]) for r in relevant]
    stds = [float(r["std"]) for r in relevant]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(scenarios, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(f"{metric} by scenario (window={window}; mean ± std)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="W7-1 figure generation — per-run or comparison PNGs.",
    )
    parser.add_argument("--input", type=Path, required=True,
                         help="metrics.csv (per-run mode) OR summary CSV (comparison mode).")
    parser.add_argument("--output", type=Path, required=True,
                         help="Output directory for PNGs.")
    parser.add_argument("--mode", choices=["per-run", "comparison"], default="per-run")
    parser.add_argument("--window", default="paper",
                         help="Window to filter on (comparison mode only).")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "per-run":
        metrics = read_long_metrics(args.input)
        wrote = 0
        if plot_wealth_trajectory(metrics, args.output / "wealth_trajectory.png"):
            wrote += 1
        if plot_trajectory(metrics, "lambda_trajectory", "λ_{t+h}",
                            "Anticipatory learning rate (paper Eq 13)",
                            args.output / "lambda_trajectory.png"):
            wrote += 1
        if plot_trajectory(metrics, "belief_trajectory", "v_{t+1}",
                            "Belief coefficient (paper Eq 20)",
                            args.output / "belief_trajectory.png"):
            wrote += 1
        print(f"[figures/per-run] wrote {wrote} PNGs to {args.output}")
        return 0 if wrote > 0 else 1

    # comparison mode
    rows = read_summary(args.input)
    if not rows:
        print(f"[error] no rows in {args.input}", file=sys.stderr)
        return 2
    wrote = 0
    for metric, ylabel in (("final_wealth", "Final wealth W_T"),
                            ("hypv", "Expected Hypv"),
                            ("pocid", "POCID")):
        if plot_metric_by_scenario(rows, metric, args.window,
                                     args.output / f"{metric}_by_scenario.png",
                                     ylabel=ylabel):
            wrote += 1
    print(f"[figures/comparison] wrote {wrote} PNGs to {args.output}")
    return 0 if wrote > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
