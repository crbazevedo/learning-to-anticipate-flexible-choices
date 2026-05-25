"""W22 Probe U — NC30 c variance_penalty α sensitivity analyzer.

Hypothesis: optimal α (variance penalty coefficient) varies across regimes;
sweep α ∈ {0.0, 0.1, 1.0, 10.0, 100.0} to characterize the trade-off
between forecast-expected-contribution and forecast-variance discount.

Per docs/W22-MASTER-BACKLOG.md Section H. Standalone analyzer; does not
require ASMS run — operates on synthetic AMFC scenarios to quantify how
α changes the AMFC argmax.

NO modifications to shared code paths (sms_emoa.py, amfc_selector.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def compute_effective_contribution(expected_contributions: np.ndarray,
                                     variance_traces: np.ndarray,
                                     alpha: float) -> np.ndarray:
    """Per NC30 c: effective_contribution = expected - α · trace(Σ).

    Args:
        expected_contributions: per-candidate E[Δ_S]
        variance_traces: per-candidate trace(Σ_h)
        alpha: variance penalty coefficient (α ≥ 0)

    Returns:
        Effective per-candidate score after variance penalty
    """
    return expected_contributions - alpha * variance_traces


def argmax_under_alpha(expected_contributions: np.ndarray,
                        variance_traces: np.ndarray,
                        alpha: float) -> int:
    """Returns the candidate index that wins under variance_penalty=α."""
    eff = compute_effective_contribution(expected_contributions, variance_traces, alpha)
    return int(np.argmax(eff))


def sweep_alpha(expected_contributions: np.ndarray,
                  variance_traces: np.ndarray,
                  alphas: list[float]) -> dict[float, int]:
    """Maps each α to the resulting argmax index.

    Args:
        expected_contributions: shape (n,) per-candidate E[Δ_S]
        variance_traces: shape (n,) per-candidate trace(Σ_h)
        alphas: list of α values to sweep

    Returns:
        Dict {α: argmax_index}
    """
    return {a: argmax_under_alpha(expected_contributions, variance_traces, a)
            for a in alphas}


def find_flip_points(expected_contributions: np.ndarray,
                       variance_traces: np.ndarray,
                       alpha_grid: Optional[np.ndarray] = None) -> list[tuple[float, int, int]]:
    """Find α values where the argmax changes.

    Returns:
        List of (α, prev_idx, new_idx) tuples for each flip detected.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-3, 3, 200)
    flips = []
    prev_idx = argmax_under_alpha(expected_contributions, variance_traces, alpha_grid[0])
    for a in alpha_grid[1:]:
        new_idx = argmax_under_alpha(expected_contributions, variance_traces, a)
        if new_idx != prev_idx:
            flips.append((float(a), prev_idx, new_idx))
            prev_idx = new_idx
    return flips


def analyze_alpha_tradeoff(expected_contributions: np.ndarray,
                             variance_traces: np.ndarray,
                             alphas: list[float] | None = None) -> str:
    """Generate a markdown report of α sensitivity.

    Args:
        expected_contributions: per-candidate E[Δ_S]
        variance_traces: per-candidate trace(Σ_h)
        alphas: α values to report. If None, uses [0, 0.1, 1, 10, 100].

    Returns:
        Markdown-formatted report string.
    """
    if alphas is None:
        alphas = [0.0, 0.1, 1.0, 10.0, 100.0]
    n = len(expected_contributions)
    lines = [
        "## NC30 c variance_penalty α sensitivity",
        "",
        f"Candidates: {n}",
        f"E[Δ_S] range: [{expected_contributions.min():.4e}, {expected_contributions.max():.4e}]",
        f"trace(Σ) range: [{variance_traces.min():.4e}, {variance_traces.max():.4e}]",
        "",
        "| α | argmax idx | E[Δ_S] of pick | trace(Σ) of pick | effective_contrib |",
        "|---|---|---|---|---|",
    ]
    for a in alphas:
        eff = compute_effective_contribution(expected_contributions, variance_traces, a)
        idx = int(np.argmax(eff))
        lines.append(
            f"| {a} | {idx} | {expected_contributions[idx]:.4e} | "
            f"{variance_traces[idx]:.4e} | {eff[idx]:.4e} |"
        )
    flips = find_flip_points(expected_contributions, variance_traces)
    lines.append("")
    if flips:
        lines.append(f"**Flip points** ({len(flips)} total):")
        for a, prev, new in flips:
            lines.append(f"  - α ≈ {a:.4f}: argmax flips {prev} → {new}")
    else:
        lines.append("**No flips detected** across α ∈ [0.001, 1000].")
    return "\n".join(lines)
