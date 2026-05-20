"""W22 Probe AI — Pareto front size |P_t| sensitivity for AMFC.

Hypothesis: AMFC argmax behavior depends on |P_t| (number of candidates).
- Very small |P_t| (e.g., 2-3): boundary candidates always win
- Medium |P_t| (5-15): middle candidates can win, AMFC behaves "richly"
- Large |P_t| (50+): contributions concentrate at extremes; AMFC degenerates

Quantify how AMFC's pick distribution changes with |P_t|.

Per docs/W22-MASTER-BACKLOG.md Section H. Standalone analyzer; pure synthetic.
NO modifications to shared code paths.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def front_contribution(idx: int, sorted_front: np.ndarray, R1: float, R2: float) -> float:
    """Per-solution HV contribution at a fixed position (matches amfc_selector formula)."""
    n = sorted_front.shape[0]
    if n == 1:
        return float((sorted_front[0, 0] - R1) * (R2 - sorted_front[0, 1]))
    roi_i, risk_i = sorted_front[idx]
    if idx == 0:
        return float((roi_i - R1) * (R2 - risk_i))
    if idx == n - 1:
        return float((roi_i - sorted_front[idx - 1, 0]) * (R2 - risk_i))
    return float((roi_i - sorted_front[idx + 1, 0]) * (sorted_front[idx - 1, 1] - risk_i))


def synthetic_front(n: int, rng: Optional[np.random.Generator] = None,
                      roi_range: tuple[float, float] = (0.0001, 0.005),
                      risk_range: tuple[float, float] = (0.001, 0.05)
                      ) -> np.ndarray:
    """Generate a synthetic Pareto front of size n, sorted by ROI ascending."""
    if rng is None:
        rng = np.random.default_rng()
    rois = np.linspace(roi_range[0], roi_range[1], n)
    risks = np.linspace(risk_range[1], risk_range[0], n)  # descending
    rois += rng.normal(0, 0.0001, size=n)
    risks += rng.normal(0, 0.001, size=n)
    idx = np.argsort(rois)
    rois = rois[idx]
    risks = risks[idx]
    return np.column_stack([rois, risks])


def pick_distribution(n_runs: int, front_size: int,
                        R1: float = 0.0, R2: float = 0.05,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate n_runs synthetic fronts of given size; return the count
    distribution of which POSITION wins (deterministic argmax of contribution).

    Returns:
        array shape (front_size,) where element i is the count of runs in
        which position i had the max contribution.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    counts = np.zeros(front_size, dtype=int)
    for _ in range(n_runs):
        front = synthetic_front(front_size, rng)
        contribs = np.array([front_contribution(i, front, R1, R2)
                              for i in range(front_size)])
        winner = int(np.argmax(contribs))
        counts[winner] += 1
    return counts


def boundary_concentration(counts: np.ndarray) -> float:
    """Fraction of runs where the winner was at a boundary (position 0 or n-1)."""
    if len(counts) <= 1:
        return 1.0
    boundary_count = counts[0] + counts[-1]
    return boundary_count / counts.sum()


def front_size_sweep(front_sizes: list[int], n_runs: int = 100,
                       R1: float = 0.0, R2: float = 0.05,
                       rng: Optional[np.random.Generator] = None) -> dict:
    """For each front_size, compute the pick-position distribution and boundary
    concentration."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for n in front_sizes:
        counts = pick_distribution(n_runs, n, R1, R2, rng)
        out[n] = {
            "counts": counts,
            "boundary_concentration": boundary_concentration(counts),
            "middle_winner_fraction": (counts.sum() - counts[0] - counts[-1]) / counts.sum() if n > 2 else 0.0,
            "argmax_position": int(np.argmax(counts)),  # most-common winner
        }
    return out


def analyze_front_size_sensitivity(front_sizes: Optional[list[int]] = None,
                                     n_runs: int = 200) -> str:
    """Markdown summary of pick-position vs front size."""
    if front_sizes is None:
        front_sizes = [2, 3, 5, 7, 10, 20, 50]
    sweep = front_size_sweep(front_sizes, n_runs=n_runs)
    lines = [
        "## AMFC pick concentration vs Pareto front size",
        "",
        f"n_runs per size: {n_runs}",
        "",
        "| |P_t| | most-common winner | boundary % | middle % |",
        "|---|---|---|---|",
    ]
    for n in front_sizes:
        r = sweep[n]
        lines.append(
            f"| {n} | pos {r['argmax_position']} | {r['boundary_concentration']:.1%} | "
            f"{r['middle_winner_fraction']:.1%} |"
        )
    lines.append("")
    # Verdict heuristic
    big_size = max(front_sizes)
    big_boundary_pct = sweep[big_size]["boundary_concentration"]
    if big_boundary_pct > 0.95:
        verdict = (
            f"**Verdict:** at |P_t|={big_size}, {big_boundary_pct:.0%} of picks are at "
            "boundaries → AMFC DEGENERATES to boundary selection on large fronts."
        )
    elif big_boundary_pct > 0.7:
        verdict = (
            f"**Verdict:** at |P_t|={big_size}, {big_boundary_pct:.0%} of picks are at "
            "boundaries → AMFC has STRONG boundary bias on large fronts."
        )
    else:
        verdict = (
            f"**Verdict:** at |P_t|={big_size}, only {big_boundary_pct:.0%} of picks are at "
            "boundaries → middle candidates remain competitive on large fronts."
        )
    lines.append(verdict)
    return "\n".join(lines)
