"""W22 Probe AI-2 — data-derived z_ref + boundary bias interaction.

Hypothesis: Probe AI showed AMFC has STRONG boundary bias on synthetic
fronts (91-100% boundary picks across |P_t|). The honest scar noted that
hardcoded R1=0, R2=0.05 was the cause. Under NC30 b (data-derived z_ref,
R1=min ROI, R2=max risk), the boundary contribution becomes:
    (boundary_ROI - R1) · (R2 - boundary_risk)
  = (boundary_ROI - boundary_ROI) · (R2 - boundary_risk)  [boundary IS min ROI]
  = 0
So the boundary candidate gets ZERO contribution.

Probe AI-2 verifies this: with data-derived z_ref, the boundary bias
VANISHES; pick concentrates in the middle.

Per docs/W22-MASTER-BACKLOG.md and Probe AI's scar.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Reuse front_contribution + synthetic_front from Probe AI
from .probe_ai_front_size_sensitivity import (
    front_contribution,
    synthetic_front,
)


def derive_zref(front: np.ndarray, margin: float = 0.0) -> tuple[float, float]:
    """NC30 b: R1 = min ROI, R2 = max risk (± margin)."""
    rois = front[:, 0]
    risks = front[:, 1]
    R1 = float(np.min(rois))
    R2 = float(np.max(risks))
    if margin > 0:
        roi_range = float(np.ptp(rois))
        risk_range = float(np.ptp(risks))
        R1 -= margin * roi_range
        R2 += margin * risk_range
    return R1, R2


def winners_under_zref_modes(n_runs: int, front_size: int,
                                fixed_R1: float = 0.0, fixed_R2: float = 0.05,
                                derived_margin: float = 0.0,
                                rng: Optional[np.random.Generator] = None
                                ) -> dict:
    """Compare argmax winner position under fixed vs derived z_ref."""
    if rng is None:
        rng = np.random.default_rng(0)
    fixed_counts = np.zeros(front_size, dtype=int)
    derived_counts = np.zeros(front_size, dtype=int)
    flips = 0
    for _ in range(n_runs):
        front = synthetic_front(front_size, rng)
        # Fixed
        c_fixed = np.array([front_contribution(i, front, fixed_R1, fixed_R2)
                              for i in range(front_size)])
        w_fixed = int(np.argmax(c_fixed))
        fixed_counts[w_fixed] += 1
        # Derived
        d_R1, d_R2 = derive_zref(front, derived_margin)
        c_derived = np.array([front_contribution(i, front, d_R1, d_R2)
                                for i in range(front_size)])
        w_derived = int(np.argmax(c_derived))
        derived_counts[w_derived] += 1
        if w_fixed != w_derived:
            flips += 1
    return {
        "fixed_counts": fixed_counts,
        "derived_counts": derived_counts,
        "fixed_boundary_pct": (fixed_counts[0] + fixed_counts[-1]) / n_runs,
        "derived_boundary_pct": (derived_counts[0] + derived_counts[-1]) / n_runs,
        "fixed_middle_pct": (n_runs - fixed_counts[0] - fixed_counts[-1]) / n_runs if front_size > 2 else 0.0,
        "derived_middle_pct": (n_runs - derived_counts[0] - derived_counts[-1]) / n_runs if front_size > 2 else 0.0,
        "argmax_disagreement_pct": flips / n_runs,
    }


def sweep_front_sizes_under_both_zrefs(front_sizes: list[int],
                                          n_runs: int = 200,
                                          rng: Optional[np.random.Generator] = None) -> dict:
    """For each front_size, compare fixed vs derived z_ref behavior."""
    if rng is None:
        rng = np.random.default_rng(0)
    out = {}
    for n in front_sizes:
        out[n] = winners_under_zref_modes(n_runs, n, rng=rng)
    return out


def analyze_zref_boundary_interaction(front_sizes: Optional[list[int]] = None,
                                         n_runs: int = 300) -> str:
    """Markdown summary."""
    if front_sizes is None:
        front_sizes = [3, 5, 10, 20]
    sweep = sweep_front_sizes_under_both_zrefs(front_sizes, n_runs)
    lines = [
        "## NC30 b data-derived z_ref vs fixed: boundary-bias interaction",
        "",
        f"n_runs per size: {n_runs}",
        "",
        "| |P_t| | fixed-zref boundary % | derived-zref boundary % | argmax disagreement |",
        "|---|---|---|---|",
    ]
    for n in front_sizes:
        r = sweep[n]
        lines.append(
            f"| {n} | {r['fixed_boundary_pct']:.1%} | {r['derived_boundary_pct']:.1%} | "
            f"{r['argmax_disagreement_pct']:.1%} |"
        )
    lines.append("")
    # Verdict — NC30 b nulls the LEFT boundary's contribution
    # (because R1 = min ROI = ROI of leftmost candidate, so
    # contribution = (ROI - R1) * (R2 - risk) = 0). The RIGHT boundary
    # still wins because (R2 - smallest_risk) = full risk range, which
    # gives the right boundary a large contribution.
    avg_fixed = np.mean([sweep[n]["fixed_boundary_pct"] for n in front_sizes])
    avg_derived = np.mean([sweep[n]["derived_boundary_pct"] for n in front_sizes])
    avg_disagree = np.mean([sweep[n]["argmax_disagreement_pct"] for n in front_sizes])

    # Look specifically at the LEFT boundary (position 0) and RIGHT (position -1)
    # to characterize the asymmetric effect.
    fixed_left = np.mean([sweep[n]["fixed_counts"][0] / n_runs for n in front_sizes])
    derived_left = np.mean([sweep[n]["derived_counts"][0] / n_runs for n in front_sizes])
    fixed_right = np.mean([sweep[n]["fixed_counts"][-1] / n_runs for n in front_sizes])
    derived_right = np.mean([sweep[n]["derived_counts"][-1] / n_runs for n in front_sizes])

    verdict = (
        f"**Verdict:** NC30 b has ASYMMETRIC effect on boundary bias.\n\n"
        f"- LEFT boundary (pos 0): fixed {fixed_left:.0%} → derived {derived_left:.0%}\n"
        f"  (NC30 b sets R1 = min ROI = leftmost ROI → contribution = 0 → left always loses)\n"
        f"- RIGHT boundary (pos -1): fixed {fixed_right:.0%} → derived {derived_right:.0%}\n"
        f"  (NC30 b sets R2 = max risk = leftmost risk → (R2 - rightmost_risk) = full risk range → right still wins)\n"
        f"- Total boundary %: fixed={avg_fixed:.0%} → derived={avg_derived:.0%}\n"
        f"- argmax disagreement: {avg_disagree:.0%}\n\n"
        "**Implication:** NC30 b structurally NULLS the LEFT (low-ROI) boundary's\n"
        "contribution but does NOT touch the RIGHT (high-ROI) boundary's advantage.\n"
        "AMFC under NC30 b will still concentrate on the high-ROI extreme. To kill\n"
        "the right-boundary bias too, R2 would need to be tighter (e.g., median risk),\n"
        "but that's geometrically inconsistent with R2 = 'risk we don't want to exceed'."
    )
    lines.append(verdict)
    return "\n".join(lines)
