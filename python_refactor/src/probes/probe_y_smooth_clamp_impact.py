"""W22 Probe Y — NC13b smooth TIP clamp impact analyzer.

Hypothesis: in TIP-saturation regimes (where raw TIP frequently lies in
tail regions [0, 0.05) ∪ (0.95, 1]), the smooth clamp (NC13b) preserves
gradient signal that the hard clamp destroys. Quantify the difference
between the two clamping policies across a TIP-distribution grid.

Per docs/W22-MASTER-BACKLOG.md Section H. Standalone analyzer; pure
synthetic, no ASMS run.

NO modifications to shared code paths.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def hard_clamp(tip: float, c_min: float = 0.05, c_max: float = 0.95) -> float:
    """Legacy hard clip: max(c_min, min(c_max, tip))."""
    return max(c_min, min(c_max, tip))


def smooth_clamp(tip: float, c_min: float = 0.05, c_max: float = 0.95,
                   k: float = 4.0) -> float:
    """NC13b smooth squash via shifted-scaled tanh."""
    center = 0.5 * (c_min + c_max)
    width = c_max - c_min
    return c_min + width * (1.0 + math.tanh(k * (tip - center))) / 2.0


def derivative_hard(tip: float, c_min: float = 0.05, c_max: float = 0.95) -> float:
    """Derivative of hard clip wrt tip: 1 inside [c_min, c_max], else 0."""
    if c_min < tip < c_max:
        return 1.0
    return 0.0


def derivative_smooth(tip: float, c_min: float = 0.05, c_max: float = 0.95,
                       k: float = 4.0) -> float:
    """Derivative of smooth clamp wrt tip: width · k · sech²(k(tip-center)) / 2."""
    center = 0.5 * (c_min + c_max)
    width = c_max - c_min
    sech_sq = 1.0 - math.tanh(k * (tip - center)) ** 2
    return width * k * sech_sq / 2.0


def compare_clamps(tips: np.ndarray, c_min: float = 0.05, c_max: float = 0.95,
                    k: float = 4.0) -> dict:
    """Compare hard vs smooth clamp outputs across a TIP grid.

    Args:
        tips: 1-D array of raw TIP values
        c_min, c_max: clamp bounds
        k: smooth squash steepness

    Returns:
        Dict with:
          - hard: clamped outputs under hard clip
          - smooth: clamped outputs under NC13b
          - diff: smooth - hard (positive in upper tail; negative in lower tail)
          - hard_deriv: gradient of hard clip (1 inside, 0 outside)
          - smooth_deriv: gradient of smooth clip (always > 0)
          - signal_recovered_fraction: fraction of tail TIPs where smooth preserves
            gradient that hard destroys
    """
    hard = np.array([hard_clamp(t, c_min, c_max) for t in tips])
    smooth = np.array([smooth_clamp(t, c_min, c_max, k) for t in tips])
    diff = smooth - hard
    hard_deriv = np.array([derivative_hard(t, c_min, c_max) for t in tips])
    smooth_deriv = np.array([derivative_smooth(t, c_min, c_max, k) for t in tips])
    # "Tail" = TIPs in [0, c_min] ∪ [c_max, 1]
    is_tail = (tips < c_min) | (tips > c_max)
    n_tail = int(np.sum(is_tail))
    n_recovered = int(np.sum(is_tail & (smooth_deriv > 1e-6)))
    return {
        "hard": hard,
        "smooth": smooth,
        "diff": diff,
        "hard_deriv": hard_deriv,
        "smooth_deriv": smooth_deriv,
        "signal_recovered_fraction": n_recovered / n_tail if n_tail > 0 else 0.0,
        "n_tail": n_tail,
        "n_total": len(tips),
    }


def saturation_regime_summary(tip_distribution: np.ndarray,
                                c_min: float = 0.05, c_max: float = 0.95,
                                k: float = 4.0) -> str:
    """Generate a markdown report on the impact of smooth-vs-hard clamp
    on a TIP distribution that is suspected to be saturated.

    Args:
        tip_distribution: empirical TIP values (e.g. from a real ASMS run)
        c_min, c_max: clamp bounds
        k: smooth steepness

    Returns:
        Markdown summary including statistics on saturation rate, signal
        recovery, and the average gradient difference.
    """
    result = compare_clamps(tip_distribution, c_min, c_max, k)
    n_below = int(np.sum(tip_distribution < c_min))
    n_above = int(np.sum(tip_distribution > c_max))
    n_inside = int(np.sum((tip_distribution >= c_min) & (tip_distribution <= c_max)))
    sat_rate = (n_below + n_above) / len(tip_distribution)
    mean_gradient_diff = float(np.mean(result["smooth_deriv"] - result["hard_deriv"]))
    avg_abs_diff = float(np.mean(np.abs(result["diff"])))
    lines = [
        "## NC13b smooth-clamp impact summary",
        "",
        f"TIP distribution: {len(tip_distribution)} samples",
        f"- below c_min ({c_min}): {n_below} ({n_below/len(tip_distribution):.1%})",
        f"- above c_max ({c_max}): {n_above} ({n_above/len(tip_distribution):.1%})",
        f"- inside: {n_inside} ({n_inside/len(tip_distribution):.1%})",
        f"- saturation rate: **{sat_rate:.1%}**",
        "",
        f"Smooth clamp impact:",
        f"- signal_recovered_fraction (tail TIPs w/ non-zero gradient): "
        f"**{result['signal_recovered_fraction']:.1%}**",
        f"- mean(smooth - hard) output difference: {float(np.mean(result['diff'])):.4e}",
        f"- mean |smooth - hard| output difference: {avg_abs_diff:.4e}",
        f"- mean gradient difference (smooth - hard): {mean_gradient_diff:.4f}",
        "",
    ]
    if sat_rate > 0.30:
        lines.append("**Verdict:** HIGH saturation regime — NC13b smooth clamp likely matters.")
    elif sat_rate > 0.10:
        lines.append("**Verdict:** Moderate saturation — NC13b may matter at the margin.")
    else:
        lines.append("**Verdict:** Low saturation — NC13b should be near-no-op in production.")
    return "\n".join(lines)
