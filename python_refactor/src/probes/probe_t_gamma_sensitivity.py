"""W22 Probe T — NC29a γ sensitivity sweep analyzer.

Per ``docs/W22-MASTER-BACKLOG.md`` (Probe T): NC29a (commit f42cc9d) replaced
the flat ``1/(H-1)`` prefactor in ``λ^H`` with a geometric ``γ^h`` discount,
tunable via env var ``W22_NC29A_GAMMA`` (default 0.9). Hypothesis: the
optimal ``γ`` varies across regimes; sweep ``γ ∈ {0.5, 0.7, 0.9, 0.99}`` and
quantify the resulting weight profile across horizons WITHOUT running a full
ASMS optimization.

This module is a pure synthetic analyzer — no I/O to disk, no shared-code
imports (only ``math`` + ``numpy``). It replicates the NC29a closed-form

    λ^H_h = clamp(γ^h · (1 - H(TIP_h)), 0, 0.5)

where ``H(p)`` is the binary entropy in bits and ``TIP_h ∈ [0, 1]``. The
goal is to produce a per-γ profile across horizons so the operator can pick
``γ*`` from the curve rather than guessing.

No modifications to shared code paths (sms_emoa.py, anticipatory_learning.py,
multi_horizon_anticipatory.py). Standalone by design.
"""
from __future__ import annotations

import math
from typing import Sequence


_LAMBDA_H_CAP = 0.5  # NC29a safety guard (matches production clamp)


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits. Matches TIPCalculator.binary_entropy.

    Returns 0.0 at the boundaries p ∈ {0, 1} (the limit, not a bug).
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def compute_lambda_h_profile(
    gamma: float, H: int, tip_value: float = 0.5
) -> list[float]:
    """Per-horizon λ^H values for a given γ and a fixed TIP value.

    Replicates the NC29a formula:

        λ^H_h = clamp(γ^h · (1 - H(TIP_h)), 0, 0.5)   for h = 1..H

    Parameters
    ----------
    gamma : float
        Geometric discount factor. Clamped to (0.01, 0.999) to match the
        production safety bounds in calculate_multi_horizon_lambda_rates.
    H : int
        Maximum horizon (number of look-ahead steps). Must be ≥ 1.
    tip_value : float
        Fixed TIP value used at every horizon. Defaults to 0.5 (maximum
        entropy → minimum confidence → minimum λ^H pre-clamp).

    Returns
    -------
    list[float]
        Per-horizon λ^H values, indexed 0..H-1 (where index 0 corresponds
        to h=1).
    """
    if H < 1:
        raise ValueError(f"H must be ≥ 1, got {H}")
    # Match production safety bounds (calculate_multi_horizon_lambda_rates).
    g = max(0.01, min(0.999, float(gamma)))
    confidence = 1.0 - _binary_entropy(tip_value)
    profile: list[float] = []
    for h in range(1, H + 1):
        raw = (g**h) * confidence
        clamped = max(0.0, min(_LAMBDA_H_CAP, raw))
        profile.append(clamped)
    return profile


def sweep_gamma(
    gammas: Sequence[float], H: int, tip_schedule: Sequence[float] | None = None
) -> dict[float, list[float]]:
    """Map each γ to its per-horizon λ^H profile.

    Parameters
    ----------
    gammas : Sequence[float]
        γ values to sweep.
    H : int
        Maximum horizon.
    tip_schedule : Sequence[float] | None
        Per-horizon TIP values (length must equal H). If None, a constant
        TIP=0.5 is used at every horizon (worst-case entropy).

    Returns
    -------
    dict[float, list[float]]
        Mapping from γ to its λ^H profile.
    """
    if tip_schedule is not None and len(tip_schedule) != H:
        raise ValueError(
            f"tip_schedule length ({len(tip_schedule)}) must equal H ({H})"
        )
    result: dict[float, list[float]] = {}
    for gamma in gammas:
        if tip_schedule is None:
            result[gamma] = compute_lambda_h_profile(gamma, H, tip_value=0.5)
        else:
            g = max(0.01, min(0.999, float(gamma)))
            profile: list[float] = []
            for h in range(1, H + 1):
                tip_h = tip_schedule[h - 1]
                confidence = 1.0 - _binary_entropy(tip_h)
                raw = (g**h) * confidence
                profile.append(max(0.0, min(_LAMBDA_H_CAP, raw)))
            result[gamma] = profile
    return result


def effective_horizon(profile: Sequence[float], threshold: float = 0.01) -> int:
    """Last horizon h (1-indexed) where λ^H_h > threshold.

    Returns 0 if no horizon exceeds threshold (degenerate / all-clamped-low).
    """
    last = 0
    for idx, val in enumerate(profile, start=1):
        if val > threshold:
            last = idx
    return last


def cumulative_anticipation_weight(profile: Sequence[float]) -> float:
    """Σ λ^H_h — total weight assigned to multi-horizon predictions."""
    return float(sum(profile))


def analyze_gamma_tradeoff(gammas: Sequence[float], H: int) -> str:
    """Markdown report comparing γ values on effective_horizon and
    cumulative_anticipation_weight, plus a balance product as candidate γ*.
    """
    swept = sweep_gamma(gammas, H)
    rows: list[tuple[float, int, float, float]] = []
    for gamma in gammas:
        profile = swept[gamma]
        eh = effective_horizon(profile)
        cw = cumulative_anticipation_weight(profile)
        balance = eh * cw
        rows.append((gamma, eh, cw, balance))

    # Identify γ* under the proposed balance metric (eh × cumulative_weight).
    star_gamma, star_eh, star_cw, star_balance = max(rows, key=lambda r: r[3])

    lines: list[str] = []
    lines.append(f"## γ sensitivity sweep (H={H}, TIP=0.5 constant)")
    lines.append("")
    lines.append(
        "| γ | effective_horizon (λ^H>0.01) | cumulative_weight (Σ λ^H) | balance (eh × Σ) |"
    )
    lines.append("|---|---|---|---|")
    for gamma, eh, cw, balance in rows:
        marker = " ← γ*" if gamma == star_gamma else ""
        lines.append(f"| {gamma:g} | {eh} | {cw:.4f} | {balance:.4f}{marker} |")
    lines.append("")
    lines.append(
        f"**Candidate γ* under balance = effective_horizon × cumulative_weight: "
        f"γ={star_gamma:g}** "
        f"(eh={star_eh}, Σ={star_cw:.4f}, balance={star_balance:.4f})"
    )
    lines.append("")
    lines.append("### Per-horizon λ^H profiles")
    lines.append("")
    header = "| h | " + " | ".join(f"γ={g:g}" for g in gammas) + " |"
    sep = "|---|" + "|".join(["---"] * len(gammas)) + "|"
    lines.append(header)
    lines.append(sep)
    for h_idx in range(H):
        cells = " | ".join(f"{swept[g][h_idx]:.4f}" for g in gammas)
        lines.append(f"| {h_idx + 1} | {cells} |")
    return "\n".join(lines)
