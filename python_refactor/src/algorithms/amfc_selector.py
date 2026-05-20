"""W22-NC30: Anticipative Maximal Flexible Choice (AMFC) selector.

Operator-defined AMFC per W22 Inspection 6 correction:

    s* = argmax_{s ∈ P_t}  E[ HV-contribution( s_{t+h} in forecast F_{t+h} ) ]

For each candidate solution s on the current Pareto frontier:
  1. Forecast each solution's (ROI, risk) at horizon h via its KF
  2. Monte-Carlo sample joint forecast frontiers (one (ROI, risk) per solution
     per MC iter, drawn from each solution's per-horizon Gaussian)
  3. For each MC sample, compute solution s's HV contribution within that
     sampled frontier (same Δ_S formula as deterministic SMS-EMOA)
  4. E[Δ_S^(s)_h] = mean over MC iters of solution s's per-iter contribution
  5. Pick s* = argmax over s of E[Δ_S^(s)_h]

This differs from the current Hv-DM (argmax of current-period Δ_S) because:
  - The contribution is computed in the FORECAST future frontier, not the
    current one
  - Uncertainty in the forecast (KF Σ) is propagated through the MC; high-
    uncertainty solutions naturally get lower expected contributions if
    their forecast distribution overlaps competing solutions
  - The "given choice s" conditioning is implicit in WHERE s lands in the
    forecast frontier (its position relative to other forecast solutions)

W22 Inspection 6 TEST 6 measured 82% argmax disagreement vs current Hv-DM
on synthetic data; this module makes that empirical on real benchmarks.

Selected via env var ``W22_NC30_AMFC=v1`` or via DM type ``Hv-DM-AMFC``.

DESIGN NOTES (from W22-NC30-CONTRACT.md):
- Option A (per-candidate MC of future contributions): IMPLEMENTED HERE
- Option B (path-dependent asset-return MC): deferred; needs market-Σ at
  this layer (same plumbing as NC26-deep)
- Option C (joint Bayesian): out of scope; research-grade

REGRESSION TESTS: tests/test_nc30_amfc_selector.py
"""
from __future__ import annotations

from typing import List

import numpy as np


def _forecast_solution_at_horizon(solution, horizon: int):
    """Return (μ_h, Σ_h) bivariate Gaussian forecast for solution at horizon h.

    Reads from the solution's existing KF state. If the solution has no
    KF state, returns a degenerate (mean=current, cov=0) forecast.

    Args:
        solution: Solution with optional .P.kalman_state (KalmanParams)
        horizon: prediction horizon (h ≥ 1)

    Returns:
        (mu_h, Sigma_h) — mu_h shape (2,), Sigma_h shape (2, 2)
    """
    current_mean = np.array([solution.P.ROI, solution.P.risk], dtype=float)

    kf = getattr(solution.P, "kalman_state", None)
    if kf is None or getattr(kf, "P", None) is None:
        return current_mean, np.zeros((2, 2))

    # Read existing KF state (matches sms_emoa.py state layout: x = [ROI, risk, vROI, vRisk])
    x = np.asarray(kf.x).flatten()
    P = np.asarray(kf.P)
    F = np.asarray(getattr(kf, "F", np.eye(len(x))))

    # h-step ahead state mean: μ_h = F^h · x
    x_h = x.copy()
    P_h = P.copy()
    for _ in range(horizon):
        x_h = F @ x_h
        P_h = F @ P_h @ F.T

    # Project to (ROI, risk) — first 2 components per paper Eq 11.
    mu_h = x_h[:2]
    Sigma_h = P_h[:2, :2]
    # Safety: ensure Sigma_h is PSD (numerical guard).
    Sigma_h = 0.5 * (Sigma_h + Sigma_h.T)
    return mu_h, Sigma_h


def _front_contribution(idx: int, sorted_front: np.ndarray, R1: float, R2: float) -> float:
    """Per-solution HV contribution in a sorted-by-ROI front.

    Matches the deterministic formula from sms_emoa.py:_compute_hypervolume_
    contributions_class:
      - First solution:  (ROI − R1) · (R2 − risk)
      - Middle solution: (ROI_i − ROI_{i+1}) · (risk_{i−1} − risk_i)
      - Last solution:   (ROI_i − ROI_{i−1}) · (R2 − risk_i)

    Args:
        idx: index of the solution in the sorted front
        sorted_front: shape (n, 2) array of (ROI, risk) sorted ascending by ROI
        R1, R2: reference point components

    Returns:
        HV contribution (may be negative if z_ref is too tight; that's a
        signal that the forecast solution sits BEHIND the reference)
    """
    n = sorted_front.shape[0]
    if n == 1:
        return float((sorted_front[0, 0] - R1) * (R2 - sorted_front[0, 1]))

    roi_i, risk_i = sorted_front[idx]
    if idx == 0:
        # No left neighbor — use R1 as the ROI floor.
        return float((roi_i - R1) * (R2 - risk_i))
    if idx == n - 1:
        return float((roi_i - sorted_front[idx - 1, 0]) * (R2 - risk_i))
    # Middle: same convention as sms_emoa.py
    #   (ROI_i − ROI_{i+1}) · (risk_{i−1} − risk_i)
    return float((roi_i - sorted_front[idx + 1, 0]) * (sorted_front[idx - 1, 1] - risk_i))


def derive_zref_from_population(population: List, margin: float = 0.0) -> tuple[float, float]:
    """W22-NC30 b: data-derive z_ref from the current population's extremes.

    Per W22 Inspection 6 honest scar — the hard-coded z_ref defaults
    (sms_emoa.py: (0.0, 0.2); main.py: (-1.0, 10.0)) DISAGREE and flip
    AMFC's argmax. This helper computes z_ref from the population so the
    forecast-HV comparison is comparable across periods.

    Args:
        population: list of solutions with .P.ROI and .P.risk
        margin: relative margin to apply outside the observed extremes
            (e.g., 0.1 = R1 ten percent below min ROI, R2 ten percent above
            max risk; default 0.0 = exact extremes)

    Returns:
        (R1, R2) tuple where R1 ≤ min ROI and R2 ≥ max risk
    """
    if not population:
        return (0.0, 0.05)
    rois = np.array([s.P.ROI for s in population])
    risks = np.array([s.P.risk for s in population])
    R1 = float(np.min(rois))
    R2 = float(np.max(risks))
    if margin > 0.0:
        roi_range = float(np.max(rois) - np.min(rois))
        risk_range = float(np.max(risks) - np.min(risks))
        R1 -= margin * roi_range
        R2 += margin * risk_range
    return R1, R2


def select_amfc(
    population: List,
    horizon: int = 1,
    n_mc: int = 200,
    R1: float | None = None,
    R2: float | None = None,
    pareto_only: bool = True,
    rng: np.random.Generator | None = None,
    derive_zref: bool = False,
    zref_margin: float = 0.0,
    tie_break_by_variance: bool = False,
    tie_epsilon: float = 0.05,
):
    """Select the AMFC solution from the population.

    Args:
        population: list of Solution objects with .P.ROI, .P.risk, .P.kalman_state, .Pareto_rank
        horizon: forecast horizon h ≥ 1
        n_mc: Monte Carlo samples for the forecast frontier
        R1, R2: reference point components (must be the same as used for HV scoring).
            If None and ``derive_zref=True``, derived from population extremes.
            If None and ``derive_zref=False``, defaults to (0.0, 0.05).
        pareto_only: restrict to current-front (Pareto_rank == 0) candidates
        rng: optional numpy Generator (for reproducibility)
        derive_zref: NC30 b — if True, derive R1/R2 from population (overrides
            R1/R2 args when they are None). When R1 or R2 is explicitly given,
            that value takes precedence over the derivation.
        zref_margin: NC30 b — relative margin for the derivation (default 0.0)
        tie_break_by_variance: NC30 d — if True, break ties (top-1 within
            ``tie_epsilon`` of top-2) by picking the candidate with the LOWEST
            forecast variance trace (most certain forecast wins).
        tie_epsilon: NC30 d — relative threshold for tie detection. A tie is
            declared if (top1 − top2) / max(|top1|, 1e-12) < tie_epsilon.

    Returns:
        Selected Solution object. If population is empty, returns None.
        If pareto_only=True and no solution has Pareto_rank == 0, falls back
        to the full population.

    Behavior contract (per W22-NC30-CONTRACT.md):
        - Identity at H=1, Σ→0: AMFC ≡ Hv-DM
        - MC stability: with n_mc≥200 + small σ, argmax is stable across calls
        - All returned solutions are FROM the input population (no synthesis)
        - With derive_zref=True: AMFC pick is invariant under uniform shifts
          of the population's ROI/risk values (because z_ref shifts with them)
    """
    if rng is None:
        rng = np.random.default_rng()
    if not population:
        return None

    # Candidate filter: Pareto front only by default (matches Hv-DM behavior).
    if pareto_only:
        candidates = [s for s in population if getattr(s, "Pareto_rank", 0) == 0]
        if not candidates:
            candidates = list(population)
    else:
        candidates = list(population)

    n = len(candidates)
    if n == 1:
        return candidates[0]

    # Resolve z_ref (NC30 b: data-derived if requested and not explicitly given).
    if R1 is None or R2 is None:
        if derive_zref:
            derived_R1, derived_R2 = derive_zref_from_population(candidates, zref_margin)
            if R1 is None:
                R1 = derived_R1
            if R2 is None:
                R2 = derived_R2
        else:
            if R1 is None:
                R1 = 0.0
            if R2 is None:
                R2 = 0.05

    # Forecast each candidate's (μ_h, Σ_h).
    mus = np.zeros((n, 2))
    sigmas = np.zeros((n, 2, 2))
    for i, sol in enumerate(candidates):
        mu_h, Sigma_h = _forecast_solution_at_horizon(sol, horizon)
        mus[i] = mu_h
        sigmas[i] = Sigma_h

    # Per-candidate Cholesky for sampling. Degenerate Σ → zero matrix → exact mean.
    cholesky = np.zeros_like(sigmas)
    for i in range(n):
        try:
            cholesky[i] = np.linalg.cholesky(sigmas[i] + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            # Σ not PSD — fall back to eigendecomposition with non-negative eigenvalues.
            w, v = np.linalg.eigh(sigmas[i])
            w_pos = np.maximum(w, 0.0)
            cholesky[i] = v @ np.diag(np.sqrt(w_pos))

    # MC: sample n_mc forecast frontiers; compute per-candidate contributions.
    contributions = np.zeros((n, n_mc))
    for j in range(n_mc):
        # Independent draws per candidate (no shared market noise — matches Option A).
        eps = rng.standard_normal(size=(n, 2))
        samples = mus + np.einsum("ijk,ik->ij", cholesky, eps)  # shape (n, 2)
        # Sort by ROI ascending (matches sms_emoa.py)
        order = np.argsort(samples[:, 0])
        sorted_front = samples[order]
        # Inverse permutation: position of original candidate i in sorted_front
        inv_order = np.argsort(order)
        for i in range(n):
            pos = int(inv_order[i])
            contributions[i, j] = _front_contribution(pos, sorted_front, R1, R2)

    # Expected contribution per candidate.
    expected_contributions = contributions.mean(axis=1)

    # NC30 d: tie-break by lowest forecast variance trace.
    if tie_break_by_variance and n >= 2:
        sorted_idx = np.argsort(-expected_contributions)  # descending
        top1, top2 = sorted_idx[0], sorted_idx[1]
        top1_val, top2_val = expected_contributions[top1], expected_contributions[top2]
        denom = max(abs(top1_val), 1e-12)
        is_tie = (top1_val - top2_val) / denom < tie_epsilon
        if is_tie:
            # Among all candidates within tie_epsilon of top-1, pick lowest trace(Σ_h)
            tie_set = [
                i for i in range(n)
                if (top1_val - expected_contributions[i]) / denom < tie_epsilon
            ]
            variances = np.array([np.trace(sigmas[i]) for i in tie_set])
            best_in_tie = tie_set[int(np.argmin(variances))]
            return candidates[best_in_tie]

    best_idx = int(np.argmax(expected_contributions))
    return candidates[best_idx]
