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


def select_amfc(
    population: List,
    horizon: int = 1,
    n_mc: int = 200,
    R1: float = 0.0,
    R2: float = 0.05,
    pareto_only: bool = True,
    rng: np.random.Generator | None = None,
):
    """Select the AMFC solution from the population.

    Args:
        population: list of Solution objects with .P.ROI, .P.risk, .P.kalman_state, .Pareto_rank
        horizon: forecast horizon h ≥ 1
        n_mc: Monte Carlo samples for the forecast frontier
        R1, R2: reference point components (must be the same as used for HV scoring)
        pareto_only: restrict to current-front (Pareto_rank == 0) candidates
        rng: optional numpy Generator (for reproducibility)

    Returns:
        Selected Solution object. If population is empty, returns None.
        If pareto_only=True and no solution has Pareto_rank == 0, falls back
        to the full population.

    Behavior contract (per W22-NC30-CONTRACT.md):
        - Identity at H=1, Σ→0: AMFC ≡ Hv-DM
          (with zero forecast variance, the MC collapses to a deterministic
          single sample = mean forecast = current state for the identity F,
          and the contributions equal the deterministic Δ_S.)
        - MC stability: with n_mc≥200, argmax is stable across calls
        - All returned solutions are FROM the input population (no synthesis)
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
    best_idx = int(np.argmax(expected_contributions))
    return candidates[best_idx]
