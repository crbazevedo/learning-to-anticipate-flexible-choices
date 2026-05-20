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

# W22-NC30 telemetry: a module-level ring buffer that callers can flush
# (e.g., once per period) to compare AMFC picks against Hv-DM picks in
# production without running parallel experiments. Default usage:
#
#   from src.algorithms.amfc_selector import (
#       select_amfc, get_amfc_telemetry, reset_amfc_telemetry,
#   )
#   reset_amfc_telemetry()
#   for period in periods:
#       pick = select_amfc(population, ..., collect_telemetry=True)
#       ... run rebalance with pick ...
#   telem = get_amfc_telemetry()  # list of per-call telemetry dicts
#
# Each telemetry dict contains:
#   - amfc_idx: index of the AMFC pick in the candidate list
#   - hv_dm_idx: index of the Hv-DM pick (argmax of solution.Delta_S)
#   - amfc_agrees_with_hv_dm: bool
#   - amfc_pick_roi, hv_dm_pick_roi: ROI of each pick
#   - n_candidates: |P_t|
#   - top1_expected_contrib: E[Δ_S] of the AMFC pick
#   - mean_forecast_variance: average trace(Σ_h) across candidates
#   - amfc_pick_forecast_variance: trace(Σ_h) of the AMFC pick
#   - tie_break_fired: bool (NC30 d telemetry)
_AMFC_TELEMETRY: List[dict] = []


def get_amfc_telemetry() -> List[dict]:
    """Return a copy of the AMFC telemetry ring buffer."""
    return list(_AMFC_TELEMETRY)


def reset_amfc_telemetry() -> None:
    """Clear the AMFC telemetry ring buffer."""
    _AMFC_TELEMETRY.clear()


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
    derive_zref: bool = True,  # NC30-v2 STRUCTURAL: default True (was False) per operator directive
    zref_margin: float = 0.0,
    tie_break_by_variance: bool = False,
    tie_epsilon: float = 0.05,
    collect_telemetry: bool = False,
    analytical: bool = True,  # NC30-v2 STRUCTURAL: deterministic E[contrib] at mean positions
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

    # NC30-v2 STRUCTURAL FIX (operator directive 2026-05-19): default to
    # ANALYTICAL E[contribution] computed at the deterministic mean-sorted
    # positions. Pre-fix used per-iter MC sort-order which produced degenerate
    # behavior in two regimes:
    #   (a) identical means + different variances → different E[Δ_S] from
    #       independent-candidate noise (CRN partially fixed, but boundary
    #       ties still chaotic)
    #   (b) σ ≈ inter-solution spread → modal pick flips across seeds because
    #       boundary contributions overlap statistically
    #
    # ANALYTICAL mode (default): sort candidates by mean ROI ONCE; compute
    # deterministic contribution at the resulting fixed positions using mean
    # forecast values. This eliminates MC sort-order noise entirely. The
    # forecast variance signal is retained via NC30 d tie-break.
    #
    # MC mode (analytical=False) is kept for backward compatibility and
    # research; uses shared-noise CRN to minimize per-iter noise but cannot
    # eliminate boundary-tie chaos.
    if analytical:
        order = np.argsort(mus[:, 0])  # sort by mean ROI
        sorted_mus = mus[order]
        inv_order = np.argsort(order)
        # Per-candidate deterministic contribution at fixed mean position.
        expected_contributions = np.zeros(n)
        for i in range(n):
            pos = int(inv_order[i])
            expected_contributions[i] = _front_contribution(pos, sorted_mus, R1, R2)

        # NC30-v2 STRUCTURAL FIX: tie-handling for equal/near-equal mean ROIs.
        # Without this, two candidates at identical means get DIFFERENT
        # contributions because the sort-tiebreak (numpy argsort stable) puts
        # them at adjacent positions with asymmetric contribution formulas
        # (position 0 uses R1-floor, position 1 uses prev-neighbor).
        #
        # Fix: detect tied-mean groups; assign the AVERAGE contribution across
        # the group's contiguous positions to every member of the group. This
        # is the rigorous expectation under uniform permutation of tied
        # candidates and restores the symmetry the operator's directive
        # (2026-05-19) demands.
        roi_sorted = sorted_mus[:, 0]
        tie_tolerance = 1e-12  # numerical-precision ties only
        # Group consecutive equal-mean-ROI positions.
        group_start = 0
        while group_start < n:
            group_end = group_start
            while group_end + 1 < n and abs(roi_sorted[group_end + 1] - roi_sorted[group_start]) < tie_tolerance:
                group_end += 1
            if group_end > group_start:
                # Multiple positions tied; average their contributions.
                tied_positions = list(range(group_start, group_end + 1))
                contrib_at_tied_positions = [
                    _front_contribution(p, sorted_mus, R1, R2) for p in tied_positions
                ]
                avg_contrib = float(np.mean(contrib_at_tied_positions))
                # Assign to every original candidate whose sorted position falls in the tied group.
                for orig_i in range(n):
                    if int(inv_order[orig_i]) in tied_positions:
                        expected_contributions[orig_i] = avg_contrib
            group_start = group_end + 1
    else:
        # Per-candidate Cholesky for sampling. Degenerate Σ → zero matrix → exact mean.
        cholesky = np.zeros((n, 2, 2))
        for i in range(n):
            try:
                cholesky[i] = np.linalg.cholesky(sigmas[i] + 1e-12 * np.eye(2))
            except np.linalg.LinAlgError:
                # Σ not PSD — eigendecomp fallback.
                w, v = np.linalg.eigh(sigmas[i])
                w_pos = np.maximum(w, 0.0)
                cholesky[i] = v @ np.diag(np.sqrt(w_pos))

        contributions = np.zeros((n, n_mc))
        for j in range(n_mc):
            # Shared-noise CRN: one direction per iter, scaled per-candidate by Cholesky.
            eps_shared = rng.standard_normal(size=2)
            samples = mus + np.einsum("ijk,k->ij", cholesky, eps_shared)
            order_j = np.argsort(samples[:, 0])
            sorted_front_j = samples[order_j]
            inv_order_j = np.argsort(order_j)
            for i in range(n):
                pos = int(inv_order_j[i])
                contributions[i, j] = _front_contribution(pos, sorted_front_j, R1, R2)
        expected_contributions = contributions.mean(axis=1)

    # NC30 d: tie-break by lowest forecast variance trace.
    tie_fired = False
    selected_idx = int(np.argmax(expected_contributions))
    if tie_break_by_variance and n >= 2:
        sorted_idx = np.argsort(-expected_contributions)  # descending
        top1, top2 = sorted_idx[0], sorted_idx[1]
        top1_val, top2_val = expected_contributions[top1], expected_contributions[top2]
        denom = max(abs(top1_val), 1e-12)
        is_tie = (top1_val - top2_val) / denom < tie_epsilon
        if is_tie:
            tie_fired = True
            tie_set = [
                i for i in range(n)
                if (top1_val - expected_contributions[i]) / denom < tie_epsilon
            ]
            variances = np.array([np.trace(sigmas[i]) for i in tie_set])
            selected_idx = tie_set[int(np.argmin(variances))]

    # NC30 telemetry: compare against the Hv-DM pick (argmax of Delta_S).
    if collect_telemetry:
        # Hv-DM equivalent: argmax of the candidate's Delta_S attribute.
        # If no Delta_S attribute, falls back to current-period rectangle
        # using the resolved R1/R2.
        def _hv_dm_score(s):
            ds = getattr(s, "Delta_S", None)
            if ds is None or ds == 0.0:
                # Fallback: deterministic single-solution rectangle
                return (s.P.ROI - R1) * (R2 - s.P.risk)
            return float(ds)

        hv_dm_idx = int(np.argmax([_hv_dm_score(s) for s in candidates]))
        _AMFC_TELEMETRY.append({
            "amfc_idx": selected_idx,
            "hv_dm_idx": hv_dm_idx,
            "amfc_agrees_with_hv_dm": selected_idx == hv_dm_idx,
            "amfc_pick_roi": float(candidates[selected_idx].P.ROI),
            "amfc_pick_risk": float(candidates[selected_idx].P.risk),
            "hv_dm_pick_roi": float(candidates[hv_dm_idx].P.ROI),
            "hv_dm_pick_risk": float(candidates[hv_dm_idx].P.risk),
            "n_candidates": n,
            "horizon": horizon,
            "n_mc": n_mc,
            "top1_expected_contrib": float(expected_contributions[selected_idx]),
            "mean_forecast_variance": float(np.mean([np.trace(s) for s in sigmas])),
            "amfc_pick_forecast_variance": float(np.trace(sigmas[selected_idx])),
            "tie_break_fired": tie_fired,
            "R1": R1,
            "R2": R2,
        })

    return candidates[selected_idx]
