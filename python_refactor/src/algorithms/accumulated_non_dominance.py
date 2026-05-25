"""W22-NC38 — accumulated multi-period non-dominance probability.

Operator directive 2026-05-20: "probability of non-dominance accumulated
across future periods" — extend single-period P(non-dominance) into a
multi-horizon sum with geometric discount.

For solution s vs current population, the single-period non-dominance
probability is computed via TIP (Defn 6.1). NC38 accumulates this over H
future periods with γ^h discount:

    Q^{ND}(s) = Σ_{h=1}^H γ^h · P(s ‖ others_{t+h} | s)

Where:
  - P(s ‖ others_{t+h}) = probability solution s is mutually non-dominated
    with every other solution at horizon t+h
  - Forecast made via existing KF state (h-step ahead F^h prediction)
  - γ default from W22_NC29A_GAMMA env var (= 0.9), shared with NC29a

Use cases:
  - Selection rule: argmax_s Q^{ND}(s) — picks the solution most likely to
    remain non-dominated over the planning horizon
  - Composition with NC34/NC35: anticipatory mutation scorer can use Q^{ND}
    as an alternative to Δ_S-based scoring
  - Diagnostic: high Q^{ND} solutions are persistent Pareto-front members

Standalone analyzer module — NO modifications to shared code.
"""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

# Reuse infrastructure
from .amfc_selector import _forecast_solution_at_horizon


def _get_gamma() -> float:
    """Read γ from W22_NC29A_GAMMA env var (default 0.9, clamped to (0.01, 0.999))."""
    try:
        gamma = float(os.environ.get("W22_NC29A_GAMMA", "0.9"))
    except ValueError:
        gamma = 0.9
    return max(0.01, min(0.999, gamma))


def pairwise_non_dominance_probability_analytical(
    mu_s: np.ndarray,  # (2,) mean of solution s forecast
    sigma_s: np.ndarray,  # (2, 2) covariance of solution s forecast
    mu_other: np.ndarray,  # (2,) mean of other solution
    sigma_other: np.ndarray,  # (2, 2) covariance of other (treated as fixed if zero)
) -> float:
    """Probability that solution s and other are mutually non-dominated, under
    independent bivariate Gaussian forecasts.

    Computes P(NOT dominate(s, other) AND NOT dominate(other, s)) under the
    joint distribution (s ~ N(μ_s, Σ_s), other ~ N(μ_other, Σ_other)).

    For independent draws, the difference d = (s_ROI - o_ROI, s_risk - o_risk)
    is bivariate normal with mean μ_s - μ_other and cov Σ_s + Σ_other.

    P(s dominates other) = P(d_ROI > 0 AND d_risk < 0)
    P(other dominates s) = P(d_ROI < 0 AND d_risk > 0)
    P(non-dominance) = 1 - P(s dom) - P(other dom)

    Uses scipy.stats.multivariate_normal for the bivariate CDF.
    """
    from scipy import stats as scipy_stats

    diff_mean = np.asarray(mu_s, dtype=float) - np.asarray(mu_other, dtype=float)
    diff_cov = np.asarray(sigma_s, dtype=float) + np.asarray(sigma_other, dtype=float)

    sigma_roi = float(np.sqrt(diff_cov[0, 0]))
    sigma_risk = float(np.sqrt(diff_cov[1, 1]))
    if sigma_roi < 1e-12 or sigma_risk < 1e-12:
        # Degenerate: point-vs-point dominance check
        s_dominates_other = (diff_mean[0] > 0) and (diff_mean[1] < 0)
        other_dominates_s = (diff_mean[0] < 0) and (diff_mean[1] > 0)
        if s_dominates_other or other_dominates_s:
            return 0.0
        return 1.0

    rho_raw = diff_cov[0, 1] / (sigma_roi * sigma_risk)
    rho = max(-0.999999, min(0.999999, rho_raw))

    # Standardize differences
    z1 = -diff_mean[0] / sigma_roi  # for P(d_ROI > 0) we need P(Z > z1) where Z ~ N(0,1)
    z2 = -diff_mean[1] / sigma_risk  # similarly

    Phi_z1 = float(scipy_stats.norm.cdf(z1))
    Phi_z2 = float(scipy_stats.norm.cdf(z2))
    bvn = scipy_stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
    Phi2_z1z2 = float(bvn.cdf([z1, z2]))

    # P(d_ROI > 0 AND d_risk < 0) = P(Z > z1 AND Z < z2)
    #                             = Φ(z2) - Φ_2(z1, z2; ρ)
    p_s_dominates = Phi_z2 - Phi2_z1z2
    p_other_dominates = Phi_z1 - Phi2_z1z2

    p_s_dominates = max(0.0, p_s_dominates)
    p_other_dominates = max(0.0, p_other_dominates)

    p_non_dominance = 1.0 - p_s_dominates - p_other_dominates
    return float(max(0.0, min(1.0, p_non_dominance)))


def joint_non_dominance_probability(
    solution_s, other_solutions: List, horizon: int = 1,
) -> float:
    """Probability that solution s is mutually non-dominated with ALL other
    solutions at horizon h.

    Approximation: assume independence across pairs (s vs other_i are
    INDEPENDENT for different i). This is conservative but tractable.

    P(s NOT dominated by anyone in others) ≈ Π_i P(s ‖ other_i)
    """
    mu_s, sigma_s = _forecast_solution_at_horizon(solution_s, horizon)
    if not other_solutions:
        return 1.0
    joint_p = 1.0
    for other in other_solutions:
        if other is solution_s:
            continue
        mu_o, sigma_o = _forecast_solution_at_horizon(other, horizon)
        p_pair = pairwise_non_dominance_probability_analytical(
            mu_s, sigma_s, mu_o, sigma_o,
        )
        joint_p *= p_pair
    return joint_p


def accumulated_non_dominance_score(
    solution_s, other_solutions: List, max_horizon: int = 3,
    gamma: Optional[float] = None,
) -> float:
    """Q^{ND}(s) = Σ_{h=1}^H γ^h · P(s ‖ others at horizon h).

    Args:
        solution_s: candidate Solution
        other_solutions: other Solutions to compare against
        max_horizon: H (number of horizons to accumulate over)
        gamma: discount factor (default from W22_NC29A_GAMMA env)

    Returns:
        Q^{ND}(s) ∈ [0, γ(1-γ^H)/(1-γ)] geometric series upper bound
    """
    if gamma is None:
        gamma = _get_gamma()
    score = 0.0
    for h in range(1, max_horizon + 1):
        p_nd = joint_non_dominance_probability(solution_s, other_solutions, horizon=h)
        score += (gamma ** h) * p_nd
    return float(score)


def select_by_accumulated_non_dominance(
    population: List,
    max_horizon: int = 3,
    gamma: Optional[float] = None,
    pareto_only: bool = True,
) -> int:
    """Select the solution with maximum accumulated non-dominance score.

    Args:
        population: list of Solutions
        max_horizon: H
        gamma: discount factor
        pareto_only: if True, restrict candidates to Pareto-rank-0

    Returns:
        Index into population of the selected solution.
    """
    if pareto_only:
        candidates_idx = [i for i, s in enumerate(population)
                           if getattr(s, "Pareto_rank", 0) == 0]
        if not candidates_idx:
            candidates_idx = list(range(len(population)))
    else:
        candidates_idx = list(range(len(population)))

    if not candidates_idx:
        return -1
    if len(candidates_idx) == 1:
        return candidates_idx[0]

    scores = np.zeros(len(candidates_idx))
    for k, i in enumerate(candidates_idx):
        scores[k] = accumulated_non_dominance_score(
            population[i],
            [population[j] for j in range(len(population)) if j != i],
            max_horizon=max_horizon,
            gamma=gamma,
        )
    best_local = int(np.argmax(scores))
    return candidates_idx[best_local]


def rank_by_accumulated_non_dominance(
    population: List,
    max_horizon: int = 3,
    gamma: Optional[float] = None,
) -> np.ndarray:
    """Return per-solution Q^{ND} scores (no selection — returns all)."""
    if gamma is None:
        gamma = _get_gamma()
    n = len(population)
    scores = np.zeros(n)
    for i, s in enumerate(population):
        others = [population[j] for j in range(n) if j != i]
        scores[i] = accumulated_non_dominance_score(s, others, max_horizon, gamma)
    return scores
