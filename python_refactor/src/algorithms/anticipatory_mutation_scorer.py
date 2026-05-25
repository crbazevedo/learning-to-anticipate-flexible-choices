"""W22-NC34: Anticipatory mutation scorer.

Operator-flagged (W22): "operators that preserve anticipatory signals or that
anticipate children or mutants performance/rank in the FNDS class or in the
Pareto frontier."

Hypothesis (H-NC34-anticipatory-beats-random)
---------------------------------------------
Standard mutation perturbs portfolio weights randomly; ASMS would benefit from
offspring SCORED BY THEIR PREDICTED FUTURE Δ_S CONTRIBUTION. Selecting mutants
that anticipate high future contribution beats blind perturbation by ≥ 20% on
average in expected Δ_S on synthetic.

Math
----
For each of K candidate mutants {m_1, ..., m_K}:
  1. Forecast each mutant's (ROI, risk) at horizon h via its existing KF state
  2. Compute predicted_Δ_S(m_k) = front-contribution(m_k forecast inserted
     into the forecasted current Pareto frontier, sorted by ROI)
Selection:
  - argmax_k predicted_Δ_S(m_k), OR
  - softmax-weighted: P(m_k) ∝ exp(β · predicted_Δ_S(m_k))

For probability-of-non-dominance scoring (Defn 6.1 / TIP-style):
  - Sample (ROI, risk) from mutant's bivariate Gaussian forecast
  - Estimate P[mutant not dominated by ANY current_front solution at horizon h]

Honest scars
------------
- Cost is O(K · n_mc) per call; tractable for K ≤ 20, n_mc ≤ 100.
- Forecast accuracy depends entirely on KF tuning (couples with NC32).
- "Best" mutant by predicted Δ_S is a heuristic; a multi-objective alternative
  (e.g., combined Pareto-rank + Δ_S, or rank-in-FNDS-class) is deferred.
- For an empty `current_front`, scoring degenerates: this module returns the
  z_ref-anchored contribution rectangle (mutant alone vs (R1, R2)). Caller
  should not use this in production without an explicit fallback.

Integration sketch
------------------
Inside SMSEMOA._mutation, after generating K candidate mutants for one parent,
call::

    from src.algorithms.anticipatory_mutation_scorer import (
        select_best_mutant_argmax,
    )
    best_idx = select_best_mutant_argmax(
        parent, mutants, current_front, horizon=1, R1=R1, R2=R2,
    )
    chosen_mutant = mutants[best_idx]

Wiring is the operator's decision; this module is standalone (no modifications
to shared code paths).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# Reuse forecast and front-contribution from the AMFC selector module.
# Importing keeps the math identical and avoids drift.
from src.algorithms.amfc_selector import (
    _forecast_solution_at_horizon,
    _front_contribution,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_zref(
    mutants: List,
    current_front: List,
    R1: Optional[float],
    R2: Optional[float],
) -> tuple[float, float]:
    """Resolve reference point if not supplied.

    Uses min ROI and max risk across the union of mutants + current_front. A
    degenerate (empty) result falls back to (0.0, 0.05) — the legacy default
    in amfc_selector.derive_zref_from_population.
    """
    if R1 is not None and R2 is not None:
        return float(R1), float(R2)
    pool = list(mutants) + list(current_front)
    if not pool:
        return (R1 if R1 is not None else 0.0), (R2 if R2 is not None else 0.05)
    rois = np.array([s.P.ROI for s in pool])
    risks = np.array([s.P.risk for s in pool])
    out_R1 = float(np.min(rois)) if R1 is None else float(R1)
    out_R2 = float(np.max(risks)) if R2 is None else float(R2)
    return out_R1, out_R2


def _forecast_positions(
    solutions: List, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast (μ_h, Σ_h) for every solution.

    Returns:
        mus: shape (n, 2)
        sigmas: shape (n, 2, 2)
    """
    n = len(solutions)
    mus = np.zeros((n, 2))
    sigmas = np.zeros((n, 2, 2))
    for i, sol in enumerate(solutions):
        mu_h, Sigma_h = _forecast_solution_at_horizon(sol, horizon)
        mus[i] = mu_h
        sigmas[i] = Sigma_h
    return mus, sigmas


def _contribution_of_inserted_point(
    point: np.ndarray, base_front: np.ndarray, R1: float, R2: float
) -> float:
    """Insert `point` into a base front (sorted-by-ROI ascending), then return
    its `_front_contribution` at the resulting position.

    If `base_front` is empty, returns the single-point rectangle
    (ROI − R1) · (R2 − risk).
    """
    if base_front.shape[0] == 0:
        return float((point[0] - R1) * (R2 - point[1]))
    combined = np.vstack([base_front, point[None, :]])
    order = np.argsort(combined[:, 0])
    sorted_front = combined[order]
    # Position of the inserted point in the sorted front.
    inserted_pos = int(np.where(order == base_front.shape[0])[0][0])
    return _front_contribution(inserted_pos, sorted_front, R1, R2)


def _probability_non_dominated_at_horizon(
    mutant_mu: np.ndarray,
    mutant_sigma: np.ndarray,
    front_mus: np.ndarray,
    front_sigmas: np.ndarray,
    n_mc: int,
    rng: np.random.Generator,
) -> float:
    """Monte-Carlo estimate of P[mutant non-dominated by every current_front
    solution at horizon h].

    Bivariate Gaussian dominance per Defn 6.1: dominance in (ROI, risk) space
    is "higher ROI AND lower risk". A mutant sample (r, k) is non-dominated by
    a front sample (r', k') iff NOT (r' >= r AND k' <= k AND (r' > r OR k' < k)).
    The mutant is "fully non-dominated" in a sample iff it survives this test
    against EVERY front member.

    Edge case: empty `front_mus` → mutant is vacuously non-dominated; returns 1.0.
    """
    n_front = front_mus.shape[0]
    if n_front == 0:
        return 1.0

    # Cholesky factors for sampling (with PSD guard).
    def _chol(Sigma: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.cholesky(Sigma + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(Sigma)
            return v @ np.diag(np.sqrt(np.maximum(w, 0.0)))

    L_mutant = _chol(mutant_sigma)
    L_front = np.stack([_chol(front_sigmas[i]) for i in range(n_front)])

    non_dominated_count = 0
    for _ in range(n_mc):
        eps_m = rng.standard_normal(2)
        m_sample = mutant_mu + L_mutant @ eps_m  # (ROI, risk)
        # Per-front independent draws
        eps_f = rng.standard_normal((n_front, 2))
        f_samples = front_mus + np.einsum("ijk,ik->ij", L_front, eps_f)
        # Dominance check: any front solution dominates the mutant?
        # Front dominates mutant iff f.ROI ≥ m.ROI AND f.risk ≤ m.risk AND strict in one.
        roi_better = f_samples[:, 0] >= m_sample[0]
        risk_better = f_samples[:, 1] <= m_sample[1]
        strict = (f_samples[:, 0] > m_sample[0]) | (f_samples[:, 1] < m_sample[1])
        dominated = np.any(roi_better & risk_better & strict)
        if not dominated:
            non_dominated_count += 1

    return non_dominated_count / float(n_mc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_mutants_by_predicted_delta_s(
    parent_solution,
    mutants: List,
    current_front: List,
    horizon: int = 1,
    R1: Optional[float] = None,
    R2: Optional[float] = None,
) -> np.ndarray:
    """Score each mutant by its predicted Δ_S contribution at horizon h.

    For each mutant m_k:
        1. Forecast m_k's (ROI, risk) at horizon h (mean from its KF state)
        2. Forecast every current_front solution at horizon h (mean only)
        3. Insert m_k's forecast into the front; compute its
           _front_contribution at the resulting position

    Args:
        parent_solution: parent solution (unused in scoring, kept for API
            symmetry — caller may want to bias scoring by parent context in
            a future extension; currently a no-op).
        mutants: list of candidate offspring solutions. Each must expose
            .P.ROI, .P.risk, .P.kalman_state (optional).
        current_front: list of current Pareto-front solutions (same shape).
        horizon: prediction horizon h ≥ 1.
        R1, R2: reference point components. If None, derived from the union
            of mutants + current_front (min ROI, max risk).

    Returns:
        scores: shape (len(mutants),) of predicted Δ_S per mutant.

    Raises:
        ValueError: if `mutants` is empty.
    """
    _ = parent_solution  # API symmetry — unused in current scoring
    if not mutants:
        raise ValueError("score_mutants_by_predicted_delta_s: `mutants` is empty")
    R1, R2 = _resolve_zref(mutants, current_front, R1, R2)

    # Forecast everyone at horizon h.
    mutant_mus, _ = _forecast_positions(mutants, horizon)
    if current_front:
        front_mus, _ = _forecast_positions(current_front, horizon)
    else:
        front_mus = np.zeros((0, 2))

    scores = np.zeros(len(mutants))
    for k in range(len(mutants)):
        scores[k] = _contribution_of_inserted_point(
            mutant_mus[k], front_mus, R1, R2
        )
    return scores


def score_mutants_by_non_dominance(
    parent_solution,
    mutants: List,
    current_front: List,
    horizon: int = 1,
    n_mc: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Score each mutant by its average P[non-dominated by any current_front
    solution at horizon h].

    Uses bivariate Gaussian dominance per Defn 6.1 (TIP-style logic inlined
    here since each Solution already exposes its KF state).

    Args:
        parent_solution: parent solution (API symmetry; unused).
        mutants: list of candidate offspring (must expose KF state for
            non-trivial output; without KF state, Σ_h = 0 and sampling
            collapses to the mean).
        current_front: list of current solutions to test dominance against.
        horizon: prediction horizon h ≥ 1.
        n_mc: number of Monte-Carlo samples per mutant.
        rng: optional numpy Generator (default: np.random.default_rng()).

    Returns:
        probabilities: shape (len(mutants),) of P_ND per mutant; each in
        [0.0, 1.0].

    Raises:
        ValueError: if `mutants` is empty.
    """
    _ = parent_solution
    if not mutants:
        raise ValueError("score_mutants_by_non_dominance: `mutants` is empty")
    if rng is None:
        rng = np.random.default_rng()

    mutant_mus, mutant_sigmas = _forecast_positions(mutants, horizon)
    if current_front:
        front_mus, front_sigmas = _forecast_positions(current_front, horizon)
    else:
        front_mus = np.zeros((0, 2))
        front_sigmas = np.zeros((0, 2, 2))

    probs = np.zeros(len(mutants))
    for k in range(len(mutants)):
        probs[k] = _probability_non_dominated_at_horizon(
            mutant_mus[k], mutant_sigmas[k],
            front_mus, front_sigmas,
            n_mc=n_mc, rng=rng,
        )
    return probs


def select_best_mutant_argmax(
    parent_solution,
    mutants: List,
    current_front: List,
    horizon: int = 1,
    R1: Optional[float] = None,
    R2: Optional[float] = None,
) -> int:
    """Select the INDEX of the best mutant by predicted Δ_S (argmax).

    Returns:
        Index in [0, len(mutants)) of the mutant with highest predicted Δ_S.

    Raises:
        ValueError: if `mutants` is empty.
    """
    scores = score_mutants_by_predicted_delta_s(
        parent_solution, mutants, current_front,
        horizon=horizon, R1=R1, R2=R2,
    )
    return int(np.argmax(scores))


def select_mutant_softmax(
    parent_solution,
    mutants: List,
    current_front: List,
    horizon: int = 1,
    beta: float = 1.0,
    R1: Optional[float] = None,
    R2: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Select a mutant INDEX via softmax over predicted Δ_S scores.

    Selection probability: P(m_k) ∝ exp(β · predicted_Δ_S(m_k)).

    Special cases:
        - β = 0: uniform random selection across mutants.
        - β → ∞: equivalent to argmax (concentrated on the best mutant).
        - Negative β: anti-selects (probability concentrates on the WORST mutant).
          Permitted but not recommended; caller's responsibility.

    Args:
        beta: softmax inverse-temperature. Default 1.0.
        rng: optional numpy Generator (default: np.random.default_rng()).

    Returns:
        Index in [0, len(mutants)) of the sampled mutant.

    Raises:
        ValueError: if `mutants` is empty.
    """
    if not mutants:
        raise ValueError("select_mutant_softmax: `mutants` is empty")
    if rng is None:
        rng = np.random.default_rng()

    n = len(mutants)
    if beta == 0.0:
        return int(rng.integers(0, n))

    scores = score_mutants_by_predicted_delta_s(
        parent_solution, mutants, current_front,
        horizon=horizon, R1=R1, R2=R2,
    )

    # Numerically stable softmax.
    z = beta * scores
    z = z - np.max(z)
    weights = np.exp(z)
    s = float(np.sum(weights))
    if s <= 0.0 or not np.isfinite(s):
        # Degenerate (all -inf, all NaN, etc.) — fall back to uniform.
        return int(rng.integers(0, n))
    probs = weights / s
    return int(rng.choice(n, p=probs))
