"""W22-NC33: Dirichlet posterior concentration → KF Q coupling.

Operator directive 2026-05-20: "direct linkage and integration between
Dirichlet and KF mechanisms, making their applications less isolated and
more coupled in terms of sequential application and/or param derivations."

Hypothesis: a KF's process noise Q should be SCALED by the precision of the
weight predictor's posterior. When the Dirichlet posterior is tight (high
concentration α), our confidence in future portfolio composition is high,
so the KF's process noise should be LOW. When the posterior is diffuse, Q
should be HIGH.

Math:
  α        = Dirichlet posterior concentration vector (length d, sum = α_sum)
  posterior precision = α_sum (large → tight posterior)
  posterior_variance_scalar = max(diag of Dirichlet Var matrix)
                            = max_i { α_i (α_sum - α_i) / (α_sum² (α_sum + 1)) }
  effective_Q_scale = base_scale · f(precision)

We propose a simple closed-form scaling:
  effective_Q_scale = Q_base · (1 + posterior_variance_scalar / posterior_variance_baseline)
where posterior_variance_baseline is from a uniform prior (Jeffreys α=0.5):
  ≈ d/(d+1)² for large α_sum (asymptotic) or computed exactly from initial α.

When the Dirichlet posterior matches uniform: scale = 2 · Q_base
When tight (low variance): scale → 1 · Q_base (no increase)
When diffuse (variance > baseline): scale > 2 · Q_base (Q grows)

This is a STANDALONE analyzer + composition function. It does NOT modify the
existing KF or Dirichlet predictors — those continue to work independently.
A future structural fix would wire this into the production KF predict step:
  Q_eff = scale_Q_by_dirichlet_posterior(Q_base, portfolio.posterior_predictor)
"""
from __future__ import annotations

import numpy as np


def dirichlet_posterior_variance_max(alpha: np.ndarray) -> float:
    """Compute the MAX of the diagonal of Dirichlet's variance matrix.

    Per the closed-form for Dirichlet:
      Var(X_i) = α_i (α_sum - α_i) / (α_sum² (α_sum + 1))

    Args:
        alpha: concentration vector, shape (d,)

    Returns:
        max_i Var(X_i)
    """
    alpha = np.asarray(alpha, dtype=float)
    alpha_sum = float(np.sum(alpha))
    if alpha_sum <= 0:
        return 0.0
    var_diag = alpha * (alpha_sum - alpha) / (alpha_sum ** 2 * (alpha_sum + 1.0))
    return float(np.max(var_diag))


def dirichlet_posterior_precision(alpha: np.ndarray) -> float:
    """Posterior precision (sum of α; higher → tighter posterior)."""
    return float(np.sum(alpha))


def baseline_posterior_variance(d: int, alpha_prior: float = 0.5) -> float:
    """Max diagonal variance for a uniform Dirichlet(α_prior, ..., α_prior).

    With α_i = α_prior for all i:
      α_sum = d · α_prior
      Var(X_i) = α_prior (d-1) · α_prior / ((d α_prior)² (d α_prior + 1))
              = (d-1) / (d² (d α_prior + 1))

    Args:
        d: dimension of the simplex
        alpha_prior: per-component prior concentration

    Returns:
        Var(X_i) for any i under uniform Dirichlet
    """
    if d <= 1:
        return 0.0
    alpha_sum = d * alpha_prior
    return (d - 1) / (d ** 2 * (alpha_sum + 1.0))


def scale_Q_by_dirichlet_posterior(
    Q_base: np.ndarray | float,
    alpha: np.ndarray,
    alpha_prior: float = 0.5,
    max_scale: float = 10.0,
) -> np.ndarray | float:
    """Scale KF process noise Q by the Dirichlet posterior precision.

    Tight posterior (low variance) → small Q scaling (Q stays low: KF trusts model).
    Diffuse posterior (high variance) → large Q scaling (Q grows: KF distrusts model).

    Scaling formula:
        scale = min(max_scale, 1 + posterior_variance / baseline_variance)
    where baseline_variance is from a uniform Dirichlet of the same dimension
    and prior. With initial alpha (uniform prior), variance ≈ baseline, so
    scale ≈ 2.0. As posterior tightens, variance → 0 → scale → 1.0. As
    posterior diffuses beyond baseline, scale grows up to max_scale.

    Args:
        Q_base: base KF process noise (matrix or scalar)
        alpha: Dirichlet posterior concentration
        alpha_prior: prior concentration for baseline calculation
        max_scale: cap on the scaling factor

    Returns:
        Q_base * scale (same type and shape as Q_base)
    """
    d = len(alpha)
    if d <= 1:
        return Q_base * 1.0
    posterior_var = dirichlet_posterior_variance_max(alpha)
    baseline_var = baseline_posterior_variance(d, alpha_prior)
    if baseline_var <= 0:
        return Q_base * 1.0
    scale = 1.0 + posterior_var / baseline_var
    scale = min(max_scale, max(1.0, scale))
    if isinstance(Q_base, np.ndarray):
        return Q_base * scale
    return float(Q_base) * scale


def kf_residual_to_dirichlet_concentration_increment(
    residual: np.ndarray | float,
    baseline_residual: float = 1.0,
    min_increment: float = 0.1,
    max_increment: float = 5.0,
) -> float:
    """Map KF innovation magnitude to Dirichlet concentration_increment.

    Operator-flagged coupling DIRECTION 2: KF residual → Dirichlet update
    aggressiveness. When the KF sees large innovations (model surprise), the
    Dirichlet predictor should update MORE AGGRESSIVELY (higher
    concentration_increment) — reflecting that recent observations carry
    extra weight because the model is mis-tracking.

    Formula:
        increment = clamp(|residual| / baseline_residual, min_increment, max_increment)

    Args:
        residual: KF innovation (scalar or vector; we use L2 norm)
        baseline_residual: scale parameter (innovation magnitude considered "normal")
        min_increment: minimum concentration increment (always observe at least this much)
        max_increment: cap on aggressive updates

    Returns:
        Dirichlet concentration_increment for use in observe_and_predict()
    """
    if isinstance(residual, np.ndarray):
        mag = float(np.linalg.norm(residual))
    else:
        mag = abs(float(residual))
    if baseline_residual <= 0:
        return min_increment
    raw = mag / baseline_residual
    return float(max(min_increment, min(max_increment, raw)))


def coupled_predict_update_cycle(
    portfolio_obs: np.ndarray,
    objective_obs: np.ndarray,  # observed (ROI, risk)
    objective_pred: np.ndarray,  # predicted (ROI, risk) — pre-update
    dirichlet_predictor,  # DirichletPosteriorPredictor instance
    kalman_state,  # KalmanParams with .x, .P, .Q
    Q_base: np.ndarray | float | None = None,
    alpha_prior: float = 0.5,
    residual_baseline: float = 0.01,  # ROI/risk scale; per-period typical innovation
) -> dict:
    """One cycle of coupled Dirichlet ↔ KF update.

    Sequence:
      1. Compute KF residual (objective_obs - objective_pred)
      2. Map residual → Dirichlet concentration_increment
      3. Update Dirichlet posterior with portfolio_obs (using the
         residual-derived increment)
      4. Compute Dirichlet posterior precision/variance
      5. Scale Q by Dirichlet posterior (effective Q for next predict step)
      6. Return effective Q + updated posterior + residual magnitude

    Args:
        portfolio_obs: realized portfolio weights this period
        objective_obs: realized (ROI, risk) this period
        objective_pred: pre-update KF-predicted (ROI, risk)
        dirichlet_predictor: instance with `observe_and_predict()` method
        kalman_state: KalmanParams (with .Q if no Q_base provided)
        Q_base: base process noise (if None, uses kalman_state.Q or default 0.01·I)
        alpha_prior: Dirichlet prior concentration

    Returns:
        Dict with:
          - residual_magnitude (float)
          - concentration_increment (float, computed from residual)
          - posterior_mean (np.ndarray, updated)
          - effective_Q (np.ndarray or float, scaled by posterior)
    """
    residual = np.asarray(objective_obs, dtype=float) - np.asarray(objective_pred, dtype=float)
    increment = kf_residual_to_dirichlet_concentration_increment(
        residual, baseline_residual=residual_baseline,
    )
    posterior_mean = dirichlet_predictor.observe_and_predict(
        portfolio_obs, concentration_increment=increment,
    )
    if Q_base is None:
        Q_base = getattr(kalman_state, "Q", 0.01 * np.eye(4))
    effective_Q = scale_Q_by_dirichlet_posterior(
        Q_base, dirichlet_predictor.alpha, alpha_prior,
    )
    return {
        "residual_magnitude": float(np.linalg.norm(residual)),
        "concentration_increment": increment,
        "posterior_mean": posterior_mean,
        "effective_Q": effective_Q,
        "dirichlet_precision": dirichlet_posterior_precision(dirichlet_predictor.alpha),
        "dirichlet_max_variance": dirichlet_posterior_variance_max(
            dirichlet_predictor.alpha
        ),
    }
