"""
W13-2 Out-of-sample future hypervolume evaluator.

Implements the protocol from Azevedo PhD thesis §7.2.2 + §7.3.3
(see `docs/THESIS-INDEX.md`):

    Eq (7.10):  ẑ_{t+1} = f(Û_t^{N*}, χ_{t+1}, m̂_{u*_t})
    Eq (7.11):  Ŝ_{t+1} = (1/E) Σ_{e=1}^E S(ẑ_{e,t+1}, z_ref)

with E=1000 Monte-Carlo scenarios, z_ref = (0.2, 0.0)^T
(risk_max=0.2, return_min=0.0).

The χ_{t+1} = (μ_{t+1}, Σ_{t+1}) parameters come from MLE-fitting a
Gaussian to a HELD-OUT block of asset returns (the period the
algorithm did NOT train on). Each MC scenario e bootstraps that
block to obtain (μ̂_e, Σ̂_e); the trained Pareto-flexible set's
portfolios are evaluated against (μ̂_e, Σ̂_e); the hypervolume of
the resulting N-point cloud is averaged across the E scenarios.

Critical distinction from in-sample EFHV (which the C++ legacy and
`sms_emoa._compute_expected_future_hypervolume` produce): in-sample
samples from the algorithm's OWN KF state projection; OOS samples
from the actually-observed future distribution. Footnote 2 of
§7.2.2 (p. 144): "Those sets of parameters [χ_{t+1}] are only used
for a post-hoc assessment of the obtained results and do not
interfere in the optimization problem."

Pure-function module. No I/O. No side effects on global state.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def fit_future_state(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """MLE-fit Gaussian on held-out daily asset returns.

    Returns:
        (mu, sigma) where mu is the mean return vector (n_assets,) and
        sigma is the sample covariance matrix (n_assets, n_assets).
    """
    arr = np.asarray(returns_df, dtype=float) if not isinstance(returns_df, np.ndarray) else returns_df
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"need at least 2 rows; got shape {arr.shape}")
    mu = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)
    return mu, cov


def evaluate_portfolio_under_future(weights: np.ndarray,
                                      mu: np.ndarray,
                                      sigma: np.ndarray) -> tuple[float, float]:
    """For one portfolio weight vector, compute (future_risk, future_return)
    under the given future Gaussian state.

    Risk = u^T Σ u (variance — matches thesis Markowitz formulation §7.2).
    Return = μ^T u.
    """
    w = np.asarray(weights, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    risk = float(w @ sigma_arr @ w)
    ret = float(mu_arr @ w)
    return risk, ret


def hypervolume_2d(points: Iterable[tuple[float, float]],
                    z_ref: tuple[float, float] = (0.2, 0.0)) -> float:
    """2D hypervolume for (risk-MIN, return-MAX) objective space.

    `z_ref = (risk_max, return_min)` is the worst-case reference point
    (thesis §7.2.3: (0.2, 0.0) for portfolio selection — max 20% risk,
    min 0% return). Each Pareto point (r, q) dominates a rectangle
    [r, z_ref[0]] × [z_ref[1], q].

    HV = total area of the staircase formed by the upper-left envelope.
    """
    pts = [(float(r), float(q)) for r, q in points
            if float(r) <= z_ref[0] and float(q) >= z_ref[1]]
    if not pts:
        return 0.0
    # Sort by risk ASC, tiebreak by return DESC.
    pts.sort(key=lambda p: (p[0], -p[1]))
    # Pareto-filter: keep points whose return strictly improves the running max.
    front: list[tuple[float, float]] = []
    max_q = float("-inf")
    for r, q in pts:
        if q > max_q:
            front.append((r, q))
            max_q = q
    # Sweep right-to-left, accumulating staircase rectangles.
    hv = 0.0
    prev_r = z_ref[0]
    for r, q in reversed(front):
        width = prev_r - r
        height = q - z_ref[1]
        hv += width * height
        prev_r = r
    return max(hv, 0.0)


# ---------------------------------------------------------------------------
# Deterministic + Monte-Carlo OOS HV
# ---------------------------------------------------------------------------

def compute_oos_hv_deterministic(pareto_weights: list[np.ndarray],
                                   mu: np.ndarray,
                                   sigma: np.ndarray,
                                   z_ref: tuple[float, float] = (0.2, 0.0)) -> float:
    """Single-shot OOS HV: evaluate Pareto front under fixed (μ, Σ).

    Useful as a debugging baseline; the thesis-faithful version is
    compute_oos_efhv (MC-averaged).
    """
    points = [evaluate_portfolio_under_future(w, mu, sigma) for w in pareto_weights]
    return hypervolume_2d(points, z_ref)


def compute_oos_efhv(pareto_weights: list[np.ndarray],
                      oos_returns: pd.DataFrame,
                      n_samples: int = 1000,
                      z_ref: tuple[float, float] = (0.2, 0.0),
                      rng: np.random.Generator | None = None) -> dict:
    """Sample-average OOS Future Hypervolume per thesis Eqs 7.10+7.11.

    For each of n_samples MC scenarios e:
      1. Bootstrap-resample oos_returns with replacement (preserves
         multivariate dependence structure)
      2. MLE-fit (μ̂_e, Σ̂_e) on the bootstrap sample
      3. For each portfolio u_i in pareto_weights, compute
         (u_i^T Σ̂_e u_i, μ̂_e^T u_i)
      4. Compute hypervolume of the N-point cloud against z_ref → S_e

    Returns:
        {
          'efhv_mean': float,   # Ŝ_{t+1} per Eq 7.11
          'efhv_std':  float,   # MC standard deviation across scenarios
          'efhv_samples': np.ndarray  # all E sample HVs (for downstream stats)
        }
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = (oos_returns.values if isinstance(oos_returns, pd.DataFrame)
            else np.asarray(oos_returns, dtype=float))
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"oos_returns must be 2D with ≥ 2 rows; got shape {arr.shape}")
    n_days, n_assets = arr.shape

    # Validate portfolio weight shapes.
    weights_arr = [np.asarray(w, dtype=float) for w in pareto_weights]
    if not weights_arr:
        return {"efhv_mean": 0.0, "efhv_std": 0.0,
                "efhv_samples": np.zeros(0, dtype=float)}
    for i, w in enumerate(weights_arr):
        if w.shape != (n_assets,):
            raise ValueError(
                f"weights[{i}] shape {w.shape} mismatches oos_returns "
                f"n_assets={n_assets}"
            )

    samples_hv = np.empty(n_samples, dtype=float)
    for e in range(n_samples):
        idx = rng.integers(0, n_days, n_days)
        sample = arr[idx]
        mu = sample.mean(axis=0)
        # ddof=1 sample covariance; fall back to ddof=0 if n_days==1
        # (the n_days < 2 guard above prevents this, but defensive).
        cov = np.cov(sample, rowvar=False, ddof=1)
        points = [
            (float(w @ cov @ w), float(mu @ w))
            for w in weights_arr
        ]
        samples_hv[e] = hypervolume_2d(points, z_ref)

    return {
        "efhv_mean": float(samples_hv.mean()),
        "efhv_std": float(samples_hv.std(ddof=1)) if n_samples >= 2 else 0.0,
        "efhv_samples": samples_hv,
    }


def compute_per_portfolio_efhv(pareto_weights: list[np.ndarray],
                                oos_returns: pd.DataFrame,
                                n_samples: int = 1000,
                                z_ref: tuple[float, float] = (0.2, 0.0),
                                rng: np.random.Generator | None = None) -> np.ndarray:
    """W17-4: per-portfolio expected single-point HV against z_ref.

    Implements thesis §6.4 Eq 6.42 (AMFC selection) on the OOS side:
    for each portfolio i in pareto_weights, return E[Hypv((u_i^T Σ̂ u_i,
    μ̂^T u_i))] averaged over n_samples MC bootstrap scenarios.

    The "single-point HV" against z_ref = (risk_max, return_min) is:
        HV_i_e = max(0, μ̂_e^T u_i - z_ref[1]) * max(0, z_ref[0] - u_i^T Σ̂_e u_i)
    i.e. the rectangular area dominated by the single point (var, return)
    against the reference corner (risk_max, return_min). Per-portfolio
    EFHV = mean over e.

    The argmax-index of the returned array is the AMFC per Eq 6.42.
    walk_forward.run_walk_forward uses this argmax as u*_{t-1} for the
    next rolling period (W16-2-CARRY-1 closure: replaces "first
    Pareto-front portfolio" proxy).

    Returns:
        np.ndarray of length len(pareto_weights), each entry is the
        portfolio's expected single-point HV (≥ 0). All-zero indicates
        a degenerate period (all portfolios dominate-or-are-dominated
        by z_ref); caller should fall back to weights[0].
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = (oos_returns.values if isinstance(oos_returns, pd.DataFrame)
            else np.asarray(oos_returns, dtype=float))
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError(f"oos_returns must be 2D with ≥ 2 rows; got shape {arr.shape}")
    n_days, n_assets = arr.shape

    weights_arr = [np.asarray(w, dtype=float) for w in pareto_weights]
    n_portfolios = len(weights_arr)
    if n_portfolios == 0:
        return np.zeros(0, dtype=float)
    for i, w in enumerate(weights_arr):
        if w.shape != (n_assets,):
            raise ValueError(
                f"weights[{i}] shape {w.shape} mismatches oos_returns "
                f"n_assets={n_assets}"
            )

    risk_ref, return_ref = z_ref[0], z_ref[1]
    # Accumulate per-portfolio HV samples across MC bootstrap.
    per_portfolio_hv = np.zeros(n_portfolios, dtype=float)
    for e in range(n_samples):
        idx = rng.integers(0, n_days, n_days)
        sample = arr[idx]
        mu = sample.mean(axis=0)
        cov = np.cov(sample, rowvar=False, ddof=1)
        for i, w in enumerate(weights_arr):
            var = float(w @ cov @ w)
            ret = float(mu @ w)
            # Single-point HV against z_ref = (risk_max, return_min).
            # Dominance corner: portfolio dominates the reference iff
            # ret > return_ref AND var < risk_ref.
            hv = max(0.0, ret - return_ref) * max(0.0, risk_ref - var)
            per_portfolio_hv[i] += hv
    return per_portfolio_hv / float(n_samples)
