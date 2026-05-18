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
from scipy import stats as _stats  # W22 Option B: standard normal CDF/PDF


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
                      rng: np.random.Generator | None = None,
                      use_closed_form: bool = False,
                      use_closed_form_expectation: bool = False,
                      use_v2_per_front: bool = False) -> dict:
    """Sample-average OOS Future Hypervolume per thesis Eqs 7.10+7.11.

    For each of n_samples MC scenarios e:
      1. Bootstrap-resample oos_returns with replacement (preserves
         multivariate dependence structure)
      2. MLE-fit (μ̂_e, Σ̂_e) on the bootstrap sample
      3. For each portfolio u_i in pareto_weights, compute
         (u_i^T Σ̂_e u_i, μ̂_e^T u_i)
      4. Compute hypervolume of the N-point cloud against z_ref → S_e

    W22 closed-form variant (use_closed_form=True): skips bootstrap MC;
    instead uses single full-window MLE (μ̂, Σ̂) computed once. Returns
    deterministic Ŝ given (oos_returns, pareto_weights, z_ref). This is
    a POINT-ESTIMATE (S(E[μ,Σ])) NOT a true closed-form expectation
    (E[S(μ,Σ)]) — by Jensen these differ for nonlinear HV. For Phase B
    parallel validation: if closed-form Ŝ tracks the MC mean within
    noise, the MC bootstrap was over-sampling for our use; otherwise
    the uncertainty model is material and a true closed-form expected
    Ŝ (Option B per W22 design doc, Black-Scholes-style truncated
    means per portfolio + HV aggregation) is needed.

    n_samples is IGNORED when use_closed_form=True (output is
    deterministic; efhv_std=0.0; efhv_samples=[the single Ŝ value]).

    Returns:
        {
          'efhv_mean': float,   # Ŝ_{t+1} per Eq 7.11
          'efhv_std':  float,   # MC standard deviation across scenarios (0 if closed-form)
          'efhv_samples': np.ndarray  # all E sample HVs (length 1 if closed-form)
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

    if use_closed_form:
        # W22 Option A: closed-form (point-estimate) variant. Single
        # full-window MLE → deterministic Ŝ. Skips bootstrap MC entirely.
        mu = arr.mean(axis=0)
        cov = np.cov(arr, rowvar=False, ddof=1)
        points = [(float(w @ cov @ w), float(mu @ w)) for w in weights_arr]
        hv = hypervolume_2d(points, z_ref)
        return {
            "efhv_mean": float(hv),
            "efhv_std": 0.0,
            "efhv_samples": np.array([hv], dtype=float),
        }

    if use_v2_per_front:
        # W22 Option C: lift v2's per-front Δ_S formula
        # (legacy-cpp-v2/source/asms_emoa.cpp:380+) directly to OOS
        # aggregation. The Python equivalent lives at
        # python_refactor/src/algorithms/sms_emoa.py:572-616.
        #
        # Setup: full-window MLE (μ̂, Σ̂).
        # Per portfolio i: mu_i = μ̂^T u_i (= ROI_i), sigma2_i = u_i^T Σ̂ u_i (= risk_i).
        # Conditional variance estimates (analogous to KF conditional
        # parameters):
        #   var_roi_i = sigma2_i / n_days (asymptotic MLE mean variance)
        #   var_risk_i = 2 * sigma2_i^2 / (n_days - 1) (Wishart approx)
        #
        # Sort portfolios by ROI ASC, apply v2's per-position formula
        # (sms_emoa.py:572-606 branching: first / middle / last):
        #   First (i=0): mean_delta_ROI = ROI_0 - ROI_1
        #                mean_delta_risk = z_ref[0] - risk_0
        #                var_delta_* = sum of variances
        #   Middle:      mean_delta_ROI = ROI_i - ROI_{i+1}
        #                mean_delta_risk = risk_{i-1} - risk_i
        #                var_delta_* = sums
        #   Last (i=N-1): mean_delta_ROI = ROI_N - return_min
        #                 mean_delta_risk = risk_{N-1} - risk_N
        #                 var_delta_* = sums
        # Then:
        #   delta_S_i = (mean_delta_ROI * var_delta_risk
        #                + mean_delta_risk * var_delta_ROI)
        #               / (var_delta_ROI + var_delta_risk)
        # Aggregate Ŝ = sum(delta_S_i)
        risk_max, return_min = z_ref[0], z_ref[1]
        mu_hat = arr.mean(axis=0)
        Sigma_hat = np.cov(arr, rowvar=False, ddof=1)
        if Sigma_hat.ndim == 0:
            Sigma_hat = np.array([[float(Sigma_hat)]])

        # Build (ROI, risk, var_ROI, var_risk) per portfolio.
        rows = []
        for w in weights_arr:
            mu_i = float(mu_hat @ w)
            sigma2_i = float(w @ Sigma_hat @ w)
            sigma2_i = max(sigma2_i, 0.0)
            var_roi = sigma2_i / max(n_days, 1)
            var_risk = 2.0 * sigma2_i * sigma2_i / max(n_days - 1, 1)
            rows.append((mu_i, sigma2_i, var_roi, var_risk))
        # Sort by ROI ASC (matches sms_emoa.py:569: solutions.sort key=ROI)
        rows.sort(key=lambda r: r[0])

        if len(rows) == 0:
            return {"efhv_mean": 0.0, "efhv_std": 0.0,
                    "efhv_samples": np.zeros(0, dtype=float)}
        if len(rows) == 1:
            # Single-portfolio degenerate case: use the bare expected
            # single-point HV vs z_ref (no neighbors to delta against).
            roi, risk, var_roi, var_risk = rows[0]
            mean_delta_roi = roi - return_min
            mean_delta_risk = risk_max - risk
            total_var = var_roi + var_risk
            delta_s = (
                (mean_delta_roi * var_risk + mean_delta_risk * var_roi) / total_var
                if total_var > 0 else 0.0
            )
            return {"efhv_mean": float(delta_s), "efhv_std": 0.0,
                    "efhv_samples": np.array([delta_s], dtype=float)}

        N = len(rows)
        total = 0.0
        for i in range(N):
            roi_i, risk_i, var_roi_i, var_risk_i = rows[i]
            if i == 0:
                roi_next, risk_next, var_roi_next, var_risk_next = rows[i + 1]
                mean_delta_roi = roi_i - roi_next
                mean_delta_risk = risk_max - risk_i
                var_delta_roi = var_roi_i + var_roi_next
                var_delta_risk = var_risk_i
            elif i == N - 1:
                roi_prev, risk_prev, var_roi_prev, var_risk_prev = rows[i - 1]
                mean_delta_roi = roi_i - return_min
                mean_delta_risk = risk_prev - risk_i
                var_delta_roi = var_roi_i
                var_delta_risk = var_risk_prev + var_risk_i
            else:
                roi_prev, risk_prev, var_roi_prev, var_risk_prev = rows[i - 1]
                roi_next, risk_next, var_roi_next, var_risk_next = rows[i + 1]
                mean_delta_roi = roi_i - roi_next
                mean_delta_risk = risk_prev - risk_i
                var_delta_roi = var_roi_i + var_roi_next
                var_delta_risk = var_risk_prev + var_risk_i
            total_var = var_delta_roi + var_delta_risk
            if total_var > 0:
                delta_s = (mean_delta_roi * var_delta_risk
                            + mean_delta_risk * var_delta_roi) / total_var
            else:
                delta_s = 0.0
            total += delta_s

        return {
            "efhv_mean": float(total),
            "efhv_std": 0.0,
            "efhv_samples": np.array([total], dtype=float),
        }

    if use_closed_form_expectation:
        # W22 Option B: TRUE closed-form expected Ŝ via per-portfolio
        # Black-Scholes-style truncated-mean call/put pricing on
        # (ROI, risk) modeled as independent Gaussians.
        #
        # Setup: full-window MLE (μ̂, Σ̂).
        # Per portfolio i:
        #   mu_i = μ̂^T u_i (mean of E[ROI_i])
        #   sigma2_i = u_i^T Σ̂ u_i (point estimate of variance)
        #   Var(ROI_i) ≈ sigma2_i / n_days (asymptotic MLE mean variance)
        #   Var(risk_i) ≈ 2 * sigma2_i^2 / (n_days - 1) (Wishart approx)
        #
        # E[max(0, ROI_i - return_min)] using BS-like formula:
        #   E[max(0, X - K)] = (μ - K) * Φ((μ - K) / σ) + σ * φ((μ - K) / σ)
        # E[max(0, risk_max - risk_i)] using put-like formula:
        #   E[max(0, K - X)] = (K - μ) * Φ((K - μ) / σ) + σ * φ((K - μ) / σ)
        #
        # E[HV_i] (independence) = product of the two truncated means.
        # Aggregate Ŝ = SUM of per-portfolio E[HV_i] (no Pareto overlap
        # correction → over-estimate; see HONEST SCAR in docstring).
        risk_max, return_min = z_ref[0], z_ref[1]
        mu_hat = arr.mean(axis=0)
        Sigma_hat = np.cov(arr, rowvar=False, ddof=1)
        # n_assets=1 → np.cov returns a 0-d array; coerce to 2D for matmul.
        if Sigma_hat.ndim == 0:
            Sigma_hat = np.array([[float(Sigma_hat)]])

        total = 0.0
        for w in weights_arr:
            mu_i = float(mu_hat @ w)
            sigma2_i = float(w @ Sigma_hat @ w)
            sigma2_i = max(sigma2_i, 0.0)
            var_roi = sigma2_i / max(n_days, 1)
            var_risk = 2.0 * sigma2_i * sigma2_i / max(n_days - 1, 1)
            sd_roi = float(np.sqrt(var_roi))
            sd_risk = float(np.sqrt(var_risk))

            # E[max(0, ROI - return_min)]: BS call
            if sd_roi <= 0.0:
                call_term = max(0.0, mu_i - return_min)
            else:
                d_roi = (mu_i - return_min) / sd_roi
                call_term = (mu_i - return_min) * _stats.norm.cdf(d_roi) \
                            + sd_roi * _stats.norm.pdf(d_roi)

            # E[max(0, risk_max - risk)]: BS put on risk
            if sd_risk <= 0.0:
                put_term = max(0.0, risk_max - sigma2_i)
            else:
                d_risk = (risk_max - sigma2_i) / sd_risk
                put_term = (risk_max - sigma2_i) * _stats.norm.cdf(d_risk) \
                           + sd_risk * _stats.norm.pdf(d_risk)

            total += call_term * put_term

        return {
            "efhv_mean": float(total),
            "efhv_std": 0.0,
            "efhv_samples": np.array([total], dtype=float),
        }

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
