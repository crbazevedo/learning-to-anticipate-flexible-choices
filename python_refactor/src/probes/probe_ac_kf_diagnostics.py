"""W22 Probe AC: Kalman-filter NIS/NEES consistency diagnostics.

Standalone module — does NOT touch shared code (``kalman_filter.py``,
``sms_emoa.py``, ``anticipatory_learning.py``). The KF instrumentation
that would call into this module is a *future, separate* operator
decision; this probe ships only the diagnostics and a unit-test
battery so we can falsify the hypothesis::

    AC-H1  The ASMS Kalman filter is properly tuned iff its Normalized
           Innovation Squared (NIS) and Normalized Estimation Error
           Squared (NEES) statistics lie within their chi-squared
           confidence intervals.

Background (Bar-Shalom, Li, Kirubarajan, *Estimation with Applications
to Tracking and Navigation*, Wiley 2001, §5.4 "Consistency of State
Estimators"):

* The innovation ``y_t = z_t - H * x_hat_{t|t-1}`` is, under a correctly
  tuned KF, a zero-mean white sequence with covariance ``S_t``.
* The Normalized Innovation Squared
  ``NIS_t = y_t^T * S_t^{-1} * y_t``
  is therefore chi-squared with ``d_meas`` degrees of freedom, where
  ``d_meas`` is the dimension of the measurement.
* Averaging ``N`` independent NIS samples and multiplying by ``N`` gives
  a chi-squared random variable with ``N * d_meas`` degrees of freedom.
  A two-sided 95% confidence band on the *average* is therefore
  ``[chi2.ppf(alpha/2, N*d_meas) / N, chi2.ppf(1-alpha/2, N*d_meas) / N]``.
* The Normalized Estimation Error Squared
  ``NEES_t = (x_true - x_hat)^T * P_t^{-1} * (x_true - x_hat)``
  is chi-squared with ``d_state`` dof under correct tuning. Without
  ground-truth state we substitute the smoothed estimate (see the
  integration sketch in ``docs/W22-PROBE-AC-KF-DIAGNOSTICS.md``).

Failure modes the test detects:

* **Under-tuned R** (assumed measurement noise too small): innovations
  are squashed by an over-confident ``S_t`` => ``NIS`` too high.
* **Over-tuned R**: ``S_t`` too large => ``NIS`` too low.
* **Under-tuned Q** (process noise): KF over-trusts the prior =>
  innovations grow => NIS high AND NEES high.
* **Mismatched F**: bias in innovation mean breaks whiteness; mean NIS
  drifts outside the band.

Numerical-safety contract
-------------------------
* covariances are symmetrised before inversion (``0.5 * (S + S.T)``)
* ``numpy.linalg.solve`` is preferred over an explicit inverse; on
  ``LinAlgError`` (singular covariance) we fall back to a pseudo-inverse
  via ``numpy.linalg.pinv`` and return the resulting quadratic form
* scalar (1-D) measurements / states are accepted as length-1 arrays
* an empty history list returns ``passes=False`` with sentinel values
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Single-timestep statistics
# ---------------------------------------------------------------------------


def _as_column(v: np.ndarray) -> np.ndarray:
    """Coerce 0-D / 1-D arrays to (d, 1) column vectors."""
    arr = np.atleast_1d(np.asarray(v, dtype=float))
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr
    raise ValueError(f"expected vector, got shape {arr.shape}")


def _as_matrix(m: np.ndarray, d: int) -> np.ndarray:
    """Coerce scalar covariance to (1,1); validate shape (d,d)."""
    arr = np.atleast_2d(np.asarray(m, dtype=float))
    if arr.shape != (d, d):
        raise ValueError(f"covariance shape {arr.shape} does not match vector dim {d}")
    return 0.5 * (arr + arr.T)  # symmetrise


def _quadratic_form(v: np.ndarray, M: np.ndarray) -> float:
    """Compute v^T * M^{-1} * v with a pseudo-inverse fallback."""
    try:
        solved = np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(M) @ v
    return float((v.T @ solved).item())


def compute_nis(innovation: np.ndarray, innovation_cov: np.ndarray) -> float:
    """Single-timestep Normalized Innovation Squared.

    ``NIS_t = y_t^T * S_t^{-1} * y_t``

    Parameters
    ----------
    innovation
        Innovation vector ``y_t = z_t - H * x_hat_{t|t-1}``. Scalar or
        1-D array; coerced to a column vector.
    innovation_cov
        Innovation covariance ``S_t = H * P_{t|t-1} * H^T + R``.

    Returns
    -------
    float
        Non-negative scalar. Chi-squared with ``d_meas`` dof under
        correct tuning.
    """
    y = _as_column(innovation)
    S = _as_matrix(innovation_cov, y.shape[0])
    return _quadratic_form(y, S)


def compute_nees(estimation_error: np.ndarray, state_cov: np.ndarray) -> float:
    """Single-timestep Normalized Estimation Error Squared.

    ``NEES_t = (x_true - x_hat)^T * P_t^{-1} * (x_true - x_hat)``

    Parameters
    ----------
    estimation_error
        Estimation error ``x_true - x_hat_{t|t}``. Without ground truth
        the caller may substitute the smoothed estimate (RTS smoother).
    state_cov
        Posterior state covariance ``P_{t|t}``.

    Returns
    -------
    float
        Non-negative scalar. Chi-squared with ``d_state`` dof under
        correct tuning.
    """
    e = _as_column(estimation_error)
    P = _as_matrix(state_cov, e.shape[0])
    return _quadratic_form(e, P)


# ---------------------------------------------------------------------------
# Confidence interval
# ---------------------------------------------------------------------------


def chi2_ci(dof: int, n_samples: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided confidence interval for the *average* of N chi-squared
    samples each with ``dof`` degrees of freedom.

    Let ``X_i ~ chi2(dof)`` independently for ``i = 1..N``. Then
    ``Y = sum_i X_i ~ chi2(N*dof)``, so the 1-alpha CI on the average
    ``Y / N`` is::

        [chi2.ppf(alpha/2, N*dof) / N, chi2.ppf(1-alpha/2, N*dof) / N]

    Parameters
    ----------
    dof
        Degrees of freedom per sample (measurement dim for NIS, state
        dim for NEES).
    n_samples
        Number of independent samples to average.
    alpha
        Significance level. Default 0.05 -> 95% CI.

    Returns
    -------
    (lower, upper)
        Bounds on the average statistic.
    """
    if dof <= 0:
        raise ValueError(f"dof must be positive, got {dof}")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")
    total_dof = dof * n_samples
    low = chi2.ppf(alpha / 2.0, total_dof) / n_samples
    high = chi2.ppf(1.0 - alpha / 2.0, total_dof) / n_samples
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------


def _consistency_test(history: Sequence[float], dof: int, alpha: float, label: str) -> dict:
    arr = np.asarray(list(history), dtype=float)
    if arr.size == 0:
        return {
            "statistic": label,
            "n_samples": 0,
            "dof": dof,
            "alpha": alpha,
            "mean": float("nan"),
            "expected_mean": float(dof),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "passes": False,
            "reason": "empty history",
        }
    mean = float(arr.mean())
    low, high = chi2_ci(dof, arr.size, alpha=alpha)
    passes = bool(low <= mean <= high)
    if passes:
        reason = "mean inside CI"
    elif mean > high:
        reason = "mean above CI (filter over-confident: noise covariance too small)"
    else:
        reason = "mean below CI (filter under-confident: noise covariance too large)"
    return {
        "statistic": label,
        "n_samples": int(arr.size),
        "dof": int(dof),
        "alpha": float(alpha),
        "mean": mean,
        "expected_mean": float(dof),
        "ci_low": float(low),
        "ci_high": float(high),
        "passes": passes,
        "reason": reason,
    }


def nis_consistency_test(nis_history: list[float], dof: int, alpha: float = 0.05) -> dict:
    """Bar-Shalom averaged-NIS consistency test.

    Parameters
    ----------
    nis_history
        Sequence of single-timestep NIS values.
    dof
        Measurement dimension (the dof of each NIS sample).
    alpha
        Significance level. Default 0.05.

    Returns
    -------
    dict
        ``{statistic, n_samples, dof, alpha, mean, expected_mean,
        ci_low, ci_high, passes, reason}``
    """
    return _consistency_test(nis_history, dof, alpha, label="NIS")


def nees_consistency_test(nees_history: list[float], dof: int, alpha: float = 0.05) -> dict:
    """Bar-Shalom averaged-NEES consistency test.

    Parameters
    ----------
    nees_history
        Sequence of single-timestep NEES values.
    dof
        State dimension (the dof of each NEES sample).
    alpha
        Significance level. Default 0.05.

    Returns
    -------
    dict
        See ``nis_consistency_test``.
    """
    return _consistency_test(nees_history, dof, alpha, label="NEES")


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def extract_innovations_from_residual_window(residual_window: list, S_window: list) -> dict:
    """Compute per-period NIS values from paired (residual, S) windows.

    Convenience for an integration step that has already collected the
    innovation residuals and their covariances during a KF predict/update
    loop. Aligns the two windows by index, computes NIS per period, and
    returns the per-period history alongside summary stats.

    Parameters
    ----------
    residual_window
        Iterable of innovation vectors (one per timestep).
    S_window
        Iterable of innovation covariances aligned with
        ``residual_window``.

    Returns
    -------
    dict
        ``{n_periods, dof, nis_history, mean, min, max}``. ``dof`` is
        inferred from the first innovation's length; an empty input
        returns ``n_periods=0``.
    """
    residuals = list(residual_window)
    covs = list(S_window)
    if len(residuals) != len(covs):
        raise ValueError(
            "residual_window and S_window must have equal length; "
            f"got {len(residuals)} and {len(covs)}"
        )
    if not residuals:
        return {
            "n_periods": 0,
            "dof": 0,
            "nis_history": [],
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    dof = int(_as_column(residuals[0]).shape[0])
    nis_history: list[float] = [compute_nis(r, S) for r, S in zip(residuals, covs)]
    arr = np.asarray(nis_history, dtype=float)
    return {
        "n_periods": int(arr.size),
        "dof": int(dof),
        "nis_history": nis_history,
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
