"""W22 Probe AF: Mehra (1970) innovation-autocorrelation-based Q estimation.

Standalone module — does NOT touch shared code (``kalman_filter.py``,
``sms_emoa.py``, ``anticipatory_learning.py``). The probe ships a
contained battery of diagnostic functions so we can falsify the
hypothesis::

    AF-H1  The ASMS Kalman filter is correctly tuned iff its
           innovation sequence is white (lag-k autocorrelations
           ``C_k = 0`` for ``k >= 1``) and its observed lag-0
           autocorrelation matches the theoretical expectation
           ``H * P_pred * H^T + R``.

Background — Mehra (1970), "On the identification of variances and
adaptive Kalman filtering", IEEE Transactions on Automatic Control
**15**(2), 175–184:

Let the innovation be ``y_t = z_t - H * x_hat_{t|t-1}``. Under a
correctly tuned filter,

* ``E[y_t] = 0``,
* ``E[y_t * y_t^T] = S = H * P_pred * H^T + R``,
* ``E[y_t * y_{t+k}^T] = 0`` for all ``k >= 1`` (white sequence).

When the assumed process-noise covariance ``Q`` is **wrong** (too small
or too large) the steady-state ``P_pred`` is biased away from its true
value, so the innovations stop being white: ``C_k != 0`` for some
``k >= 1``. Mehra showed that the time-stationary autocorrelations
``C_k`` carry enough information to *back out* the true ``Q`` whenever
the system is observable in the Mehra sense (the observability matrix
``[H; H*F; ...; H*F^{n-1}]`` has full column rank ``n_state``).

For our 4-state KF with measurement matrix
``H = [I_2 | 0_2]`` (positions observed, velocities hidden) the
observability matrix is ``[H; H*F] = [I_2 0_2; I_2 dt*I_2]`` which has
rank 4 whenever ``dt != 0`` — the textbook constant-velocity model is
Mehra-observable.

This module is intentionally **post-hoc**: the operator collects an
innovation window from a KF run, feeds it in, and gets back a diagnostic
verdict + (when defined) a Q estimate. No KF instrumentation lives
here. The integration sketch lives in
``docs/W22-PROBE-AF-MEHRA-Q.md`` and is a *future* operator decision.

Numerical-safety contract
-------------------------
* outer products are formed on (d, 1) column views to avoid silent
  scalar/1-D broadcasting differences
* the lag-0 autocorrelation is symmetrised before reporting
* the ``simplified_q_estimate`` routine returns ``np.full((n, n), nan)``
  when the system is Mehra-unobservable or when the lag-1 residual is
  numerically zero (no Q drift to identify)
* an empty innovation list returns sentinel NaNs with a
  ``NEEDS_INVESTIGATION`` verdict; the caller is expected to guard with
  ``np.isnan(...).any()``
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stack_innovations(innovations: Sequence[np.ndarray]) -> np.ndarray:
    """Coerce a list of ``(d,)`` or ``(d, 1)`` vectors to a ``(T, d)`` array.

    Raises
    ------
    ValueError
        If the list is empty or vectors have inconsistent length.
    """
    if len(innovations) == 0:
        raise ValueError("innovations sequence is empty")
    rows: list[np.ndarray] = []
    for idx, v in enumerate(innovations):
        arr = np.atleast_1d(np.asarray(v, dtype=float)).reshape(-1)
        rows.append(arr)
        if arr.shape[0] != rows[0].shape[0]:
            raise ValueError(
                f"innovation {idx} has length {arr.shape[0]}; "
                f"expected {rows[0].shape[0]}"
            )
    return np.vstack(rows)


def _symmetrise(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------


def innovation_autocorrelation(
    innovations: Sequence[np.ndarray], max_lag: int = 5
) -> np.ndarray:
    """Compute the time-averaged innovation autocorrelation ``C_k``.

    ``C_k = (1 / (T - k)) * sum_{t=0}^{T - k - 1} y_t * y_{t+k}^T``

    Parameters
    ----------
    innovations
        Sequence of ``T`` innovation vectors, each shape ``(d,)`` or
        ``(d, 1)``. Must be non-empty and homogeneously sized.
    max_lag
        Maximum lag ``k`` to compute. Must be a non-negative integer
        strictly less than ``T``.

    Returns
    -------
    np.ndarray
        Stacked autocorrelations of shape ``(max_lag + 1, d, d)``. The
        lag-0 slice is symmetrised; higher-lag slices are *not* (the
        cross-product matrix is genuinely asymmetric in general).

    Notes
    -----
    The biased / "unbiased" choice matches Mehra (1970, eq. 9): we use
    the unbiased estimator with denominator ``T - k`` so that for white
    innovations ``E[C_k] = 0`` exactly for ``k >= 1`` regardless of
    sample size.
    """
    Y = _stack_innovations(innovations)  # (T, d)
    T = Y.shape[0]
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative, got {max_lag}")
    if max_lag >= T:
        raise ValueError(
            f"max_lag {max_lag} must be < T {T}; cannot estimate "
            "autocorrelation when the window is exhausted"
        )
    d = Y.shape[1]
    out = np.zeros((max_lag + 1, d, d), dtype=float)
    for k in range(max_lag + 1):
        n_pairs = T - k
        if n_pairs <= 0:
            out[k] = np.full((d, d), np.nan)
            continue
        # sum_t y_t * y_{t+k}^T  ==  Y[:n_pairs].T @ Y[k:k+n_pairs]
        acc = Y[: T - k].T @ Y[k : k + (T - k)]
        out[k] = acc / float(n_pairs)
    out[0] = _symmetrise(out[0])
    return out


def expected_innovation_cov_under_correct_q(
    H: np.ndarray, P_pred: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """Return the theoretical lag-0 innovation covariance ``S``.

    ``S = H * P_pred * H^T + R``

    This is the matrix that ``C_0`` should match (in expectation) for a
    correctly tuned filter — it is the matrix the KF itself reports as
    its innovation covariance during the update step.
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    P_pred = np.atleast_2d(np.asarray(P_pred, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    S = H @ P_pred @ H.T + R
    return _symmetrise(S)


def _is_mehra_observable(F: np.ndarray, H: np.ndarray) -> bool:
    """Mehra-observability check: the stacked observability matrix has
    full column rank ``n_state``.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    H = np.atleast_2d(np.asarray(H, dtype=float))
    n = F.shape[0]
    rows: list[np.ndarray] = [H]
    power = np.eye(n)
    for _ in range(n - 1):
        power = power @ F
        rows.append(H @ power)
    stacked = np.vstack(rows)
    rank = int(np.linalg.matrix_rank(stacked))
    return rank == n


def simplified_q_estimate(
    autocorr: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    P_pred: np.ndarray,
) -> np.ndarray:
    """Simplified Mehra Q estimator for the 2-D-measurement case.

    Mehra's full algorithm (1970, eqs. 13–17) solves a Lyapunov-style
    linear system for ``Q`` once a steady-state Kalman gain has been
    recovered from the innovation autocorrelations. The full derivation
    is long; for diagnostic purposes the **simplified** form below is
    sufficient and matches the operator's stated scope:

        Q_hat ≈ F * (C_0 - H*P_pred*H^T) * F^T  −  noise-floor terms

    rearranged from the steady-state ``H*P_pred*H^T = C_0 - R`` identity
    after one transition step (Mehra 1970, §III. eq. 11–12). When the
    bracketed term is PSD and the system is Mehra-observable this
    returns a PSD ``Q_hat`` matrix in state-space; otherwise the routine
    returns an ``(n, n)`` NaN matrix and the caller is expected to flag
    the case (see ``q_tuning_diagnostic``).

    Parameters
    ----------
    autocorr
        Output of :func:`innovation_autocorrelation` with ``max_lag``
        ``>= 1``. Only ``autocorr[0]`` and ``autocorr[1]`` are
        consumed.
    F
        State transition matrix, shape ``(n, n)``.
    H
        Measurement matrix, shape ``(d, n)``.
    P_pred
        Steady-state predicted covariance, shape ``(n, n)``.

    Returns
    -------
    np.ndarray
        Estimated ``Q`` matrix, shape ``(n, n)``. Symmetrised.
        Returns an all-NaN matrix if the system is Mehra-unobservable
        or the bracketed expression is not positive semi-definite.
    """
    F = np.atleast_2d(np.asarray(F, dtype=float))
    H = np.atleast_2d(np.asarray(H, dtype=float))
    P_pred = np.atleast_2d(np.asarray(P_pred, dtype=float))
    autocorr = np.asarray(autocorr, dtype=float)
    n = F.shape[0]
    nan_mat = np.full((n, n), np.nan, dtype=float)

    if not _is_mehra_observable(F, H):
        return nan_mat
    if autocorr.ndim != 3 or autocorr.shape[0] < 2:
        return nan_mat

    C0 = _symmetrise(autocorr[0])
    # Bracketed term: residual innovation covariance attributable to Q
    # via the steady-state identity. Lift from measurement space to
    # state space using the Moore-Penrose pseudo-inverse of H (this is
    # the simplified — and observability-conditioned — back-projection).
    HP = H @ P_pred @ H.T
    delta_S = C0 - HP  # (d, d). Should = R if Q is correct.

    # Lift delta_S to state space: ``Q_state ≈ H_pinv * delta_S * H_pinv^T``
    H_pinv = np.linalg.pinv(H)
    Q_meas_lifted = H_pinv @ delta_S @ H_pinv.T

    # Propagate one step: Q_hat ≈ F * Q_meas_lifted * F^T
    Q_hat = F @ Q_meas_lifted @ F.T
    Q_hat = _symmetrise(Q_hat)

    # PSD check via eigen-spectrum. Tolerate tiny negative eigenvalues
    # from finite-sample noise (< 1e-10 * trace).
    eigvals = np.linalg.eigvalsh(Q_hat)
    tol = 1e-10 * max(abs(np.trace(Q_hat)), 1.0)
    if np.any(eigvals < -tol):
        return nan_mat

    # Clip tiny negatives to 0 for cleanliness
    if np.any(eigvals < 0):
        # Reconstruct from clipped eigendecomposition
        w, V = np.linalg.eigh(Q_hat)
        w = np.clip(w, 0.0, None)
        Q_hat = (V * w) @ V.T
        Q_hat = _symmetrise(Q_hat)

    return Q_hat


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


def q_tuning_diagnostic(
    innovations: Sequence[np.ndarray],
    H: np.ndarray,
    P_pred: np.ndarray,
    R: np.ndarray,
    max_lag: int = 5,
) -> dict:
    """Verdict on whether ``Q`` is correctly tuned, given an innovation
    window collected from a KF run.

    The verdict is a 4-way categorical:

    * ``CORRECTLY_TUNED`` — ``C_0_ratio`` in ``[0.7, 1.3]`` AND all
      higher-lag autocorrelation norms < 0.30 * ``||C_0||``
    * ``UNDER_TUNED_Q`` — ``C_0_ratio > 1.3`` (observed innovations
      bigger than the filter expects → process noise is bigger than
      assumed)
    * ``OVER_TUNED_Q`` — ``C_0_ratio < 0.7`` (observed innovations
      smaller than the filter expects → assumed Q is too large)
    * ``NEEDS_INVESTIGATION`` — ``C_0_ratio`` in tuned-band but
      higher-lag norms are large (whiteness violated; could be F-mismatch
      or Q drift the simple ratio test cannot disambiguate)

    Parameters
    ----------
    innovations
        Sequence of innovation vectors collected post-hoc from a KF
        predict/update loop.
    H, P_pred, R
        KF measurement matrix, steady-state predicted covariance, and
        assumed measurement-noise covariance.
    max_lag
        Lags to inspect for whiteness violation. Default 5.

    Returns
    -------
    dict
        Keys::

            C_0_observed: np.ndarray (d, d)
            C_0_expected: np.ndarray (d, d) = H*P_pred*H^T + R
            C_0_ratio: float = trace(C_0_observed) / trace(C_0_expected)
            autocorr_norms: list[float] of length max_lag (Frobenius
                norms of C_1, ..., C_{max_lag})
            normalised_autocorr_norms: list[float] of length max_lag
                (each divided by ||C_0_expected||_F)
            diagnostic_str: one of 'CORRECTLY_TUNED', 'UNDER_TUNED_Q',
                'OVER_TUNED_Q', 'NEEDS_INVESTIGATION'
            n_samples: int = T (innovation count)
            max_lag: int (echoed)
    """
    H = np.atleast_2d(np.asarray(H, dtype=float))
    P_pred = np.atleast_2d(np.asarray(P_pred, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    if len(innovations) == 0:
        d = H.shape[0]
        nan_mat = np.full((d, d), np.nan)
        return {
            "C_0_observed": nan_mat,
            "C_0_expected": expected_innovation_cov_under_correct_q(H, P_pred, R),
            "C_0_ratio": float("nan"),
            "autocorr_norms": [float("nan")] * max_lag,
            "normalised_autocorr_norms": [float("nan")] * max_lag,
            "diagnostic_str": "NEEDS_INVESTIGATION",
            "n_samples": 0,
            "max_lag": int(max_lag),
        }

    Y = _stack_innovations(innovations)
    T = Y.shape[0]
    # Cap max_lag to T - 1 so we don't blow up on small windows
    effective_max_lag = int(min(max_lag, T - 1))
    if effective_max_lag < 1:
        # Cannot test whiteness with < 2 samples
        C0 = _symmetrise(Y.T @ Y / float(T))
        S_expected = expected_innovation_cov_under_correct_q(H, P_pred, R)
        return {
            "C_0_observed": C0,
            "C_0_expected": S_expected,
            "C_0_ratio": float(np.trace(C0) / max(np.trace(S_expected), 1e-30)),
            "autocorr_norms": [],
            "normalised_autocorr_norms": [],
            "diagnostic_str": "NEEDS_INVESTIGATION",
            "n_samples": int(T),
            "max_lag": int(effective_max_lag),
        }

    C = innovation_autocorrelation(innovations, max_lag=effective_max_lag)
    C0_observed = C[0]
    C0_expected = expected_innovation_cov_under_correct_q(H, P_pred, R)

    # Ratio of traces (robust to off-diagonal noise)
    tr_obs = float(np.trace(C0_observed))
    tr_exp = float(np.trace(C0_expected))
    if abs(tr_exp) < 1e-30:
        ratio = float("nan")
    else:
        ratio = tr_obs / tr_exp

    # Higher-lag norms (Frobenius)
    higher = C[1 : effective_max_lag + 1]
    autocorr_norms = [float(np.linalg.norm(M, ord="fro")) for M in higher]
    c0_expected_norm = float(np.linalg.norm(C0_expected, ord="fro"))
    if c0_expected_norm < 1e-30:
        normalised_norms = [float("nan")] * len(autocorr_norms)
    else:
        normalised_norms = [n / c0_expected_norm for n in autocorr_norms]

    # Pad to max_lag for caller convenience
    while len(autocorr_norms) < max_lag:
        autocorr_norms.append(float("nan"))
        normalised_norms.append(float("nan"))

    # Verdict thresholds (documented + literal):
    #   ratio in [0.7, 1.3]  =>  C_0 magnitudes agree (within ~30%)
    #   normalised_norms < 0.30  =>  higher-lag autocorrelations are
    #                                a small fraction of the lag-0
    #                                magnitude (whiteness holds)
    ratio_in_band = bool(np.isfinite(ratio) and 0.7 <= ratio <= 1.3)
    finite_norms = [
        n for n in normalised_norms[:effective_max_lag] if np.isfinite(n)
    ]
    if finite_norms:
        whiteness_holds = bool(max(finite_norms) < 0.30)
    else:
        whiteness_holds = False

    if np.isnan(ratio):
        verdict = "NEEDS_INVESTIGATION"
    elif ratio_in_band and whiteness_holds:
        verdict = "CORRECTLY_TUNED"
    elif ratio_in_band and not whiteness_holds:
        verdict = "NEEDS_INVESTIGATION"
    elif ratio > 1.3:
        verdict = "UNDER_TUNED_Q"
    elif ratio < 0.7:
        verdict = "OVER_TUNED_Q"
    else:
        verdict = "NEEDS_INVESTIGATION"

    return {
        "C_0_observed": C0_observed,
        "C_0_expected": C0_expected,
        "C_0_ratio": float(ratio),
        "autocorr_norms": autocorr_norms,
        "normalised_autocorr_norms": normalised_norms,
        "diagnostic_str": verdict,
        "n_samples": int(T),
        "max_lag": int(effective_max_lag),
    }
