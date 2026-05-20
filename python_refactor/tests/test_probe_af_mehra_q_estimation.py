"""Unit tests for ``src.probes.probe_af_mehra_q_estimation`` (W22 Probe AF).

Two synthetic-KF generators are used:

1. ``_synthetic_white_innovations`` — directly samples i.i.d. zero-mean
   Gaussian innovations to exercise the autocorrelation maths without
   the confounding factor of a full KF loop.
2. ``_run_2d_const_velocity_kf`` — a textbook 4-state constant-velocity
   KF (position-velocity pair) with ``H = [I_2 | 0_2]``. This is the
   minimal Mehra-observable model and matches the architecture of the
   ASMS KF described in the W22 master backlog.

The Mehra observability requirement (``[H; H*F]`` full rank) is
satisfied by the constant-velocity model whenever ``dt != 0``; the test
``test_q_estimate_returns_nan_for_unobservable_case`` deliberately
breaks observability by feeding ``H`` with a column of zeros and an
``F = I_4`` so that ``[H; H*F]`` collapses to two stacked copies of
``H`` with rank 2 < 4.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.probes.probe_af_mehra_q_estimation import (
    expected_innovation_cov_under_correct_q,
    innovation_autocorrelation,
    q_tuning_diagnostic,
    simplified_q_estimate,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _synthetic_white_innovations(
    *, cov: np.ndarray, n_samples: int, seed: int
) -> list[np.ndarray]:
    """Draw i.i.d. zero-mean Gaussian innovations with covariance ``cov``."""
    rng = np.random.default_rng(seed)
    d = cov.shape[0]
    L = np.linalg.cholesky(cov)
    samples = []
    for _ in range(n_samples):
        z = rng.standard_normal(d)
        samples.append(L @ z)
    return samples


def _synthetic_ar1_innovations(
    *, phi: float, sigma: float, n_samples: int, seed: int
) -> list[np.ndarray]:
    """Generate scalar AR(1) innovations: y_t = phi*y_{t-1} + sigma*e_t."""
    rng = np.random.default_rng(seed)
    samples = []
    prev = 0.0
    for _ in range(n_samples):
        e = rng.normal(0.0, sigma)
        cur = phi * prev + e
        samples.append(np.array([cur]))
        prev = cur
    return samples


def _run_2d_const_velocity_kf(
    *,
    Q_true: float,
    R_true: float,
    Q_assumed: float,
    R_assumed: float,
    dt: float = 1.0,
    n_steps: int,
    seed: int,
) -> dict:
    """Run a 4-state constant-velocity KF with isotropic Q, R.

    State ``x = [px, py, vx, vy]^T``. Dynamics::

        F = [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1,  0],
             [0, 0, 0,  1]]
        H = [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        Q = Q_scalar * I_4
        R = R_scalar * I_2

    Returns the per-step innovation list AND the steady-state P_pred
    snapshot at the final step (so the caller can compute the expected
    innovation covariance under the assumed-Q model).
    """
    F = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    Q_a = Q_assumed * np.eye(4)
    R_a = R_assumed * np.eye(2)
    Q_t = Q_true * np.eye(4)
    R_t = R_true * np.eye(2)

    rng = np.random.default_rng(seed)
    x_true = np.zeros(4)
    x_post = np.zeros(4)
    P_post = np.eye(4)

    L_Q = np.linalg.cholesky(Q_t) if Q_true > 0 else np.zeros((4, 4))
    L_R = np.linalg.cholesky(R_t) if R_true > 0 else np.zeros((2, 2))

    innovations: list[np.ndarray] = []
    P_pred_final = None
    for _ in range(n_steps):
        # True dynamics
        w = L_Q @ rng.standard_normal(4) if Q_true > 0 else np.zeros(4)
        x_true = F @ x_true + w
        v = L_R @ rng.standard_normal(2) if R_true > 0 else np.zeros(2)
        z = H @ x_true + v
        # Predict
        x_pred = F @ x_post
        P_pred = F @ P_post @ F.T + Q_a
        # Innovation
        y = z - H @ x_pred
        innovations.append(y.copy())
        # Update
        S = H @ P_pred @ H.T + R_a
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_post = x_pred + K @ y
        P_post = (np.eye(4) - K @ H) @ P_pred
        P_pred_final = P_pred
    assert P_pred_final is not None
    return {
        "innovations": innovations,
        "F": F,
        "H": H,
        "R_assumed": R_a,
        "P_pred_final": P_pred_final,
    }


# ---------------------------------------------------------------------------
# Autocorrelation sanity
# ---------------------------------------------------------------------------


def test_autocorrelation_lag_zero_is_outer_product_mean():
    """C_0 should equal the symmetrised mean of outer products."""
    rng = np.random.default_rng(11)
    n = 500
    d = 3
    Y = rng.standard_normal((n, d))
    innovations = [Y[i] for i in range(n)]
    C = innovation_autocorrelation(innovations, max_lag=2)
    expected_C0 = 0.5 * ((Y.T @ Y) + (Y.T @ Y).T) / n
    np.testing.assert_allclose(C[0], expected_C0, rtol=1e-12, atol=1e-12)


def test_autocorrelation_white_noise_is_zero_off_diagonal_lag():
    """White innovations: lag-k C_k for k>=1 should be ~0 in Frobenius norm.

    Threshold scales like 1/sqrt(T) per CLT. With T = 4000, d = 2,
    Frobenius norm of C_k is dominated by ~ sqrt(d^2) / sqrt(T) ≈ 0.03.
    """
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    innovations = _synthetic_white_innovations(cov=cov, n_samples=4000, seed=7)
    C = innovation_autocorrelation(innovations, max_lag=5)
    # Lag 0 should be close to cov in magnitude
    np.testing.assert_allclose(C[0], cov, atol=0.08)
    # All higher lags should be near zero
    for k in range(1, 6):
        norm = np.linalg.norm(C[k], ord="fro")
        assert norm < 0.10, f"lag {k} norm {norm} too high for white innovations"


def test_autocorrelation_ar1_innovations_nonzero_lag_one():
    """AR(1) sequence with phi=0.8: C_1 / C_0 should ≈ phi (>> 0)."""
    phi = 0.8
    sigma = 1.0
    innovations = _synthetic_ar1_innovations(
        phi=phi, sigma=sigma, n_samples=4000, seed=13
    )
    C = innovation_autocorrelation(innovations, max_lag=3)
    c0 = float(C[0, 0, 0])
    c1 = float(C[1, 0, 0])
    assert c0 > 0
    # Theoretical: c1/c0 = phi for AR(1)
    ratio = c1 / c0
    assert abs(ratio - phi) < 0.10, (
        f"expected lag-1/lag-0 ratio ≈ {phi}, got {ratio}"
    )
    # And c1 magnitude should be a non-trivial fraction of c0
    assert abs(c1) > 0.3 * abs(c0)


# ---------------------------------------------------------------------------
# Expected covariance formula
# ---------------------------------------------------------------------------


def test_expected_cov_formula_matches_HPH_plus_R():
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    P_pred = np.diag([2.0, 3.0, 5.0, 7.0])
    R = np.array([[0.5, 0.0], [0.0, 0.25]])
    S = expected_innovation_cov_under_correct_q(H, P_pred, R)
    expected = H @ P_pred @ H.T + R
    np.testing.assert_allclose(S, 0.5 * (expected + expected.T), rtol=1e-12)
    np.testing.assert_allclose(S, np.array([[2.5, 0.0], [0.0, 3.25]]), rtol=1e-12)


# ---------------------------------------------------------------------------
# q_tuning_diagnostic — three-way verdict on synthetic KF data
# ---------------------------------------------------------------------------


def test_diagnostic_correctly_tuned_on_synthetic_well_tuned():
    """Q_assumed == Q_true, R_assumed == R_true → CORRECTLY_TUNED."""
    out = _run_2d_const_velocity_kf(
        Q_true=0.01, R_true=0.10, Q_assumed=0.01, R_assumed=0.10,
        n_steps=2500, seed=21,
    )
    # Drop warm-up (filter takes ~50 steps to reach steady state with
    # an identity initial covariance)
    innovations = out["innovations"][200:]
    diag = q_tuning_diagnostic(
        innovations,
        H=out["H"],
        P_pred=out["P_pred_final"],
        R=out["R_assumed"],
        max_lag=4,
    )
    assert diag["diagnostic_str"] == "CORRECTLY_TUNED", (
        f"expected CORRECTLY_TUNED, got {diag['diagnostic_str']} "
        f"(ratio={diag['C_0_ratio']:.3f}, "
        f"norms={diag['normalised_autocorr_norms']})"
    )


def test_diagnostic_under_tuned_on_synthetic_small_q():
    """Q_assumed << Q_true → observed innovations bigger than filter expects.

    Verdict should be UNDER_TUNED_Q or NEEDS_INVESTIGATION (the latter
    is acceptable if whiteness is borderline-violated alongside the
    ratio test).
    """
    out = _run_2d_const_velocity_kf(
        Q_true=0.10, R_true=0.10, Q_assumed=0.001, R_assumed=0.10,
        n_steps=2500, seed=29,
    )
    innovations = out["innovations"][200:]
    diag = q_tuning_diagnostic(
        innovations,
        H=out["H"],
        P_pred=out["P_pred_final"],
        R=out["R_assumed"],
        max_lag=4,
    )
    assert diag["diagnostic_str"] in {"UNDER_TUNED_Q", "NEEDS_INVESTIGATION"}, (
        f"expected UNDER_TUNED_Q or NEEDS_INVESTIGATION, "
        f"got {diag['diagnostic_str']} (ratio={diag['C_0_ratio']:.3f})"
    )
    # Under-tuned Q: observed innovations are larger than expected
    # ⇒ ratio should be > 1 even if not > 1.3
    assert diag["C_0_ratio"] > 1.0, (
        f"under-tuned-Q ratio should be > 1.0, got {diag['C_0_ratio']}"
    )


def test_diagnostic_over_tuned_on_synthetic_large_q():
    """Q_assumed >> Q_true → filter expects bigger innovations than it sees.

    Verdict should be OVER_TUNED_Q or NEEDS_INVESTIGATION.
    """
    out = _run_2d_const_velocity_kf(
        Q_true=0.001, R_true=0.10, Q_assumed=1.0, R_assumed=0.10,
        n_steps=2500, seed=37,
    )
    innovations = out["innovations"][200:]
    diag = q_tuning_diagnostic(
        innovations,
        H=out["H"],
        P_pred=out["P_pred_final"],
        R=out["R_assumed"],
        max_lag=4,
    )
    assert diag["diagnostic_str"] in {"OVER_TUNED_Q", "NEEDS_INVESTIGATION"}, (
        f"expected OVER_TUNED_Q or NEEDS_INVESTIGATION, "
        f"got {diag['diagnostic_str']} (ratio={diag['C_0_ratio']:.3f})"
    )
    # Over-tuned Q: observed innovations are smaller than expected
    # ⇒ ratio should be < 1
    assert diag["C_0_ratio"] < 1.0, (
        f"over-tuned-Q ratio should be < 1.0, got {diag['C_0_ratio']}"
    )


# ---------------------------------------------------------------------------
# simplified_q_estimate — PSD-ness and unobservable case
# ---------------------------------------------------------------------------


def test_q_estimate_returns_psd_matrix():
    """When the system is Mehra-observable and the residual is PSD, the
    estimator should return a PSD Q_hat.
    """
    out = _run_2d_const_velocity_kf(
        Q_true=0.05, R_true=0.10, Q_assumed=0.001, R_assumed=0.10,
        n_steps=3000, seed=43,
    )
    innovations = out["innovations"][200:]
    C = innovation_autocorrelation(innovations, max_lag=3)
    # Use the actual constant-velocity F (Mehra-observable). With F = I_4
    # the observability matrix [H; H*F] = [H; H] has rank 2 < 4 and the
    # estimator (correctly) returns NaN — that's the unobservable case
    # tested separately below.
    F_cv = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    Q_hat = simplified_q_estimate(
        C, F=F_cv, H=out["H"], P_pred=out["P_pred_final"]
    )
    # Should not be NaN
    assert not np.isnan(Q_hat).any(), f"Q_hat unexpectedly NaN: {Q_hat}"
    # Should be PSD (all eigenvalues >= -tol)
    eigvals = np.linalg.eigvalsh(Q_hat)
    tol = 1e-9 * max(abs(np.trace(Q_hat)), 1.0)
    assert (eigvals >= -tol).all(), (
        f"Q_hat eigenvalues {eigvals} contain a strict negative beyond tol"
    )
    # Symmetric within fp precision
    np.testing.assert_allclose(Q_hat, Q_hat.T, atol=1e-12)


def test_q_estimate_returns_nan_for_unobservable_case():
    """Force Mehra-unobservable system: F = I_4 + H drops velocity columns.

    With ``F = I_4`` we have ``H*F = H``, so the observability matrix
    ``[H; H*F]`` has rank 2 (= rank of H), strictly less than n_state=4.
    The estimator should return an all-NaN matrix.
    """
    F_bad = np.eye(4)  # state is "frozen"; velocities never coupled to position
    H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    # Dummy autocorrelation
    rng = np.random.default_rng(101)
    fake_innovs = [rng.standard_normal(2) for _ in range(200)]
    C = innovation_autocorrelation(fake_innovs, max_lag=2)
    P_pred = np.eye(4)
    Q_hat = simplified_q_estimate(C, F=F_bad, H=H, P_pred=P_pred)
    assert np.isnan(Q_hat).all(), (
        f"expected all-NaN Q_hat for unobservable system, got {Q_hat}"
    )
