"""Unit tests for ``src.probes.probe_ac_kf_diagnostics`` (W22 Probe AC).

The synthetic-KF scenarios use a 1-D constant-state model

    x_t = x_{t-1} + w_t,    w_t ~ N(0, Q_true)
    z_t = x_t + v_t,         v_t ~ N(0, R_true)

with H = F = 1. The filter assumes ``Q_assumed``, ``R_assumed`` and we
sweep them up/down to demonstrate the NIS/NEES bands triggering.

NIS samples are time-uncorrelated for a tuned filter (the innovation
sequence is white) so the single-realization averaged-NIS test is valid
out of the box. NEES samples are NOT independent across time inside one
realization — the standard fix (Bar-Shalom §5.4) is Monte-Carlo
averaging across M independent realizations at a fixed timestep; we use
that pattern in the NEES tests.

A known caveat documented in ``docs/W22-PROBE-AC-KF-DIAGNOSTICS.md``:
in this minimal 1-D constant-state model, *over-tuned Q* does not push
NEES below the CI because the steady-state filter degenerates to
trusting the measurement (K -> 1, P_post -> R_assumed -> NEES -> 1).
Over-tuned Q failure is exercised against NIS only; the under-tuned Q
case is exercised against NEES.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2
from src.probes.probe_ac_kf_diagnostics import (
    chi2_ci,
    compute_nees,
    compute_nis,
    extract_innovations_from_residual_window,
    nees_consistency_test,
    nis_consistency_test,
)

# ---------------------------------------------------------------------------
# Synthetic-KF helper (1-D constant state, scalar measurement)
# ---------------------------------------------------------------------------


def _run_scalar_kf(
    *,
    Q_true: float,
    R_true: float,
    Q_assumed: float,
    R_assumed: float,
    n_steps: int,
    seed: int,
) -> dict:
    """Run a textbook 1-D constant-state KF and return per-step NIS, NEES.

    The filter has F = H = 1, so its book-keeping is::

        x_pred = x_post                  (prior = previous posterior)
        P_pred = P_post + Q_assumed
        S = P_pred + R_assumed
        K = P_pred / S
        innovation y = z - x_pred
        x_post = x_pred + K * y
        P_post = (1 - K) * P_pred
    """
    rng = np.random.default_rng(seed)
    x_true = 0.0
    x_post = 0.0
    P_post = 1.0  # initial covariance
    nis: list[float] = []
    nees: list[float] = []
    for _ in range(n_steps):
        # True state propagation
        x_true = x_true + rng.normal(0.0, np.sqrt(Q_true))
        z = x_true + rng.normal(0.0, np.sqrt(R_true))
        # Predict
        x_pred = x_post
        P_pred = P_post + Q_assumed
        # Innovation
        S = P_pred + R_assumed
        y = z - x_pred
        nis.append(compute_nis(np.array([y]), np.array([[S]])))
        # Update
        K = P_pred / S
        x_post = x_pred + K * y
        P_post = (1.0 - K) * P_pred
        # NEES uses ground-truth error (this is a *synthetic* test;
        # production wiring would substitute the smoothed estimate)
        err = x_true - x_post
        nees.append(compute_nees(np.array([err]), np.array([[P_post]])))
    return {"nis": nis, "nees": nees}


# ---------------------------------------------------------------------------
# Single-timestep statistic sanity
# ---------------------------------------------------------------------------


def test_nis_chi2_one_for_normalized_innovation():
    """y = 1 unit-stddev innovation, S = 1 => NIS = 1 (one chi-squared "unit")."""
    assert compute_nis(np.array([1.0]), np.array([[1.0]])) == pytest.approx(1.0)


def test_nees_chi2_one_for_normalized_error():
    """e = 1, P = 1 => NEES = 1."""
    assert compute_nees(np.array([1.0]), np.array([[1.0]])) == pytest.approx(1.0)


def test_nis_two_dim_known_quadratic_form():
    """Two-dimensional sanity check using a known quadratic form."""
    y = np.array([1.0, 2.0])
    S = np.array([[2.0, 0.0], [0.0, 4.0]])
    expected = 1.0 / 2.0 + 4.0 / 4.0  # 0.5 + 1.0 = 1.5
    assert compute_nis(y, S) == pytest.approx(expected)


def test_compute_nis_handles_singular_covariance():
    """A singular S falls back to pseudo-inverse without raising."""
    y = np.array([1.0, 0.0])
    S = np.array([[1.0, 0.0], [0.0, 0.0]])
    value = compute_nis(y, S)
    assert np.isfinite(value)
    assert value == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CI sanity
# ---------------------------------------------------------------------------


def test_chi2_ci_known_values():
    """Cross-check chi2_ci against direct scipy.stats.chi2.ppf calls.

    For dof=1 and N=100 the *average*-NIS 95% CI is
    ``[chi2.ppf(0.025, 100)/100, chi2.ppf(0.975, 100)/100]``.
    """
    low, high = chi2_ci(dof=1, n_samples=100, alpha=0.05)
    assert low == pytest.approx(chi2.ppf(0.025, 100) / 100.0)
    assert high == pytest.approx(chi2.ppf(0.975, 100) / 100.0)
    # the average-NIS band brackets the per-sample mean (1.0)
    assert low < 1.0 < high


def test_chi2_ci_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        chi2_ci(dof=0, n_samples=10)
    with pytest.raises(ValueError):
        chi2_ci(dof=1, n_samples=0)
    with pytest.raises(ValueError):
        chi2_ci(dof=1, n_samples=10, alpha=0.0)
    with pytest.raises(ValueError):
        chi2_ci(dof=1, n_samples=10, alpha=1.0)


def test_chi2_ci_narrows_with_more_samples():
    """More samples ⇒ tighter band around the dof mean."""
    low_small, high_small = chi2_ci(dof=1, n_samples=20)
    low_large, high_large = chi2_ci(dof=1, n_samples=500)
    assert (high_large - low_large) < (high_small - low_small)
    for low, high in [(low_small, high_small), (low_large, high_large)]:
        assert low < 1.0 < high


# ---------------------------------------------------------------------------
# Consistency tests on a tuned filter
# ---------------------------------------------------------------------------


def test_nis_consistency_passes_on_tuned_kf():
    """Q_assumed = Q_true and R_assumed = R_true ⇒ averaged NIS in CI."""
    out = _run_scalar_kf(
        Q_true=0.01,
        R_true=0.1,
        Q_assumed=0.01,
        R_assumed=0.1,
        n_steps=500,
        seed=20260520,
    )
    verdict = nis_consistency_test(out["nis"], dof=1)
    assert verdict["passes"], verdict


def _monte_carlo_nees_at_final_step(
    *,
    Q_true: float,
    R_true: float,
    Q_assumed: float,
    R_assumed: float,
    n_steps: int,
    n_runs: int,
    base_seed: int,
) -> list[float]:
    """Collect the final-step NEES from ``n_runs`` independent realizations.

    Cross-realization samples are i.i.d. (each uses an independent RNG
    stream) so the resulting list satisfies the assumptions of the
    averaged chi-squared CI.
    """
    samples = []
    for r in range(n_runs):
        out = _run_scalar_kf(
            Q_true=Q_true,
            R_true=R_true,
            Q_assumed=Q_assumed,
            R_assumed=R_assumed,
            n_steps=n_steps,
            seed=base_seed + r,
        )
        samples.append(out["nees"][-1])
    return samples


def test_nees_consistency_passes_on_tuned_kf():
    """Tuned filter ⇒ Monte-Carlo NEES at final timestep in CI."""
    samples = _monte_carlo_nees_at_final_step(
        Q_true=0.01,
        R_true=0.1,
        Q_assumed=0.01,
        R_assumed=0.1,
        n_steps=50,
        n_runs=300,
        base_seed=20260520,
    )
    verdict = nees_consistency_test(samples, dof=1)
    assert verdict["passes"], verdict


# ---------------------------------------------------------------------------
# Mis-tuned filter — NIS falls outside CI
# ---------------------------------------------------------------------------


def test_nis_consistency_fails_on_under_tuned_R():
    """R_assumed too small ⇒ S over-confident ⇒ NIS mean too high."""
    out = _run_scalar_kf(
        Q_true=0.01,
        R_true=0.1,
        Q_assumed=0.01,
        R_assumed=0.001,  # 100x too small
        n_steps=500,
        seed=20260520,
    )
    verdict = nis_consistency_test(out["nis"], dof=1)
    assert not verdict["passes"], verdict
    assert verdict["mean"] > verdict["ci_high"], verdict


def test_nis_consistency_fails_on_over_tuned_R():
    """R_assumed too large ⇒ S inflated ⇒ NIS mean too low."""
    out = _run_scalar_kf(
        Q_true=0.01,
        R_true=0.1,
        Q_assumed=0.01,
        R_assumed=10.0,  # 100x too large
        n_steps=500,
        seed=20260520,
    )
    verdict = nis_consistency_test(out["nis"], dof=1)
    assert not verdict["passes"], verdict
    assert verdict["mean"] < verdict["ci_low"], verdict


def test_nees_consistency_fails_on_under_tuned_Q():
    """Q_assumed too small ⇒ P_post under-states uncertainty ⇒ NEES too high.

    Uses Monte-Carlo NEES at the final timestep (i.i.d. across runs).
    """
    samples = _monte_carlo_nees_at_final_step(
        Q_true=0.05,
        R_true=0.1,
        Q_assumed=1e-5,  # 5000x too small
        R_assumed=0.1,
        n_steps=50,
        n_runs=300,
        base_seed=20260520,
    )
    verdict = nees_consistency_test(samples, dof=1)
    assert not verdict["passes"], verdict
    assert verdict["mean"] > verdict["ci_high"], verdict


def test_nees_consistency_fails_on_under_tuned_R():
    """R_assumed too small in the NEES sense (under-confident P).

    With R_assumed << R_true the filter gain K -> 1 too aggressively,
    so x_post tracks the noisy z and the residual error variance
    exceeds P_post -> averaged NEES > 1 mean.
    """
    samples = _monte_carlo_nees_at_final_step(
        Q_true=0.01,
        R_true=0.5,
        Q_assumed=0.01,
        R_assumed=0.001,  # 500x too small relative to truth
        n_steps=50,
        n_runs=300,
        base_seed=20260520,
    )
    verdict = nees_consistency_test(samples, dof=1)
    assert not verdict["passes"], verdict
    assert verdict["mean"] > verdict["ci_high"], verdict


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def test_extract_innovations_from_residual_window_matches_compute_nis():
    """Per-period NIS via the wrapper matches the elementwise scalar values."""
    out = _run_scalar_kf(
        Q_true=0.01,
        R_true=0.1,
        Q_assumed=0.01,
        R_assumed=0.1,
        n_steps=50,
        seed=42,
    )
    # Rebuild residual + S streams that would have been emitted by the KF
    rng = np.random.default_rng(42)
    x_true = 0.0
    x_post = 0.0
    P_post = 1.0
    Q, R = 0.01, 0.1
    residuals = []
    S_window = []
    for _ in range(50):
        x_true = x_true + rng.normal(0.0, np.sqrt(Q))
        z = x_true + rng.normal(0.0, np.sqrt(R))
        x_pred = x_post
        P_pred = P_post + Q
        S = P_pred + R
        y = z - x_pred
        residuals.append(np.array([y]))
        S_window.append(np.array([[S]]))
        K = P_pred / S
        x_post = x_pred + K * y
        P_post = (1.0 - K) * P_pred
    summary = extract_innovations_from_residual_window(residuals, S_window)
    assert summary["n_periods"] == 50
    assert summary["dof"] == 1
    assert summary["nis_history"] == pytest.approx(out["nis"])
    assert summary["mean"] == pytest.approx(float(np.mean(out["nis"])))


def test_extract_innovations_rejects_misaligned_windows():
    with pytest.raises(ValueError):
        extract_innovations_from_residual_window(
            [np.array([0.0])], [np.array([[1.0]]), np.array([[1.0]])]
        )


def test_extract_innovations_handles_empty_window():
    summary = extract_innovations_from_residual_window([], [])
    assert summary["n_periods"] == 0
    assert summary["nis_history"] == []


def test_consistency_test_empty_history_returns_failure_sentinel():
    verdict = nis_consistency_test([], dof=1)
    assert verdict["passes"] is False
    assert verdict["n_samples"] == 0
