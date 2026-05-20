"""W22 Probe Q-v1: tests for ``AR1AssetPredictor``.

Covers the spec in ``docs/W22-PROBE-Q-V1-AR1.md``:

* construction / shape contract
* observe + buffer rolling
* no-change baseline
* AR(1) recovery on a synthetic AR(1) process with a known rho
* falsification check: AR(1) RMSE < no-change RMSE on AR-generated data
* degenerate cases: constant series, window < 2, NaN handling
"""
from __future__ import annotations

import numpy as np
import pytest

from src.probes.probe_q_ar1_predictor import AR1AssetPredictor


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestInitialization:
    def test_initialization(self):
        p = AR1AssetPredictor(d=5, window_size=60)
        assert p.d == 5
        assert p.window_size == 60
        assert p.n_observed == 0
        # Buffers should be empty per asset.
        for k in range(5):
            assert p.window_length(k) == 0

    def test_default_window(self):
        p = AR1AssetPredictor(d=3)
        assert p.window_size == 60

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError):
            AR1AssetPredictor(d=0)
        with pytest.raises(ValueError):
            AR1AssetPredictor(d=-2)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            AR1AssetPredictor(d=3, window_size=-1)


# --------------------------------------------------------------------- #
# observe                                                               #
# --------------------------------------------------------------------- #


class TestObserve:
    def test_observe_accepts_returns(self):
        p = AR1AssetPredictor(d=4, window_size=10)
        r = np.array([0.01, -0.02, 0.005, 0.0])
        p.observe(r)
        assert p.n_observed == 1
        for k in range(4):
            assert p.window_length(k) == 1

    def test_observe_rejects_wrong_length(self):
        p = AR1AssetPredictor(d=3, window_size=10)
        with pytest.raises(ValueError):
            p.observe(np.array([0.01, 0.02]))  # length 2 != d=3

    def test_observe_rolls_window(self):
        p = AR1AssetPredictor(d=2, window_size=3)
        for i in range(5):
            p.observe(np.array([float(i), -float(i)]))
        # window_size=3 → only last three observations retained.
        assert p.window_length(0) == 3
        assert p.n_observed == 5

    def test_observe_handles_nonfinite(self):
        p = AR1AssetPredictor(d=2, window_size=5)
        p.observe(np.array([np.nan, 0.01]))
        p.observe(np.array([np.inf, 0.02]))
        p.observe(np.array([0.005, 0.03]))
        # Buffer length counts every observe, including NaN entries.
        assert p.window_length(0) == 3
        # No-change should still find the most recent FINITE value.
        nc = p.predict_no_change()
        assert nc[0] == pytest.approx(0.005)
        assert nc[1] == pytest.approx(0.03)


# --------------------------------------------------------------------- #
# no-change baseline                                                    #
# --------------------------------------------------------------------- #


class TestPredictNoChange:
    def test_predict_no_change_returns_last(self):
        p = AR1AssetPredictor(d=3, window_size=10)
        p.observe(np.array([0.01, 0.02, 0.03]))
        p.observe(np.array([0.04, 0.05, 0.06]))
        nc = p.predict_no_change()
        np.testing.assert_allclose(nc, [0.04, 0.05, 0.06])

    def test_predict_no_change_zero_when_empty(self):
        p = AR1AssetPredictor(d=3, window_size=10)
        nc = p.predict_no_change()
        np.testing.assert_allclose(nc, np.zeros(3))


# --------------------------------------------------------------------- #
# AR(1) prediction                                                      #
# --------------------------------------------------------------------- #


def _generate_ar1(rho: float, mu: float, sigma: float, n: int,
                  seed: int) -> np.ndarray:
    """Generate a single-asset AR(1) trajectory of length n."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype=float)
    x[0] = mu + rng.normal(0, sigma)
    for t in range(1, n):
        x[t] = mu + rho * (x[t - 1] - mu) + rng.normal(0, sigma)
    return x


class TestPredictNext:
    def test_predict_next_uses_ar_coefficient(self):
        """With rho close to 1, prediction tracks the last observation."""
        # Build a clean AR(1) with high persistence so estimated rho is
        # close to 1 and prediction ~= last observation.
        series = _generate_ar1(rho=0.95, mu=0.0, sigma=1e-3, n=200, seed=0)
        p = AR1AssetPredictor(d=1, window_size=200)
        for x in series:
            p.observe(np.array([x]))
        pred = p.predict_next()
        last = series[-1]
        # With rho ~= 0.95, mu ~= 0, prediction ~ 0.95 * last.
        assert abs(pred[0] - 0.95 * last) < 0.5 * abs(last) + 1e-4

    def test_ar_recovers_known_ar_process(self):
        """Estimated rho recovers the generating rho within tolerance."""
        true_rho = 0.7
        true_mu = 0.002
        series = _generate_ar1(rho=true_rho, mu=true_mu, sigma=0.01,
                                n=4000, seed=42)
        p = AR1AssetPredictor(d=1, window_size=4000)
        for x in series:
            p.observe(np.array([x]))
        # Back out rho from the prediction:
        # pred = mu + rho * (r_t - mu)  =>  rho = (pred - mu) / (r_t - mu)
        pred = float(p.predict_next()[0])
        # Use the predictor's own mu (window mean) for the inversion.
        # That's by construction what the predictor used.
        mu_hat = float(np.mean(series))
        r_t = float(series[-1])
        if abs(r_t - mu_hat) < 1e-9:
            pytest.skip("degenerate sample: r_t == mu_hat")
        rho_hat = (pred - mu_hat) / (r_t - mu_hat)
        assert abs(rho_hat - true_rho) < 0.05, (
            f"rho_hat={rho_hat:.4f} not within 0.05 of true {true_rho}"
        )

    def test_ar_beats_no_change_on_ar_data(self):
        """AR(1) RMSE < no-change RMSE on AR-generated data (Q-M1).

        Generates 100 one-step-ahead predictions on an AR(1) process
        with rho=0.7. Per the probe falsifier, AR(1) must beat the
        persistence baseline on this synthetic data — otherwise the
        estimator is broken, irrespective of real-market performance.
        """
        true_rho = 0.7
        true_mu = 0.0
        sigma = 0.01
        n_warmup = 200
        n_steps = 100
        full = _generate_ar1(
            rho=true_rho, mu=true_mu, sigma=sigma,
            n=n_warmup + n_steps + 1, seed=7,
        )
        p = AR1AssetPredictor(d=1, window_size=n_warmup)
        for x in full[:n_warmup]:
            p.observe(np.array([x]))

        ar_errs: list[float] = []
        nc_errs: list[float] = []
        for t in range(n_warmup, n_warmup + n_steps):
            pred_ar = float(p.predict_next()[0])
            pred_nc = float(p.predict_no_change()[0])
            actual = float(full[t + 1])
            # `actual` is the realised next return; observe brings the
            # buffer forward by appending the CURRENT t's return.
            ar_errs.append((pred_ar - actual) ** 2)
            nc_errs.append((pred_nc - actual) ** 2)
            p.observe(np.array([full[t]]))

        rmse_ar = float(np.sqrt(np.mean(ar_errs)))
        rmse_nc = float(np.sqrt(np.mean(nc_errs)))
        assert rmse_ar < rmse_nc, (
            f"AR(1) failed to beat no-change on synthetic AR(1) "
            f"(rmse_ar={rmse_ar:.6f}, rmse_nc={rmse_nc:.6f})"
        )


# --------------------------------------------------------------------- #
# Degenerate cases                                                      #
# --------------------------------------------------------------------- #


class TestDegenerateCases:
    def test_handles_constant_series(self):
        """Zero-variance window → predict the constant (= mean)."""
        p = AR1AssetPredictor(d=2, window_size=20)
        c = 0.0042
        for _ in range(20):
            p.observe(np.array([c, -c]))
        pred = p.predict_next()
        # Variance is zero → fallback returns the mean for each asset.
        np.testing.assert_allclose(pred, [c, -c], atol=1e-9)

    def test_handles_short_window(self):
        """window_size < 2 → no-change fallback."""
        p = AR1AssetPredictor(d=2, window_size=1)
        p.observe(np.array([0.01, 0.02]))
        pred = p.predict_next()
        # window_size=1 forces no-change.
        np.testing.assert_allclose(pred, [0.01, 0.02])

    def test_handles_single_observation(self):
        """One observation in a normal window → no-change fallback."""
        p = AR1AssetPredictor(d=3, window_size=10)
        p.observe(np.array([0.1, 0.2, 0.3]))
        pred = p.predict_next()
        np.testing.assert_allclose(pred, [0.1, 0.2, 0.3])

    def test_handles_all_nan_window(self):
        """All-NaN window → no-change fallback (zeros, since no finite r_t)."""
        p = AR1AssetPredictor(d=2, window_size=5)
        for _ in range(5):
            p.observe(np.array([np.nan, np.nan]))
        pred = p.predict_next()
        np.testing.assert_allclose(pred, [0.0, 0.0])

    def test_zero_window_is_always_no_change(self):
        """window_size=0 is valid but degenerate; never extrapolates."""
        p = AR1AssetPredictor(d=2, window_size=0)
        p.observe(np.array([0.1, 0.2]))
        pred = p.predict_next()
        # With buf_len floored at 1, no-change returns the last value.
        np.testing.assert_allclose(pred, [0.1, 0.2])
