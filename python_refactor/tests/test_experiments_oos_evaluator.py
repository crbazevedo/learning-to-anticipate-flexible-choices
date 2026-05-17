"""
W13-2 regression tests for the OOS future hypervolume evaluator.

Pins the implementation of thesis Eqs (7.10)+(7.11). Includes:
  - hand-computed 2D HV cases (rectangle, staircase, infeasible)
  - deterministic OOS HV against fixed (μ, Σ)
  - MC stability (E=1000 mean within tolerance of deterministic)
  - bootstrap reproducibility under fixed RNG seed
  - thesis z_ref = (0.2, 0.0) sanity
"""

import unittest

import numpy as np
import pandas as pd

from experiments.oos_evaluator import (
    compute_oos_efhv,
    compute_oos_hv_deterministic,
    evaluate_portfolio_under_future,
    fit_future_state,
    hypervolume_2d,
)


# ---------------------------------------------------------------------------
# 2D hypervolume
# ---------------------------------------------------------------------------

class TestHypervolume2D(unittest.TestCase):
    """Hand-computed 2D HV cases."""

    def test_single_point_rectangle(self):
        # Portfolio at (risk=0.05, return=0.1); z_ref = (0.2, 0.0)
        # HV = (0.2 - 0.05) * (0.1 - 0.0) = 0.015
        hv = hypervolume_2d([(0.05, 0.1)], z_ref=(0.2, 0.0))
        self.assertAlmostEqual(hv, 0.015, places=8)

    def test_two_point_staircase(self):
        # Pareto front: (0.05, 0.05), (0.15, 0.10) — truly non-dominated.
        # Lower risk has lower return; higher risk has higher return.
        # z_ref = (0.2, 0.0).
        # Sweep right-to-left:
        #   (0.15, 0.10): width = 0.20 - 0.15 = 0.05; height = 0.10 - 0 = 0.10 → 0.005
        #   (0.05, 0.05): width = 0.15 - 0.05 = 0.10; height = 0.05 - 0 = 0.005 → 0.0005 wait
        # Recompute: 0.10 * 0.05 = 0.005. Total = 0.005 + 0.005 = 0.010.
        hv = hypervolume_2d([(0.05, 0.05), (0.15, 0.10)], z_ref=(0.2, 0.0))
        self.assertAlmostEqual(hv, 0.010, places=8)

    def test_dominated_second_point_collapses_to_first(self):
        # (0.05, 0.10) dominates (0.15, 0.05) on both axes (lower risk +
        # higher return). HV must equal the single-point rectangle.
        hv = hypervolume_2d([(0.05, 0.10), (0.15, 0.05)], z_ref=(0.2, 0.0))
        self.assertAlmostEqual(hv, 0.015, places=8)

    def test_dominated_point_excluded(self):
        # (0.10, 0.05) is dominated by (0.05, 0.10). HV must NOT count it.
        hv_with_dom = hypervolume_2d(
            [(0.05, 0.10), (0.10, 0.05)], z_ref=(0.2, 0.0))
        hv_without = hypervolume_2d([(0.05, 0.10)], z_ref=(0.2, 0.0))
        # Dominated point doesn't add to staircase; HV matches single-point case.
        self.assertAlmostEqual(hv_with_dom, hv_without, places=8)

    def test_infeasible_points_excluded(self):
        # (0.5, 0.1) — risk > z_ref[0] (0.2). (0.05, -0.1) — return < z_ref[1].
        hv = hypervolume_2d(
            [(0.5, 0.1), (0.05, -0.1)], z_ref=(0.2, 0.0))
        self.assertEqual(hv, 0.0)

    def test_empty_input(self):
        self.assertEqual(hypervolume_2d([], z_ref=(0.2, 0.0)), 0.0)


# ---------------------------------------------------------------------------
# fit_future_state + portfolio evaluation
# ---------------------------------------------------------------------------

class TestFitFutureState(unittest.TestCase):
    def test_mle_gaussian_recovers_known_stats(self):
        rng = np.random.default_rng(42)
        true_mu = np.array([0.001, 0.002, -0.0005])
        true_cov = np.array([[0.0004, 0.0001, 0.0],
                              [0.0001, 0.0003, 0.0001],
                              [0.0, 0.0001, 0.0002]])
        n = 5000
        sample = rng.multivariate_normal(true_mu, true_cov, size=n)
        mu_hat, cov_hat = fit_future_state(pd.DataFrame(sample))
        np.testing.assert_allclose(mu_hat, true_mu, atol=5e-4)
        np.testing.assert_allclose(cov_hat, true_cov, atol=5e-5)

    def test_too_few_rows_raises(self):
        with self.assertRaises(ValueError):
            fit_future_state(pd.DataFrame([[0.01, 0.02]]))


class TestEvaluatePortfolio(unittest.TestCase):
    def test_equal_weight_portfolio_under_known_state(self):
        mu = np.array([0.01, 0.02, 0.03])
        cov = np.array([[0.04, 0.0, 0.0],
                         [0.0, 0.01, 0.0],
                         [0.0, 0.0, 0.09]])
        w = np.array([1/3, 1/3, 1/3])
        risk, ret = evaluate_portfolio_under_future(w, mu, cov)
        # Expected risk = (1/9) * (0.04 + 0.01 + 0.09) = 0.01555...
        self.assertAlmostEqual(risk, (0.04 + 0.01 + 0.09) / 9, places=10)
        # Expected return = (0.01 + 0.02 + 0.03) / 3 = 0.02
        self.assertAlmostEqual(ret, 0.02, places=10)


# ---------------------------------------------------------------------------
# Deterministic OOS HV
# ---------------------------------------------------------------------------

class TestDeterministicOOSHV(unittest.TestCase):
    def test_single_portfolio_matches_hypervolume_2d(self):
        mu = np.array([0.01, 0.02, 0.03])
        cov = 0.01 * np.eye(3)
        w = np.array([0.5, 0.3, 0.2])
        # (risk, return) = (0.01 * (0.25+0.09+0.04), 0.5*0.01+0.3*0.02+0.2*0.03)
        #               = (0.0038, 0.017)
        hv_via_eval = compute_oos_hv_deterministic([w], mu, cov, z_ref=(0.2, 0.0))
        hv_direct = hypervolume_2d([(0.0038, 0.017)], z_ref=(0.2, 0.0))
        self.assertAlmostEqual(hv_via_eval, hv_direct, places=10)


# ---------------------------------------------------------------------------
# MC-averaged OOS EFHV
# ---------------------------------------------------------------------------

class TestComputeOOSEFHV(unittest.TestCase):
    def test_bootstrap_reproducible_under_fixed_seed(self):
        rng_data = np.random.default_rng(0)
        oos_returns = pd.DataFrame(
            rng_data.normal(loc=0.0005, scale=0.012, size=(100, 4)))
        weights = [np.array([0.25, 0.25, 0.25, 0.25])]
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        a = compute_oos_efhv(weights, oos_returns, n_samples=50, rng=rng1)
        b = compute_oos_efhv(weights, oos_returns, n_samples=50, rng=rng2)
        self.assertEqual(a["efhv_mean"], b["efhv_mean"])
        np.testing.assert_array_equal(a["efhv_samples"], b["efhv_samples"])

    def test_efhv_mean_within_tolerance_of_deterministic(self):
        """With E=1000 bootstrap samples drawn from a large enough oos
        window, the MC mean should converge to the deterministic HV
        computed against the full-window MLE."""
        rng_data = np.random.default_rng(7)
        n_days = 500
        oos_returns_arr = rng_data.normal(loc=0.001, scale=0.015, size=(n_days, 5))
        oos_returns = pd.DataFrame(oos_returns_arr)

        weights = [
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            np.array([0.4, 0.3, 0.2, 0.1, 0.0]),
            np.array([0.0, 0.5, 0.0, 0.5, 0.0]),
        ]
        # Deterministic baseline.
        mu_full, cov_full = fit_future_state(oos_returns)
        det_hv = compute_oos_hv_deterministic(weights, mu_full, cov_full)
        # MC estimate.
        rng = np.random.default_rng(99)
        result = compute_oos_efhv(weights, oos_returns, n_samples=1000, rng=rng)
        # MC mean should land within ~3 standard errors of the deterministic.
        se = result["efhv_std"] / np.sqrt(1000)
        self.assertLess(abs(result["efhv_mean"] - det_hv), 4 * se + 1e-9,
                         f"MC mean {result['efhv_mean']:.6g} vs deterministic "
                         f"{det_hv:.6g} (3σ tolerance: {3*se:.6g})")

    def test_efhv_returns_dict_with_expected_keys(self):
        rng_data = np.random.default_rng(5)
        oos_returns = pd.DataFrame(rng_data.normal(0, 0.01, size=(50, 3)))
        weights = [np.array([1/3, 1/3, 1/3])]
        result = compute_oos_efhv(weights, oos_returns, n_samples=10,
                                    rng=np.random.default_rng(0))
        self.assertIn("efhv_mean", result)
        self.assertIn("efhv_std", result)
        self.assertIn("efhv_samples", result)
        self.assertEqual(len(result["efhv_samples"]), 10)

    def test_empty_pareto_front_returns_zero(self):
        oos_returns = pd.DataFrame(np.zeros((10, 3)))
        result = compute_oos_efhv([], oos_returns, n_samples=100)
        self.assertEqual(result["efhv_mean"], 0.0)
        self.assertEqual(result["efhv_std"], 0.0)
        self.assertEqual(len(result["efhv_samples"]), 0)

    def test_weight_shape_mismatch_raises(self):
        oos_returns = pd.DataFrame(np.zeros((10, 3)))
        bad_weights = [np.array([0.5, 0.5])]  # only 2 weights for 3 assets
        with self.assertRaises(ValueError):
            compute_oos_efhv(bad_weights, oos_returns, n_samples=10)

    def test_thesis_zref_is_default(self):
        """Per thesis §7.2.3: z_ref = (0.2, 0.0)^T. Verify default."""
        from inspect import signature
        sig = signature(compute_oos_efhv)
        default_zref = sig.parameters["z_ref"].default
        self.assertEqual(default_zref, (0.2, 0.0))


if __name__ == "__main__":
    unittest.main()
