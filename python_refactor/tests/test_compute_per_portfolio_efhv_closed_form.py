"""W22: tests for compute_per_portfolio_efhv use_closed_form flag.

Per W22 calibration smoke finding: per_portfolio_efhv was always using
MC bootstrap regardless of which Ŝ estimator was active in
compute_oos_efhv, producing degenerate "all EFHV are 0/NaN" fallback
warnings when paired with a closed-form Ŝ estimator. Fix: thread the
closed-form flag through compute_per_portfolio_efhv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oos_evaluator import compute_per_portfolio_efhv


def _make_oos_returns(n_days: int = 50, n_assets: int = 5,
                       seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.normal(0.001, 0.01, (n_days, n_assets))
    return pd.DataFrame(arr, columns=[f"a{i}" for i in range(n_assets)])


def _make_pareto_weights(n_portfolios: int = 5, n_assets: int = 5,
                          seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(n_portfolios):
        w = rng.dirichlet(np.ones(n_assets))
        weights.append(w)
    return weights


class TestDefaultFlagFalse:
    """Default behavior preserved (bootstrap MC)."""

    def test_default_uses_mc(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result = compute_per_portfolio_efhv(weights, oos, n_samples=10,
                                              rng=np.random.default_rng(42))
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)
        assert np.all(result >= 0.0)


class TestClosedFormFlag:
    """use_closed_form=True → deterministic per-portfolio single-point HV."""

    def test_closed_form_returns_correct_shape(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result = compute_per_portfolio_efhv(weights, oos,
                                              use_closed_form=True)
        assert result.shape == (5,)
        assert np.all(result >= 0.0)

    def test_closed_form_is_deterministic(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result_a = compute_per_portfolio_efhv(weights, oos,
                                                use_closed_form=True,
                                                rng=np.random.default_rng(1))
        result_b = compute_per_portfolio_efhv(weights, oos,
                                                use_closed_form=True,
                                                rng=np.random.default_rng(999))
        assert np.allclose(result_a, result_b, atol=0)

    def test_closed_form_matches_manual_single_point_hv(self):
        oos = _make_oos_returns(n_days=30, n_assets=3, seed=7)
        weights = [np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.5, 0.3])]
        z_ref = (0.001, 0.0001)  # tight enough to be in-the-money

        result = compute_per_portfolio_efhv(weights, oos,
                                              use_closed_form=True,
                                              z_ref=z_ref)

        # Hand-compute
        arr = oos.values
        mu = arr.mean(axis=0)
        cov = np.cov(arr, rowvar=False, ddof=1)
        expected = []
        for w in weights:
            var = float(w @ cov @ w)
            ret = float(mu @ w)
            hv = max(0.0, ret - z_ref[1]) * max(0.0, z_ref[0] - var)
            expected.append(hv)
        assert np.allclose(result, np.array(expected), atol=1e-15)

    def test_closed_form_argmax_consistent_with_default_mc(self):
        """For low-noise fixture, closed-form argmax should match MC argmax."""
        oos = _make_oos_returns(n_days=200, seed=42)  # plenty of data
        weights = _make_pareto_weights(n_portfolios=10, seed=42)
        closed = compute_per_portfolio_efhv(weights, oos, use_closed_form=True)
        mc = compute_per_portfolio_efhv(weights, oos, n_samples=500,
                                          rng=np.random.default_rng(42))
        # AMFC argmax should agree for sufficient MC samples
        assert int(np.argmax(closed)) == int(np.argmax(mc)) or \
                abs(closed[int(np.argmax(closed))] - mc[int(np.argmax(mc))]) < 0.1 * mc[int(np.argmax(mc))]


class TestEmptyWeights:
    """Empty weights returns empty array."""

    def test_empty_returns_empty(self):
        oos = _make_oos_returns()
        result = compute_per_portfolio_efhv([], oos, use_closed_form=True)
        assert result.shape == (0,)
