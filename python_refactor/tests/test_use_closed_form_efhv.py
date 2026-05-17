"""W22: tests for use_closed_form_efhv flag in compute_oos_efhv.

Methodology pivot per operator directive 2026-05-17: when flag is True,
compute_oos_efhv skips bootstrap MC and uses a single full-window MLE
(μ̂, Σ̂) → deterministic Ŝ point-estimate. This is S(E[μ,Σ]) rather
than the true closed-form expectation E[S(μ,Σ)] — by Jensen these
differ for nonlinear HV.
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

from experiments.oos_evaluator import compute_oos_efhv, fit_future_state, hypervolume_2d


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


class TestClosedFormFlagDefaultsFalse:
    """Default behavior (use_closed_form=False) preserves MC bootstrap."""

    def test_default_uses_mc(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result = compute_oos_efhv(weights, oos, n_samples=10,
                                    rng=np.random.default_rng(42))
        # MC with n_samples=10 should have non-zero std (probabilistic)
        assert "efhv_mean" in result
        assert "efhv_std" in result
        assert len(result["efhv_samples"]) == 10  # MC produced 10 samples


class TestClosedFormFlagTrue:
    """When flag=True: single MLE → deterministic Ŝ; n_samples ignored."""

    def test_closed_form_returns_single_sample(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result = compute_oos_efhv(weights, oos, n_samples=1000,
                                    use_closed_form=True)
        # Single sample in output (n_samples ignored)
        assert len(result["efhv_samples"]) == 1
        # std = 0 since single point
        assert result["efhv_std"] == 0.0

    def test_closed_form_is_deterministic(self):
        """Same (weights, returns) → same Ŝ regardless of rng seed."""
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result1 = compute_oos_efhv(weights, oos, use_closed_form=True,
                                     rng=np.random.default_rng(1))
        result2 = compute_oos_efhv(weights, oos, use_closed_form=True,
                                     rng=np.random.default_rng(99999))
        result3 = compute_oos_efhv(weights, oos, use_closed_form=True,
                                     rng=None)
        # Identical Ŝ across different RNGs (deterministic per fixture)
        assert result1["efhv_mean"] == pytest.approx(result2["efhv_mean"], abs=0)
        assert result2["efhv_mean"] == pytest.approx(result3["efhv_mean"], abs=0)

    def test_closed_form_matches_manual_full_mle_hv(self):
        """Closed-form Ŝ equals HV of (var, mean) cloud under full-window MLE."""
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        # Compute the expected closed-form result manually.
        mu, cov = fit_future_state(oos)
        points = [(float(w @ cov @ w), float(mu @ w)) for w in weights]
        expected_hv = hypervolume_2d(points, z_ref=(0.2, 0.0))
        result = compute_oos_efhv(weights, oos, use_closed_form=True)
        assert result["efhv_mean"] == pytest.approx(expected_hv, abs=1e-15)


class TestClosedFormVsMC:
    """Closed-form Ŝ vs MC mean Ŝ: should be in the same ballpark for
    reasonable n_samples, but they differ structurally (Jensen)."""

    def test_closed_form_within_mc_distribution(self):
        """Closed-form Ŝ should be within a few stds of the MC mean."""
        oos = _make_oos_returns(n_days=100, seed=42)
        weights = _make_pareto_weights(n_portfolios=10, seed=42)
        closed = compute_oos_efhv(weights, oos, use_closed_form=True)
        mc = compute_oos_efhv(weights, oos, n_samples=500,
                                rng=np.random.default_rng(42))
        # The closed-form point estimate should not be ridiculously far
        # from the MC mean (within 5 stds is loose but safe)
        if mc["efhv_std"] > 0:
            z = abs(closed["efhv_mean"] - mc["efhv_mean"]) / mc["efhv_std"]
            assert z < 5.0, f"closed-form Ŝ={closed['efhv_mean']} is {z:.2f} stds away from MC mean"


class TestEmptyWeights:
    """Edge case: empty pareto_weights returns zero Ŝ + empty samples."""

    def test_empty_returns_zero_closed_form(self):
        oos = _make_oos_returns()
        result = compute_oos_efhv([], oos, use_closed_form=True)
        assert result["efhv_mean"] == 0.0
        assert result["efhv_std"] == 0.0
        assert len(result["efhv_samples"]) == 0
