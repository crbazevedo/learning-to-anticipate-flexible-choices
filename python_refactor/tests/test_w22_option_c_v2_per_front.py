"""W22 Option C: tests for use_v2_per_front flag in compute_oos_efhv.

Lifts v2's per-front Δ_S formula
(legacy-cpp-v2/source/asms_emoa.cpp:380+, Python equivalent at
src/algorithms/sms_emoa.py:572-616) directly to the OOS aggregation.

Per portfolio i (sorted by ROI ASC):
  delta_S_i = (mean_delta_ROI * var_delta_risk
               + mean_delta_risk * var_delta_ROI)
              / (var_delta_ROI + var_delta_risk)

with first / middle / last branching for the neighbor deltas.
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

from experiments.oos_evaluator import compute_oos_efhv


def _make_oos_returns(n_days: int = 100, n_assets: int = 5,
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


class TestFlagDefaultsFalse:
    """Default behavior: use_v2_per_front=False → MC bootstrap."""

    def test_default_uses_mc(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result = compute_oos_efhv(weights, oos, n_samples=10,
                                    rng=np.random.default_rng(42))
        assert len(result["efhv_samples"]) == 10


class TestFlagOnDeterministic:
    """v2-per-front is deterministic."""

    def test_deterministic_across_rng_seeds(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result_a = compute_oos_efhv(weights, oos,
                                      use_v2_per_front=True,
                                      rng=np.random.default_rng(1))
        result_b = compute_oos_efhv(weights, oos,
                                      use_v2_per_front=True,
                                      rng=np.random.default_rng(999))
        assert result_a["efhv_mean"] == pytest.approx(result_b["efhv_mean"], abs=0)
        assert result_a["efhv_std"] == 0.0


class TestSinglePortfolio:
    """Single-portfolio degenerate case: returns the bare per-portfolio Δ_S."""

    def test_single_portfolio_does_not_crash(self):
        oos = _make_oos_returns()
        weights = [np.full(5, 1.0 / 5)]  # equal-weight portfolio
        result = compute_oos_efhv(weights, oos, use_v2_per_front=True)
        # Should produce a finite scalar
        assert np.isfinite(result["efhv_mean"])
        assert result["efhv_std"] == 0.0


class TestEmptyWeights:
    """Empty pareto_weights returns zero (existing guard in compute_oos_efhv)."""

    def test_empty_returns_zero(self):
        oos = _make_oos_returns()
        result = compute_oos_efhv([], oos, use_v2_per_front=True)
        # The early-empty guard at top of compute_oos_efhv handles this
        assert result["efhv_mean"] == 0.0


class TestThreePortfolioManualV2Formula:
    """Verify the 3-portfolio per-position formula matches a hand-computed v2 reference."""

    def test_three_portfolio_first_middle_last_match_manual(self):
        # 3 portfolios with known weights so we can hand-compute Δ_S
        # using the v2 formula (sms_emoa.py:572-606).
        oos = _make_oos_returns(n_days=50, n_assets=3, seed=99)
        weights = [
            np.array([0.6, 0.3, 0.1]),  # lowest ROI expected
            np.array([0.3, 0.4, 0.3]),
            np.array([0.1, 0.3, 0.6]),  # highest ROI expected (asset 2 has highest mean)
        ]
        result = compute_oos_efhv(weights, oos, use_v2_per_front=True,
                                    z_ref=(0.2, 0.0))

        # Compute manually (same formula as in impl)
        arr = oos.values
        mu_hat = arr.mean(axis=0)
        Sigma_hat = np.cov(arr, rowvar=False, ddof=1)
        n_days = arr.shape[0]
        rows = []
        for w in weights:
            mu_i = float(mu_hat @ w)
            sigma2_i = float(w @ Sigma_hat @ w)
            sigma2_i = max(sigma2_i, 0.0)
            var_roi = sigma2_i / max(n_days, 1)
            var_risk = 2.0 * sigma2_i * sigma2_i / max(n_days - 1, 1)
            rows.append((mu_i, sigma2_i, var_roi, var_risk))
        rows.sort(key=lambda r: r[0])

        # First (i=0):
        roi0, risk0, vr0, vk0 = rows[0]
        roi1, risk1, vr1, vk1 = rows[1]
        roi2, risk2, vr2, vk2 = rows[2]
        # First
        mdr_0 = roi0 - roi1
        mdrr_0 = 0.2 - risk0  # risk_max - risk_0
        vdr_0 = vr0 + vr1
        vdrr_0 = vk0
        ds_0 = (mdr_0 * vdrr_0 + mdrr_0 * vdr_0) / (vdr_0 + vdrr_0)
        # Middle (i=1)
        mdr_1 = roi1 - roi2
        mdrr_1 = risk0 - risk1
        vdr_1 = vr1 + vr2
        vdrr_1 = vk0 + vk1
        ds_1 = (mdr_1 * vdrr_1 + mdrr_1 * vdr_1) / (vdr_1 + vdrr_1)
        # Last (i=2)
        mdr_2 = roi2 - 0.0  # return_min
        mdrr_2 = risk1 - risk2
        vdr_2 = vr2
        vdrr_2 = vk1 + vk2
        ds_2 = (mdr_2 * vdrr_2 + mdrr_2 * vdr_2) / (vdr_2 + vdrr_2)
        expected_total = ds_0 + ds_1 + ds_2

        assert result["efhv_mean"] == pytest.approx(expected_total, rel=1e-12)
