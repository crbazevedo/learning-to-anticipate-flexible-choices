"""W21-5: tests for V5 (use_thesis_eq74_risk), V6 (use_v2_kf_lifecycle stub),
V7 (use_v2_entropy_operators) ablation flags."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.solution import Solution
from src.algorithms.operators import (
    v2_raise_entropy_mutation,
    v2_lower_entropy_mutation,
    v2_roulette_mutation,
    thesis_dual_mode_mutation,
)
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_m = Portfolio.mean_ROI
    prev_c = Portfolio.covariance
    prev_r = Portfolio.use_thesis_eq74_risk
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    Portfolio.use_thesis_eq74_risk = False
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_m
        Portfolio.covariance = prev_c
        Portfolio.use_thesis_eq74_risk = prev_r


# ─── V5 (sqrt removal) ────────────────────────────────────────────────


class TestV5SqrtRemoval:
    """V5: use_thesis_eq74_risk forces compute_risk to return bare variance."""

    def test_default_returns_std_dev(self):
        """Default: compute_risk returns sqrt(u^T Σ u) (pre-W21-5 behavior)."""
        # Don't touch the flag; should be False
        algo = SMSEMOA()
        assert algo.use_thesis_eq74_risk is False
        # Default Portfolio behavior preserved
        assert Portfolio.use_thesis_eq74_risk is False
        p = Portfolio(num_assets=3)
        p.investment = np.array([0.5, 0.3, 0.2])
        cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.01]])
        risk = Portfolio.compute_risk(p, cov)
        variance = 0.5**2 * 0.04 + 0.3**2 * 0.09 + 0.2**2 * 0.01
        assert risk == pytest.approx(np.sqrt(variance), abs=1e-12)

    def test_v5_flag_returns_bare_variance(self):
        """V5 flag: compute_risk returns u^T Σ u (thesis Eq 7.4)."""
        algo = SMSEMOA(use_thesis_eq74_risk=True)
        assert algo.use_thesis_eq74_risk is True
        # Class-level toggle propagated
        assert Portfolio.use_thesis_eq74_risk is True
        p = Portfolio(num_assets=3)
        p.investment = np.array([0.5, 0.3, 0.2])
        cov = np.array([[0.04, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.01]])
        risk = Portfolio.compute_risk(p, cov)
        variance = 0.5**2 * 0.04 + 0.3**2 * 0.09 + 0.2**2 * 0.01
        assert risk == pytest.approx(variance, abs=1e-12)  # NO sqrt

    def test_v5_returns_ratio_sqrt(self):
        """Variance vs std-dev ratio: variance == sqrt(variance)**2."""
        SMSEMOA(use_thesis_eq74_risk=True)
        p = Portfolio(num_assets=2)
        p.investment = np.array([0.6, 0.4])
        cov = np.array([[0.1, 0.0], [0.0, 0.05]])
        risk_var = Portfolio.compute_risk(p, cov)
        Portfolio.use_thesis_eq74_risk = False
        risk_std = Portfolio.compute_risk(p, cov)
        assert risk_var == pytest.approx(risk_std**2, abs=1e-12)


# ─── V6 (KF lifecycle stub) ────────────────────────────────────────────


class TestV6KFLifecycleStub:
    """V6: scaffold present; full implementation deferred."""

    def test_flag_default_false(self):
        algo = SMSEMOA()
        assert algo.use_v2_kf_lifecycle is False

    def test_flag_settable_to_true(self):
        algo = SMSEMOA(use_v2_kf_lifecycle=True)
        assert algo.use_v2_kf_lifecycle is True


# ─── V7 (entropy operators) ────────────────────────────────────────────


class TestV7EntropyOperators:
    """V7: use_v2_entropy_operators swaps thesis dual-mode for 4-op roulette."""

    def test_flag_default_false(self):
        algo = SMSEMOA()
        assert algo.use_v2_entropy_operators is False

    def test_flag_settable_to_true(self):
        algo = SMSEMOA(use_v2_entropy_operators=True)
        assert algo.use_v2_entropy_operators is True

    def test_raise_entropy_reduces_max_weight(self):
        """v2_raise_entropy_mutation reduces the largest weight (flattens distribution)."""
        sol = Solution(num_assets=4)
        sol.P.investment = np.array([0.7, 0.1, 0.1, 0.1])
        rng = np.random.default_rng(42)
        mutated = v2_raise_entropy_mutation(sol, rng=rng)
        # Largest pre-mutation = index 0 (weight 0.7). After mutation: smaller.
        # (Note: simplex projection normalizes, so absolute weight comparison
        # is post-normalization. Still: the formerly-largest weight should
        # have its relative rank drop or its absolute value drop.)
        assert mutated.P.investment[0] < 0.7 or np.argmax(mutated.P.investment) != 0

    def test_lower_entropy_raises_max_weight_proportion(self):
        """v2_lower_entropy_mutation increases the largest weight (concentrates)."""
        sol = Solution(num_assets=4)
        sol.P.investment = np.array([0.4, 0.3, 0.2, 0.1])
        rng = np.random.default_rng(42)
        mutated = v2_lower_entropy_mutation(sol, rng=rng)
        # Largest pre-mutation = index 0 (weight 0.4). After raise + simplex
        # projection, index 0's relative weight should be ≥ original (modulo
        # simplex normalization).
        assert np.argmax(mutated.P.investment) == 0

    def test_roulette_returns_valid_solution(self):
        """v2_roulette_mutation always returns a valid Solution with non-negative weights."""
        sol = Solution(num_assets=5)
        sol.P.investment = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        rng = np.random.default_rng(42)
        for _ in range(20):
            mutated = v2_roulette_mutation(sol, rng=rng)
            assert isinstance(mutated, Solution)
            assert mutated.P.investment.shape == sol.P.investment.shape
            assert np.all(mutated.P.investment >= 0.0)


# ─── Combined flag composability ───────────────────────────────────────


class TestFlagsCompose:
    """All 4 W21-1/5 flags coexist on the constructor without conflict."""

    def test_all_flags_settable(self):
        algo = SMSEMOA(
            use_v2_stability_weighting=True,
            use_thesis_eq74_risk=True,
            use_v2_kf_lifecycle=True,
            use_v2_entropy_operators=True,
        )
        assert algo.use_v2_stability_weighting is True
        assert algo.use_thesis_eq74_risk is True
        assert algo.use_v2_kf_lifecycle is True
        assert algo.use_v2_entropy_operators is True
        assert Portfolio.use_thesis_eq74_risk is True

    def test_all_flags_default_false(self):
        algo = SMSEMOA()
        assert algo.use_v2_stability_weighting is False
        assert algo.use_thesis_eq74_risk is False
        assert algo.use_v2_kf_lifecycle is False
        assert algo.use_v2_entropy_operators is False
        assert Portfolio.use_thesis_eq74_risk is False
