"""W20-1: tests for use_v2_anticipative_rate flag (Reading-E experimental flag)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.anticipatory_learning import (
    AnticipatoryLearning,
    TIPIntegratedAnticipatoryLearning,
)
from src.algorithms.multi_horizon_anticipatory import MultiHorizonAnticipatoryLearning
from src.algorithms.solution import Solution
from src.algorithms.temporal_incomparability_probability import (
    TemporalIncomparabilityCalculator,
)
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_m, prev_c = Portfolio.mean_ROI, Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_m
        Portfolio.covariance = prev_c


def _make_solution(num_assets: int = 5) -> Solution:
    sol = Solution(num_assets=num_assets)
    sol.P.ROI = 0.05
    sol.P.risk = 0.10
    sol.prediction_error = 0.5
    sol.alpha = 0.5
    return sol


class _StubTIPCalc:
    """TIP calculator that returns a fixed value (for deterministic test)."""
    def __init__(self, tip_value: float = 0.5):
        self.tip_value = tip_value

    def calculate_tip(self, current, predicted):
        return self.tip_value

    def calculate_anticipatory_learning_rate_tip(self, tip, horizon):
        # Standard Shannon/binary form: (1 - binary_entropy(tip)) / (H-1)
        if tip == 0.0 or tip == 1.0:
            ent = 0.0
        else:
            ent = -(tip * np.log2(tip) + (1.0 - tip) * np.log2(1.0 - tip))
        return (1.0 - ent) / max(1, horizon - 1)


class TestFlagDefaultsFalseAndUsesEq716:
    """Default behavior: use_v2_anticipative_rate=False → Eq 7.16 formula."""

    def test_default_uses_eq716(self):
        al = AnticipatoryLearning(window_size=2)
        assert al.use_v2_anticipative_rate is False

        sol = _make_solution()
        sol_pred = _make_solution()
        # At TIP=0.5, λ^H=0, λ^K via _compute_lambda_k → some value,
        # combined = 0.5*(0 + λ^K)
        tip_stub = _StubTIPCalc(tip_value=0.5)
        for _ in range(2):
            al.record_kf_residual(1.0)
        lam = al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
            tip_calculator=tip_stub, predicted_solution=sol_pred, horizon=2,
        )
        # The trace row should have lambda_h ≈ 0 (since TIP=0.5)
        row = al.get_lambda_trace()[-1]
        assert row["lambda_h"] == pytest.approx(0.0, abs=1e-15)
        # combined = 0.5 * (0 + λ^K)
        assert lam == pytest.approx(0.5 * row["lambda_k"], abs=1e-9)


class TestFlagTrueUsesV2Formula:
    """Flag=True → α = 1 - TIP (v2 monotonic formula)."""

    def test_v2_formula_at_tip_05(self):
        al = AnticipatoryLearning(window_size=2, use_v2_anticipative_rate=True)
        assert al.use_v2_anticipative_rate is True

        sol = _make_solution()
        sol_pred = _make_solution()
        tip_stub = _StubTIPCalc(tip_value=0.5)
        lam = al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
            tip_calculator=tip_stub, predicted_solution=sol_pred, horizon=2,
        )
        # v2 formula: 1 - tip = 1 - 0.5 = 0.5
        assert lam == pytest.approx(0.5, abs=1e-15)

    def test_v2_formula_at_tip_0(self):
        al = AnticipatoryLearning(window_size=2, use_v2_anticipative_rate=True)
        sol = _make_solution()
        sol_pred = _make_solution()
        tip_stub = _StubTIPCalc(tip_value=0.0)
        lam = al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
            tip_calculator=tip_stub, predicted_solution=sol_pred, horizon=2,
        )
        assert lam == pytest.approx(1.0, abs=1e-15)

    def test_v2_formula_at_tip_1(self):
        al = AnticipatoryLearning(window_size=2, use_v2_anticipative_rate=True)
        sol = _make_solution()
        sol_pred = _make_solution()
        tip_stub = _StubTIPCalc(tip_value=1.0)
        lam = al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
            tip_calculator=tip_stub, predicted_solution=sol_pred, horizon=2,
        )
        assert lam == pytest.approx(0.0, abs=1e-15)


class TestFlagForwardsThroughSubclasses:
    """Flag propagates through TIPIntegrated + MultiHorizon constructors."""

    def test_tip_integrated_forwards_flag(self):
        learner = TIPIntegratedAnticipatoryLearning(
            window_size=2, monte_carlo_samples=100,
            use_v2_anticipative_rate=True,
        )
        assert learner.use_v2_anticipative_rate is True

    def test_multi_horizon_forwards_flag(self):
        learner = MultiHorizonAnticipatoryLearning(
            max_horizon=2, monte_carlo_samples=100, window_size=3,
            use_v2_anticipative_rate=True,
        )
        assert learner.use_v2_anticipative_rate is True

    def test_default_is_false_through_subclasses(self):
        learner = MultiHorizonAnticipatoryLearning(
            max_horizon=2, monte_carlo_samples=100, window_size=3,
        )
        assert learner.use_v2_anticipative_rate is False
