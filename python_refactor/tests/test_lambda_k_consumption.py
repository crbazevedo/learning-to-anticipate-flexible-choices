"""
W16-1: regression tests for λ^K consumption in
compute_anticipatory_learning_rate per thesis Eq 7.16 + Eq 6.9.

Closes BACKLOG.md H2 + W15-3-CARRY-1 consumption half.

Thesis grounding (verbatim):

§7.2.3 Eq (7.16), p. 146:
    "λ_{t+h} = (1/2)(λ_{t+h}^(H) + λ_{t+h}^(K)), for which the
     anticipation horizon is H = 2 (one-step-ahead prediction)."

§7.1.1 (p. 140):
    "The anticipation factor is controlled by four levels of window
     size (K): K ∈ {0, 1, 2, 3}. For K = 0, we have the myopic
     baseline SMS-EMOA (SMS, in short)..."

§6.1.4 Eq (6.9), p. 119:
    λ^K_{t+h} is defined as the normalized sum of KF squared
    residuals over the historical window of size K.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.anticipatory_learning import (
    AnticipatoryLearning,
    TIPIntegratedAnticipatoryLearning,
)
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    """
    Portfolio.mean_ROI / covariance are CLASS variables cached across
    tests. Without isolation, earlier tests may leave a length-3
    mean_ROI cached and our Solution(num_assets=5) crashes on matmul.
    Reset to None so Solution.__init__ skips compute_efficiency.
    """
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


def _make_solution(num_assets: int = 5,
                   roi: float = 0.05,
                   risk: float = 0.10,
                   prediction_error: float = 0.5,
                   alpha: float = 0.5) -> Solution:
    """Helper: minimal Solution with the fields the rate computation reads."""
    sol = Solution(num_assets=num_assets)
    sol.P.ROI = roi
    sol.P.risk = risk
    sol.prediction_error = prediction_error
    sol.alpha = alpha
    return sol


class TestLambdaKConsumptionK0:
    """K = 0: λ^K must be identically 0 (myopic SMS-EMOA per §7.1.1)."""

    def test_k0_lambda_k_is_zero(self):
        """At K=0, _compute_lambda_k returns exactly 0.0."""
        al = AnticipatoryLearning(window_size=0)
        sol = _make_solution()
        lambda_k = al._compute_lambda_k(
            sol,
            min_error=0.0,
            max_error=1.0,
            min_alpha=0.0,
            max_alpha=1.0,
            current_time=5,
        )
        assert lambda_k == 0.0, "K=0 must give λ^K=0 per §7.1.1 myopic baseline"

    def test_k0_record_kf_residual_noop(self):
        """At K=0, recording residuals must NOT populate the window."""
        al = AnticipatoryLearning(window_size=0)
        for _ in range(10):
            al.record_kf_residual(1.5)
        assert al._kf_residual_window == [], "K=0 must not accumulate residuals"

    def test_k0_combined_lambda_equals_half_lambda_h(self):
        """At K=0, λ = 0.5*(λ^H + 0) = 0.5*λ^H — both arms still average."""
        al = AnticipatoryLearning(window_size=0)
        sol = _make_solution()
        # No TIP calculator → λ^H also defaults to 0
        lam = al.compute_anticipatory_learning_rate(
            sol, min_error=0.0, max_error=1.0,
            min_alpha=0.0, max_alpha=1.0, current_time=5,
        )
        assert lam == 0.0, "K=0 + no TIP → both arms 0 → λ=0"


class TestLambdaKConsumptionKPositive:
    """K > 0: λ^K consumes the residual window per Eq 6.9."""

    def test_kpos_residual_window_truncated_to_k(self):
        """record_kf_residual truncates buffer to length K."""
        K = 3
        al = AnticipatoryLearning(window_size=K)
        for i, val in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
            al.record_kf_residual(val)
        assert al._kf_residual_window == [3.0, 4.0, 5.0], \
            f"window must keep last K={K} residuals, got {al._kf_residual_window}"

    def test_kpos_lambda_k_fires_after_residuals_recorded(self):
        """With K>0 and residuals recorded, λ^K is strictly > 0."""
        al = AnticipatoryLearning(window_size=3)
        sol = _make_solution()
        for _ in range(3):
            al.record_kf_residual(1.5)
        lambda_k = al._compute_lambda_k(
            sol,
            min_error=0.0, max_error=1.0,
            min_alpha=0.0, max_alpha=1.0,
            current_time=5,
        )
        assert lambda_k > 0.0, "Non-zero residuals must produce λ^K > 0"
        assert lambda_k <= 0.5, "λ^K range must be [0, 0.5] to match λ^H"

    def test_kpos_warmup_falls_back_to_traditional_rate(self):
        """K>0 + empty residual buffer → fall back to traditional rate."""
        al = AnticipatoryLearning(window_size=3)
        sol = _make_solution(prediction_error=0.5, alpha=1.0)
        lambda_k = al._compute_lambda_k(
            sol,
            min_error=0.0, max_error=1.0,
            min_alpha=0.0, max_alpha=1.0,
            current_time=5,  # current_time > 0 + K > 0 → rate_upb = 0.5
        )
        # Traditional rate at uncertainty=1.0, accuracy=0.5 →
        # 0.0 + 0.5*1.0*0.5 + 0.5*0.5*0.5 = 0.25 + 0.125 = 0.375
        assert 0.0 < lambda_k <= 0.5, \
            f"Warm-up fallback must be in [0, 0.5], got {lambda_k}"

    def test_kpos_combined_lambda_is_average(self):
        """λ = 0.5*(λ^H + λ^K) holds verbatim per Eq 7.16."""
        al = AnticipatoryLearning(window_size=2)
        sol = _make_solution()
        for _ in range(2):
            al.record_kf_residual(2.0)
        # Compute λ^K alone
        lambda_k = al._compute_lambda_k(
            sol, min_error=0.0, max_error=1.0,
            min_alpha=0.0, max_alpha=1.0, current_time=5,
        )
        # Compute λ with no TIP (λ^H = 0)
        lam = al.compute_anticipatory_learning_rate(
            sol, min_error=0.0, max_error=1.0,
            min_alpha=0.0, max_alpha=1.0, current_time=5,
        )
        # Eq 7.16 verbatim: λ = 0.5*(0 + λ^K)
        assert lam == pytest.approx(0.5 * lambda_k, abs=1e-9), \
            f"Eq 7.16: λ must equal 0.5*(λ^H + λ^K); got λ={lam} but expected {0.5 * lambda_k}"


class TestLambdaTracePlumbing:
    """W16-1 trace plumbing (W16-4 builds CSV emit on top)."""

    def test_trace_appended_per_call(self):
        """Each compute_anticipatory_learning_rate call appends one trace row."""
        al = AnticipatoryLearning(window_size=2)
        sol = _make_solution()
        for _ in range(3):
            al.compute_anticipatory_learning_rate(
                sol, min_error=0.0, max_error=1.0,
                min_alpha=0.0, max_alpha=1.0, current_time=5,
                generation=42, solution_rank=7,
            )
        rows = al.get_lambda_trace()
        assert len(rows) == 3
        for r in rows:
            # W17-2 added lambda_k_branch column; verify ⊆ relationship
            # so this test stays robust against future schema growth.
            expected_keys = {
                "period", "generation", "solution_rank",
                "lambda_h", "lambda_k", "lambda", "tip",
                "lambda_k_branch",  # W17-2 addition
            }
            assert set(r.keys()) == expected_keys
            assert r["generation"] == 42
            assert r["solution_rank"] == 7
            assert r["period"] == 5

    def test_reset_trace_clears(self):
        """reset_lambda_trace empties the buffer."""
        al = AnticipatoryLearning(window_size=2)
        sol = _make_solution()
        al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
        )
        assert len(al.get_lambda_trace()) == 1
        al.reset_lambda_trace()
        assert al.get_lambda_trace() == []

    def test_trace_records_lambda_formula(self):
        """Trace row's λ equals 0.5*(λ^H + λ^K) verbatim."""
        al = AnticipatoryLearning(window_size=2)
        sol = _make_solution()
        for _ in range(2):
            al.record_kf_residual(1.0)
        al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
        )
        row = al.get_lambda_trace()[-1]
        assert row["lambda"] == pytest.approx(
            0.5 * (row["lambda_h"] + row["lambda_k"]), abs=1e-9,
        )
