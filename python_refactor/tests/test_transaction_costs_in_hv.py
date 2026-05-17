"""
W16-2: regression tests for transaction-cost integration into the
anticipatory HV objective per thesis §7.2 Eqs (7.4)-(7.5) +
Table 7.1 (BACKLOG H1).

Thesis grounding (verbatim):

§7.2 Eqs (7.4)-(7.5), p. 142:
    "z_t|u*_{t-1} = g(u_t, χ_t) + h(u_t, u*_{t-1}),
     where: g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T
     and h(u_t, u*_{t-1}) = (0, h(u_t, u*_{t-1}))^T,
     in which h is a cost function representing all incurring
     transaction fees and commissions."

§7.2.3 Table 7.1, p. 144 — Brazilian SEC fee schedule:

    Traded value         Proportional   Fixed
    < 135.07             0.0%           2.70
    ≥ 135.08, < 498.62   2.0%           0.00
    ≥ 498.63, < 1514.69  1.5%           2.49
    ≥ 1514.70, < 3029.38 1.0%           10.06
    ≥ 3029.39            0.5%           25.21
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio
from src.config.thesis_parameters import ThesisParameters


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    """Reset Portfolio class variables between tests (W16-1 hygiene)."""
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


class TestThesisTable71Brackets:
    """Bracket lookups match thesis Table 7.1 verbatim."""

    def test_brackets_loaded_into_parameters(self):
        """ThesisParameters().__post_init__ populates the brackets list."""
        params = ThesisParameters()
        assert params.THESIS_TABLE_71_BRACKETS is not None
        assert len(params.THESIS_TABLE_71_BRACKETS) == 5

    def test_bracket_lower_bound_lt_135(self):
        """Traded value < 135.07 → 0% proportional + 2.70 fixed."""
        # 100 traded → 100 * 0.0 + 2.70 = 2.70 cost
        weights_new = np.array([0.50, 0.50])
        weights_prev = np.array([0.45, 0.55])  # |Δ| = [0.05, 0.05] × 1000 = 50, 50
        cost = Portfolio.compute_thesis_transaction_cost(
            weights_new, weights_prev, portfolio_value=1000.0,
        )
        # Each asset's traded value = 50 (in bracket 1: 0.0% + 2.70 fixed)
        # Wait — 50 < 135.07 → bracket 0 → 0% + 2.70. Two assets:
        # total_fee = 2*2.70 = 5.40 ; cost_frac = 5.40 / 1000 = 0.0054
        assert cost == pytest.approx(0.0054, abs=1e-6)

    def test_bracket_proportional_active(self):
        """Traded value in bracket 1 [135.08, 498.62) → 2% proportional."""
        # |Δw_1| * V = 200 → bracket 1 (≥ 135.08, < 498.62)
        # Fee = 200 * 0.02 + 0.00 = 4.00
        weights_new = np.array([0.50, 0.50])
        weights_prev = np.array([0.30, 0.70])  # |Δ| = [0.20, 0.20] × 1000 = 200, 200
        cost = Portfolio.compute_thesis_transaction_cost(
            weights_new, weights_prev, portfolio_value=1000.0,
        )
        # Each asset: 200 * 0.02 + 0.00 = 4.00; total = 8.00 / 1000 = 0.008
        assert cost == pytest.approx(0.008, abs=1e-6)

    def test_high_bracket_low_proportional(self):
        """Traded value ≥ 3029.39 → 0.5% proportional + 25.21 fixed."""
        # |Δw_1| * V = 5000 → bracket 4 (≥ 3029.39): 0.5% + 25.21
        # Fee = 5000 * 0.005 + 25.21 = 25.00 + 25.21 = 50.21
        weights_new = np.array([1.00, 0.00])
        weights_prev = np.array([0.50, 0.50])  # |Δ| = [0.50, 0.50] × 10000 = 5000, 5000
        cost = Portfolio.compute_thesis_transaction_cost(
            weights_new, weights_prev, portfolio_value=10000.0,
        )
        # Each asset: 5000 * 0.005 + 25.21 = 50.21; total = 100.42 / 10000 = 0.010042
        assert cost == pytest.approx(0.010042, abs=1e-6)

    def test_zero_churn_zero_cost(self):
        """weights_new == weights_prev → traded_value = 0 → cost = 0."""
        weights = np.array([0.30, 0.40, 0.30])
        cost = Portfolio.compute_thesis_transaction_cost(
            weights, weights, portfolio_value=1000.0,
        )
        assert cost == 0.0

    def test_none_prev_zero_cost(self):
        """previous_weights = None (first period) → cost = 0."""
        weights = np.array([0.30, 0.40, 0.30])
        cost = Portfolio.compute_thesis_transaction_cost(
            weights, None, portfolio_value=1000.0,
        )
        assert cost == 0.0


class TestSMSEMOATxnCostIntegration:
    """W16-2: _evaluate_solution subtracts txn cost from objective ROI."""

    def test_default_no_previous_weights_no_cost_subtraction(self):
        """Without set_previous_weights, objective ROI == gross ROI."""
        sms = SMSEMOA(population_size=4, generations=1)
        sol = Solution(num_assets=5)
        sol.P.investment = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        sol.P.ROI = 0.10
        sol.P.risk = 0.05
        # previous_weights is None by default
        sms._evaluate_solution(sol, data={})
        assert sol.objectives == [0.10, 0.05]
        assert sol.transaction_cost == 0.0
        assert sol.roi_gross == 0.10
        assert sol.roi_net == 0.10

    def test_set_previous_weights_then_objective_is_net(self):
        """With previous_weights set, objective ROI = gross ROI - txn cost."""
        sms = SMSEMOA(population_size=4, generations=1)
        sol = Solution(num_assets=5)
        sol.P.investment = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        sol.P.ROI = 0.10
        sol.P.risk = 0.05

        # u*_{t-1} = uniform [0.5, 0.5, 0, 0, 0]
        prev = np.array([0.50, 0.50, 0.00, 0.00, 0.00])
        sms.set_previous_weights(prev, portfolio_value=1000.0)

        sms._evaluate_solution(sol, data={})

        # |Δ| per asset (× 1000) = [300, 300, 200, 200, 200]
        # Assets 0,1 in bracket 1 (300 ∈ [135.08, 498.62)): 300 * 0.02 + 0 = 6.00
        # Assets 2,3,4 in bracket 1: 200 * 0.02 + 0 = 4.00
        # Total = 2*6.00 + 3*4.00 = 12.00 + 12.00 = 24.00
        # Cost fraction = 24.00 / 1000 = 0.024
        # ROI_net = 0.10 - 0.024 = 0.076
        assert sol.transaction_cost == pytest.approx(0.024, abs=1e-6)
        assert sol.roi_gross == 0.10
        assert sol.roi_net == pytest.approx(0.076, abs=1e-6)
        assert sol.objectives[0] == pytest.approx(0.076, abs=1e-6)
        assert sol.objectives[1] == 0.05  # risk unchanged

    def test_optimizer_sees_cost_low_churn_wins(self):
        """Two solutions with same gross ROI/risk; low-churn one has higher objective ROI."""
        sms = SMSEMOA(population_size=4, generations=1)

        prev = np.array([0.50, 0.50, 0.00])
        sms.set_previous_weights(prev, portfolio_value=1000.0)

        sol_low_churn = Solution(num_assets=3)
        sol_low_churn.P.investment = np.array([0.55, 0.45, 0.00])  # |Δ| = [50, 50, 0]
        sol_low_churn.P.ROI = 0.10
        sol_low_churn.P.risk = 0.05

        sol_high_churn = Solution(num_assets=3)
        sol_high_churn.P.investment = np.array([0.00, 0.00, 1.00])  # |Δ| = [500, 500, 1000]
        sol_high_churn.P.ROI = 0.10
        sol_high_churn.P.risk = 0.05

        sms._evaluate_solution(sol_low_churn, data={})
        sms._evaluate_solution(sol_high_churn, data={})

        assert sol_low_churn.transaction_cost < sol_high_churn.transaction_cost
        assert sol_low_churn.objectives[0] > sol_high_churn.objectives[0], \
            "Low-churn solution must have higher NET ROI (objective)"

    def test_set_previous_weights_none_resets(self):
        """set_previous_weights(None) clears prior u*_{t-1}."""
        sms = SMSEMOA(population_size=4, generations=1)
        sms.set_previous_weights(np.array([0.5, 0.5]), portfolio_value=100.0)
        assert sms.previous_weights is not None
        sms.set_previous_weights(None)
        assert sms.previous_weights is None
