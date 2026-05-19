"""W22-NC8c regression: cross-period KF state persistence via Portfolio.carried_velocity.

Pre-NC8c, KF state was born/dies per walk-forward period: each new period
created fresh portfolios with x[2:4] = [0, 0] (zero velocity). The KF
prediction x_next = F·x = [x[0], x[1], 0, 0] = persistence by construction.

NC8c: walk_forward.run_walk_forward saves the AMFC portfolio's KF state
after each period and passes it as `previous_kf_state` to the next
period's training. sms_emoa._initialize_population reads it from
`data['previous_kf_state']` and sets `Portfolio.carried_velocity` (class-
level), which `Portfolio.initialize_kalman_filter` consumes when creating
fresh KF states for BOTH initial-pop solutions and offspring (via
_finalize_offspring_objectives).

This test pins the contract: with carried_velocity set, fresh KF state
x[2:4] = carried_velocity (not zero).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution


@pytest.fixture(autouse=True)
def reset_portfolio_state():
    """Reset Portfolio class state between tests to avoid pollution."""
    saved_velocity = Portfolio.carried_velocity
    saved_velocity_cov = Portfolio.carried_velocity_covariance
    yield
    Portfolio.carried_velocity = saved_velocity
    Portfolio.carried_velocity_covariance = saved_velocity_cov


def _init_portfolio_class_stats(n_assets: int = 8) -> None:
    """Set Portfolio class-level statistics so compute_efficiency fires."""
    rng = np.random.default_rng(123)
    returns = rng.normal(loc=0.001, scale=0.01, size=(60, n_assets))
    Portfolio.mean_ROI = Portfolio.estimate_assets_mean_ROI(returns)
    Portfolio.median_ROI = Portfolio.estimate_assets_median_ROI(returns)
    Portfolio.covariance = Portfolio.estimate_covariance(
        Portfolio.mean_ROI, returns
    )
    Portfolio.robust_covariance = Portfolio.covariance


class TestNC8cCarriedVelocity:
    """Pin the NC8c contract: Portfolio.carried_velocity propagates to
    KF state x[2:4] when Portfolio.initialize_kalman_filter fires."""

    def test_default_carried_velocity_is_none_zero_velocity(self):
        """Without carried_velocity set, fresh KF state has zero velocity."""
        _init_portfolio_class_stats(n_assets=8)
        Portfolio.carried_velocity = None  # explicit
        sol = Solution(num_assets=8)
        assert sol.P.kalman_state is not None
        assert float(sol.P.kalman_state.x[2]) == 0.0
        assert float(sol.P.kalman_state.x[3]) == 0.0

    def test_carried_velocity_is_applied_to_fresh_kf_state(self):
        """With carried_velocity set, fresh KF state x[2:4] = carried velocity."""
        _init_portfolio_class_stats(n_assets=8)
        Portfolio.carried_velocity = np.array([0.005, -0.002])
        Portfolio.carried_velocity_covariance = np.array(
            [[100.0, 0.0], [0.0, 100.0]]
        )

        sol = Solution(num_assets=8)
        assert sol.P.kalman_state is not None
        np.testing.assert_allclose(sol.P.kalman_state.x[2], 0.005)
        np.testing.assert_allclose(sol.P.kalman_state.x[3], -0.002)

        # Velocity-block covariance also applied
        np.testing.assert_allclose(sol.P.kalman_state.P[2, 2], 100.0)
        np.testing.assert_allclose(sol.P.kalman_state.P[3, 3], 100.0)

    def test_carried_velocity_makes_x_next_diverge_from_persistence(self):
        """Key NC8c invariant: with carried velocity, x_next != x (KF
        prediction ≠ persistence). Without carry, x_next == x (persistence)."""
        _init_portfolio_class_stats(n_assets=8)

        # WITHOUT carry
        Portfolio.carried_velocity = None
        sol_no_carry = Solution(num_assets=8)
        x_next_no_carry = sol_no_carry.P.kalman_state.F @ sol_no_carry.P.kalman_state.x
        # x_next[0:2] = x[0:2] + x[2:4] = x[0:2] + 0 = persistence
        assert float(x_next_no_carry[0]) == float(sol_no_carry.P.kalman_state.x[0])
        assert float(x_next_no_carry[1]) == float(sol_no_carry.P.kalman_state.x[1])

        # WITH carry (non-zero velocity)
        Portfolio.carried_velocity = np.array([0.005, -0.002])
        Portfolio.carried_velocity_covariance = np.array(
            [[100.0, 0.0], [0.0, 100.0]]
        )
        sol_carry = Solution(num_assets=8)
        x_next_carry = sol_carry.P.kalman_state.F @ sol_carry.P.kalman_state.x
        # x_next[0] = x[0] + 0.005 ≠ x[0]
        assert abs(float(x_next_carry[0]) - float(sol_carry.P.kalman_state.x[0]) - 0.005) < 1e-9
        assert abs(float(x_next_carry[1]) - float(sol_carry.P.kalman_state.x[1]) + 0.002) < 1e-9

    def test_offspring_finalize_inherits_carried_velocity(self):
        """W22-NC8b's `_finalize_offspring_objectives` calls
        Portfolio.initialize_kalman_filter, so offspring also pick up
        carried velocity. This test confirms the offspring path."""
        from src.algorithms.operators import (
            thesis_uniform_crossover,
        )

        _init_portfolio_class_stats(n_assets=8)
        Portfolio.carried_velocity = np.array([0.003, -0.001])
        Portfolio.carried_velocity_covariance = np.array(
            [[50.0, 0.0], [0.0, 50.0]]
        )

        rng = np.random.default_rng(7)
        p1 = Solution(num_assets=8)
        p1.P.investment = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0])
        Portfolio.compute_efficiency(p1.P)
        p2 = Solution(num_assets=8)
        p2.P.investment = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.3])
        Portfolio.compute_efficiency(p2.P)

        c1, c2 = thesis_uniform_crossover(p1, p2, p=0.5, rng=rng)

        for child in (c1, c2):
            assert child.P.kalman_state is not None, (
                "Offspring missing KF state — NC8b finalize did not run"
            )
            # After NC8b finalize + NC8c carry, offspring's KF velocity
            # MUST be the carried value, not the original zero from
            # create_kalman_params default.
            np.testing.assert_allclose(child.P.kalman_state.x[2], 0.003)
            np.testing.assert_allclose(child.P.kalman_state.x[3], -0.001)
