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
    """Reset Portfolio class state between tests to avoid pollution.

    Saves+restores carried_velocity, carried_velocity_covariance,
    carried_position, carried_position_covariance (NC8c-v2 additions).
    Also resets Portfolio.mean_ROI/covariance to None to prevent
    `_init_portfolio_class_stats` from one test leaking to others.
    """
    saved = (
        Portfolio.carried_velocity,
        Portfolio.carried_velocity_covariance,
        getattr(Portfolio, "carried_position", None),
        getattr(Portfolio, "carried_position_covariance", None),
        Portfolio.mean_ROI,
        Portfolio.covariance,
    )
    yield
    (
        Portfolio.carried_velocity,
        Portfolio.carried_velocity_covariance,
        Portfolio.carried_position,
        Portfolio.carried_position_covariance,
        Portfolio.mean_ROI,
        Portfolio.covariance,
    ) = saved


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

    def test_nc8c_v2_carried_position_is_applied(self):
        """W22-NC8c-v2: carried_position overrides x[0:2] in fresh KF state."""
        _init_portfolio_class_stats(n_assets=8)
        Portfolio.carried_position = np.array([0.0042, 0.0035])
        Portfolio.carried_position_covariance = np.array(
            [[0.01, 0.0], [0.0, 0.01]]
        )
        sol = Solution(num_assets=8)
        np.testing.assert_allclose(sol.P.kalman_state.x[0], 0.0042)
        np.testing.assert_allclose(sol.P.kalman_state.x[1], 0.0035)
        # Position-block covariance also applied
        np.testing.assert_allclose(sol.P.kalman_state.P[0, 0], 0.01)

    def test_nc8d_predict_before_first_update_consistency(self):
        """W22-NC8d: when carry is active, x_next must equal F @ x and
        P_next must equal F @ P @ F^T. This ensures cross-terms exist in
        P_next so K[2, 0] can be non-zero on the first kalman_update
        → velocity actually learns from observations."""
        _init_portfolio_class_stats(n_assets=8)
        Portfolio.carried_position = np.array([0.001, 0.001])
        Portfolio.carried_position_covariance = np.array(
            [[0.1, 0.0], [0.0, 0.1]]
        )
        Portfolio.carried_velocity = np.array([0.0005, 0.0002])
        Portfolio.carried_velocity_covariance = np.array(
            [[100.0, 0.0], [0.0, 100.0]]
        )

        sol = Solution(num_assets=8)
        kf = sol.P.kalman_state
        # NC8d invariant: x_next = F @ x
        np.testing.assert_array_almost_equal(kf.x_next, kf.F @ kf.x)
        # NC8d invariant: P_next = F @ P @ F^T
        np.testing.assert_array_almost_equal(
            kf.P_next, kf.F @ kf.P @ kf.F.T
        )
        # And P_next has cross-terms (P_next[0, 2] != 0) — this is the
        # KEY enabler for velocity learning on the first kalman_update.
        assert abs(float(kf.P_next[0, 2])) > 0.5, (
            f"NC8d INVARIANT VIOLATED: P_next[0, 2] = {kf.P_next[0, 2]:.4e}, "
            f"expected ≥ 0.5 (≈ carried_velocity_covariance diag = 100). "
            f"Without this cross-term, K[2, 0] = 0 and velocity cannot learn."
        )

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
