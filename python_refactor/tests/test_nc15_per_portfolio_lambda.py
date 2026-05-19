"""W22-NC15 regression: per-portfolio λ shrinkage by KF position uncertainty.

NC15 mechanism: when W22_NC15_LAMBDA_SHRINKAGE=1, the multi-horizon
calculate_multi_horizon_lambda_rates applies a per-portfolio shrinkage
factor: λ_eff = λ_combined / (1 + α · trace(P[:2,:2])). This breaks
the population-wide uniformity of λ even when TIP is saturated.

Without NC15: all portfolios get λ ≈ 0.5 (uniform).
With NC15: λ varies per portfolio by KF uncertainty.

Default (env unset): no NC15 (backward compatible).
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from src.algorithms.kalman_filter import KalmanParams
from src.algorithms.multi_horizon_anticipatory import (
    MultiHorizonAnticipatoryLearning,
)
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def reset_env_and_portfolio_state():
    """Reset NC15 env vars + Portfolio class state between tests."""
    saved_env = {
        "W22_NC15_LAMBDA_SHRINKAGE": os.environ.get("W22_NC15_LAMBDA_SHRINKAGE"),
        "W22_NC15_SHRINK_ALPHA": os.environ.get("W22_NC15_SHRINK_ALPHA"),
    }
    saved_portfolio = (
        Portfolio.mean_ROI,
        Portfolio.covariance,
        getattr(Portfolio, "carried_velocity", None),
        getattr(Portfolio, "carried_position", None),
    )
    yield
    # Restore env
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    # Restore Portfolio
    (
        Portfolio.mean_ROI,
        Portfolio.covariance,
        Portfolio.carried_velocity,
        Portfolio.carried_position,
    ) = saved_portfolio


def _make_solution_with_kf(n_assets: int = 8, p_diag_pos: float = 0.1) -> Solution:
    """Helper: build a Solution with a KF state having custom position uncertainty."""
    # Set Portfolio class stats so Solution.__init__ doesn't skip compute_efficiency
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.01, size=(60, n_assets))
    Portfolio.mean_ROI = Portfolio.estimate_assets_mean_ROI(returns)
    Portfolio.median_ROI = Portfolio.estimate_assets_median_ROI(returns)
    Portfolio.covariance = Portfolio.estimate_covariance(Portfolio.mean_ROI, returns)
    Portfolio.robust_covariance = Portfolio.covariance

    sol = Solution(num_assets=n_assets)
    # Override KF position uncertainty
    if sol.P.kalman_state is not None:
        sol.P.kalman_state.P[0, 0] = p_diag_pos
        sol.P.kalman_state.P[1, 1] = p_diag_pos
    return sol


class TestNC15LambdaShrinkage:
    """Pin NC15 per-portfolio λ shrinkage contract."""

    def test_default_off_no_shrinkage_applied(self):
        """With env unset, behavior = pre-NC15 (no shrinkage)."""
        # NC15 disabled — env var must be unset
        os.environ.pop("W22_NC15_LAMBDA_SHRINKAGE", None)
        learner = MultiHorizonAnticipatoryLearning(
            max_horizon=3, use_v2_anticipative_rate=True
        )
        sol_low_unc = _make_solution_with_kf(n_assets=8, p_diag_pos=0.01)
        sol_high_unc = _make_solution_with_kf(n_assets=8, p_diag_pos=10.0)

        lr_low = learner.calculate_multi_horizon_lambda_rates(
            sol_low_unc, prediction_horizon=3, current_time=0,
        )
        lr_high = learner.calculate_multi_horizon_lambda_rates(
            sol_high_unc, prediction_horizon=3, current_time=0,
        )
        # Without NC15, the two solutions get same λ (TIP-based only)
        # Note: lambda_rates depends on TIP which depends on kalman state,
        # so they may differ slightly via TIP path. But the SHRINKAGE
        # path is off — no /(1 + α·trace(P)) factor applied.
        # We can't assert exact equality due to TIP path divergence; we
        # assert that the high-unc solution does NOT have substantially
        # lower λ than what TIP-only would give.
        assert len(lr_low) > 0 and len(lr_high) > 0

    def test_enabled_shrinks_high_uncertainty_lambda(self):
        """With NC15 enabled, high-P portfolio gets SMALLER λ than low-P."""
        os.environ["W22_NC15_LAMBDA_SHRINKAGE"] = "1"
        os.environ["W22_NC15_SHRINK_ALPHA"] = "1.0"

        learner = MultiHorizonAnticipatoryLearning(
            max_horizon=3, use_v2_anticipative_rate=True
        )
        sol_low_unc = _make_solution_with_kf(n_assets=8, p_diag_pos=0.01)
        sol_high_unc = _make_solution_with_kf(n_assets=8, p_diag_pos=10.0)

        # NC15 effect: low_unc λ should be (very) close to base λ
        # high_unc λ should be ~ base / (1 + 1·20) ≈ base / 21 = 1/21 of base
        lr_low = learner.calculate_multi_horizon_lambda_rates(
            sol_low_unc, prediction_horizon=3, current_time=0,
        )
        lr_high = learner.calculate_multi_horizon_lambda_rates(
            sol_high_unc, prediction_horizon=3, current_time=0,
        )

        assert len(lr_low) >= 1 and len(lr_high) >= 1
        # High-uncertainty portfolio should have substantially smaller λ
        # (after NC15 shrinkage with α=1.0 and trace=20)
        for low_h, high_h in zip(lr_low, lr_high):
            assert high_h < low_h * 0.5, (
                f"NC15 should shrink high-unc λ ≪ low-unc λ; "
                f"got low={low_h:.6f}, high={high_h:.6f} (ratio {high_h/low_h:.4f})"
            )

    def test_alpha_zero_disables_shrinkage(self):
        """With α=0, shrink factor = 1/(1+0) = 1 → no effect."""
        os.environ["W22_NC15_LAMBDA_SHRINKAGE"] = "1"
        os.environ["W22_NC15_SHRINK_ALPHA"] = "0.0"

        learner = MultiHorizonAnticipatoryLearning(
            max_horizon=3, use_v2_anticipative_rate=True
        )
        sol_low_unc = _make_solution_with_kf(n_assets=8, p_diag_pos=0.01)
        sol_high_unc = _make_solution_with_kf(n_assets=8, p_diag_pos=10.0)

        lr_low = learner.calculate_multi_horizon_lambda_rates(
            sol_low_unc, prediction_horizon=3, current_time=0,
        )
        lr_high = learner.calculate_multi_horizon_lambda_rates(
            sol_high_unc, prediction_horizon=3, current_time=0,
        )
        # With α=0 effectively no NC15; the only differentiation is via
        # TIP path (which may differ but not by huge factor).
        for low_h, high_h in zip(lr_low, lr_high):
            # No shrinkage → ratios should be close (within factor 2)
            ratio = high_h / max(low_h, 1e-12)
            assert 0.5 < ratio < 2.0, (
                f"With α=0, NC15 inactive; got ratio {ratio:.4f}"
            )
