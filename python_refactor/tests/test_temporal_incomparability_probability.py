"""
Unit tests for Temporal Incomparability Probability (TIP) Implementation

Tests the implementation of Definition 6.1 and related functionality.
"""

import unittest
import numpy as np
from types import SimpleNamespace

# W1-4 import-style fix (follows W1-2's pattern in test_tip_integration.py
# and W1-3's pattern in test_eq14_integration.py): use the standard
# `from src.algorithms.X` style so pytest's rootdir collection works
# without the legacy sys.path.insert hack. This closes 1 more of the
# W1-3-CARRY-1 collection-error count.
from src.algorithms.temporal_incomparability_probability import TemporalIncomparabilityCalculator


class TestTemporalIncomparabilityProbability(unittest.TestCase):
    """Test cases for TemporalIncomparabilityCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tip_calculator = TemporalIncomparabilityCalculator(monte_carlo_samples=100)
        
        # Create mock solutions for testing
        self.current_solution = self._create_mock_solution(0.1, 0.05)  # 10% ROI, 5% risk
        self.predicted_solution = self._create_mock_solution(0.12, 0.06)  # 12% ROI, 6% risk
        
    def _create_mock_solution(self, roi: float, risk: float):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
        
        return MockSolution(roi, risk)
    
    def test_initialization(self):
        """Test TIP calculator initialization."""
        self.assertEqual(self.tip_calculator.monte_carlo_samples, 100)
        self.assertEqual(len(self.tip_calculator.historical_tips), 0)
        
    def test_calculate_tip_basic(self):
        """Test basic TIP calculation."""
        tip = self.tip_calculator.calculate_tip(
            self.current_solution, self.predicted_solution
        )
        
        # TIP should be between 0 and 1
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
        # Should be stored in history
        self.assertEqual(len(self.tip_calculator.historical_tips), 1)
        self.assertEqual(self.tip_calculator.historical_tips[0], tip)
        
    def test_calculate_tip_with_uncertainty(self):
        """Test TIP calculation with prediction uncertainty."""
        tip = self.tip_calculator.calculate_tip(
            self.current_solution, self.predicted_solution,
            prediction_uncertainty=0.05
        )
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_with_dominance_relationships(self):
        """Test TIP calculation with different dominance relationships."""
        # Case 1: Current dominates predicted
        current_dominates = self._create_mock_solution(0.15, 0.03)  # Higher ROI, lower risk
        predicted_dominated = self._create_mock_solution(0.10, 0.05)
        
        tip1 = self.tip_calculator.calculate_tip(current_dominates, predicted_dominated)
        
        # Case 2: Predicted dominates current
        current_dominated = self._create_mock_solution(0.10, 0.05)
        predicted_dominates = self._create_mock_solution(0.15, 0.03)
        
        tip2 = self.tip_calculator.calculate_tip(current_dominated, predicted_dominates)
        
        # Case 3: Mutually non-dominated
        current_non_dom = self._create_mock_solution(0.12, 0.04)  # Higher ROI, higher risk
        predicted_non_dom = self._create_mock_solution(0.10, 0.03)  # Lower ROI, lower risk
        
        tip3 = self.tip_calculator.calculate_tip(current_non_dom, predicted_non_dom)
        
        # TIP should be higher for non-dominated case
        self.assertGreater(tip3, tip1)
        self.assertGreater(tip3, tip2)
        
    def test_binary_entropy(self):
        """Test binary entropy function."""
        # Test known values
        self.assertAlmostEqual(self.tip_calculator.binary_entropy(0.5), 1.0, places=5)
        self.assertAlmostEqual(self.tip_calculator.binary_entropy(0.0), 0.0, places=5)
        self.assertAlmostEqual(self.tip_calculator.binary_entropy(1.0), 0.0, places=5)
        
        # Test symmetry
        p = 0.3
        entropy_p = self.tip_calculator.binary_entropy(p)
        entropy_1_minus_p = self.tip_calculator.binary_entropy(1.0 - p)
        self.assertAlmostEqual(entropy_p, entropy_1_minus_p, places=5)
        
    def test_anticipatory_learning_rate_calculation(self):
        """Test anticipatory learning rate calculation (Equation 6.6)."""
        tip = 0.5  # Maximum uncertainty
        horizon = 3
        
        learning_rate = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip, horizon)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(learning_rate, 0.0)
        self.assertLessEqual(learning_rate, 1.0)
        
        # Test with different TIP values
        tip_low = 0.1  # Low uncertainty (same entropy as 0.9)
        tip_medium = 0.3  # Medium uncertainty
        tip_high = 0.5  # Maximum uncertainty
        
        lr_low = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip_low, horizon)
        lr_medium = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip_medium, horizon)
        lr_high = self.tip_calculator.calculate_anticipatory_learning_rate_tip(tip_high, horizon)
        
        # Higher TIP (closer to 0.5) should result in lower learning rate
        self.assertLess(lr_high, lr_medium)
        self.assertLess(lr_medium, lr_low)
        
    def test_anticipatory_learning_rate_edge_cases(self):
        """Test edge cases for learning rate calculation."""
        # Horizon = 1 should return 0
        lr_horizon_1 = self.tip_calculator.calculate_anticipatory_learning_rate_tip(0.5, 1)
        self.assertEqual(lr_horizon_1, 0.0)
        
        # Horizon = 2 should work
        lr_horizon_2 = self.tip_calculator.calculate_anticipatory_learning_rate_tip(0.5, 2)
        self.assertGreaterEqual(lr_horizon_2, 0.0)
        self.assertLessEqual(lr_horizon_2, 1.0)
        
    def test_historical_tip_trend(self):
        """Test historical TIP trend calculation."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        trend = self.tip_calculator.get_historical_tip_trend()
        
        # Should be positive (increasing trend)
        self.assertGreater(trend, 0.0)
        
        # Test with decreasing trend
        self.tip_calculator.historical_tips = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        trend_decreasing = self.tip_calculator.get_historical_tip_trend()
        self.assertLess(trend_decreasing, 0.0)
        
    def test_average_tip(self):
        """Test average TIP calculation."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        avg_tip = self.tip_calculator.get_average_tip(window_size=5)
        expected_avg = np.mean([0.3, 0.4, 0.5, 0.6, 0.7])
        
        self.assertAlmostEqual(avg_tip, expected_avg, places=5)
        
        # Test with smaller window
        avg_tip_small = self.tip_calculator.get_average_tip(window_size=3)
        expected_avg_small = np.mean([0.5, 0.6, 0.7])
        
        self.assertAlmostEqual(avg_tip_small, expected_avg_small, places=5)
        
    def test_tip_statistics(self):
        """Test TIP statistics calculation."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        stats = self.tip_calculator.get_tip_statistics()
        
        self.assertEqual(stats['count'], 5)
        self.assertAlmostEqual(stats['mean'], 0.5, places=5)
        self.assertGreaterEqual(stats['min'], 0.0)
        self.assertLessEqual(stats['max'], 1.0)
        self.assertGreaterEqual(stats['std'], 0.0)
        
    def test_reset_history(self):
        """Test history reset functionality."""
        # Add some historical TIP values
        self.tip_calculator.historical_tips = [0.3, 0.4, 0.5]
        
        # Reset
        self.tip_calculator.reset_history()
        
        self.assertEqual(len(self.tip_calculator.historical_tips), 0)
        
    def test_tip_with_kalman_covariance(self):
        """Test TIP calculation with Kalman filter covariance."""
        # Create solutions with mock Kalman states
        current_sol = self._create_mock_solution(0.1, 0.05)
        predicted_sol = self._create_mock_solution(0.12, 0.06)
        
        # Mock Kalman state with covariance
        class MockKalmanState:
            def __init__(self):
                self.P = np.eye(4) * 0.01
        
        current_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state.P = np.eye(4) * 0.02
        
        tip = self.tip_calculator.calculate_tip(current_sol, predicted_sol)
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_simple_fallback(self):
        """Test simple TIP calculation fallback."""
        # Test the simple calculation directly
        tip = self.tip_calculator._calculate_tip_simple(0.1, 0.05, 0.12, 0.06)
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_monte_carlo(self):
        """Test Monte Carlo TIP calculation."""
        tip = self.tip_calculator._calculate_tip_monte_carlo(
            0.1, 0.05, 0.12, 0.06, prediction_uncertainty=0.05
        )
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_with_covariance_error_handling(self):
        """Test TIP calculation with invalid covariance matrix."""
        # Create solutions with invalid covariance
        current_sol = self._create_mock_solution(0.1, 0.05)
        predicted_sol = self._create_mock_solution(0.12, 0.06)
        
        class MockKalmanState:
            def __init__(self):
                # Set invalid covariance matrix (not positive definite)
                self.P = np.array([[1.0, 2.0], [2.0, 1.0], [0.0, 0.0], [0.0, 0.0]])  # Not positive definite
        
        current_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state = MockKalmanState()
        predicted_sol.P.kalman_state.P = np.eye(4) * 0.01
        
        # Should fallback to simple calculation
        tip = self.tip_calculator.calculate_tip(current_sol, predicted_sol)
        
        self.assertGreaterEqual(tip, 0.0)
        self.assertLessEqual(tip, 1.0)
        
    def test_tip_constraints(self):
        """Test that TIP values are properly constrained."""
        # Test with extreme values
        extreme_current = self._create_mock_solution(0.0, 0.0)
        extreme_predicted = self._create_mock_solution(1.0, 1.0)
        
        tip = self.tip_calculator.calculate_tip(extreme_current, extreme_predicted)

        # Should be constrained between 0.05 and 0.95
        self.assertGreaterEqual(tip, 0.05)
        self.assertLessEqual(tip, 0.95)


def _make_kalman_state(roi_var: float, risk_var: float, cov: float = 0.0):
    """Build a minimal kalman_state stand-in with a 4x4 P matrix whose
    top-left 2x2 block is the (ROI, risk) covariance.

    TIP._calculate_tip_with_covariance reads P[:2, :2] only.
    """
    P = np.zeros((4, 4))
    P[0, 0] = roi_var
    P[1, 1] = risk_var
    P[0, 1] = cov
    P[1, 0] = cov
    return SimpleNamespace(P=P)


def _make_mock_solution(roi: float, risk: float,
                        roi_var: float = 1e-4, risk_var: float = 1e-4):
    """Build a minimal Solution-shaped mock with .P.ROI, .P.risk,
    .P.kalman_state.P[:2, :2] — the only attributes TIP reads.
    """
    portfolio = SimpleNamespace(
        ROI=roi,
        risk=risk,
        kalman_state=_make_kalman_state(roi_var, risk_var),
    )
    return SimpleNamespace(P=portfolio)


class TestPaperEq12TIPKnownAnalytical(unittest.TestCase):
    """W1-4 regression tests for paper Eq (12) — TIP definition.

    Pin the calculator against KNOWN-ANALYTICAL Gaussian cases (not
    bounds-only assertions) so a future "optimization" can't silently
    break the equation while leaving bounds intact. The audit's
    "tests assert bounds, not equations" finding for TIP closes here.

    Paper canon:
        p_{t,t+h} = Pr[ẑ_t, ẑ_{t+h}|ẑ_t are mutually incomparable]   (12)

    Two known-analytical bookend cases:
      * Disjoint Gaussians with σ ≪ |μ_1 − μ_2| → mutual non-dominance
        is rare → TIP near 0.
      * Identical Gaussians → mutual non-dominance is the expected
        outcome for half the draws → TIP near 0.5.

    Both tests use `clamp_range=None` so the (default) [0.05, 0.95]
    clamp doesn't hide the near-0 case.
    """

    def test_tip_near_zero_for_disjoint_gaussians(self):
        """Two Gaussians far apart in both objectives → TIP should be
        near 0 (current strictly dominates predicted; very few MC
        samples produce mutual non-dominance).
        """
        np.random.seed(42)  # bound MC variance
        calc = TemporalIncomparabilityCalculator(
            monte_carlo_samples=2000,
            clamp_range=None,  # disable W1-4 clamp to observe analytical truth
        )

        # current: high ROI (1.0), low risk (0.0) — clearly dominant
        # predicted: low ROI (0.0), high risk (1.0) — clearly dominated
        # With σ = 0.05 in each, the two distributions are >> 10σ apart
        # → mutual non-dominance is rare → TIP near 0.
        current = _make_mock_solution(roi=1.0, risk=0.0,
                                       roi_var=0.0025, risk_var=0.0025)
        predicted = _make_mock_solution(roi=0.0, risk=1.0,
                                         roi_var=0.0025, risk_var=0.0025)

        tip = calc.calculate_tip(current, predicted)
        self.assertLess(tip, 0.1,
                        f"Disjoint Gaussians should give TIP near 0, got {tip:.4f}")

    def test_tip_near_half_for_identical_gaussians(self):
        """Two IDENTICAL Gaussians → for any sampled pair, on average
        about half the time one is mutually non-dominated with the
        other (the other half splits between current-dominates and
        predicted-dominates). Expected TIP near 0.5.

        Uses a loose tolerance (0.3 < TIP < 0.7) because MC variance
        is real even at 2000 samples; the point is to pin the
        equation's behaviour against analytical expectation, not to
        prove the calculator is high-precision.
        """
        np.random.seed(42)
        calc = TemporalIncomparabilityCalculator(
            monte_carlo_samples=2000,
            clamp_range=None,
        )

        current = _make_mock_solution(roi=0.5, risk=0.5,
                                       roi_var=0.01, risk_var=0.01)
        predicted = _make_mock_solution(roi=0.5, risk=0.5,
                                         roi_var=0.01, risk_var=0.01)

        tip = calc.calculate_tip(current, predicted)
        self.assertGreater(tip, 0.3,
                           f"Identical Gaussians should give TIP near 0.5, got {tip:.4f}")
        self.assertLess(tip, 0.7,
                        f"Identical Gaussians should give TIP near 0.5, got {tip:.4f}")

    def test_clamp_range_default_preserves_pre_w1_4_behaviour(self):
        """Constructor opt-out: the default clamp_range=(0.05, 0.95)
        still bounds TIP (W1-4 doesn't change live-caller behaviour).
        """
        np.random.seed(42)
        calc_default = TemporalIncomparabilityCalculator(monte_carlo_samples=500)
        # Disjoint case → without clamp would give TIP near 0; with
        # default clamp should give >= 0.05.
        current = _make_mock_solution(roi=1.0, risk=0.0,
                                       roi_var=0.0025, risk_var=0.0025)
        predicted = _make_mock_solution(roi=0.0, risk=1.0,
                                         roi_var=0.0025, risk_var=0.0025)
        tip = calc_default.calculate_tip(current, predicted)
        self.assertGreaterEqual(tip, 0.05)
        self.assertLessEqual(tip, 0.95)


class TestPaperEq13LambdaBinaryEntropy(unittest.TestCase):
    """W1-4 regression tests for paper Eq (13).

    Paper canon:
        λ^(H)_{t+h} = (1/(H-1)) [1 - H(p_{t,t+h})]              (13)

    where H is the binary entropy function:
        H(p) = -p log_2(p) - (1-p) log_2(1-p)

    Verified table:
        H(0)    = 0    (degenerate)
        H(0.25) ≈ 0.811
        H(0.5)  = 1    (max uncertainty)
        H(0.75) ≈ 0.811
        H(1)    = 0    (degenerate)

    These give λ^(H) (for H=2 horizons, 1/(H-1) = 1):
        λ(tip=0)    = 1 - 0 = 1.0
        λ(tip=0.5)  = 1 - 1 = 0.0
        λ(tip=1)    = 1 - 0 = 1.0

    Pinned here against known-analytical truth, not bounds.
    """

    def setUp(self):
        # clamp doesn't matter — these test the entropy + rate fns directly
        self.calc = TemporalIncomparabilityCalculator(monte_carlo_samples=10)

    def test_binary_entropy_table(self):
        """Pin H(p) at known analytical points."""
        self.assertAlmostEqual(self.calc.binary_entropy(0.0), 0.0, places=10)
        self.assertAlmostEqual(self.calc.binary_entropy(1.0), 0.0, places=10)
        self.assertAlmostEqual(self.calc.binary_entropy(0.5), 1.0, places=10)
        # H(0.25) = -0.25 log2(0.25) - 0.75 log2(0.75)
        #        = -0.25 * (-2) - 0.75 * log2(0.75)
        #        = 0.5 + 0.75 * 0.4150375 = 0.5 + 0.3113 = 0.8113
        self.assertAlmostEqual(
            self.calc.binary_entropy(0.25), 0.8112781244591328, places=8,
        )

    def test_lambda_eq13_for_h_equals_2(self):
        """For H=2 (one-step-ahead), (1/(H-1)) = 1, so
        λ(tip) = 1 - H(tip).
        """
        # tip=0 → H(0)=0 → λ = 1.0
        self.assertAlmostEqual(
            self.calc.calculate_anticipatory_learning_rate_tip(0.0, 2), 1.0, places=10,
        )
        # tip=1 → H(1)=0 → λ = 1.0
        self.assertAlmostEqual(
            self.calc.calculate_anticipatory_learning_rate_tip(1.0, 2), 1.0, places=10,
        )
        # tip=0.5 → H(0.5)=1 → λ = 0.0
        self.assertAlmostEqual(
            self.calc.calculate_anticipatory_learning_rate_tip(0.5, 2), 0.0, places=10,
        )

    def test_lambda_eq13_for_h_equals_3_normalisation(self):
        """For H=3, (1/(H-1)) = 0.5 — verify normalisation."""
        # tip=0 → H(0)=0 → λ = 0.5 * 1 = 0.5
        self.assertAlmostEqual(
            self.calc.calculate_anticipatory_learning_rate_tip(0.0, 3), 0.5, places=10,
        )
        # tip=0.5 → H(0.5)=1 → λ = 0.5 * 0 = 0.0
        self.assertAlmostEqual(
            self.calc.calculate_anticipatory_learning_rate_tip(0.5, 3), 0.0, places=10,
        )

    def test_lambda_eq13_returns_zero_for_horizon_le_1(self):
        """Edge case: H ≤ 1 has no future horizons, λ must be 0."""
        self.assertEqual(self.calc.calculate_anticipatory_learning_rate_tip(0.3, 1), 0.0)
        self.assertEqual(self.calc.calculate_anticipatory_learning_rate_tip(0.3, 0), 0.0)


if __name__ == '__main__':
    unittest.main()
