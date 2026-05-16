"""
Equation-level integration test for paper Eq (14) — the multi-horizon
anticipatory learning convex combination — implemented by
MultiHorizonAnticipatoryLearning.apply_anticipatory_learning_rule.

Paper canon:
    Azevedo, C. R. B., & Von Zuben, F. J. (2015). Learning to Anticipate
    Flexible Choices in Multiple Criteria Decision-Making Under
    Uncertainty. IEEE Transactions on Cybernetics. §IV-A.2 Eq (14):

        ẑ_t | ẑ_{t+1:t+H-1} ≡ ẑ_t + Σ_{h=1}^{H-1} λ_{t+h} (ẑ_{t+h} | ẑ_t − ẑ_t)

    which, after distributing, is algebraically equivalent to:

        (1 − Σ_{h=1}^{H-1} λ_{t+h}) ẑ_t + Σ_{h=1}^{H-1} λ_{t+h} ẑ_{t+h}

The implementation in `multi_horizon_anticipatory.py:apply_anticipatory_
learning_rule` materialises the second form directly. This test pins
that materialisation against a 3-horizon hand-computed case.
"""

import unittest
import numpy as np

# W1-3: use the canonical `from src.algorithms.X` style (matches W1-2's
# test_tip_integration fix). After this unit lands the relative-import
# refactor for src/algorithms/, this import resolves cleanly.
from src.algorithms.multi_horizon_anticipatory import MultiHorizonAnticipatoryLearning


class TestPaperEq14MultiHorizonConvexCombo(unittest.TestCase):
    """W1-3 regression test for paper Eq (14)."""

    def test_apply_anticipatory_learning_rule_matches_paper_eq14(self):
        """3-horizon hand-computed case: assert the convex combination
        produces the expected vector under paper Eq (14).

        Setup (H=3, so we have h=1 and h=2 future horizons):
            current_state    = [1.0, 0.5]    # ROI=1.0, risk=0.5 per Eq (11)
            predicted[h=1]   = [1.2, 0.4]
            predicted[h=2]   = [1.4, 0.3]
            λ_{t+1}          = 0.2
            λ_{t+2}          = 0.3
            Σλ               = 0.5
            (1 − Σλ)         = 0.5

        Expected per paper Eq (14):
            ẑ_t|ẑ_{t+1:t+2} = (1−Σλ)·ẑ_t + λ_{t+1}·ẑ_{t+1} + λ_{t+2}·ẑ_{t+2}
                            = 0.5·[1.0, 0.5] + 0.2·[1.2, 0.4] + 0.3·[1.4, 0.3]
                            = [0.50+0.24+0.42, 0.25+0.08+0.09]
                            = [1.16, 0.42]
        """
        learner = MultiHorizonAnticipatoryLearning(max_horizon=3, monte_carlo_samples=50)

        current_state = np.array([1.0, 0.5])
        predicted_states = [
            np.array([1.2, 0.4]),  # ẑ_{t+1}
            np.array([1.4, 0.3]),  # ẑ_{t+2}
        ]
        lambda_rates = [0.2, 0.3]  # Σλ = 0.5; (1 − Σλ) = 0.5

        result = learner.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates,
        )

        expected = (
            0.5 * current_state
            + 0.2 * predicted_states[0]
            + 0.3 * predicted_states[1]
        )
        np.testing.assert_allclose(expected, np.array([1.16, 0.42]), atol=1e-12)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_h_equals_2_collapses_to_two_term_form_paper_eq16(self):
        """Cross-check: H=2 reduces Eq (14) to the H=2 Gaussian-mean form
        of paper Eq (16): λ_t · ẑ_t + λ_{t+1} · ẑ_{t+1|t} with λ_t = 1 − λ_{t+1}.

        Setup:
            current_state = [0.8, 0.6]
            predicted[1]  = [1.0, 0.4]
            λ_{t+1}       = 0.4   ⇒   λ_t = 0.6
        Expected:
            0.6 · [0.8, 0.6] + 0.4 · [1.0, 0.4] = [0.88, 0.52]
        """
        learner = MultiHorizonAnticipatoryLearning(max_horizon=2, monte_carlo_samples=50)
        current = np.array([0.8, 0.6])
        predicted = [np.array([1.0, 0.4])]
        lambdas = [0.4]

        result = learner.apply_anticipatory_learning_rule(current, predicted, lambdas)

        np.testing.assert_allclose(result, np.array([0.88, 0.52]), atol=1e-9)

    def test_zero_horizon_returns_current_state_unchanged(self):
        """Edge case: when there are no future horizons, the rule must
        return current_state unchanged (the convex combo degenerates)."""
        learner = MultiHorizonAnticipatoryLearning(max_horizon=1, monte_carlo_samples=50)
        current = np.array([0.42, 0.17])

        result = learner.apply_anticipatory_learning_rule(current, [], [])

        np.testing.assert_allclose(result, current, atol=1e-12)

    def test_multi_horizon_inherits_from_anticipatory_learning(self):
        """W1-3 wiring guarantee: MultiHorizonAnticipatoryLearning must be
        usable as a `set_learning(...)` target on SMS-EMOA / NSGA-II, which
        requires it to expose the parent class's learn_population /
        learn_single_solution interfaces. Verified via isinstance().
        """
        from src.algorithms.anticipatory_learning import AnticipatoryLearning
        learner = MultiHorizonAnticipatoryLearning(max_horizon=3, monte_carlo_samples=50)
        self.assertIsInstance(learner, AnticipatoryLearning)
        self.assertTrue(hasattr(learner, 'learn_population'))
        self.assertTrue(hasattr(learner, 'learn_single_solution'))


if __name__ == '__main__':
    unittest.main()
