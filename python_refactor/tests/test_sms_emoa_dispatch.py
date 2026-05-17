"""
W11-1 regression tests for SMS-EMOA's _apply_anticipatory_learning.

Pins the W10-CARRY-2 fix: when the learner's class overrides
`learn_population`, _apply_anticipatory_learning calls it ONCE for
the unlearned subset. When no override exists (base AnticipatoryLearning),
it falls back to the per-solution loop calling learn_single_solution.
"""

import types
import unittest
from unittest.mock import MagicMock

from src.algorithms.sms_emoa import SMSEMOA


# Mimic the base / subclass shape from anticipatory_learning.py without
# importing the full graph (which drags in numpy, pandas, etc. that
# the dispatch logic doesn't actually need to validate).

class AnticipatoryLearning:
    """Stand-in for the base class. Per the real shape (verified
    during W11-1 implementation), base defines ONLY
    learn_single_solution; learn_population is added by subclasses
    (TIPIntegrated, MultiHorizon) only."""
    def learn_single_solution(self, solution, generation):
        pass


class OverridingLearner(AnticipatoryLearning):
    """Stand-in for TIPIntegrated / MultiHorizon — overrides learn_population."""
    def __init__(self):
        self.population_calls = []
        self.single_calls = []

    def learn_population(self, solutions, generation):
        self.population_calls.append((tuple(solutions), generation))

    def learn_single_solution(self, solution, generation):
        self.single_calls.append((solution, generation))


class BaseLearner(AnticipatoryLearning):
    """Stand-in for base — does NOT override learn_population."""
    def __init__(self):
        self.single_calls = []

    def learn_single_solution(self, solution, generation):
        self.single_calls.append((solution, generation))


def _make_solution(anticipated: bool):
    sol = types.SimpleNamespace()
    sol.anticipation = anticipated
    return sol


def _make_algo(learner, population):
    """Build a minimal SMSEMOA-shaped object exposing only what
    _apply_anticipatory_learning needs."""
    # Don't run SMSEMOA.__init__ — too many deps.
    algo = SMSEMOA.__new__(SMSEMOA)
    algo.anticipatory_learning = learner
    algo.population = population
    return algo


class TestDispatch(unittest.TestCase):

    def test_override_learner_uses_learn_population(self):
        learner = OverridingLearner()
        pop = [_make_solution(False), _make_solution(False), _make_solution(True)]
        algo = _make_algo(learner, pop)
        algo._apply_anticipatory_learning(generation=3)
        # learn_population called once with the 2 unlearned solutions.
        self.assertEqual(len(learner.population_calls), 1)
        called_solutions, called_gen = learner.population_calls[0]
        self.assertEqual(len(called_solutions), 2)
        self.assertEqual(called_gen, 3)
        # Per-solution fallback never fires when override exists.
        self.assertEqual(learner.single_calls, [])

    def test_base_learner_falls_back_to_per_solution_loop(self):
        learner = BaseLearner()
        pop = [_make_solution(False), _make_solution(False), _make_solution(True)]
        algo = _make_algo(learner, pop)
        algo._apply_anticipatory_learning(generation=7)
        # Per-solution called for the 2 unlearned solutions.
        self.assertEqual(len(learner.single_calls), 2)
        for _, gen in learner.single_calls:
            self.assertEqual(gen, 7)

    def test_no_learner_is_noop(self):
        algo = _make_algo(None, [_make_solution(False)])
        # Should not raise.
        algo._apply_anticipatory_learning(generation=0)

    def test_all_solutions_anticipated_is_noop(self):
        learner = OverridingLearner()
        pop = [_make_solution(True), _make_solution(True)]
        algo = _make_algo(learner, pop)
        algo._apply_anticipatory_learning(generation=5)
        self.assertEqual(learner.population_calls, [])
        self.assertEqual(learner.single_calls, [])

    def test_real_subclasses_have_learn_population(self):
        """Sanity: real TIP and MultiHorizon classes define learn_population
        (per W11-1 audit); base AnticipatoryLearning does NOT. Skip if
        import fails (offline / partial env)."""
        try:
            from src.algorithms.anticipatory_learning import (
                AnticipatoryLearning as RealBase,
                TIPIntegratedAnticipatoryLearning,
            )
        except ImportError:
            self.skipTest("anticipatory_learning module not importable in this env")
        # Base lacks learn_population entirely.
        self.assertFalse(hasattr(RealBase, 'learn_population'),
                          "Base AnticipatoryLearning should NOT have learn_population")
        # TIPIntegrated defines it (subclass-only).
        self.assertTrue(hasattr(TIPIntegratedAnticipatoryLearning, 'learn_population'),
                         "TIPIntegrated must define learn_population for W11-1 routing")
        # MultiHorizon lives in a different module; tolerate absence here.
        try:
            from src.algorithms.multi_horizon_anticipatory import (
                MultiHorizonAnticipatoryLearning,
            )
            self.assertTrue(hasattr(MultiHorizonAnticipatoryLearning, 'learn_population'),
                             "MultiHorizon must define learn_population for W11-1 routing")
        except ImportError:
            pass  # Optional check


if __name__ == "__main__":
    unittest.main()
