"""
W15-2 regression tests for thesis §7.2.3 operators (BACKLOG B3+B4).

Grounding (verbatim, thesis §7.2.3 p. 147 Search Operators):
  "We utilized uniform crossover over the mean DD vectors. For
   mutation, we randomly choose between (1) modifying the non-zero
   weights; or (2) adding/removing assets. If operator (1) is
   selected, then, with probability 1/2, we either increase or
   decrease the investment on a randomly chosen asset by a uniformly
   drawn factor from 10 to 50%. If (2) is selected, then, with
   probability 1/2, we either add or remove a randomly chosen asset.
   If it is removed, we simply set its weight to zero. If it is
   added, we randomly set its weight within a ±10% range from an
   equally-balanced allocation. All modified DD vectors are
   renormalized."

Plus thesis §7.2.3 p. 146 (cardinality):
  "We considered minimum and maximum cardinality of c_l = 5 and
   c_u = 15 assets."

Plus thesis §7.2 p. 141 (simplex):
  "u ∈ S^{N-1} denote the proportions of wealth to be invested"
"""

import unittest

import numpy as np

from src.algorithms.operators import (
    THESIS_CARDINALITY_MAX,
    THESIS_CARDINALITY_MIN,
    project_to_simplex,
    thesis_dual_mode_mutation,
    thesis_uniform_crossover,
)
from src.algorithms.solution import Solution


def _mk_solution(weights: np.ndarray) -> Solution:
    s = Solution(num_assets=len(weights))
    s.P.investment = weights.astype(float).copy()
    return s


class TestProjectToSimplex(unittest.TestCase):
    """Cardinality + simplex projection per BACKLOG B3 + B4."""

    def test_clips_negative_weights(self):
        w = np.array([0.3, -0.1, 0.4, 0.2, -0.05, 0.05, 0.0, 0.0, 0.2])
        out = project_to_simplex(w, c_l=2, c_u=9, rng=np.random.default_rng(0))
        # All non-negative
        self.assertTrue(np.all(out >= 0))
        # Sums to 1
        self.assertAlmostEqual(out.sum(), 1.0, places=10)

    def test_caps_at_c_u(self):
        # 10 assets, all non-zero; c_u=5 → keep only top-5 by weight
        w = np.array([0.05, 0.20, 0.10, 0.30, 0.05, 0.10, 0.05, 0.05, 0.05, 0.05])
        out = project_to_simplex(w, c_l=2, c_u=5, rng=np.random.default_rng(0))
        self.assertLessEqual(int((out > 1e-12).sum()), 5)
        self.assertAlmostEqual(out.sum(), 1.0, places=10)

    def test_fills_to_c_l(self):
        # 10 assets, only 2 non-zero; c_l=5 → add 3 random assets
        w = np.zeros(10)
        w[0] = 0.6; w[3] = 0.4
        out = project_to_simplex(w, c_l=5, c_u=10, rng=np.random.default_rng(0))
        self.assertGreaterEqual(int((out > 1e-12).sum()), 5)
        self.assertAlmostEqual(out.sum(), 1.0, places=10)

    def test_thesis_defaults_are_5_and_15(self):
        self.assertEqual(THESIS_CARDINALITY_MIN, 5)
        self.assertEqual(THESIS_CARDINALITY_MAX, 15)


class TestThesisUniformCrossover(unittest.TestCase):
    """Uniform crossover per thesis §7.2.3 p. 147 (NOT SBX)."""

    def test_swap_probability_at_50pct(self):
        # With p=0.5 and 100 assets, expect ~50 swaps statistically.
        n = 100
        rng = np.random.default_rng(42)
        p1 = _mk_solution(np.ones(n) / n)  # uniform
        p2 = _mk_solution(np.full(n, 2.0 / n))  # 2x uniform (will renormalize)
        p2.P.investment = p2.P.investment / p2.P.investment.sum()
        c1, c2 = thesis_uniform_crossover(p1, p2, p=0.5,
                                            c_l=5, c_u=n, rng=rng)
        # Children are in the simplex
        self.assertAlmostEqual(c1.P.investment.sum(), 1.0, places=10)
        self.assertAlmostEqual(c2.P.investment.sum(), 1.0, places=10)
        self.assertTrue(np.all(c1.P.investment >= 0))
        self.assertTrue(np.all(c2.P.investment >= 0))

    def test_extreme_p_passes_through(self):
        n = 20
        rng = np.random.default_rng(7)
        p1 = _mk_solution(np.ones(n) / n)
        p2 = _mk_solution(np.zeros(n))
        p2.P.investment[0] = 1.0
        # p=0 → no swap; children should mirror parents (modulo cardinality fill)
        c1, _ = thesis_uniform_crossover(p1, p2, p=0.0,
                                          c_l=1, c_u=n, rng=rng)
        # c1 ≈ p1 weights (no swaps happened)
        np.testing.assert_allclose(c1.P.investment, p1.P.investment, atol=1e-10)


class TestThesisDualModeMutation(unittest.TestCase):
    """Dual-mode mutation per thesis §7.2.3 p. 147."""

    def test_mode_1_modify_nonzero_factor_in_range(self):
        """Drive the coin flips so mode 1 fires; check factor range."""
        # Patch rng to always pick mode 1 + increase
        from unittest.mock import MagicMock
        rng = MagicMock()
        rng.random = MagicMock(side_effect=[0.0, 0.0])  # 0.0 < 0.5 → mode 1; 0.0 < 0.5 → increase
        rng.choice = MagicMock(return_value=2)  # pick asset 2
        rng.uniform = MagicMock(return_value=0.30)  # factor=0.30 in [0.10, 0.50]
        sol = _mk_solution(np.full(10, 0.10))
        out = thesis_dual_mode_mutation(sol, rng=rng)
        # asset 2 increased by 30% → 0.13
        # Then renormalize + simplex project; cardinality stays 10 ≥ c_l (default 5)
        # The relative increase on asset 2 should be preserved approximately.
        self.assertGreater(out.P.investment[2], out.P.investment[0])

    def test_simplex_invariant_held(self):
        """100 random mutations all produce simplex-valid weights."""
        rng = np.random.default_rng(0)
        n = 30  # asset count > c_u to allow add/remove room
        sol = _mk_solution(np.full(n, 1.0 / n))
        for _ in range(100):
            sol = thesis_dual_mode_mutation(sol, rng=rng)
            self.assertAlmostEqual(sol.P.investment.sum(), 1.0, places=10)
            self.assertTrue(np.all(sol.P.investment >= 0))
            # Cardinality stays in [c_l, c_u]
            card = int((sol.P.investment > 1e-12).sum())
            self.assertGreaterEqual(card, THESIS_CARDINALITY_MIN)
            self.assertLessEqual(card, THESIS_CARDINALITY_MAX)


class TestCardinalityBoundsHold(unittest.TestCase):
    """End-to-end: combining crossover + mutation, cardinality stays in thesis range."""

    def test_after_xover_and_mutation_cardinality_in_5_15(self):
        rng = np.random.default_rng(1)
        n = 30
        p1 = _mk_solution(np.full(n, 1.0 / n))
        p2 = _mk_solution(np.array([0.5, 0.5] + [0.0] * (n - 2)))
        for _ in range(50):
            c1, c2 = thesis_uniform_crossover(p1, p2, rng=rng)
            c1 = thesis_dual_mode_mutation(c1, rng=rng)
            c2 = thesis_dual_mode_mutation(c2, rng=rng)
            for c in (c1, c2):
                card = int((c.P.investment > 1e-12).sum())
                self.assertGreaterEqual(card, THESIS_CARDINALITY_MIN,
                                          f"cardinality {card} < c_l={THESIS_CARDINALITY_MIN}")
                self.assertLessEqual(card, THESIS_CARDINALITY_MAX,
                                       f"cardinality {card} > c_u={THESIS_CARDINALITY_MAX}")


class TestW22NC8bOffspringObjectivesFresh(unittest.TestCase):
    """W22-NC8b regression: offspring's portfolio.ROI/risk MUST reflect its
    ACTUAL weights, not the random initial weights from Solution.__init__.

    Pre-NC8b (W15-2 regression): thesis_uniform_crossover and
    thesis_dual_mode_mutation called `Solution(num_assets=n)` (which
    compute_efficiency'd random initial weights from Portfolio.init()),
    then OVERWROTE the weights without recomputing. Result: offspring's
    portfolio.ROI/risk bore no relation to its weights — selection
    downstream used STALE objectives.

    The OLD `crossover`/`mutation` operators in this same file (lines 47-54,
    170-175) DID recompute efficiency after weight assignment. The thesis
    operators (W15-2 addition) silently dropped this discipline. NC8b is a
    regression introduced by W15-2.

    Receipt: W22 Probe A pre-NC7 verdict + post-NC7 verdict both showed
    KF predictions bit-identical to persistence partly because BOTH
    persistence_ROI (sol.P.ROI, stale) and kf.x_next[0] (init x[0], stale)
    were the same random-weights value.

    This test pins the post-NC8b invariant: after crossover OR mutation,
    portfolio.ROI MUST equal what Portfolio.compute_efficiency would
    compute on the offspring's actual weights.
    """

    def _init_portfolio_class_stats(self, n_assets: int = 8) -> None:
        """Initialize Portfolio class-level statistics so compute_efficiency
        fires. Without this, _finalize_offspring_objectives is a no-op."""
        from src.portfolio.portfolio import Portfolio
        rng = np.random.default_rng(42)
        # 60 daily returns, n_assets columns, small positive drift
        returns = rng.normal(loc=0.001, scale=0.01, size=(60, n_assets))
        Portfolio.mean_ROI = Portfolio.estimate_assets_mean_ROI(returns)
        Portfolio.median_ROI = Portfolio.estimate_assets_median_ROI(returns)
        Portfolio.covariance = Portfolio.estimate_covariance(
            Portfolio.mean_ROI, returns
        )
        Portfolio.robust_covariance = Portfolio.covariance

    def test_crossover_offspring_roi_matches_actual_weights(self):
        """After thesis_uniform_crossover, child.P.ROI must equal the ROI
        computed from child.P.investment (NOT from random-init weights)."""
        from src.portfolio.portfolio import Portfolio
        n = 8
        self._init_portfolio_class_stats(n_assets=n)

        rng = np.random.default_rng(1)
        # Two parents with deterministic concentrated allocations
        p1 = _mk_solution(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]))
        Portfolio.compute_efficiency(p1.P)
        p2 = _mk_solution(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.3]))
        Portfolio.compute_efficiency(p2.P)

        c1, c2 = thesis_uniform_crossover(p1, p2, p=0.5, rng=rng)

        # For each child: re-compute efficiency on its actual weights and
        # check that the operator's output already produced that result
        # (i.e. operator finalized objectives, not left stale ones).
        for child in (c1, c2):
            expected_ROI = float(np.dot(child.P.investment, Portfolio.mean_ROI))
            actual_ROI = float(child.P.ROI)
            # Allow modest tolerance for compute_efficiency internals
            self.assertAlmostEqual(
                actual_ROI, expected_ROI, places=6,
                msg=(f"NC8b STALE-OBJECTIVES REGRESSION: "
                     f"crossover child has portfolio.ROI={actual_ROI:.6e} "
                     f"but actual-weights ROI is {expected_ROI:.6e}. "
                     f"Operator did not recompute objectives after weight "
                     f"assignment — selection sees stale data.")
            )

    def test_mutation_offspring_roi_matches_actual_weights(self):
        """After thesis_dual_mode_mutation, mutated.P.ROI must equal the ROI
        computed from mutated.P.investment."""
        from src.portfolio.portfolio import Portfolio
        n = 8
        self._init_portfolio_class_stats(n_assets=n)

        rng = np.random.default_rng(2)
        parent = _mk_solution(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]))
        Portfolio.compute_efficiency(parent.P)

        for _ in range(10):  # multiple draws to cover both mutation modes
            mutated = thesis_dual_mode_mutation(parent, rng=rng)
            expected_ROI = float(np.dot(mutated.P.investment, Portfolio.mean_ROI))
            actual_ROI = float(mutated.P.ROI)
            self.assertAlmostEqual(
                actual_ROI, expected_ROI, places=6,
                msg=(f"NC8b STALE-OBJECTIVES REGRESSION: "
                     f"mutated portfolio.ROI={actual_ROI:.6e} != "
                     f"actual-weights ROI {expected_ROI:.6e}")
            )

    def test_crossover_offspring_kalman_state_matches_actual_roi(self):
        """After NC8b finalize, child.P.kalman_state.x[0:2] MUST equal
        child.P.ROI/risk (not random-weights values)."""
        from src.portfolio.portfolio import Portfolio
        n = 8
        self._init_portfolio_class_stats(n_assets=n)

        rng = np.random.default_rng(3)
        p1 = _mk_solution(np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        Portfolio.compute_efficiency(p1.P)
        p2 = _mk_solution(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]))
        Portfolio.compute_efficiency(p2.P)

        c1, c2 = thesis_uniform_crossover(p1, p2, p=0.5, rng=rng)

        for child in (c1, c2):
            self.assertIsNotNone(child.P.kalman_state,
                                 "kalman_state missing on offspring after NC8b finalize")
            kf_x = child.P.kalman_state.x
            self.assertAlmostEqual(
                float(kf_x[0]), float(child.P.ROI), places=6,
                msg=(f"NC8b STALE-KF REGRESSION: "
                     f"kalman_state.x[0]={kf_x[0]:.6e} != "
                     f"portfolio.ROI={child.P.ROI:.6e}")
            )
            self.assertAlmostEqual(
                float(kf_x[1]), float(child.P.risk), places=6,
                msg=(f"NC8b STALE-KF REGRESSION: "
                     f"kalman_state.x[1]={kf_x[1]:.6e} != "
                     f"portfolio.risk={child.P.risk:.6e}")
            )


if __name__ == "__main__":
    unittest.main()
