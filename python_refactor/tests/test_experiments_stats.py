"""
W8-2 equation-level tests for stats helpers.

Each function asserted against a known-analytical input where the
expected output is computable by hand or by a different code path.
"""

import unittest

import numpy as np

from experiments.stats import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    holm_bonferroni,
    mann_whitney_u,
    one_way_anova,
    welch_t,
    wilcoxon_signed_rank,
)


class TestStatsHelpers(unittest.TestCase):
    """Equation-level regression tests for W8-2 stats helpers."""

    # -- inferential tests --

    def test_mann_whitney_two_disjoint_groups_p_near_zero(self):
        """When x and y are strictly disjoint (x all greater), MWU p → 0."""
        x = [10.0, 11.0, 12.0, 13.0, 14.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = mann_whitney_u(x, y, alternative="greater")
        self.assertLess(p, 0.05)

    def test_mann_whitney_identical_distributions_p_high(self):
        """When x and y are identical samples, MWU p ≈ 1 (no detectable diff)."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = list(x)
        _, p = mann_whitney_u(x, y, alternative="two-sided")
        self.assertGreater(p, 0.5)

    def test_wilcoxon_requires_equal_length(self):
        """Paired test must reject mismatched lengths."""
        with self.assertRaises(ValueError):
            wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_welch_t_identical_samples_t_near_zero(self):
        """Identical samples → t ≈ 0, p high."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = list(x)
        t, p = welch_t(x, y)
        self.assertAlmostEqual(t, 0.0, places=6)
        self.assertGreater(p, 0.5)

    def test_one_way_anova_two_groups_matches_t_squared(self):
        """For 2 groups, one-way ANOVA F = t² (Welch). Cross-check."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 6.0, 7.0, 8.0, 9.0]
        f, p_anova = one_way_anova(x, y)
        # Use equal-variance t for the F = t² identity.
        from scipy import stats as sps
        t_eq = sps.ttest_ind(x, y, equal_var=True).statistic
        self.assertAlmostEqual(f, t_eq ** 2, places=6)

    def test_one_way_anova_requires_two_groups(self):
        with self.assertRaises(ValueError):
            one_way_anova([1.0, 2.0])

    # -- effect sizes --

    def test_cohens_d_identical_samples_zero(self):
        """Identical samples → d = 0."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = list(x)
        self.assertAlmostEqual(cohens_d(x, y), 0.0, places=6)

    def test_cohens_d_large_effect(self):
        """When mean(x) - mean(y) ≈ 1 pooled std, d ≈ 1.0."""
        x = [10.0, 11.0, 12.0, 13.0, 14.0]  # mean=12, std=√2.5
        y = [8.0, 9.0, 10.0, 11.0, 12.0]    # mean=10, std=√2.5
        d = cohens_d(x, y)
        # diff = 2; pooled std = √2.5; d = 2/√2.5 ≈ 1.265
        self.assertAlmostEqual(d, 2.0 / np.sqrt(2.5), places=4)

    def test_cliffs_delta_x_dominates_y_returns_one(self):
        """When every x > every y, δ = 1."""
        x = [10.0, 11.0, 12.0]
        y = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(cliffs_delta(x, y), 1.0, places=6)

    def test_cliffs_delta_identical_returns_zero(self):
        x = [1.0, 2.0, 3.0]
        y = [1.0, 2.0, 3.0]
        # P(X>Y) = P(X<Y) for identical; δ = 0 modulo ties.
        self.assertAlmostEqual(cliffs_delta(x, y), 0.0, places=6)

    # -- bootstrap CI --

    def test_bootstrap_ci_contains_mean(self):
        """CI of the mean should contain the sample mean for normal data."""
        rng = np.random.default_rng(42)
        values = rng.normal(loc=10.0, scale=1.0, size=100)
        lo, hi = bootstrap_ci(values, statistic=np.mean, n_resamples=500, rng=rng)
        sample_mean = float(np.mean(values))
        self.assertLessEqual(lo, sample_mean)
        self.assertGreaterEqual(hi, sample_mean)

    def test_bootstrap_ci_empty_returns_nan(self):
        lo, hi = bootstrap_ci([], statistic=np.mean, n_resamples=10)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    # -- multi-comparison --

    def test_holm_bonferroni_single_pvalue_unchanged(self):
        self.assertEqual(holm_bonferroni([0.05]), [0.05])

    def test_holm_bonferroni_multiple_pvalues_step_down(self):
        """Standard textbook example: p = [0.01, 0.04, 0.03], m = 3.
        Sorted: 0.01, 0.03, 0.04.
        Step adjusted: 3*0.01=0.03; 2*0.03=0.06; 1*0.04=0.04.
        Monotone: 0.03, 0.06, 0.06 (the third's raw 0.04 < running max 0.06).
        Restored to original order: [0.03, 0.06, 0.06].
        """
        adjusted = holm_bonferroni([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adjusted[0], 0.03, places=6)
        self.assertAlmostEqual(adjusted[1], 0.06, places=6)
        self.assertAlmostEqual(adjusted[2], 0.06, places=6)

    def test_holm_bonferroni_clips_to_one(self):
        """If raw p × m > 1, clip to 1."""
        adjusted = holm_bonferroni([0.5, 0.6, 0.9])
        for adj in adjusted:
            self.assertLessEqual(adj, 1.0)

    def test_holm_bonferroni_empty(self):
        self.assertEqual(holm_bonferroni([]), [])


if __name__ == "__main__":
    unittest.main()
