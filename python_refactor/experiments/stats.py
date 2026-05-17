"""
W8-2 statistical-test helpers for the analytics plan.

Pure functions over numpy arrays. No I/O. No global state.

Each function implements one of the tests specified in
[`docs/ANALYTICS-PLAN.md` §3 + §4](../../docs/ANALYTICS-PLAN.md). The
docstrings cite the methodology source.

Exposed surface:
    mann_whitney_u(x, y, alternative='two-sided') → (U, p)
    wilcoxon_signed_rank(x, y) → (W, p)         # paired-by-seed
    welch_t(x, y) → (t, p)
    one_way_anova(*groups) → (F, p)
    cohens_d(x, y) → float
    cliffs_delta(x, y) → float in [-1, 1]
    bootstrap_ci(values, statistic, n_resamples, alpha) → (lo, hi)
    holm_bonferroni(p_values) → list[float]
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Inferential tests
# ---------------------------------------------------------------------------

def mann_whitney_u(x: Sequence[float], y: Sequence[float],
                    alternative: str = "two-sided") -> tuple[float, float]:
    """Mann-Whitney U test for two independent samples.

    Methodology: paper §V-D specifies Mann-Whitney U for pairwise
    scenario comparison because it's distribution-free and robust to
    the heavy-tailed wealth distribution.

    Args:
        x, y: independent samples.
        alternative: 'two-sided' (default), 'less', or 'greater'.

    Returns:
        (U statistic, p-value). p is the asymptotic-normal p-value
        for n + m > 20.
    """
    result = sps.mannwhitneyu(x, y, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.

    Use when samples are matched (same seed across scenarios). Tighter
    than Mann-Whitney because seed-variance is removed from the
    comparison.

    Args:
        x, y: matched pairs (must be same length).

    Returns:
        (W statistic, p-value).
    """
    if len(x) != len(y):
        raise ValueError(f"Paired samples must match length: |x|={len(x)}, |y|={len(y)}")
    result = sps.wilcoxon(x, y)
    return float(result.statistic), float(result.pvalue)


def welch_t(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Welch's t-test (unequal variances) for two independent samples.

    Sanity-check companion to mann_whitney_u when normality is plausible
    (skewness < 1, kurtosis < 5). Disagreement with Mann-Whitney is
    itself a finding (non-normality).
    """
    result = sps.ttest_ind(x, y, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def one_way_anova(*groups: Sequence[float]) -> tuple[float, float]:
    """One-way ANOVA across ≥ 2 independent groups.

    Used in ANALYTICS-PLAN §4.1 for the Multi-Horizon-level ANOVA over
    {S1, S2, S3}. Parametric — assumes Gaussian + equal variance.

    Returns:
        (F statistic, p-value).
    """
    if len(groups) < 2:
        raise ValueError("ANOVA requires ≥ 2 groups")
    result = sps.f_oneway(*groups)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------

def cohens_d(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d — standardized mean difference.

    d = (mean(x) - mean(y)) / pooled_std

    Pooled std uses n-1 (sample). Magnitude interpretation:
        |d| < 0.2 trivial; 0.2 small; 0.5 medium; 0.8 large.

    Note: x is the "treatment" / numerator; positive d means x > y.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    nx, ny = len(x_arr), len(y_arr)
    vx = x_arr.var(ddof=1)
    vy = y_arr.var(ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((x_arr.mean() - y_arr.mean()) / pooled)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's δ — non-parametric effect size in [-1, 1].

    δ = P(X > Y) - P(X < Y), estimated empirically over all pairs.
    Magnitude interpretation (Romano et al. 2006):
        |δ| < 0.147 negligible; 0.147 small; 0.33 medium; 0.474 large.

    Robust to outliers; orthogonal to Mann-Whitney's p-value.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    # Use broadcasting to compute all pairs at once.
    diff_sign = np.sign(x_arr[:, None] - y_arr[None, :])
    return float(diff_sign.mean())


def bootstrap_ci(values: Sequence[float],
                  statistic: Callable[[np.ndarray], float] = np.mean,
                  n_resamples: int = 10000,
                  alpha: float = 0.05,
                  rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI for `statistic` of `values`.

    Default: 95% CI on the mean (10,000 resamples). Non-parametric —
    robust to the wealth metric's heavy-tailed distribution where the
    Gaussian-assumption CI on mean ± 1.96·SEM under-covers.

    Args:
        values: input sample.
        statistic: callable on a numpy array → scalar.
        n_resamples: bootstrap iterations.
        alpha: significance level; CI is (alpha/2, 1-alpha/2) percentile.
        rng: optional numpy Generator for reproducible resampling.

    Returns:
        (lower, upper) percentile bounds.
    """
    values_arr = np.asarray(values, dtype=float)
    n = len(values_arr)
    if n == 0:
        return float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng()
    bs_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        bs_stats[i] = statistic(values_arr[idx])
    lo, hi = np.percentile(bs_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Multi-comparison correction
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Uniformly more powerful than naive Bonferroni; controls family-wise
    error rate at the same α level.

    Algorithm:
        1. Sort p-values ascending; keep original indices.
        2. For i = 1..m: adjusted p_(i) = (m - i + 1) * p_(i)
           but enforced monotone non-decreasing.
        3. Clip to ≤ 1.
        4. Restore original ordering.

    Returns:
        list of adjusted p-values in original input order.
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return []
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        adj = (m - i) * sorted_p[i]
        running_max = max(running_max, adj)
        adjusted_sorted[i] = min(running_max, 1.0)
    # Restore original order.
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]
