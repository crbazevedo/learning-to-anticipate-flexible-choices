"""W22 Probe U regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_u_alpha_sensitivity import (
    analyze_alpha_tradeoff,
    argmax_under_alpha,
    compute_effective_contribution,
    find_flip_points,
    sweep_alpha,
)


def test_effective_contribution_at_zero_alpha_is_expected():
    """α=0 → effective == expected."""
    expected = np.array([0.5, 0.3, 0.8])
    variances = np.array([0.01, 0.5, 0.001])
    eff = compute_effective_contribution(expected, variances, alpha=0.0)
    np.testing.assert_allclose(eff, expected)


def test_argmax_at_zero_alpha_matches_argmax_expected():
    expected = np.array([0.5, 0.3, 0.8])
    variances = np.array([0.01, 0.5, 0.001])
    idx = argmax_under_alpha(expected, variances, alpha=0.0)
    assert idx == 2  # 0.8 is largest


def test_argmax_at_large_alpha_picks_lowest_variance():
    """At very high α, variance dominates → argmax = argmin variance."""
    expected = np.array([0.5, 0.3, 0.8])
    variances = np.array([0.01, 0.5, 0.001])
    idx = argmax_under_alpha(expected, variances, alpha=1000.0)
    # All eff values are very negative; the LEAST negative wins:
    # eff[i] = expected[i] - 1000 * variance[i]
    # eff[0] = 0.5 - 10 = -9.5
    # eff[1] = 0.3 - 500 = -499.7
    # eff[2] = 0.8 - 1 = -0.2  ← largest (smallest |negative|)
    assert idx == 2  # candidate with both high expected AND low variance wins


def test_sweep_alpha_returns_dict_keyed_by_alpha():
    expected = np.array([0.5, 0.3])
    variances = np.array([0.01, 0.5])
    result = sweep_alpha(expected, variances, [0.0, 0.5, 10.0])
    assert set(result.keys()) == {0.0, 0.5, 10.0}
    assert all(isinstance(v, int) for v in result.values())


def test_find_flip_points_detects_argmax_change():
    """Two candidates: low-σ-low-mean vs high-σ-high-mean.
    Low α → high-mean wins; high α → low-σ wins. So a flip exists."""
    expected = np.array([0.3, 0.5])  # idx 1 has higher mean
    variances = np.array([0.001, 1.0])  # idx 0 has lower variance
    flips = find_flip_points(expected, variances)
    assert len(flips) >= 1
    # The first flip should go from idx 1 (high-mean wins at α=0)
    # to idx 0 (low-var wins at large α)
    first_flip = flips[0]
    assert first_flip[1] == 1  # prev_idx
    assert first_flip[2] == 0  # new_idx


def test_find_flip_points_no_flip_when_one_dominates_both():
    """If one candidate is strictly better in BOTH mean and variance, no flip."""
    expected = np.array([0.3, 0.8])  # idx 1 has higher mean
    variances = np.array([0.5, 0.001])  # idx 1 ALSO has lower variance
    # idx 1 dominates in both dimensions → argmax always = 1 regardless of α
    flips = find_flip_points(expected, variances)
    assert flips == []


def test_analyze_alpha_tradeoff_returns_markdown_with_keys():
    expected = np.array([0.5, 0.3, 0.8])
    variances = np.array([0.01, 0.5, 0.001])
    md = analyze_alpha_tradeoff(expected, variances)
    assert "α sensitivity" in md
    assert "| α |" in md
    assert "0.0" in md or "0.1" in md


def test_analyze_alpha_tradeoff_custom_alphas():
    expected = np.array([1.0, 2.0])
    variances = np.array([0.0, 0.5])
    md = analyze_alpha_tradeoff(expected, variances, alphas=[0.5, 5.0])
    assert "| 0.5 |" in md or "| 0.5" in md
    assert "| 5.0 |" in md or "| 5.0" in md


def test_default_alphas_include_zero_and_one():
    """Default α list includes 0.0 and 1.0 (the most common operating points)."""
    expected = np.array([0.5, 0.3])
    variances = np.array([0.01, 0.5])
    md = analyze_alpha_tradeoff(expected, variances)
    assert "| 0.0 |" in md
    assert "| 1.0 |" in md
