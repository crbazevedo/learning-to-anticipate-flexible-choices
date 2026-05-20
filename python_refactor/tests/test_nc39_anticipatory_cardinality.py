"""W22-NC39 regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.anticipatory_cardinality_handler import (
    anticipatory_top_k_projection,
    compare_projections,
    naive_top_k_projection,
    sweep_anticipation_strength,
)


def test_naive_top_k_keeps_top_K_by_weight():
    w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    proj = naive_top_k_projection(w, cardinality=3)
    # Top 3 by weight: indices 0, 1, 2
    assert proj[0] > 0 and proj[1] > 0 and proj[2] > 0
    assert proj[3] == 0 and proj[4] == 0
    np.testing.assert_allclose(np.sum(proj), 1.0)


def test_naive_top_k_full_cardinality_renormalizes():
    """K = n → keep all, just renormalize."""
    w = np.array([2.0, 4.0, 4.0])  # unnormalized
    proj = naive_top_k_projection(w, cardinality=3)
    np.testing.assert_allclose(proj, [0.2, 0.4, 0.4])


def test_naive_top_k_zero_total_fallback():
    """If all weights are zero, fallback to uniform over kept."""
    w = np.zeros(5)
    proj = naive_top_k_projection(w, cardinality=3)
    # Should be a valid simplex over some 3 entries
    assert abs(np.sum(proj) - 1.0) < 1e-10
    assert np.sum(proj > 0) == 3


def test_anticipatory_at_alpha_zero_matches_naive():
    """α=0 → identical to naive_top_k_projection."""
    w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    g = np.array([0.01, 0.02, 0.05, 0.10, 0.03])
    proj_naive = naive_top_k_projection(w, cardinality=3)
    proj_antic = anticipatory_top_k_projection(w, cardinality=3,
                                                  predicted_growth=g,
                                                  anticipation_strength=0.0)
    np.testing.assert_allclose(proj_antic, proj_naive)


def test_anticipatory_no_growth_uses_naive():
    """predicted_growth=None → naive behavior."""
    w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    proj = anticipatory_top_k_projection(w, cardinality=3,
                                            predicted_growth=None,
                                            anticipation_strength=2.0)
    np.testing.assert_allclose(proj, naive_top_k_projection(w, 3))


def test_anticipatory_can_flip_top_k_selection():
    """With high anticipation_strength and outsized growth on a low-weight asset,
    the anticipatory projection should INCLUDE that asset in the top-K
    where naive would not."""
    w = np.array([0.3, 0.3, 0.2, 0.15, 0.05])  # top-3 by weight = [0,1,2]
    # Asset 4 has tiny weight but massive predicted growth
    g = np.array([0.01, 0.01, 0.01, 0.01, 100.0])
    naive = naive_top_k_projection(w, cardinality=3)
    antic = anticipatory_top_k_projection(w, cardinality=3,
                                              predicted_growth=g,
                                              anticipation_strength=10.0)
    naive_idx = set(np.where(naive > 1e-12)[0])
    antic_idx = set(np.where(antic > 1e-12)[0])
    # Anticipatory should include asset 4 (high growth)
    assert 4 in antic_idx
    # Naive does NOT include asset 4
    assert 4 not in naive_idx


def test_anticipatory_negative_alpha_avoids_growth():
    """Negative anticipation_strength = defensive: avoid high-growth assets."""
    w = np.array([0.3, 0.3, 0.2, 0.15, 0.05])
    # Asset 0 has high growth, others low/neutral
    g = np.array([100.0, 0.01, 0.01, 0.01, 0.01])
    antic_defensive = anticipatory_top_k_projection(w, cardinality=2,
                                                       predicted_growth=g,
                                                       anticipation_strength=-1.0)
    # Defensive should exclude the high-growth asset 0
    assert antic_defensive[0] == 0


def test_anticipatory_output_on_simplex():
    """Output always on simplex (sums to 1, ≥0) for various inputs."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        n = rng.integers(3, 15)
        w = rng.uniform(0, 1, size=n)
        g = rng.normal(0, 0.1, size=n)
        K = int(rng.integers(1, n))
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            proj = anticipatory_top_k_projection(w, K, g, alpha)
            np.testing.assert_allclose(np.sum(proj), 1.0, atol=1e-10)
            assert np.all(proj >= 0)


def test_anticipatory_keeps_exactly_K_assets():
    w = np.array([0.4, 0.3, 0.2, 0.1])
    g = np.array([0.01, 0.05, 0.1, 0.2])
    for K in [1, 2, 3, 4]:
        proj = anticipatory_top_k_projection(w, K, g, anticipation_strength=1.0)
        n_nonzero = np.sum(proj > 1e-12)
        assert n_nonzero == K, f"K={K}, expected {K} non-zeros; got {n_nonzero}"


def test_compare_projections_returns_all_keys():
    w = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    g = np.array([0.01, 0.02, 0.10, 0.05, 0.03])
    result = compare_projections(w, 3, g, anticipation_strength=1.0)
    required = {
        "naive_weights", "anticipatory_weights",
        "naive_kept_indices", "anticipatory_kept_indices",
        "jaccard_kept",
        "naive_growth_weighted_score", "anticipatory_growth_weighted_score",
        "signed_growth_uplift",
    }
    assert required.issubset(result.keys())


def test_compare_projections_anticipatory_growth_higher_when_alpha_positive():
    """With α > 0, anticipatory growth-weighted score should be ≥ naive (it
    deliberately tilts toward high-growth assets)."""
    w = np.array([0.3, 0.3, 0.2, 0.15, 0.05])
    g = np.array([0.01, 0.01, 0.10, 0.20, 0.05])  # assets 2, 3 have high growth
    result = compare_projections(w, 3, g, anticipation_strength=2.0)
    assert result["signed_growth_uplift"] >= 0.0


def test_sweep_returns_dict_keyed_by_strength():
    w = np.array([0.4, 0.3, 0.2, 0.1])
    g = np.array([0.01, 0.05, 0.1, 0.2])
    sweep = sweep_anticipation_strength(w, 2, g, strengths=[0.0, 1.0, 5.0])
    assert set(sweep.keys()) == {0.0, 1.0, 5.0}


def test_dimension_mismatch_raises():
    w = np.array([0.5, 0.5])
    g = np.array([0.01, 0.02, 0.03])  # wrong dim
    import pytest
    with pytest.raises(ValueError):
        anticipatory_top_k_projection(w, 2, g, anticipation_strength=1.0)
