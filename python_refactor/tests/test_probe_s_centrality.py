"""Tests for Probe S-v1 — asset network centrality.

Covers the contained module
``python_refactor/src/probes/probe_s_centrality.py``.

No network calls, no external data. All graphs are synthetic and
small (K <= 8) for deterministic verification.
"""

import numpy as np
import pytest

from src.probes.probe_s_centrality import (
    betweenness_centrality,
    build_correlation_network,
    centrality_gini,
    eigenvector_centrality,
    hub_tilt_weights,
)


# ---------------------------------------------------------------------------
# build_correlation_network
# ---------------------------------------------------------------------------


def test_correlation_network_diagonal_one():
    """Diagonal of the correlation network is exactly 1 for every asset."""
    rng = np.random.default_rng(seed=42)
    returns = rng.standard_normal(size=(100, 5))
    adj = build_correlation_network(returns)
    np.testing.assert_allclose(np.diag(adj), np.ones(5), atol=1e-12)


def test_correlation_network_symmetric():
    """Output adjacency is exactly symmetric."""
    rng = np.random.default_rng(seed=7)
    returns = rng.standard_normal(size=(80, 6))
    adj = build_correlation_network(returns)
    np.testing.assert_allclose(adj, adj.T, atol=1e-12)


def test_correlation_network_top_k_sparsification():
    """Top-K sparsification keeps the strongest edges and zeros others."""
    # Construct a network with one obvious "hub" asset 0 correlated to all,
    # and the rest uncorrelated.
    rng = np.random.default_rng(seed=11)
    n_t = 500
    n_k = 5
    hub = rng.standard_normal(size=n_t)
    returns = np.column_stack([hub] + [
        0.95 * hub + 0.05 * rng.standard_normal(size=n_t)
        for _ in range(n_k - 1)
    ])
    adj = build_correlation_network(returns, top_k_edges_per_node=1)
    # Diagonal stays 1.
    np.testing.assert_allclose(np.diag(adj), np.ones(n_k), atol=1e-12)
    # Asset 0 must have at least one non-zero off-diagonal (its top edge).
    assert np.any(adj[0, 1:] != 0)


# ---------------------------------------------------------------------------
# eigenvector_centrality
# ---------------------------------------------------------------------------


def test_eigenvector_centrality_on_known_graph():
    """On a star graph, the center has the highest centrality."""
    # K = 5 star: node 0 is connected to 1..4; periphery is unconnected.
    n = 5
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = 1.0
        adj[i, 0] = 1.0
    np.fill_diagonal(adj, 1.0)

    cent = eigenvector_centrality(adj)
    assert cent.argmax() == 0
    # The four periphery nodes are interchangeable -> equal centrality.
    np.testing.assert_allclose(cent[1:], np.full(n - 1, cent[1]), atol=1e-6)


def test_eigenvector_centrality_sum_to_one():
    """L1-normalized output sums to 1 on a generic graph."""
    rng = np.random.default_rng(seed=2026)
    returns = rng.standard_normal(size=(200, 8))
    adj = build_correlation_network(returns)
    cent = eigenvector_centrality(adj)
    assert cent.shape == (8,)
    assert np.all(cent >= 0)
    assert pytest.approx(cent.sum(), rel=1e-6) == 1.0


def test_eigenvector_centrality_degenerate_graph_is_uniform():
    """All-zero adjacency -> uniform distribution (graceful degenerate case)."""
    adj = np.zeros((4, 4))
    cent = eigenvector_centrality(adj)
    np.testing.assert_allclose(cent, np.full(4, 0.25), atol=1e-12)


# ---------------------------------------------------------------------------
# betweenness_centrality
# ---------------------------------------------------------------------------


def test_betweenness_star_graph_center_dominates():
    """On a star graph, only the center lies on any shortest path."""
    n = 5
    adj = np.zeros((n, n))
    for i in range(1, n):
        adj[0, i] = 1.0
        adj[i, 0] = 1.0
    np.fill_diagonal(adj, 1.0)
    bw = betweenness_centrality(adj)
    assert bw.argmax() == 0
    # Periphery nodes are NOT on any s-t shortest path between other peripheries.
    np.testing.assert_allclose(bw[1:], np.zeros(n - 1), atol=1e-12)


# ---------------------------------------------------------------------------
# centrality_gini
# ---------------------------------------------------------------------------


def test_gini_uniform_distribution_is_zero():
    """Uniform distribution -> Gini == 0."""
    x = np.ones(10)
    assert centrality_gini(x) == pytest.approx(0.0, abs=1e-12)


def test_gini_concentrated_distribution_is_near_one():
    """All mass on one element -> Gini approaches 1 as K grows."""
    n = 100
    x = np.zeros(n)
    x[0] = 1.0
    gini = centrality_gini(x)
    # Theoretical maximum for finite n is (n - 1) / n.
    assert gini == pytest.approx((n - 1) / n, abs=1e-6)
    assert gini > 0.95


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError):
        centrality_gini(np.array([-0.1, 0.5, 0.5]))


# ---------------------------------------------------------------------------
# hub_tilt_weights
# ---------------------------------------------------------------------------


def test_hub_tilt_gamma_zero_preserves_weights():
    """gamma == 0 returns the renormalized base weights bit-exactly."""
    base = np.array([0.4, 0.3, 0.2, 0.1])
    cent = np.array([0.7, 0.1, 0.1, 0.1])
    out = hub_tilt_weights(base, cent, gamma=0.0)
    np.testing.assert_allclose(out, base / base.sum(), atol=1e-12)


def test_hub_tilt_positive_gamma_increases_hub_weight():
    """Positive gamma raises the weight of the high-centrality asset."""
    base = np.array([0.25, 0.25, 0.25, 0.25])
    cent = np.array([0.7, 0.1, 0.1, 0.1])
    out = hub_tilt_weights(base, cent, gamma=0.5)
    assert out[0] > 0.25
    # The other three move down (or stay equal).
    assert out[1] < 0.25
    assert out[2] < 0.25
    assert out[3] < 0.25


def test_hub_tilt_negative_gamma_decreases_hub_weight():
    """Negative gamma lowers the weight of the high-centrality asset."""
    base = np.array([0.25, 0.25, 0.25, 0.25])
    cent = np.array([0.7, 0.1, 0.1, 0.1])
    out = hub_tilt_weights(base, cent, gamma=-0.5)
    assert out[0] < 0.25
    # The low-centrality assets compensate upward.
    assert out[1] > 0.25
    assert out[2] > 0.25
    assert out[3] > 0.25


def test_hub_tilt_output_is_simplex():
    """For any gamma in a reasonable range, output is on the simplex."""
    base = np.array([0.4, 0.3, 0.2, 0.1])
    cent = np.array([0.6, 0.2, 0.15, 0.05])
    for gamma in [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
        out = hub_tilt_weights(base, cent, gamma=gamma)
        assert np.all(out >= 0), f"negative entry at gamma={gamma}: {out}"
        assert pytest.approx(out.sum(), rel=1e-9) == 1.0, (
            f"sum != 1 at gamma={gamma}: {out.sum()}"
        )
