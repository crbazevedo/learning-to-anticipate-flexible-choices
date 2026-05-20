"""Unit tests for ``src.probes.probe_ab_per_portfolio_lambda_k`` (W22 Probe AB).

Falsifies the closed-enum behavior of the analyzer:

* Shared-window formula matches the production
  ``_compute_lambda_k`` line-for-line and lives in ``[0, 0.5]``.
* Per-portfolio output shape and range are well-defined.
* Heterogeneous tracking regimes (heterogeneity → 1.0) produce
  per-portfolio std > 0.05 — the claim that motivates Inspection 6.
* Homogeneous regimes (heterogeneity = 0) collapse to the shared value
  within noise.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.probes.probe_ab_per_portfolio_lambda_k import (
    COMPARE_KEYS,
    DISCRIMINATION_THRESHOLDS,
    compare_lambda_k_modes,
    compute_per_portfolio_lambda_k,
    compute_shared_lambda_k,
    synthesize_residual_windows,
)


# ---------------------------------------------------------------------------
# Shared-window λ^K — formula faithfulness + bounds
# ---------------------------------------------------------------------------
def test_compute_shared_lambda_k_zero_residuals_returns_zero():
    # All zeros — residual_sum = 0 → normalized = 1 - exp(0) = 0 → λ^K = 0.
    val = compute_shared_lambda_k([0.0, 0.0, 0.0, 0.0])
    assert val == pytest.approx(0.0, abs=1e-12)


def test_compute_shared_lambda_k_empty_window_returns_zero():
    # Empty input — analyzer must not crash; returns 0.0 by contract.
    assert compute_shared_lambda_k([]) == 0.0


def test_compute_shared_lambda_k_increases_with_residuals():
    # Larger residuals → larger residual_sum → larger normalized → larger λ^K.
    small = compute_shared_lambda_k([0.5, 0.5, 0.5, 0.5])
    large = compute_shared_lambda_k([5.0, 5.0, 5.0, 5.0])
    assert large > small


def test_compute_shared_lambda_k_bounded_in_zero_half():
    # Stress with widely-varying magnitudes — output must remain in [0, 0.5].
    rng = np.random.default_rng(20260520)
    for mean in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        win = list(rng.uniform(0.0, mean * 3.0, size=10))
        val = compute_shared_lambda_k(win)
        assert 0.0 <= val <= 0.5, f"λ^K out of [0, 0.5] at mean={mean}: {val}"


def test_compute_shared_lambda_k_matches_canonical_formula():
    # Pin the exact production formula: residual_sum / (K * scale) where
    # scale = max(1, mean(window)).
    win = [1.0, 2.0, 3.0, 4.0]
    residual_sum = sum(win)
    scale = max(1.0, float(np.mean(win)))
    expected = 0.5 * (1.0 - math.exp(-residual_sum / (len(win) * scale)))
    assert compute_shared_lambda_k(win) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Per-portfolio λ^K — shape + per-window independence
# ---------------------------------------------------------------------------
def test_compute_per_portfolio_lambda_k_returns_array():
    windows = [[0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [5.0, 5.0, 5.0]]
    out = compute_per_portfolio_lambda_k(windows)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert out.dtype == np.float64


def test_compute_per_portfolio_lambda_k_empty_input_returns_empty_array():
    out = compute_per_portfolio_lambda_k([])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0,)


def test_compute_per_portfolio_lambda_k_each_window_independent():
    # Per-portfolio values must equal compute_shared_lambda_k applied to
    # each window individually (zero cross-portfolio coupling).
    windows = [[0.5, 1.0], [2.0, 3.0], [10.0, 10.0]]
    out = compute_per_portfolio_lambda_k(windows)
    for i, w in enumerate(windows):
        assert out[i] == pytest.approx(compute_shared_lambda_k(w), rel=1e-12)


def test_per_portfolio_matches_shared_when_homogeneous():
    # heterogeneity = 0 → all portfolio means identical → all λ^K values
    # essentially identical AND close to the shared baseline.
    rng = np.random.default_rng(20260520)
    windows = synthesize_residual_windows(
        n_portfolios=20,
        window_size=10,
        mean_residual=2.0,
        std_residual=0.1,
        heterogeneity=0.0,
        rng=rng,
    )
    per = compute_per_portfolio_lambda_k(windows)
    shared = compute_shared_lambda_k([r for w in windows for r in w])
    # All per-portfolio values within 0.05 of the shared baseline.
    assert float(np.std(per)) < 0.05
    assert all(abs(v - shared) < 0.1 for v in per)


def test_per_portfolio_differs_from_shared_when_heterogeneous():
    # heterogeneity = 1.0 → portfolio means spread widely → per-portfolio
    # std should be > 0.05 (i.e. exceeds NEGLIGIBLE band).
    rng = np.random.default_rng(20260520)
    windows = synthesize_residual_windows(
        n_portfolios=20,
        window_size=10,
        mean_residual=2.0,
        std_residual=0.2,
        heterogeneity=1.0,
        rng=rng,
    )
    per = compute_per_portfolio_lambda_k(windows)
    assert float(np.std(per)) > 0.05, (
        f"expected per-portfolio std > 0.05 in heterogeneous regime, got "
        f"{float(np.std(per))} (per={per})"
    )


# ---------------------------------------------------------------------------
# compare_lambda_k_modes — closed-enum keys + significance classification
# ---------------------------------------------------------------------------
def test_compare_modes_returns_all_keys():
    rng = np.random.default_rng(20260520)
    windows = synthesize_residual_windows(
        n_portfolios=5,
        window_size=8,
        mean_residual=1.0,
        std_residual=0.1,
        heterogeneity=0.3,
        rng=rng,
    )
    result = compare_lambda_k_modes(windows)
    for key in COMPARE_KEYS:
        assert key in result, f"missing key in compare_lambda_k_modes output: {key}"
    # Type contracts on each key.
    assert isinstance(result["shared_lambda_k"], float)
    assert isinstance(result["per_portfolio_lambda_k"], np.ndarray)
    assert isinstance(result["per_portfolio_std"], float)
    assert isinstance(result["per_portfolio_range"], float)
    assert result["discrimination_significance"] in DISCRIMINATION_THRESHOLDS


def test_compare_modes_empty_input_safe():
    result = compare_lambda_k_modes([])
    assert result["shared_lambda_k"] == 0.0
    assert result["per_portfolio_lambda_k"].shape == (0,)
    assert result["per_portfolio_std"] == 0.0
    assert result["per_portfolio_range"] == 0.0
    assert result["discrimination_significance"] == "NEGLIGIBLE"


def test_discrimination_negligible_at_zero_heterogeneity():
    # Homogeneous regime → range tiny → NEGLIGIBLE.
    rng = np.random.default_rng(20260520)
    windows = synthesize_residual_windows(
        n_portfolios=20,
        window_size=10,
        mean_residual=2.0,
        std_residual=0.1,
        heterogeneity=0.0,
        rng=rng,
    )
    result = compare_lambda_k_modes(windows)
    assert result["discrimination_significance"] == "NEGLIGIBLE"


def test_discrimination_strong_at_high_heterogeneity():
    # heterogeneity = 1.0 across enough portfolios at low intra-window noise
    # → range across per-portfolio λ^K should be MODEST or STRONG.
    rng = np.random.default_rng(20260520)
    windows = synthesize_residual_windows(
        n_portfolios=30,
        window_size=20,
        mean_residual=2.0,
        std_residual=0.05,
        heterogeneity=1.0,
        rng=rng,
    )
    result = compare_lambda_k_modes(windows)
    assert result["discrimination_significance"] in {"MODEST", "STRONG"}, (
        f"expected MODEST or STRONG at high heterogeneity, got "
        f"{result['discrimination_significance']} (range="
        f"{result['per_portfolio_range']}, std={result['per_portfolio_std']})"
    )


# ---------------------------------------------------------------------------
# Synthesizer — shape + non-negativity + reproducibility
# ---------------------------------------------------------------------------
def test_synthesize_returns_correct_shape():
    rng = np.random.default_rng(20260520)
    out = synthesize_residual_windows(
        n_portfolios=7,
        window_size=11,
        mean_residual=1.0,
        std_residual=0.2,
        heterogeneity=0.5,
        rng=rng,
    )
    assert isinstance(out, list)
    assert len(out) == 7
    for w in out:
        assert isinstance(w, list)
        assert len(w) == 11
        assert all(isinstance(x, float) for x in w)
        assert all(x >= 0.0 for x in w)


def test_synthesize_is_reproducible_with_seeded_rng():
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    out_a = synthesize_residual_windows(
        n_portfolios=5,
        window_size=6,
        mean_residual=1.5,
        std_residual=0.3,
        heterogeneity=0.7,
        rng=rng_a,
    )
    out_b = synthesize_residual_windows(
        n_portfolios=5,
        window_size=6,
        mean_residual=1.5,
        std_residual=0.3,
        heterogeneity=0.7,
        rng=rng_b,
    )
    assert out_a == out_b


def test_synthesize_zero_heterogeneity_collapses_means():
    # heterogeneity = 0 → portfolio means all == mean_residual (no spread).
    rng = np.random.default_rng(20260520)
    out = synthesize_residual_windows(
        n_portfolios=10,
        window_size=50,
        mean_residual=3.0,
        std_residual=0.01,
        heterogeneity=0.0,
        rng=rng,
    )
    per_portfolio_means = [float(np.mean(w)) for w in out]
    # All portfolio means within 0.05 of mean_residual=3.0 (intra-window noise
    # only).
    for m in per_portfolio_means:
        assert abs(m - 3.0) < 0.05


def test_synthesize_rejects_negative_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        synthesize_residual_windows(
            n_portfolios=3,
            window_size=3,
            mean_residual=-1.0,
            std_residual=0.1,
            heterogeneity=0.0,
            rng=rng,
        )
    with pytest.raises(ValueError):
        synthesize_residual_windows(
            n_portfolios=3,
            window_size=3,
            mean_residual=1.0,
            std_residual=-0.1,
            heterogeneity=0.0,
            rng=rng,
        )
    with pytest.raises(ValueError):
        synthesize_residual_windows(
            n_portfolios=3,
            window_size=3,
            mean_residual=1.0,
            std_residual=0.1,
            heterogeneity=-0.5,
            rng=rng,
        )


def test_synthesize_zero_portfolios_or_window_returns_empty():
    rng = np.random.default_rng(0)
    assert (
        synthesize_residual_windows(
            n_portfolios=0,
            window_size=10,
            mean_residual=1.0,
            std_residual=0.1,
            heterogeneity=0.0,
            rng=rng,
        )
        == []
    )
    assert (
        synthesize_residual_windows(
            n_portfolios=5,
            window_size=0,
            mean_residual=1.0,
            std_residual=0.1,
            heterogeneity=0.0,
            rng=rng,
        )
        == []
    )
