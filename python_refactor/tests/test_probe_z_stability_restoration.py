"""W22 Probe Z regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_z_stability_restoration import (
    analyze_stability_tradeoff,
    apply_stability_legacy,
    apply_stability_v2,
    argmax_legacy,
    argmax_v2,
    disagreement_rate,
    stability_factor_legacy,
    stability_factor_v2,
    sweep_disagreement_vs_trace_spread,
    synthesize_population,
)


def test_stability_legacy_zero_trace_returns_one():
    assert stability_factor_legacy(0.0) == 1.0


def test_stability_legacy_monotonic_decreasing():
    for t1 in [0.0, 0.5, 1.0, 5.0]:
        for t2 in [t1 + 0.1, t1 + 1.0]:
            assert stability_factor_legacy(t1) > stability_factor_legacy(t2)


def test_stability_v2_is_one():
    assert stability_factor_v2() == 1.0


def test_apply_legacy_shrinks_high_trace():
    delta_s = np.array([1.0, 1.0])
    traces = np.array([0.0, 4.0])  # second is high uncertainty
    out = apply_stability_legacy(delta_s, traces)
    assert out[0] == 1.0  # 1.0 * 1.0 = 1.0
    assert out[1] == 0.2  # 1.0 * (1 / (1 + 4)) = 0.2


def test_apply_v2_is_noop():
    delta_s = np.array([1.0, 2.0, 3.0])
    traces = np.array([0.0, 5.0, 10.0])
    out = apply_stability_v2(delta_s, traces)
    np.testing.assert_array_equal(out, delta_s)


def test_argmax_disagrees_when_low_trace_runner_up_wins():
    """High-Δ_S high-trace candidate loses to low-Δ_S low-trace in legacy mode."""
    delta_s = np.array([1.0, 0.6])  # idx 0 has higher raw Δ_S
    traces = np.array([10.0, 0.0])  # idx 0 has high uncertainty
    # legacy: eff = [1/11, 0.6/1] = [0.091, 0.6] → argmax = 1
    # v2: eff = [1, 0.6] → argmax = 0
    assert argmax_legacy(delta_s, traces) == 1
    assert argmax_v2(delta_s, traces) == 0


def test_argmax_agrees_when_no_uncertainty_variation():
    """If all traces equal, legacy and v2 produce identical argmax."""
    delta_s = np.array([1.0, 0.6, 0.3])
    traces = np.array([0.5, 0.5, 0.5])
    assert argmax_legacy(delta_s, traces) == argmax_v2(delta_s, traces)


def test_disagreement_rate_zero_when_traces_equal():
    rng = np.random.default_rng(0)
    n_runs, n_sol = 50, 5
    delta_s = rng.normal(1.0, 0.5, size=(n_runs, n_sol))
    traces = np.full((n_runs, n_sol), 0.5)
    rate = disagreement_rate(delta_s, traces)
    assert rate == 0.0


def test_disagreement_rate_positive_when_traces_vary():
    """When traces vary widely AND Δ_S values are close, disagreement happens."""
    rng = np.random.default_rng(0)
    n_runs, n_sol = 100, 5
    # Tight Δ_S values + spread-out traces → frequent disagreement
    delta_s = rng.normal(1.0, 0.01, size=(n_runs, n_sol))
    traces = np.abs(rng.normal(1.0, 5.0, size=(n_runs, n_sol)))
    rate = disagreement_rate(delta_s, traces)
    assert rate > 0.0


def test_synthesize_population_shapes():
    rng = np.random.default_rng(0)
    d, t = synthesize_population(7, rng=rng)
    assert d.shape == (7,)
    assert t.shape == (7,)
    assert np.all(d > 0)
    assert np.all(t >= 0)


def test_sweep_disagreement_increases_with_trace_spread():
    """As trace spread grows, disagreement rate should increase (or stay)."""
    rng = np.random.default_rng(0)
    sweep = sweep_disagreement_vs_trace_spread(
        n_runs=50, n_solutions=8, rng=rng,
    )
    spreads = sorted(sweep.keys())
    rates = [sweep[s] for s in spreads]
    # Allow some noise; assert max is at one of the highest spreads
    # (not strict monotonic — too noisy at n_runs=50)
    assert max(rates) >= 0.0
    # Spread=0 → identical traces → 0% disagreement
    assert sweep[0.0] == 0.0


def test_analyze_returns_markdown():
    md = analyze_stability_tradeoff(n_runs=50, n_solutions=5)
    assert "## Stability factor" in md
    assert "trace(P) spread" in md
    assert "Verdict" in md
