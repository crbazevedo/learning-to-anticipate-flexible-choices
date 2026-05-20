"""W22 Probe T — tests for γ sensitivity sweep analyzer.

Pure synthetic analyzer tests. No shared-code imports.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_t_gamma_sensitivity import (
    analyze_gamma_tradeoff,
    compute_lambda_h_profile,
    cumulative_anticipation_weight,
    effective_horizon,
    sweep_gamma,
)


def test_lambda_h_profile_geometric_ratio():
    """Consecutive λ^H ratios equal γ (when no clamp is hit)."""
    gamma = 0.7
    # TIP=0.0 → entropy 0 → confidence 1.0. Profile starts at γ^1 = 0.7
    # which is clamped to 0.5; γ^2 = 0.49, γ^3 = 0.343, γ^4 = 0.2401, ...
    # Pairwise ratios beyond the clamp must equal γ exactly.
    profile = compute_lambda_h_profile(gamma=gamma, H=6, tip_value=0.0)
    # Skip the first clamped entry; check ratios on the unclamped tail.
    for i in range(2, len(profile) - 1):
        ratio = profile[i + 1] / profile[i]
        assert math.isclose(ratio, gamma, rel_tol=1e-12), (
            f"ratio at h={i + 1}→{i + 2} = {ratio}, expected γ={gamma}"
        )


def test_lambda_h_profile_clamped():
    """No λ^H value exceeds the 0.5 cap, even at γ→1 and full confidence."""
    # γ=0.999, TIP=0.0 → raw λ^H_1 ≈ 0.999, must clamp to 0.5.
    profile = compute_lambda_h_profile(gamma=0.999, H=5, tip_value=0.0)
    assert all(v <= 0.5 + 1e-12 for v in profile), (
        f"profile {profile} violates clamp ≤ 0.5"
    )
    assert profile[0] == pytest.approx(0.5, abs=1e-9), (
        "γ^1 · 1.0 ≈ 0.999 should clamp exactly to 0.5"
    )


def test_sweep_gamma_returns_dict_keyed_by_gamma():
    """sweep_gamma maps each γ to a profile of length H."""
    gammas = [0.5, 0.7, 0.9, 0.99]
    H = 5
    result = sweep_gamma(gammas, H)
    assert isinstance(result, dict)
    assert set(result.keys()) == set(gammas)
    for gamma, profile in result.items():
        assert len(profile) == H, f"γ={gamma} profile length {len(profile)} ≠ H={H}"


def test_effective_horizon_higher_gamma_longer():
    """γ=0.99 must have at least as long an effective horizon as γ=0.5."""
    H = 10
    # Use TIP=0.0 so the only decay source is γ^h (not entropy).
    eh_low = effective_horizon(compute_lambda_h_profile(0.5, H, tip_value=0.0))
    eh_high = effective_horizon(compute_lambda_h_profile(0.99, H, tip_value=0.0))
    assert eh_high > eh_low, (
        f"effective_horizon(γ=0.99)={eh_high} should exceed "
        f"effective_horizon(γ=0.5)={eh_low}"
    )


def test_cumulative_weight_monotonic_in_gamma_at_fixed_h():
    """Σ λ^H_h is non-decreasing as γ increases (at fixed H, TIP)."""
    H = 8
    gammas = [0.3, 0.5, 0.7, 0.9, 0.99]
    cw_values = [
        cumulative_anticipation_weight(compute_lambda_h_profile(g, H, tip_value=0.0))
        for g in gammas
    ]
    for i in range(len(cw_values) - 1):
        assert cw_values[i + 1] >= cw_values[i] - 1e-12, (
            f"cumulative weight not monotonic: γ={gammas[i]} → {cw_values[i]}, "
            f"γ={gammas[i + 1]} → {cw_values[i + 1]}"
        )


def test_analyze_gamma_tradeoff_returns_markdown_with_all_gammas():
    """The report mentions every γ value supplied."""
    gammas = [0.5, 0.7, 0.9, 0.99]
    report = analyze_gamma_tradeoff(gammas, H=5)
    assert isinstance(report, str)
    assert report.startswith("##"), "report should open with a markdown header"
    for g in gammas:
        assert f"γ={g:g}" in report or f"| {g:g} " in report, (
            f"report missing γ={g} reference"
        )
    # Must surface a γ* recommendation.
    assert "γ*" in report, "report should identify a candidate γ*"
