"""W22-NC13b structural-fix tests: smooth TIP clamp.

Per operator directive 2026-05-19 — degenerate behavior shouldn't happen.
Pre-fix: TIP clamp at [0.05, 0.95] was a HARD clip, losing tail signal.
Post-fix (NC13b): smooth squash via shifted-scaled tanh preserves tail
gradient while still bounding output in (c_min, c_max).

Opt-in via W22_NC13B_SMOOTH_CLAMP=1 env var; default unchanged for
backward compatibility.

These tests verify:
1. Default (no env var) behavior unchanged — hard clip
2. With env var: identity-ish in center, smooth tails
3. Output always strictly inside (c_min, c_max)
4. Monotonic in TIP (preserves ordering)
5. K parameter controls steepness
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.temporal_incomparability_probability import TemporalIncomparabilityCalculator


def test_default_is_hard_clip():
    """Without env var set, the clamp is the legacy hard clip."""
    calc = TemporalIncomparabilityCalculator()
    # Below clamp_min → c_min exactly
    assert calc._clamp_tip(0.01) == 0.05
    # Above clamp_max → c_max exactly
    assert calc._clamp_tip(0.99) == 0.95
    # Inside → identity
    assert calc._clamp_tip(0.5) == 0.5


def test_smooth_clamp_preserves_center(monkeypatch):
    """With env var: TIP=0.5 → output=0.5 (center is identity)."""
    monkeypatch.setenv("W22_NC13B_SMOOTH_CLAMP", "1")
    calc = TemporalIncomparabilityCalculator()
    out = calc._clamp_tip(0.5)
    # Should be exactly 0.5 because tanh(k * 0) = 0
    assert abs(out - 0.5) < 1e-10


def test_smooth_clamp_strictly_inside_bounds_for_realistic_tip(monkeypatch):
    """Output is always strictly inside (c_min, c_max) for realistic TIP ∈ [0, 1].

    For mathematical asymptotes (tip << 0 or tip >> 1), tanh saturates to
    ±1 exactly in float arithmetic, so output reaches the clamp boundary.
    This is acceptable because TIP from MC sampling is always in [0, 1].
    """
    monkeypatch.setenv("W22_NC13B_SMOOTH_CLAMP", "1")
    calc = TemporalIncomparabilityCalculator()
    # Realistic TIP range from MC sampling
    for tip in [0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1.0]:
        out = calc._clamp_tip(tip)
        assert 0.05 <= out <= 0.95, (
            f"TIP={tip} → out={out} outside closed interval [0.05, 0.95]"
        )
        # For non-boundary inputs, output should be STRICTLY inside the open interval
        if 0 < tip < 1:
            assert 0.05 < out < 0.95, (
                f"TIP={tip} (in (0,1)) → out={out} not strictly inside (0.05, 0.95)"
            )


def test_smooth_clamp_monotonic(monkeypatch):
    """Output is monotonically increasing in TIP."""
    monkeypatch.setenv("W22_NC13B_SMOOTH_CLAMP", "1")
    calc = TemporalIncomparabilityCalculator()
    tips = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    outs = [calc._clamp_tip(t) for t in tips]
    for i in range(len(outs) - 1):
        assert outs[i] < outs[i + 1], (
            f"Monotonicity violated: tip[{i}]={tips[i]} → {outs[i]} vs "
            f"tip[{i+1}]={tips[i+1]} → {outs[i+1]}"
        )


def test_smooth_clamp_preserves_signal_in_tail(monkeypatch):
    """Two tail inputs that hard-clip to the same value get DIFFERENT smooth outputs."""
    monkeypatch.setenv("W22_NC13B_SMOOTH_CLAMP", "1")
    calc = TemporalIncomparabilityCalculator()
    # Hard clip: both 0.01 and 0.001 → 0.05
    # Smooth: should produce different outputs (preserving the tail-signal)
    out_001 = calc._clamp_tip(0.001)
    out_01 = calc._clamp_tip(0.01)
    out_001_default = 0.05
    assert out_001 != out_001_default
    assert out_01 != out_001
    # Order preserved: smaller tip → smaller output
    assert out_001 < out_01


def test_smooth_clamp_k_env_var_controls_steepness(monkeypatch):
    """Higher K → steeper squash (output gets closer to boundaries for tail inputs)."""
    monkeypatch.setenv("W22_NC13B_SMOOTH_CLAMP", "1")
    calc = TemporalIncomparabilityCalculator()
    # Default k=4
    out_default = calc._clamp_tip(0.05)
    # Higher k=10 → steeper, output closer to clamp boundary
    monkeypatch.setenv("W22_NC13B_K", "10.0")
    out_steep = calc._clamp_tip(0.05)
    # Higher k pushes output further from center, so closer to c_min
    # (the boundary), so smaller value
    assert out_steep < out_default


def test_disabled_clamp_passes_through(monkeypatch):
    """When clamp_range=None, _clamp_tip is identity (env var irrelevant)."""
    monkeypatch.setenv("W22_NC13B_SMOOTH_CLAMP", "1")
    calc = TemporalIncomparabilityCalculator(clamp_range=None)
    assert calc._clamp_tip(0.01) == 0.01
    assert calc._clamp_tip(0.99) == 0.99
    assert calc._clamp_tip(0.5) == 0.5
