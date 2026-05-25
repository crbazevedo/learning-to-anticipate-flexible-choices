"""W22 Probe Y regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_y_smooth_clamp_impact import (
    compare_clamps,
    derivative_hard,
    derivative_smooth,
    hard_clamp,
    saturation_regime_summary,
    smooth_clamp,
)


def test_hard_clamp_inside_identity():
    assert hard_clamp(0.5) == 0.5
    assert hard_clamp(0.10) == 0.10


def test_hard_clamp_below_min():
    assert hard_clamp(0.01) == 0.05


def test_hard_clamp_above_max():
    assert hard_clamp(0.99) == 0.95


def test_smooth_clamp_center_is_identity():
    # tanh(0) = 0 → output = c_min + width/2 = (c_min+c_max)/2 = 0.5
    assert abs(smooth_clamp(0.5) - 0.5) < 1e-10


def test_smooth_clamp_strictly_inside_for_realistic_tip():
    for t in [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]:
        out = smooth_clamp(t)
        assert 0.05 <= out <= 0.95
        if 0 < t < 1:
            assert 0.05 < out < 0.95


def test_smooth_clamp_monotonic():
    tips = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99])
    outs = np.array([smooth_clamp(t) for t in tips])
    diffs = np.diff(outs)
    assert np.all(diffs > 0), f"Monotonicity violated: outs={outs}"


def test_derivative_hard_zero_outside_inside_one():
    assert derivative_hard(0.01) == 0.0
    assert derivative_hard(0.99) == 0.0
    assert derivative_hard(0.5) == 1.0


def test_derivative_smooth_positive_everywhere():
    """Smooth clamp has positive derivative everywhere (key property)."""
    for t in [0.001, 0.05, 0.5, 0.95, 0.999]:
        assert derivative_smooth(t) > 0.0


def test_compare_clamps_recovers_tail_signal():
    """Tail TIPs (outside [c_min, c_max]) have zero hard-deriv but positive smooth-deriv."""
    tips = np.array([0.01, 0.02, 0.5, 0.97, 0.99])
    result = compare_clamps(tips)
    # 4 of 5 are in tails; signal_recovered_fraction should be 1.0 (all tails recover)
    assert result["n_tail"] == 4
    assert result["signal_recovered_fraction"] > 0.99


def test_compare_clamps_no_tail_returns_zero_recovery():
    """All inside the clamp → tail fraction is 0; signal_recovered_fraction undefined / 0."""
    tips = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    result = compare_clamps(tips)
    assert result["n_tail"] == 0
    assert result["signal_recovered_fraction"] == 0.0


def test_saturation_summary_returns_markdown():
    tips = np.linspace(0, 1, 100)
    md = saturation_regime_summary(tips)
    assert "## NC13b smooth-clamp impact summary" in md
    assert "saturation rate" in md


def test_saturation_summary_high_regime_flagged():
    """If 50% of TIPs are tails, the summary should flag HIGH regime."""
    np.random.seed(0)
    # 60% in tails, 40% inside
    tips = np.concatenate([
        np.random.uniform(0, 0.04, 30),  # below c_min
        np.random.uniform(0.96, 1, 30),  # above c_max
        np.random.uniform(0.1, 0.9, 40),  # inside
    ])
    md = saturation_regime_summary(tips)
    assert "HIGH saturation" in md


def test_saturation_summary_low_regime_flagged():
    """If 90% of TIPs are inside, summary flags LOW regime."""
    np.random.seed(0)
    tips = np.random.uniform(0.1, 0.9, 100)  # all inside
    md = saturation_regime_summary(tips)
    assert "Low saturation" in md
