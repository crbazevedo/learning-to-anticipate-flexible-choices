"""W22 Probe AI regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_ai_front_size_sensitivity import (
    analyze_front_size_sensitivity,
    boundary_concentration,
    front_contribution,
    front_size_sweep,
    pick_distribution,
    synthetic_front,
)


def test_synthetic_front_returns_correct_shape():
    rng = np.random.default_rng(0)
    front = synthetic_front(5, rng=rng)
    assert front.shape == (5, 2)


def test_synthetic_front_sorted_by_roi():
    rng = np.random.default_rng(0)
    front = synthetic_front(10, rng=rng)
    rois = front[:, 0]
    assert np.all(np.diff(rois) >= 0)


def test_front_contribution_single_element():
    front = np.array([[0.005, 0.010]])
    c = front_contribution(0, front, R1=0.0, R2=0.02)
    assert c == 0.005 * (0.02 - 0.010)  # = 5e-5


def test_front_contribution_two_elements_boundary():
    front = np.array([[0.001, 0.015], [0.005, 0.005]])
    # idx 0 (boundary): (ROI - R1)(R2 - risk) = 0.001 * 0.005 = 5e-6
    c0 = front_contribution(0, front, R1=0.0, R2=0.02)
    # idx 1 (last): (ROI - prev_ROI)(R2 - risk) = 0.004 * 0.015 = 6e-5
    c1 = front_contribution(1, front, R1=0.0, R2=0.02)
    np.testing.assert_allclose(c0, 0.001 * 0.005, atol=1e-15)
    np.testing.assert_allclose(c1, 0.004 * 0.015, atol=1e-15)


def test_pick_distribution_returns_correct_shape():
    counts = pick_distribution(n_runs=10, front_size=5,
                                 rng=np.random.default_rng(0))
    assert counts.shape == (5,)
    assert counts.sum() == 10


def test_boundary_concentration_all_boundary():
    counts = np.array([5, 0, 5])  # all winners at boundaries
    assert boundary_concentration(counts) == 1.0


def test_boundary_concentration_none_boundary():
    counts = np.array([0, 10, 0])
    assert boundary_concentration(counts) == 0.0


def test_boundary_concentration_half():
    counts = np.array([5, 0, 5])  # 10 of 10 boundary
    assert boundary_concentration(counts) == 1.0
    counts = np.array([3, 4, 3])  # 6 of 10 boundary
    assert abs(boundary_concentration(counts) - 0.6) < 1e-12


def test_front_size_sweep_returns_dict_keyed_by_size():
    sweep = front_size_sweep([3, 5, 10], n_runs=20)
    assert set(sweep.keys()) == {3, 5, 10}
    for k, v in sweep.items():
        assert "counts" in v
        assert "boundary_concentration" in v
        assert "middle_winner_fraction" in v


def test_large_front_boundary_concentration_high():
    """At |P_t|=20, the rightmost (high-ROI) position wins ~always due to
    the (ROI - prev)(R2 - risk) formula's boundary advantage."""
    sweep = front_size_sweep([20], n_runs=100, R1=0.0, R2=0.05,
                               rng=np.random.default_rng(0))
    bc = sweep[20]["boundary_concentration"]
    # Empirically high — our analyzer should agree with Inspection 6 finding
    assert bc > 0.8, (
        f"Expected high boundary concentration (>0.8) on |P_t|=20; got {bc}"
    )


def test_analyze_returns_markdown_with_verdict():
    md = analyze_front_size_sensitivity(front_sizes=[3, 10, 50], n_runs=50)
    assert "## AMFC pick concentration" in md
    assert "Verdict" in md
    assert "|P_t|" in md or "|P_t|=" in md
