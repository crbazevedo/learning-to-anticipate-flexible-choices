"""W22 Probe AI-2 regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_ai2_zref_boundary_invariance import (
    analyze_zref_boundary_interaction,
    derive_zref,
    sweep_front_sizes_under_both_zrefs,
    winners_under_zref_modes,
)


def test_derive_zref_returns_min_roi_max_risk():
    front = np.array([[0.001, 0.012], [0.003, 0.008], [0.005, 0.004]])
    R1, R2 = derive_zref(front, margin=0.0)
    assert R1 == 0.001  # min ROI
    assert R2 == 0.012  # max risk


def test_derive_zref_margin_widens():
    front = np.array([[0.001, 0.005], [0.005, 0.001]])
    R1, R2 = derive_zref(front, margin=0.1)
    # ROI range 0.004 → R1 = 0.001 - 0.1*0.004 = 0.0006
    # risk range 0.004 → R2 = 0.005 + 0.1*0.004 = 0.0054
    np.testing.assert_allclose(R1, 0.0006, atol=1e-10)
    np.testing.assert_allclose(R2, 0.0054, atol=1e-10)


def test_winners_under_zref_modes_returns_all_keys():
    rng = np.random.default_rng(0)
    r = winners_under_zref_modes(n_runs=10, front_size=5, rng=rng)
    required = {"fixed_counts", "derived_counts", "fixed_boundary_pct",
                "derived_boundary_pct", "fixed_middle_pct",
                "derived_middle_pct", "argmax_disagreement_pct"}
    assert required.issubset(r.keys())


def test_winners_count_arrays_sum_to_n_runs():
    rng = np.random.default_rng(0)
    r = winners_under_zref_modes(n_runs=20, front_size=5, rng=rng)
    assert r["fixed_counts"].sum() == 20
    assert r["derived_counts"].sum() == 20


def test_derived_zref_nulls_left_boundary():
    """ASYMMETRIC finding: NC30 b's data-derived R1 = min ROI = leftmost
    candidate's ROI → contribution_left = (ROI - R1) * (R2 - risk) = 0
    → LEFT boundary NEVER wins. (Right boundary still wins because (R2 -
    smallest_risk) = full risk range.)"""
    rng = np.random.default_rng(0)
    r = winners_under_zref_modes(n_runs=200, front_size=10, rng=rng)
    derived_left = r["derived_counts"][0]
    assert derived_left == 0, (
        f"Expected 0 wins at LEFT boundary under derived z_ref; got {derived_left}"
    )


def test_derived_zref_does_not_reduce_total_boundary_bias():
    """HONEST SCAR captured by test: NC30 b doesn't reduce TOTAL boundary
    bias because the RIGHT boundary still wins on the (R2 - smallest_risk)
    factor. Only the asymmetric (LEFT) side is killed."""
    rng = np.random.default_rng(0)
    r = winners_under_zref_modes(n_runs=200, front_size=10, rng=rng)
    # The right boundary should still dominate
    right_pct = r["derived_counts"][-1] / 200
    assert right_pct > 0.8, (
        f"Expected right boundary still wins ≥80% under derived zref; got {right_pct:.1%}"
    )


def test_sweep_returns_dict_keyed_by_size():
    sweep = sweep_front_sizes_under_both_zrefs([3, 5, 10], n_runs=20)
    assert set(sweep.keys()) == {3, 5, 10}


def test_analyze_returns_markdown():
    md = analyze_zref_boundary_interaction(front_sizes=[5, 10], n_runs=50)
    assert "## NC30 b data-derived z_ref" in md
    assert "Verdict" in md
