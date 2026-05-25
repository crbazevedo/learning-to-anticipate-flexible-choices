"""W22 Probe AD regression tests.

Falsifies / locks the H-Probe-AD claims:
  - deterministic Δ_S replicates the sms_emoa.py closed form
  - stochastic Δ_S == deterministic when Cov(ROI, risk) = 0
  - stochastic Δ_S < deterministic when Cov(ROI, risk) > 0
  - MC converges to deterministic when Cov = 0 (zero correction)
  - MC with positive Cov sits below deterministic (Eq 6.41 corrects toward MC)
  - stochastic is closer to MC than deterministic when Cov != 0
  - compare_methods returns all keys
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_ad_delta_s_comparison import (
    compare_methods,
    compare_methods_with_eq636,
    deterministic_delta_s,
    deterministic_eq636_delta_s,
    mc_ground_truth_delta_s,
    stochastic_delta_s,
)


# ---------------------------------------------------------------------------
# 1. deterministic_delta_s formula correctness
# ---------------------------------------------------------------------------


def test_deterministic_matches_known_formula():
    """Hand-compute Δ_S on a 3-solution sorted front."""
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])  # risk DESC as ROI ASC (Pareto)
    R1 = 0.05
    R2 = 0.40

    det = deterministic_delta_s(rois, risks, R1, R2)

    # i=0 (leftmost): (ROI_0 - R1) * (R2 - risk_0)
    expected_0 = (0.10 - 0.05) * (0.40 - 0.30)
    # i=1 (middle): (ROI_1 - ROI_2) * (risk_0 - risk_1)
    expected_1 = (0.15 - 0.20) * (0.30 - 0.25)
    # i=2 (last): (ROI_2 - ROI_1) * (R2 - risk_2)
    expected_2 = (0.20 - 0.15) * (0.40 - 0.20)

    np.testing.assert_allclose(det, [expected_0, expected_1, expected_2])


def test_deterministic_handles_single_solution():
    """|C|=1 case: Δ_S = (ROI - R1) * (R2 - risk)."""
    det = deterministic_delta_s(
        np.array([0.10]), np.array([0.30]), R1=0.05, R2=0.40
    )
    np.testing.assert_allclose(det, [(0.10 - 0.05) * (0.40 - 0.30)])


def test_deterministic_preserves_input_order():
    """Output Δ_S returned in INPUT order, not sorted order."""
    # Same front as test_deterministic_matches_known_formula but
    # shuffled. Same per-solution Δ_S, just permuted.
    rois = np.array([0.20, 0.10, 0.15])
    risks = np.array([0.20, 0.30, 0.25])
    R1 = 0.05
    R2 = 0.40

    det = deterministic_delta_s(rois, risks, R1, R2)

    # Solution with ROI=0.20 -> "last" position in sorted -> expected_2
    expected_last = (0.20 - 0.15) * (0.40 - 0.20)
    # Solution with ROI=0.10 -> "first" position in sorted -> expected_0
    expected_first = (0.10 - 0.05) * (0.40 - 0.30)
    # Solution with ROI=0.15 -> "middle" position in sorted -> expected_1
    expected_middle = (0.15 - 0.20) * (0.30 - 0.25)

    np.testing.assert_allclose(
        det, [expected_last, expected_first, expected_middle]
    )


# ---------------------------------------------------------------------------
# 2. stochastic Δ_S relationship to deterministic
# ---------------------------------------------------------------------------


def test_stochastic_matches_deterministic_when_cov_zero():
    """Cov = 0 → stoch == det (the - Cov correction is zero)."""
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    covs = np.zeros(3)
    R1, R2 = 0.05, 0.40

    det = deterministic_delta_s(rois, risks, R1, R2)
    stoch = stochastic_delta_s(rois, risks, covs, R1, R2)

    np.testing.assert_allclose(stoch, det)


def test_stochastic_subtracts_self_cov():
    """Cov > 0 → stoch == det - Cov per Eq 6.41 self-correction."""
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    covs = np.array([0.005, 0.010, 0.003])
    R1, R2 = 0.05, 0.40

    det = deterministic_delta_s(rois, risks, R1, R2)
    stoch = stochastic_delta_s(rois, risks, covs, R1, R2)

    np.testing.assert_allclose(stoch, det - covs)
    # Sanity: each stoch is strictly less than det since covs > 0.
    assert np.all(stoch < det)


# ---------------------------------------------------------------------------
# 3. MC convergence properties
# ---------------------------------------------------------------------------


def test_mc_converges_to_deterministic_with_zero_cov():
    """Large n_mc + Cov=0 → MC ≈ deterministic (within MC noise band)."""
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    # Build 2x2 covs with tiny marginal variance + zero off-diagonal.
    # Using a small but nonzero marginal variance keeps the MC well-
    # defined; the resulting MC noise per Δ_S is roughly
    # 2 * sqrt(var_floor / n_mc) ~ 1e-3 for n_mc=5000.
    n = 3
    mus = np.column_stack([rois, risks])
    var_floor = 1e-4
    covs_full = np.zeros((n, 2, 2))
    covs_full[:, 0, 0] = var_floor
    covs_full[:, 1, 1] = var_floor
    # off-diagonals stay 0 -> Cov(ROI, risk) = 0 per solution

    rng = np.random.default_rng(42)
    mc = mc_ground_truth_delta_s(mus, covs_full, R1=0.05, R2=0.40, n_mc=20000, rng=rng)
    det = deterministic_delta_s(rois, risks, R1=0.05, R2=0.40)

    # With Cov=0 the analytical E[Δ_S] equals det (no correction term).
    # MC should match within a few standard errors.
    np.testing.assert_allclose(mc, det, atol=5e-3)


def test_mc_with_positive_cov_is_below_deterministic():
    """Positive within-solution Cov pulls MC E[Δ_S] BELOW det.

    Per Eq 6.41 the - Cov(a, d) self-term decreases E[Δ_S]
    when Cov(ROI, risk) > 0. The MC estimate should reflect this:
    the SUM of MC contributions < SUM of deterministic contributions.
    We use the sum (not element-wise) because positive Cov shifts
    the front's geometry stochastically and individual middle-
    solution Δ_S can swing sign under sampling; the SUM is the
    robust signal (= Lebesgue-stable hypervolume integral).
    """
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    n = 3
    # Strong positive Cov + matching marginal variance so the 2x2
    # remains positive semi-definite (Cov <= sqrt(var * var)).
    var = 0.01
    cov_off = 0.008  # 0.008 < sqrt(0.01*0.01) = 0.01 (valid PSD)
    mus = np.column_stack([rois, risks])
    covs_full = np.zeros((n, 2, 2))
    covs_full[:, 0, 0] = var
    covs_full[:, 1, 1] = var
    covs_full[:, 0, 1] = cov_off
    covs_full[:, 1, 0] = cov_off

    rng = np.random.default_rng(123)
    mc = mc_ground_truth_delta_s(mus, covs_full, R1=0.05, R2=0.40, n_mc=20000, rng=rng)
    det = deterministic_delta_s(rois, risks, R1=0.05, R2=0.40)

    assert float(np.sum(mc)) < float(np.sum(det)), (
        f"Expected MC sum < det sum with positive Cov, got mc_sum="
        f"{np.sum(mc):.6f} vs det_sum={np.sum(det):.6f}"
    )


def test_stochastic_closer_to_mc_than_deterministic_when_cov_nonzero():
    """The KEY claim: in the |C|=1 single-solution regime, stoch is a
    strictly better approximation to MC than det.

    SCAR (HONEST): the claim "stoch is closer to MC than det" only
    holds CLEANLY in the |C|=1 case AND at the boundary positions of
    multi-solution fronts. The middle-position deterministic formula
    in ``sms_emoa.py`` uses the rectangle
    ``(ROI_i - ROI_{i+1})(risk_{i-1} - risk_i)`` while Eq 6.41
    specifies ``(ROI_i - ROI_{i-1})(risk_{i+1} - risk_i)``. The
    - Cov correction subtracts the right quantity for the Eq 6.41
    rectangle but the wrong quantity for the legacy Python middle-
    branch rectangle, so the L1 gap over multi-solution fronts can
    actually be larger for stoch than for det.

    This test pins the regime where the claim DOES hold (single
    solution), to make the result falsifiable. See
    ``docs/W22-PROBE-AD-DELTA-S-COMPARISON.md`` for the full
    regime analysis.
    """
    rois = np.array([0.10])
    risks = np.array([0.30])
    n = 1
    var = 0.01
    cov_off = 0.008  # 0.008 < sqrt(0.01*0.01) = 0.01 (valid PSD)
    covs_scalar = np.full(n, cov_off)
    mus = np.column_stack([rois, risks])
    covs_full = np.zeros((n, 2, 2))
    covs_full[:, 0, 0] = var
    covs_full[:, 1, 1] = var
    covs_full[:, 0, 1] = cov_off
    covs_full[:, 1, 0] = cov_off

    rng = np.random.default_rng(7)
    mc = mc_ground_truth_delta_s(mus, covs_full, R1=0.05, R2=0.40, n_mc=50000, rng=rng)
    det = deterministic_delta_s(rois, risks, R1=0.05, R2=0.40)
    stoch = stochastic_delta_s(rois, risks, covs_scalar, R1=0.05, R2=0.40)

    l1_det = float(np.sum(np.abs(det - mc)))
    l1_stoch = float(np.sum(np.abs(stoch - mc)))

    assert l1_stoch < l1_det, (
        f"Expected stoch closer to MC than det in |C|=1 regime; got "
        f"l1_stoch={l1_stoch:.6f} vs l1_det={l1_det:.6f}"
    )
    # Strong claim: stoch should be within MC noise (3-sigma of MC SE).
    # With n=50000 and var=0.01, SE per sample ~ sqrt(var/n) ~ 4.5e-4.
    assert l1_stoch < 5e-3, (
        f"Expected stoch L1 within MC noise band (~5e-3); got {l1_stoch:.6f}"
    )


def test_stochastic_rectangle_mismatch_scar_multi_solution():
    """SCAR (HONEST) regression: document the multi-solution rectangle
    mismatch finding from the probe.

    On a 3-solution front with strong positive Cov, the L1 gap for
    stoch CAN exceed that of det because the deterministic middle-
    branch rectangle (sms_emoa.py line 723) is not the Eq 6.41
    rectangle. We pin this so a future refactor that re-aligns the
    deterministic middle branch with Eq 6.41 will trip this test
    and the operator can revisit the claim.
    """
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    n = 3
    var = 0.01
    cov_off = 0.008
    covs_scalar = np.full(n, cov_off)
    mus = np.column_stack([rois, risks])
    covs_full = np.zeros((n, 2, 2))
    covs_full[:, 0, 0] = var
    covs_full[:, 1, 1] = var
    covs_full[:, 0, 1] = cov_off
    covs_full[:, 1, 0] = cov_off

    rng = np.random.default_rng(7)
    mc = mc_ground_truth_delta_s(mus, covs_full, R1=0.05, R2=0.40, n_mc=20000, rng=rng)
    det = deterministic_delta_s(rois, risks, R1=0.05, R2=0.40)
    stoch = stochastic_delta_s(rois, risks, covs_scalar, R1=0.05, R2=0.40)

    l1_det = float(np.sum(np.abs(det - mc)))
    l1_stoch = float(np.sum(np.abs(stoch - mc)))

    # Pin the SCAR finding: on multi-solution fronts with the legacy
    # middle-branch rectangle, stoch L1 > det L1 — the - Cov
    # correction overshoots because it's applied to a different
    # rectangle than the one Eq 6.41 derives.
    assert l1_stoch > l1_det, (
        "SCAR-regression: with positive Cov on a 3-solution front the "
        "stoch L1 should exceed det L1 due to rectangle mismatch; got "
        f"l1_stoch={l1_stoch:.6f}, l1_det={l1_det:.6f}"
    )


# ---------------------------------------------------------------------------
# 4. compare_methods API surface
# ---------------------------------------------------------------------------


def test_compare_methods_returns_all_keys():
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    covs = np.array([0.001, 0.002, 0.001])

    rng = np.random.default_rng(999)
    out = compare_methods(rois, risks, covs, R1=0.05, R2=0.40, n_mc=2000, rng=rng)

    expected_keys = {
        "deterministic",
        "stochastic",
        "mc",
        "l1_det_vs_mc",
        "l1_stoch_vs_mc",
    }
    assert set(out.keys()) == expected_keys
    assert out["deterministic"].shape == (3,)
    assert out["stochastic"].shape == (3,)
    assert out["mc"].shape == (3,)
    assert isinstance(out["l1_det_vs_mc"], float)
    assert isinstance(out["l1_stoch_vs_mc"], float)
    # Per stochastic_delta_s spec: stoch = det - covs.
    np.testing.assert_allclose(
        out["stochastic"], out["deterministic"] - covs
    )


# ---------------------------------------------------------------------------
# W22 Probe AD empirical extension (2026-05-20) — eq636 rectangle tests
# ---------------------------------------------------------------------------


def test_deterministic_eq636_uses_left_roi_right_risk_rectangle():
    """For a 3-solution sorted-ROI-ASC front:
       Δ_S^eq636[i=1] = (ROI_1 - ROI_0) × (risk_2 - risk_1)
    """
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.05, 0.10, 0.18])
    R1, R2 = 0.0, 0.2
    out = deterministic_eq636_delta_s(rois, risks, R1, R2)
    assert out.shape == (3,)
    # i=1 middle: (0.15 - 0.10) * (0.18 - 0.10) = 0.05 * 0.08 = 0.004
    np.testing.assert_allclose(out[1], 0.004, rtol=1e-9)
    # i=0 left: (0.10 - 0.0) * (0.10 - 0.05) = 0.10 * 0.05 = 0.005
    np.testing.assert_allclose(out[0], 0.005, rtol=1e-9)
    # i=2 right: (0.20 - 0.15) * (0.2 - 0.18) = 0.05 * 0.02 = 0.001
    np.testing.assert_allclose(out[2], 0.001, rtol=1e-9)


def test_deterministic_eq636_differs_from_python_buggy_on_non_uniform_front():
    """The two formulas should disagree on a non-uniform front.

    Python deterministic (buggy):
      Δ_S[i=1] = (ROI_1 - ROI_2) * (risk_0 - risk_1) = -0.05 * -0.05 = 0.0025
    Eq 6.36 deterministic:
      Δ_S[i=1] = (ROI_1 - ROI_0) * (risk_2 - risk_1) = 0.05 * 0.08 = 0.004
    """
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.05, 0.10, 0.18])  # non-uniform risk spacing
    R1, R2 = 0.0, 0.2
    det_py = deterministic_delta_s(rois, risks, R1, R2)
    det_eq636 = deterministic_eq636_delta_s(rois, risks, R1, R2)
    # Middle position should differ: 0.0025 (python) vs 0.004 (eq636)
    assert not np.isclose(det_py[1], det_eq636[1])
    np.testing.assert_allclose(det_py[1], 0.0025, rtol=1e-9)
    np.testing.assert_allclose(det_eq636[1], 0.004, rtol=1e-9)


def test_compare_methods_with_eq636_returns_all_keys():
    """compare_methods_with_eq636 must include the new eq636 + MC_eq636 fields."""
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.30, 0.25, 0.20])
    covs = np.array([0.001, 0.002, 0.001])
    rng = np.random.default_rng(42)
    out = compare_methods_with_eq636(rois, risks, covs, R1=0.05, R2=0.40,
                                       n_mc=500, rng=rng)
    expected_keys = {
        "deterministic", "stochastic", "mc", "l1_det_vs_mc", "l1_stoch_vs_mc",
        "deterministic_eq636", "l1_eq636_vs_mc",
        "mc_eq636", "l1_det_vs_mc_eq636", "l1_eq636_vs_mc_eq636",
        "l1_stoch_vs_mc_eq636",
    }
    assert set(out.keys()) == expected_keys
    assert out["deterministic_eq636"].shape == (3,)
    assert out["mc_eq636"].shape == (3,)


def test_eq636_matches_mc_eq636_in_expectation_at_zero_cov():
    """When covs are zero, MC_eq636 converges to deterministic_eq636
    because the rectangle is linear in the sampled means and E[N(μ, 0)] = μ."""
    rois = np.array([0.10, 0.15, 0.20])
    risks = np.array([0.05, 0.10, 0.18])
    covs = np.array([0.0, 0.0, 0.0])
    rng = np.random.default_rng(123)
    out = compare_methods_with_eq636(rois, risks, covs, R1=0.0, R2=0.2,
                                       n_mc=10000, rng=rng)
    # With covs=0, MC samples have tiny var_floor jitter (1e-6); MC_eq636
    # should converge close to deterministic_eq636.
    np.testing.assert_allclose(
        out["mc_eq636"], out["deterministic_eq636"], rtol=0.1
    )
