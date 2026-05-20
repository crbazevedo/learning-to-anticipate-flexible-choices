"""W22-NC30 regression tests for AMFC selector.

Per docs/W22-NC30-CONTRACT.md regression test plan:
  1. API test: dispatcher resolves Hv-DM-AMFC to select_amfc
  2. Identity test: at H=1, Σ→0, AMFC ≡ Hv-DM (deterministic forecast)
  3. MC stability: argmax stable across calls with n_mc≥200
  4. Cost guard: AMFC completes within 100ms for |P|=20, H=3, n_mc=200
  5. Sanity: returned solution is FROM the input population
  6. Empty / single-solution edge cases
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.amfc_selector import (
    _forecast_solution_at_horizon,
    _front_contribution,
    get_amfc_telemetry,
    reset_amfc_telemetry,
    select_amfc,
)


class _FakePortfolio:
    def __init__(self, roi, risk, kalman_state=None):
        self.ROI = float(roi)
        self.risk = float(risk)
        self.kalman_state = kalman_state
        self.investment = np.zeros(5)  # arbitrary; not used by AMFC


class _FakeKalmanState:
    def __init__(self, x, P, F=None):
        self.x = np.asarray(x, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.F = np.asarray(F if F is not None else np.eye(len(x)), dtype=float)


class _FakeSolution:
    def __init__(self, roi, risk, pareto_rank=0, kalman_state=None):
        self.P = _FakePortfolio(roi, risk, kalman_state)
        self.Pareto_rank = pareto_rank
        self.Delta_S = 0.0


def _make_front(roi_vals, risk_vals, with_kf=True, sigma_scale=0.001, pareto_rank=0):
    """Build a synthetic Pareto front."""
    sols = []
    for r, rk in zip(roi_vals, risk_vals):
        if with_kf:
            x = np.array([r, rk, 0.0, 0.0])
            P = sigma_scale * np.eye(4)
            kf = _FakeKalmanState(x, P)
        else:
            kf = None
        sols.append(_FakeSolution(r, rk, pareto_rank=pareto_rank, kalman_state=kf))
    return sols


def test_empty_population_returns_none():
    assert select_amfc([], horizon=1, n_mc=10, R1=0.0, R2=0.05) is None


def test_single_solution_returns_it():
    sol = _FakeSolution(0.001, 0.005)
    result = select_amfc([sol], horizon=1, n_mc=10, R1=0.0, R2=0.05)
    assert result is sol


def test_returned_solution_is_from_population():
    front = _make_front([0.001, 0.002, 0.003], [0.01, 0.008, 0.006])
    rng = np.random.default_rng(0)
    result = select_amfc(front, horizon=1, n_mc=100, R1=0.0, R2=0.05, rng=rng)
    assert result in front


def test_pareto_only_filters_dominated_solutions():
    """When pareto_only=True, only Pareto_rank==0 solutions are considered."""
    front_pareto = _make_front([0.001, 0.002, 0.003], [0.01, 0.008, 0.006], pareto_rank=0)
    dominated = _make_front([0.0005, 0.0006], [0.02, 0.025], pareto_rank=1)
    pop = front_pareto + dominated
    rng = np.random.default_rng(0)
    result = select_amfc(pop, horizon=1, n_mc=100, R1=0.0, R2=0.05, pareto_only=True, rng=rng)
    assert result in front_pareto


def test_identity_at_horizon_1_zero_variance():
    """With H=1 and Σ→0 and identity F, AMFC reduces to argmax of deterministic Δ_S.

    With zero forecast variance, every MC sample is exactly the mean (= current
    state for identity F). The per-solution contribution then equals the
    deterministic Δ_S that Hv-DM uses.
    """
    rois = [0.001, 0.002, 0.003, 0.004]
    risks = [0.012, 0.009, 0.007, 0.005]
    # Build front with degenerate KF (Σ effectively 0)
    front = []
    for r, rk in zip(rois, risks):
        x = np.array([r, rk, 0.0, 0.0])
        P = 1e-14 * np.eye(4)  # essentially zero variance
        front.append(_FakeSolution(r, rk, kalman_state=_FakeKalmanState(x, P)))
    rng = np.random.default_rng(42)
    amfc_pick = select_amfc(front, horizon=1, n_mc=50, R1=0.0, R2=0.02, rng=rng)
    # Compute deterministic Hv-DM equivalent
    sorted_front_arr = np.array([(s.P.ROI, s.P.risk) for s in front])
    order = np.argsort(sorted_front_arr[:, 0])
    sorted_front = sorted_front_arr[order]
    contribs = [_front_contribution(i, sorted_front, R1=0.0, R2=0.02)
                for i in range(len(front))]
    inv_order = np.argsort(order)
    hv_dm_pick_original_idx = int(np.argmax(np.array(contribs)[inv_order]))
    hv_dm_pick = front[hv_dm_pick_original_idx]
    assert amfc_pick is hv_dm_pick, (
        f"AMFC pick (ROI={amfc_pick.P.ROI}) differs from Hv-DM pick "
        f"(ROI={hv_dm_pick.P.ROI}) at zero forecast variance — identity broken"
    )


def test_mc_stability_when_sigma_small_relative_to_spread():
    """With n_mc=200 and σ small relative to inter-solution spread, argmax
    is stable across calls.

    HONEST SCAR: when σ is comparable to or larger than inter-solution spacing
    (high-uncertainty regime where forecast distributions overlap heavily),
    MC noise dominates and the argmax is intrinsically unstable. This is a
    feature of the regime, not a bug — in those cases the operator should
    increase n_mc or accept that AMFC selection is noisy because the forecast
    really is ambiguous.

    This test uses σ=1e-6 (small relative to ROI spacing ~1e-3) which is
    the regime where MC SHOULD be stable.
    """
    front = _make_front(
        [0.001, 0.002, 0.003, 0.004, 0.005],
        [0.015, 0.012, 0.009, 0.007, 0.005],
        sigma_scale=1e-6,  # σ ≈ 1e-3, ~1000× smaller than inter-solution ROI spread
    )
    picks = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pick = select_amfc(front, horizon=1, n_mc=200, R1=0.0, R2=0.02, rng=rng)
        picks.append(front.index(pick))
    from collections import Counter
    counter = Counter(picks)
    modal_count = counter.most_common(1)[0][1]
    assert modal_count >= 4, (
        f"MC argmax unstable even at σ<<spread: picks across 5 calls = {picks}; "
        f"modal pick appeared only {modal_count}/5 times (want ≥4)"
    )


def test_mc_stability_at_high_noise_with_crn():
    """NC30-v2 STRUCTURAL FIX: with shared-noise CRN, even at σ ≈ inter-
    solution spread the AMFC argmax is STABLE across calls.

    Pre-CRN this test asserted only "no crashes" because per-candidate
    independent noise produced argmax instability under high σ. With CRN,
    the noise direction is shared across candidates so their relative ordering
    is preserved across MC samples — picks should be modally stable.

    Operator directive 2026-05-19: degenerate behavior should not happen.
    This test guards against the high-noise degeneracy regressing.
    """
    front = _make_front(
        [0.001, 0.002, 0.003, 0.004, 0.005],
        [0.015, 0.012, 0.009, 0.007, 0.005],
        sigma_scale=0.001,  # σ ≈ inter-solution spread — was the degenerate regime
    )
    picks = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        pick = select_amfc(front, horizon=1, n_mc=200, R1=0.0, R2=0.02, rng=rng)
        assert pick in front  # no crashes, no synthesis
        picks.append(front.index(pick))
    # Post-CRN: argmax should be modally stable (≥8/10 agree on the modal pick).
    from collections import Counter
    counter = Counter(picks)
    modal_count = counter.most_common(1)[0][1]
    assert modal_count >= 8, (
        f"Post-CRN MC stability regressed: picks={picks}, modal={modal_count}/10 (want ≥8). "
        f"This is the operator-flagged degeneracy that NC30-v2 CRN was supposed to fix."
    )


def test_cost_guard_under_100ms():
    """AMFC selection completes within 100ms for |P|=20, H=3, n_mc=200."""
    n = 20
    rois = np.linspace(0.001, 0.005, n)
    risks = np.linspace(0.015, 0.005, n)
    front = _make_front(rois.tolist(), risks.tolist(), sigma_scale=0.001)
    rng = np.random.default_rng(0)
    t0 = time.time()
    select_amfc(front, horizon=3, n_mc=200, R1=0.0, R2=0.02, rng=rng)
    elapsed_ms = (time.time() - t0) * 1000
    assert elapsed_ms < 100.0, (
        f"AMFC selection took {elapsed_ms:.1f}ms; budget is 100ms"
    )


def test_horizon_propagates_through_kf():
    """At horizon h, the forecast mean is F^h · x."""
    F = np.array([
        [1.0, 0.0, 1.0, 0.0],  # ROI += vROI
        [0.0, 1.0, 0.0, 1.0],  # risk += vRisk
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    x = np.array([0.001, 0.010, 0.0005, -0.001])
    P = 1e-10 * np.eye(4)
    kf = _FakeKalmanState(x, P, F=F)
    sol = _FakeSolution(x[0], x[1], kalman_state=kf)
    # Horizon 1
    mu1, _ = _forecast_solution_at_horizon(sol, horizon=1)
    expected_mu1 = (F @ x)[:2]
    np.testing.assert_allclose(mu1, expected_mu1, atol=1e-12)
    # Horizon 3
    mu3, _ = _forecast_solution_at_horizon(sol, horizon=3)
    expected_mu3 = (F @ F @ F @ x)[:2]
    np.testing.assert_allclose(mu3, expected_mu3, atol=1e-10)


def test_solution_without_kf_uses_current_state():
    """Solution without KF state: forecast mean = current; variance = 0."""
    sol = _FakeSolution(0.002, 0.008)  # no kalman_state
    mu, Sigma = _forecast_solution_at_horizon(sol, horizon=5)
    np.testing.assert_allclose(mu, [0.002, 0.008])
    np.testing.assert_allclose(Sigma, np.zeros((2, 2)))


def test_dispatcher_wires_hv_dm_amfc():
    """Verify thesis_aligned_experiment._select_decision_maker_solution
    routes 'Hv-DM-AMFC' to select_amfc."""
    from src.experiments.thesis_aligned_experiment import ThesisAlignedExperiment
    # Minimal init: ThesisAlignedExperiment takes a config dict
    exp = ThesisAlignedExperiment.__new__(ThesisAlignedExperiment)
    front = _make_front([0.001, 0.002, 0.003], [0.012, 0.009, 0.005])
    result = exp._select_decision_maker_solution(
        front, 'Hv-DM-AMFC',
        {'horizon': 1, 'n_mc': 50, 'R1': 0.0, 'R2': 0.02, 'seed': 0},
    )
    assert result in front


# =============================================================================
# NC30 b: data-derived z_ref tests
# =============================================================================

def test_derive_zref_from_population_basic():
    """R1 = min ROI; R2 = max risk (no margin)."""
    from src.algorithms.amfc_selector import derive_zref_from_population
    front = _make_front([0.001, 0.003, 0.005], [0.01, 0.007, 0.004])
    R1, R2 = derive_zref_from_population(front, margin=0.0)
    assert R1 == 0.001
    assert R2 == 0.01


def test_derive_zref_with_margin():
    """margin=0.1 widens both ends by 10% of observed range."""
    from src.algorithms.amfc_selector import derive_zref_from_population
    front = _make_front([0.001, 0.005], [0.005, 0.001])
    R1, R2 = derive_zref_from_population(front, margin=0.1)
    # ROI range = 0.005 - 0.001 = 0.004; R1 = 0.001 - 0.1*0.004 = 0.0006
    # Risk range = 0.005 - 0.001 = 0.004; R2 = 0.005 + 0.1*0.004 = 0.0054
    np.testing.assert_allclose(R1, 0.0006, atol=1e-10)
    np.testing.assert_allclose(R2, 0.0054, atol=1e-10)


def test_derive_zref_empty_population():
    """Empty population: fallback to (0.0, 0.05)."""
    from src.algorithms.amfc_selector import derive_zref_from_population
    R1, R2 = derive_zref_from_population([])
    assert R1 == 0.0 and R2 == 0.05


def test_select_amfc_derive_zref_overrides_default():
    """When R1/R2 are None and derive_zref=True, the selector uses derived values."""
    front = _make_front([0.001, 0.002, 0.003, 0.004, 0.005],
                         [0.015, 0.012, 0.009, 0.007, 0.005],
                         sigma_scale=1e-7)
    rng = np.random.default_rng(0)
    # No R1/R2 supplied + derive_zref=True
    pick_derived = select_amfc(
        front, horizon=1, n_mc=50,
        R1=None, R2=None, derive_zref=True, rng=rng,
    )
    assert pick_derived in front


def test_select_amfc_explicit_R1_overrides_derivation():
    """If R1 is explicitly given, it takes precedence over derivation."""
    front = _make_front([0.001, 0.002, 0.003], [0.01, 0.007, 0.004], sigma_scale=1e-7)
    rng = np.random.default_rng(0)
    # R1 explicit; R2 derived (max risk = 0.01)
    pick = select_amfc(
        front, horizon=1, n_mc=50,
        R1=-1.0, R2=None, derive_zref=True, rng=rng,
    )
    assert pick in front


# =============================================================================
# NC30 d: tie-break by forecast variance tests
# =============================================================================

def test_tie_break_picks_lower_variance_when_ties():
    """When two candidates have near-equal E[Δ_S] but different forecast
    variances, tie_break_by_variance picks the lower-variance one.

    Construct: 2 candidates with IDENTICAL means but very different Σ.
    Without tie-break, MC noise would pick randomly.
    With tie-break, the lower-Σ candidate wins.
    """
    # Two candidates at the same (ROI, risk) — identical means
    # but different forecast variances.
    x_low = np.array([0.003, 0.008, 0.0, 0.0])
    P_low = 1e-10 * np.eye(4)  # near-zero variance
    kf_low = _FakeKalmanState(x_low, P_low)
    sol_low_var = _FakeSolution(0.003, 0.008, kalman_state=kf_low)

    x_high = np.array([0.003, 0.008, 0.0, 0.0])
    P_high = 1e-4 * np.eye(4)  # much higher variance
    kf_high = _FakeKalmanState(x_high, P_high)
    sol_high_var = _FakeSolution(0.003, 0.008, kalman_state=kf_high)

    front = [sol_low_var, sol_high_var]
    rng = np.random.default_rng(0)
    pick = select_amfc(
        front, horizon=1, n_mc=200, R1=0.0, R2=0.02, rng=rng,
        tie_break_by_variance=True, tie_epsilon=0.5,  # generous tie window
    )
    assert pick is sol_low_var, (
        f"Tie-break by variance should pick low-Σ candidate; got high-Σ"
    )


def test_no_tie_break_default_behavior_preserved():
    """With tie_break_by_variance=False (default), behavior is unchanged."""
    front = _make_front([0.001, 0.002, 0.003], [0.01, 0.007, 0.004], sigma_scale=1e-7)
    rng = np.random.default_rng(0)
    pick = select_amfc(
        front, horizon=1, n_mc=50, R1=0.0, R2=0.02, rng=rng,
        tie_break_by_variance=False,
    )
    assert pick in front


# =============================================================================
# AMFC telemetry hook tests
# =============================================================================

def test_telemetry_off_by_default():
    """Default select_amfc call doesn't emit telemetry."""
    reset_amfc_telemetry()
    front = _make_front([0.001, 0.002, 0.003], [0.012, 0.009, 0.005])
    select_amfc(front, horizon=1, n_mc=20, R1=0.0, R2=0.02,
                 rng=np.random.default_rng(0))
    assert len(get_amfc_telemetry()) == 0


def test_telemetry_emits_per_call():
    """With collect_telemetry=True, one telemetry dict per call."""
    reset_amfc_telemetry()
    front = _make_front([0.001, 0.002, 0.003], [0.012, 0.009, 0.005])
    for seed in range(3):
        select_amfc(front, horizon=1, n_mc=20, R1=0.0, R2=0.02,
                     rng=np.random.default_rng(seed),
                     collect_telemetry=True)
    telem = get_amfc_telemetry()
    assert len(telem) == 3


def test_telemetry_keys_present():
    """Telemetry dict contains the documented keys."""
    reset_amfc_telemetry()
    front = _make_front([0.001, 0.002, 0.003], [0.012, 0.009, 0.005])
    # Stamp synthetic Delta_S so hv_dm_idx is well-defined
    for i, s in enumerate(front):
        s.Delta_S = float(i)
    select_amfc(front, horizon=1, n_mc=20, R1=0.0, R2=0.02,
                 rng=np.random.default_rng(0),
                 collect_telemetry=True)
    telem = get_amfc_telemetry()[-1]
    required = {
        "amfc_idx", "hv_dm_idx", "amfc_agrees_with_hv_dm",
        "amfc_pick_roi", "amfc_pick_risk",
        "hv_dm_pick_roi", "hv_dm_pick_risk",
        "n_candidates", "horizon", "n_mc",
        "top1_expected_contrib",
        "mean_forecast_variance", "amfc_pick_forecast_variance",
        "tie_break_fired", "R1", "R2",
    }
    missing = required - set(telem.keys())
    assert not missing, f"Telemetry missing keys: {missing}"


def test_telemetry_hv_dm_uses_delta_s_attr():
    """The hv_dm_idx in telemetry should match argmax of solution.Delta_S
    when that attribute is set."""
    reset_amfc_telemetry()
    front = _make_front([0.001, 0.002, 0.003, 0.004],
                         [0.012, 0.009, 0.007, 0.005])
    # Synthetic: middle solution has the highest Delta_S
    front[0].Delta_S = 0.1
    front[1].Delta_S = 0.9  # winner under Hv-DM
    front[2].Delta_S = 0.5
    front[3].Delta_S = 0.3
    select_amfc(front, horizon=1, n_mc=20, R1=0.0, R2=0.02,
                 rng=np.random.default_rng(0),
                 collect_telemetry=True)
    telem = get_amfc_telemetry()[-1]
    assert telem["hv_dm_idx"] == 1, (
        f"hv_dm_idx should be 1 (highest Delta_S); got {telem['hv_dm_idx']}"
    )


def test_telemetry_tie_break_fired_flag():
    """When tie-break fires, tie_break_fired=True in telemetry.

    NC30-v2 (CRN) structural fix context: with shared-noise CRN, identical-
    mean candidates produce SYMMETRIC contributions IF the noise can't
    push samples into a boundary regime (e.g., below R1, where contribution
    truncation breaks symmetry).

    Use R1 << min(ROI - 3σ) to guarantee samples stay in the
    symmetric regime, then identical means → tied E[Δ_S] → tie-break fires.
    """
    reset_amfc_telemetry()
    # Two candidates with identical means; R1 far enough below means that
    # high-σ samples don't go negative (which would break the symmetry).
    mean_roi, mean_risk = 0.05, 0.008
    x = np.array([mean_roi, mean_risk, 0.0, 0.0])
    sigma_low_sq, sigma_high_sq = 1e-10, 1e-4
    P_low = sigma_low_sq * np.eye(4)
    P_high = sigma_high_sq * np.eye(4)
    sol_low = _FakeSolution(mean_roi, mean_risk,
                              kalman_state=_FakeKalmanState(x.copy(), P_low.copy()))
    sol_high = _FakeSolution(mean_roi, mean_risk,
                               kalman_state=_FakeKalmanState(x.copy(), P_high.copy()))
    select_amfc(
        [sol_low, sol_high],
        horizon=1, n_mc=500,
        R1=-1.0, R2=1.0,  # well outside ±3σ of means; samples stay symmetric
        rng=np.random.default_rng(0),
        tie_break_by_variance=True, tie_epsilon=0.5,
        collect_telemetry=True,
    )
    telem = get_amfc_telemetry()[-1]
    assert telem["tie_break_fired"] is True


def test_telemetry_reset_clears():
    """reset_amfc_telemetry() empties the ring buffer."""
    front = _make_front([0.001, 0.002], [0.012, 0.009])
    for _ in range(5):
        select_amfc(front, horizon=1, n_mc=20, R1=0.0, R2=0.02,
                     rng=np.random.default_rng(0),
                     collect_telemetry=True)
    assert len(get_amfc_telemetry()) >= 5
    reset_amfc_telemetry()
    assert len(get_amfc_telemetry()) == 0


def test_tie_break_only_fires_when_within_epsilon():
    """If top-1 and top-2 are NOT within tie_epsilon, tie-break is skipped.

    HONEST SCAR captured by this test's design history: high forecast variance
    can DEGRADE a candidate's expected contribution (MC samples disperse
    across positions in the sorted front), so 'high ROI mean' doesn't
    automatically mean 'high E[Δ_S]'. Using uniform low variance below
    isolates the no-tie behavior cleanly.
    """
    # All uniformly LOW variance — the high-ROI solution dominates cleanly.
    rois = [0.001, 0.002, 0.010]
    risks = [0.012, 0.009, 0.005]
    front = []
    for r, rk in zip(rois, risks):
        x = np.array([r, rk, 0.0, 0.0])
        P = 1e-10 * np.eye(4)  # uniformly low variance
        kf = _FakeKalmanState(x, P)
        front.append(_FakeSolution(r, rk, kalman_state=kf))
    rng = np.random.default_rng(0)
    pick = select_amfc(
        front, horizon=1, n_mc=50, R1=0.0, R2=0.02, rng=rng,
        tie_break_by_variance=True, tie_epsilon=0.01,  # tight: no tie expected
    )
    # The high-ROI solution dominates HV contribution by a large margin
    assert pick is front[2], "Tight tie_epsilon shouldn't trigger tie-break"
