"""W22-NC34 regression tests for anticipatory mutation scorer.

Hypothesis (H-NC34-anticipatory-beats-random):
  Expected Δ_S of anticipatory-scored mutant ≥ 20% higher than random mean
  on synthetic.

Coverage targets (per docs/W22-NEXT-STEPS-NC32-36.md §B3 and operator spec):
  - score shape, value sanity (better position → higher score)
  - empty / single / zero-mutant edges
  - argmax returns valid index
  - softmax invariants (β=0 uniform; β→∞ concentrates on argmax)
  - non-dominance probability ∈ [0, 1]
  - anticipatory beats random by ≥ 20% on synthetic
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.anticipatory_mutation_scorer import (
    score_mutants_by_predicted_delta_s,
    score_mutants_by_non_dominance,
    select_best_mutant_argmax,
    select_mutant_softmax,
)


# ---------------------------------------------------------------------------
# Local fixtures (same shape as test_nc30_amfc_selector for consistency)
# ---------------------------------------------------------------------------

class _FakePortfolio:
    def __init__(self, roi, risk, kalman_state=None):
        self.ROI = float(roi)
        self.risk = float(risk)
        self.kalman_state = kalman_state
        self.investment = np.zeros(5)


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


def _make_solution(roi, risk, sigma_scale=1e-8):
    """Solution with KF state at the given (ROI, risk), tight Σ."""
    x = np.array([roi, risk, 0.0, 0.0])
    P = sigma_scale * np.eye(4)
    kf = _FakeKalmanState(x, P)
    return _FakeSolution(roi, risk, kalman_state=kf)


def _make_front(rois, risks, sigma_scale=1e-8):
    return [_make_solution(r, k, sigma_scale=sigma_scale) for r, k in zip(rois, risks)]


# ---------------------------------------------------------------------------
# Test 1: shape
# ---------------------------------------------------------------------------

def test_score_mutants_returns_correct_shape():
    parent = _make_solution(0.003, 0.008)
    mutants = _make_front([0.001, 0.002, 0.004, 0.005], [0.010, 0.009, 0.007, 0.006])
    front = _make_front([0.002, 0.003], [0.009, 0.008])
    scores = score_mutants_by_predicted_delta_s(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(mutants),)


# ---------------------------------------------------------------------------
# Test 2: better position → higher score
# ---------------------------------------------------------------------------

def test_score_mutants_higher_for_better_position():
    """A mutant with clearly higher ROI (and same risk) gets a higher Δ_S
    contribution when inserted at the high end of the front.

    The contribution at the high-ROI end is dominated by (ROI − ROI_neighbor)
    × (R2 − risk), so a higher ROI strictly grows the rectangle.
    """
    parent = _make_solution(0.003, 0.008)
    # Front spans ROI 0.001..0.003; mutant_low at 0.0005 (below front, low ROI),
    # mutant_high at 0.010 (well above front).
    front = _make_front([0.001, 0.002, 0.003], [0.012, 0.010, 0.008])
    mutant_low = _make_solution(0.0005, 0.011)
    mutant_high = _make_solution(0.010, 0.005)
    mutants = [mutant_low, mutant_high]
    scores = score_mutants_by_predicted_delta_s(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    assert scores[1] > scores[0], (
        f"High-ROI mutant should beat low-ROI mutant; got {scores}"
    )


# ---------------------------------------------------------------------------
# Test 3: empty front degenerates gracefully
# ---------------------------------------------------------------------------

def test_score_mutants_handles_empty_front():
    """No Pareto to compare against → fallback to single-point rectangle
    (mutant_ROI − R1) · (R2 − mutant_risk)."""
    parent = _make_solution(0.003, 0.008)
    mutants = _make_front([0.001, 0.005], [0.010, 0.006])
    scores = score_mutants_by_predicted_delta_s(
        parent, mutants, current_front=[], horizon=1, R1=0.0, R2=0.02,
    )
    # Both finite, both > 0 (mutants are inside the (R1, R2) box).
    assert np.all(np.isfinite(scores))
    # The high-ROI low-risk mutant should get the larger rectangle.
    assert scores[1] > scores[0]


# ---------------------------------------------------------------------------
# Test 4: argmax returns valid index
# ---------------------------------------------------------------------------

def test_select_best_mutant_returns_valid_index():
    parent = _make_solution(0.003, 0.008)
    mutants = _make_front([0.001, 0.002, 0.003, 0.004], [0.012, 0.010, 0.008, 0.006])
    front = _make_front([0.002, 0.003], [0.009, 0.007])
    idx = select_best_mutant_argmax(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    assert isinstance(idx, int)
    assert 0 <= idx < len(mutants)


# ---------------------------------------------------------------------------
# Test 5: softmax returns valid index
# ---------------------------------------------------------------------------

def test_select_softmax_returns_valid_index():
    parent = _make_solution(0.003, 0.008)
    mutants = _make_front([0.001, 0.002, 0.003, 0.004], [0.012, 0.010, 0.008, 0.006])
    front = _make_front([0.002, 0.003], [0.009, 0.007])
    rng = np.random.default_rng(0)
    idx = select_mutant_softmax(
        parent, mutants, front, horizon=1, beta=1.0, R1=0.0, R2=0.02, rng=rng,
    )
    assert isinstance(idx, int)
    assert 0 <= idx < len(mutants)


# ---------------------------------------------------------------------------
# Test 6: β=0 → uniform random
# ---------------------------------------------------------------------------

def test_select_softmax_beta_zero_uniform_random():
    """β=0 disables score-weighting; sampling is uniform across mutants.

    Empirically: across N draws, the expected count per bucket is N/K with
    binomial variance. We check that no bucket dominates (chi-square style)
    and that every bucket gets at least one hit.
    """
    parent = _make_solution(0.003, 0.008)
    # 4 mutants with very different predicted Δ_S (would dominate at β>0)
    mutants = _make_front([0.0001, 0.001, 0.005, 0.020], [0.020, 0.015, 0.005, 0.001])
    front = _make_front([0.002, 0.003], [0.009, 0.007])
    rng = np.random.default_rng(123)
    n_draws = 400
    picks = Counter()
    for _ in range(n_draws):
        idx = select_mutant_softmax(
            parent, mutants, front, horizon=1, beta=0.0, R1=0.0, R2=0.02, rng=rng,
        )
        picks[idx] += 1
    # All 4 buckets seen
    assert set(picks.keys()) == {0, 1, 2, 3}
    # Each bucket within 50–200 (expected ~100 each); generous bound for variance
    for k in range(4):
        assert 50 <= picks[k] <= 200, (
            f"β=0 should be uniform; bucket {k} got {picks[k]} (want ~100). picks={dict(picks)}"
        )


# ---------------------------------------------------------------------------
# Test 7: β → ∞ → softmax ≡ argmax
# ---------------------------------------------------------------------------

def test_select_softmax_high_beta_concentrates_on_argmax():
    """At β=1e6 the softmax collapses to argmax — every draw returns the
    argmax index."""
    parent = _make_solution(0.003, 0.008)
    mutants = _make_front([0.001, 0.002, 0.005, 0.010], [0.012, 0.010, 0.006, 0.003])
    front = _make_front([0.002, 0.003], [0.009, 0.007])
    argmax_idx = select_best_mutant_argmax(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        idx = select_mutant_softmax(
            parent, mutants, front, horizon=1, beta=1e6, R1=0.0, R2=0.02, rng=rng,
        )
        assert idx == argmax_idx, (
            f"At β→∞ every draw should hit argmax={argmax_idx}; got {idx}"
        )


# ---------------------------------------------------------------------------
# Test 8: non-dominance probability in [0, 1]
# ---------------------------------------------------------------------------

def test_score_non_dominance_returns_probability_in_zero_one():
    parent = _make_solution(0.003, 0.008, sigma_scale=1e-6)
    mutants = _make_front(
        [0.001, 0.003, 0.005], [0.010, 0.008, 0.006],
        sigma_scale=1e-6,
    )
    front = _make_front(
        [0.002, 0.004], [0.009, 0.007],
        sigma_scale=1e-6,
    )
    rng = np.random.default_rng(0)
    probs = score_mutants_by_non_dominance(
        parent, mutants, front, horizon=1, n_mc=200, rng=rng,
    )
    assert probs.shape == (len(mutants),)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Test 9: anticipatory ≥ 20% higher than random mean on synthetic
# (H-NC34-anticipatory-beats-random falsifiable criterion)
# ---------------------------------------------------------------------------

def test_anticipatory_beats_random_on_synthetic():
    """10 mutants with monotonically improving predicted positions: the
    anticipatory-picked mutant's Δ_S must be ≥ 20% higher than the mean
    of all mutants' Δ_S (random-pick baseline).

    Construction:
      - 10 mutants with linearly increasing ROI (0.001..0.010) and decreasing
        risk (0.015..0.001)
      - A small reference front against which contributions are computed
    """
    parent = _make_solution(0.003, 0.008)
    rois = np.linspace(0.001, 0.010, 10)
    risks = np.linspace(0.015, 0.001, 10)
    mutants = [_make_solution(r, k) for r, k in zip(rois, risks)]
    front = _make_front([0.003, 0.005], [0.008, 0.006])

    scores = score_mutants_by_predicted_delta_s(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    mean_score = float(np.mean(scores))
    # Falsifiable: best ≥ 1.2 × mean
    assert best_score >= 1.2 * mean_score, (
        f"H-NC34 falsified: best={best_score:.4e}, mean={mean_score:.4e}, "
        f"ratio={best_score / mean_score:.3f} (want ≥ 1.20). scores={scores}"
    )


# ---------------------------------------------------------------------------
# Test 10: single mutant handled cleanly
# ---------------------------------------------------------------------------

def test_handles_single_mutant():
    parent = _make_solution(0.003, 0.008)
    mutants = [_make_solution(0.005, 0.006)]
    front = _make_front([0.002, 0.003], [0.009, 0.007])
    scores = score_mutants_by_predicted_delta_s(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])
    # Argmax of a single-element array is 0
    idx = select_best_mutant_argmax(
        parent, mutants, front, horizon=1, R1=0.0, R2=0.02,
    )
    assert idx == 0
    # Softmax: single mutant → always picks 0
    rng = np.random.default_rng(0)
    idx_sm = select_mutant_softmax(
        parent, mutants, front, horizon=1, beta=1.0, R1=0.0, R2=0.02, rng=rng,
    )
    assert idx_sm == 0


# ---------------------------------------------------------------------------
# Test 11: zero mutants raises
# ---------------------------------------------------------------------------

def test_handles_zero_mutants_raises_or_returns_none():
    parent = _make_solution(0.003, 0.008)
    front = _make_front([0.002, 0.003], [0.009, 0.007])
    with pytest.raises(ValueError):
        score_mutants_by_predicted_delta_s(
            parent, [], front, horizon=1, R1=0.0, R2=0.02,
        )
    with pytest.raises(ValueError):
        select_best_mutant_argmax(
            parent, [], front, horizon=1, R1=0.0, R2=0.02,
        )
    with pytest.raises(ValueError):
        select_mutant_softmax(
            parent, [], front, horizon=1, beta=1.0, R1=0.0, R2=0.02,
        )
    with pytest.raises(ValueError):
        score_mutants_by_non_dominance(
            parent, [], front, horizon=1, n_mc=20,
        )


# ---------------------------------------------------------------------------
# Bonus: non-dominance score collapses to 1.0 for empty front
# ---------------------------------------------------------------------------

def test_non_dominance_empty_front_returns_one():
    """With no comparison front, a mutant is vacuously non-dominated."""
    parent = _make_solution(0.003, 0.008)
    mutants = _make_front([0.001, 0.005], [0.010, 0.006])
    rng = np.random.default_rng(0)
    probs = score_mutants_by_non_dominance(
        parent, mutants, current_front=[], horizon=1, n_mc=50, rng=rng,
    )
    np.testing.assert_allclose(probs, [1.0, 1.0])


# ---------------------------------------------------------------------------
# Bonus: dominated mutant gets P_ND lower than a dominating one
# ---------------------------------------------------------------------------

def test_non_dominance_dominated_lower_than_dominating():
    """A clearly-dominated mutant gets a strictly lower non-dominance
    probability than a clearly-dominating mutant."""
    parent = _make_solution(0.003, 0.008, sigma_scale=1e-8)
    # Front member at (ROI=0.005, risk=0.005) with tight forecast
    front = _make_front([0.005], [0.005], sigma_scale=1e-8)
    # Dominated mutant: lower ROI AND higher risk
    dominated = _make_solution(0.001, 0.010, sigma_scale=1e-8)
    # Dominating mutant: higher ROI AND lower risk
    dominating = _make_solution(0.010, 0.001, sigma_scale=1e-8)
    rng = np.random.default_rng(0)
    probs = score_mutants_by_non_dominance(
        parent, [dominated, dominating], front, horizon=1, n_mc=200, rng=rng,
    )
    assert probs[1] > probs[0], (
        f"Dominating mutant should have higher P_ND than dominated; got {probs}"
    )
    # Strong form: dominating ≥ 0.95, dominated ≤ 0.05
    assert probs[1] >= 0.95
    assert probs[0] <= 0.05
