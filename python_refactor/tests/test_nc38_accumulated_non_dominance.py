"""W22-NC38 regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.accumulated_non_dominance import (
    _get_gamma,
    accumulated_non_dominance_score,
    joint_non_dominance_probability,
    pairwise_non_dominance_probability_analytical,
    rank_by_accumulated_non_dominance,
    select_by_accumulated_non_dominance,
)


# Test fixtures
class _FakeKalmanState:
    def __init__(self, x, P, F=None):
        self.x = np.asarray(x, dtype=float)
        self.P = np.asarray(P, dtype=float)
        self.F = np.asarray(F if F is not None else np.eye(len(x)), dtype=float)


class _FakePortfolio:
    def __init__(self, roi, risk, kf=None):
        self.ROI = float(roi)
        self.risk = float(risk)
        self.kalman_state = kf
        self.investment = np.zeros(5)


class _FakeSolution:
    def __init__(self, roi, risk, pareto_rank=0, sigma_scale=1e-6):
        x = np.array([roi, risk, 0.0, 0.0])
        P = sigma_scale * np.eye(4)
        self.P = _FakePortfolio(roi, risk, _FakeKalmanState(x, P))
        self.Pareto_rank = pareto_rank


def test_pairwise_non_dominance_equal_means_returns_half_or_one():
    """Two solutions with IDENTICAL means + non-zero variance: under symmetry,
    P(s dom o) = P(o dom s) → P(ND) = 1 - 2·P(dom). For diagonal cov σ²=0.01,
    P(dom) ≈ 0.25 → P(ND) ≈ 0.5."""
    mu = np.array([0.001, 0.005])
    sigma = np.diag([0.01, 0.01])
    p = pairwise_non_dominance_probability_analytical(mu, sigma, mu, sigma)
    assert 0.4 < p < 0.6, f"Expected ~0.5; got {p}"


def test_pairwise_non_dominance_dominating_pair_returns_low_prob():
    """If s strictly dominates o (higher ROI, lower risk), P(non-dominance) → 0."""
    mu_s = np.array([0.010, 0.001])  # high ROI, low risk
    mu_o = np.array([0.001, 0.010])  # low ROI, high risk (dominated)
    cov_tight = np.diag([1e-12, 1e-12])
    p = pairwise_non_dominance_probability_analytical(mu_s, cov_tight, mu_o, cov_tight)
    assert p < 0.1, f"Expected ~0; got {p}"


def test_pairwise_non_dominance_non_dominating_pair_returns_one():
    """If s is on one corner and o on opposite (both higher in different dims),
    P(non-dominance) → 1."""
    mu_s = np.array([0.010, 0.010])  # high ROI, high risk
    mu_o = np.array([0.001, 0.001])  # low ROI, low risk
    cov_tight = np.diag([1e-12, 1e-12])
    # Neither dominates the other (s higher ROI but also higher risk)
    p = pairwise_non_dominance_probability_analytical(mu_s, cov_tight, mu_o, cov_tight)
    assert p > 0.9, f"Expected ~1; got {p}"


def test_joint_non_dominance_empty_others_returns_one():
    """No others to compare against → P(ND) = 1."""
    sol = _FakeSolution(0.001, 0.005)
    p = joint_non_dominance_probability(sol, [], horizon=1)
    assert p == 1.0


def test_joint_non_dominance_single_other():
    """One other solution → joint = pairwise."""
    s = _FakeSolution(0.005, 0.005, sigma_scale=1e-6)
    other = _FakeSolution(0.001, 0.010, sigma_scale=1e-6)  # dominated by s
    p = joint_non_dominance_probability(s, [other], horizon=1)
    # s dominates other → P(non-dom) → 0
    assert p < 0.1


def test_joint_non_dominance_decreases_with_more_others():
    """Adding more dominating others → joint P(ND) decreases monotonically
    (or stays same)."""
    s = _FakeSolution(0.005, 0.005, sigma_scale=1e-6)
    # Three solutions that all dominate s
    o1 = _FakeSolution(0.010, 0.001, sigma_scale=1e-6)
    o2 = _FakeSolution(0.011, 0.001, sigma_scale=1e-6)
    o3 = _FakeSolution(0.012, 0.001, sigma_scale=1e-6)
    p_1 = joint_non_dominance_probability(s, [o1], horizon=1)
    p_2 = joint_non_dominance_probability(s, [o1, o2], horizon=1)
    p_3 = joint_non_dominance_probability(s, [o1, o2, o3], horizon=1)
    # Each pair has independent probability < 1, so joint product decreases
    assert p_2 <= p_1 + 1e-9
    assert p_3 <= p_2 + 1e-9


def test_accumulated_score_zero_horizon_is_zero():
    """max_horizon=0 → no terms in the sum → score = 0."""
    s = _FakeSolution(0.005, 0.005)
    o = _FakeSolution(0.001, 0.010)
    score = accumulated_non_dominance_score(s, [o], max_horizon=0)
    assert score == 0.0


def test_accumulated_score_single_horizon_uses_only_h1():
    """max_horizon=1 → only h=1 contributes."""
    s = _FakeSolution(0.005, 0.005)
    o = _FakeSolution(0.010, 0.010)  # neither dominates
    p_1 = joint_non_dominance_probability(s, [o], horizon=1)
    gamma = _get_gamma()
    score = accumulated_non_dominance_score(s, [o], max_horizon=1)
    np.testing.assert_allclose(score, gamma * p_1, atol=1e-10)


def test_accumulated_score_multi_horizon_increases_with_h():
    """Adding more horizons should INCREASE the score (since each term
    contributes a positive γ^h · P(ND))."""
    s = _FakeSolution(0.010, 0.010)
    o = _FakeSolution(0.001, 0.001)  # mutual non-dominance
    score_h1 = accumulated_non_dominance_score(s, [o], max_horizon=1)
    score_h3 = accumulated_non_dominance_score(s, [o], max_horizon=3)
    assert score_h3 > score_h1


def test_select_by_accumulated_returns_valid_index():
    pop = [_FakeSolution(0.001 + 0.001*i, 0.010 - 0.001*i) for i in range(5)]
    idx = select_by_accumulated_non_dominance(pop, max_horizon=2)
    assert 0 <= idx < len(pop)


def test_select_by_accumulated_pareto_only_restricts():
    """When pareto_only=True, only Pareto-rank-0 considered."""
    pop = [_FakeSolution(0.001, 0.010, pareto_rank=0)]
    dominated = [_FakeSolution(0.0005, 0.020, pareto_rank=1)]
    pop_all = pop + dominated
    idx = select_by_accumulated_non_dominance(pop_all, pareto_only=True, max_horizon=2)
    # Should select from pop (rank 0), not from dominated
    assert idx < len(pop)  # idx 0 is the only rank-0


def test_rank_by_accumulated_returns_per_solution_scores():
    pop = [_FakeSolution(0.001 + 0.002*i, 0.010 - 0.002*i) for i in range(4)]
    scores = rank_by_accumulated_non_dominance(pop, max_horizon=2)
    assert scores.shape == (4,)
    assert np.all(scores >= 0.0)


def test_gamma_env_var_controls_discount(monkeypatch):
    """W22_NC29A_GAMMA env var controls γ for NC38."""
    monkeypatch.setenv("W22_NC29A_GAMMA", "0.5")
    s = _FakeSolution(0.010, 0.010)
    o = _FakeSolution(0.001, 0.001)
    # With γ=0.5 and H=2: score = 0.5·p_1 + 0.25·p_2
    # Both p's are ~1 (mutual non-dom) → score ≈ 0.75
    score = accumulated_non_dominance_score(s, [o], max_horizon=2)
    monkeypatch.setenv("W22_NC29A_GAMMA", "0.9")
    score_high_gamma = accumulated_non_dominance_score(s, [o], max_horizon=2)
    # γ=0.9 → 0.9·p_1 + 0.81·p_2 ≈ 1.71
    assert score_high_gamma > score
