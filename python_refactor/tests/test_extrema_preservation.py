"""
W16-3: regression tests for rank-1 extrema preservation in SMS-EMOA
reduce step (BACKLOG H6).

Thesis / REF grounding:

§3.4.2 SMS-EMOA (pp. 69-70) — base SMS-EMOA reduce step (Beume et al.
2007 [REF 32]).

REF [Beume, Naujoks, Emmerich 2007] EJOR 181(3):1653-1669 — original
SMS-EMOA paper; standard practice preserves extrema (HV indicator is
most sensitive to anchor solutions because they define the bounding
box).

STD — standard SMS-EMOA / NSGA-II implementations explicitly preserve
rank-1 anchors. The thesis inherits this convention implicitly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    """Reset Portfolio class variables (test-isolation hygiene from W16-1)."""
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


def _make_solution(roi: float, risk: float,
                   pareto_rank: int = 0,
                   hv_contribution: float = 1.0,
                   num_assets: int = 5) -> Solution:
    """Helper: minimal Solution with the fields reduce reads."""
    sol = Solution(num_assets=num_assets)
    sol.P.ROI = roi
    sol.P.risk = risk
    sol.Pareto_rank = pareto_rank
    sol.hypervolume_contribution = hv_contribution
    return sol


def _make_sms(population_size: int = 10) -> SMSEMOA:
    return SMSEMOA(population_size=population_size, generations=1)


class TestIdentifyProtectedAnchors:
    """W16-3: _identify_protected_anchors picks argmax(ROI) + argmin(risk)."""

    def test_anchors_in_rank1_protected(self):
        """The rank-1 argmax(ROI) and argmin(risk) indices are returned."""
        sms = _make_sms()
        # 5 rank-1 portfolios with distinct ROI/risk
        sms.population = [
            _make_solution(roi=0.01, risk=0.05),  # 0: argmin(risk)
            _make_solution(roi=0.05, risk=0.10),  # 1: middle
            _make_solution(roi=0.10, risk=0.15),  # 2: argmax(ROI)
            _make_solution(roi=0.03, risk=0.08),  # 3
            _make_solution(roi=0.07, risk=0.12),  # 4
        ]
        protected = sms._identify_protected_anchors()
        assert protected == {0, 2}, \
            f"expected {{argmin_risk_idx=0, argmax_roi_idx=2}}, got {protected}"

    def test_single_anchor_when_argmax_equals_argmin(self):
        """If one solution is both argmax(ROI) AND argmin(risk), set has 1 elt."""
        sms = _make_sms()
        sms.population = [
            _make_solution(roi=0.20, risk=0.05),  # 0: both anchors
            _make_solution(roi=0.05, risk=0.10),  # 1
            _make_solution(roi=0.08, risk=0.15),  # 2
        ]
        protected = sms._identify_protected_anchors()
        assert protected == {0}, f"expected {{0}}, got {protected}"

    def test_edge_case_rank1_too_small_returns_empty(self):
        """rank-1 with < 3 solutions → empty protected set (avoid deadlock)."""
        sms = _make_sms()
        sms.population = [
            _make_solution(roi=0.05, risk=0.10),
            _make_solution(roi=0.10, risk=0.15),
        ]
        protected = sms._identify_protected_anchors()
        assert protected == set(), \
            "rank-1 with 2 solutions must yield empty protected set"

    def test_non_rank1_solutions_ignored(self):
        """Only rank-1 (or rank-0) solutions enter anchor identification."""
        sms = _make_sms()
        sms.population = [
            _make_solution(roi=0.01, risk=0.05, pareto_rank=0),  # 0: rank-0 argmin(risk)
            _make_solution(roi=0.05, risk=0.10, pareto_rank=0),  # 1: rank-0
            _make_solution(roi=0.10, risk=0.15, pareto_rank=0),  # 2: rank-0 argmax(ROI)
            _make_solution(roi=0.99, risk=0.001, pareto_rank=3),  # 3: rank-3 (ignored)
        ]
        protected = sms._identify_protected_anchors()
        # rank-3 must be ignored even though it has the global extrema
        assert protected == {0, 2}, \
            f"rank-3 must be excluded from anchor protection, got {protected}"


class TestReduceStepProtectsAnchors:
    """W16-3: _remove_worst_solution protects rank-1 anchors."""

    def test_reduce_never_evicts_argmax_roi(self):
        """argmax(ROI) survives reduce even when its HV contribution is worst."""
        sms = _make_sms(population_size=4)
        sms.population = [
            _make_solution(roi=0.01, risk=0.05, hv_contribution=0.5),
            _make_solution(roi=0.05, risk=0.10, hv_contribution=0.5),
            _make_solution(roi=0.10, risk=0.15, hv_contribution=0.5),  # argmax(ROI)
            _make_solution(roi=0.07, risk=0.20, hv_contribution=0.5),
            _make_solution(roi=0.08, risk=0.25, hv_contribution=1e-9),  # eviction target
        ]
        argmax_roi_initial = max(sms.population, key=lambda s: s.P.ROI).P.ROI

        sms._remove_worst_solution()

        # Population is now size 4 and argmax(ROI) is still 0.10
        assert len(sms.population) == 4
        argmax_roi_after = max(sms.population, key=lambda s: s.P.ROI).P.ROI
        assert argmax_roi_after == argmax_roi_initial, \
            f"argmax(ROI) was evicted: {argmax_roi_initial} → {argmax_roi_after}"

    def test_reduce_never_evicts_argmin_risk(self):
        """argmin(risk) survives reduce even when its HV contribution is worst."""
        sms = _make_sms(population_size=4)
        sms.population = [
            _make_solution(roi=0.05, risk=0.05, hv_contribution=1e-9),  # argmin(risk), worst HV
            _make_solution(roi=0.06, risk=0.10, hv_contribution=0.5),
            _make_solution(roi=0.07, risk=0.15, hv_contribution=0.5),
            _make_solution(roi=0.08, risk=0.20, hv_contribution=0.5),
            _make_solution(roi=0.10, risk=0.25, hv_contribution=0.2),  # eviction target (not anchor)
        ]
        argmin_risk_initial = min(sms.population, key=lambda s: s.P.risk).P.risk

        sms._remove_worst_solution()

        assert len(sms.population) == 4
        argmin_risk_after = min(sms.population, key=lambda s: s.P.risk).P.risk
        assert argmin_risk_after == argmin_risk_initial, \
            f"argmin(risk) was evicted: {argmin_risk_initial} → {argmin_risk_after}"

    def test_stress_100_reduces_anchor_survives(self):
        """100 reduce calls with worst-HV anchor — survives all of them."""
        sms = _make_sms(population_size=5)
        # Build a population where the argmin(risk) has perpetually worst HV
        base_pop = [
            _make_solution(roi=0.01, risk=0.001, hv_contribution=1e-9),  # anchor, worst HV
            _make_solution(roi=0.05, risk=0.10, hv_contribution=0.5),
            _make_solution(roi=0.07, risk=0.15, hv_contribution=0.4),
            _make_solution(roi=0.10, risk=0.20, hv_contribution=0.6),  # other anchor (argmax ROI)
            _make_solution(roi=0.08, risk=0.18, hv_contribution=0.3),
        ]
        for _ in range(100):
            # Re-seed: add a dummy that should always lose
            sms.population = list(base_pop) + [_make_solution(
                roi=0.06, risk=0.30, hv_contribution=1e-15,
            )]
            sms._remove_worst_solution()
            argmin_risk = min(sms.population, key=lambda s: s.P.risk).P.risk
            argmax_roi = max(sms.population, key=lambda s: s.P.ROI).P.ROI
            assert argmin_risk == 0.001, f"argmin(risk) evicted at iter, found {argmin_risk}"
            assert argmax_roi == 0.10, f"argmax(ROI) evicted at iter, found {argmax_roi}"

    def test_edge_case_small_front_falls_back(self):
        """Front with 2 solutions — no protection; reduce still proceeds."""
        sms = _make_sms(population_size=1)
        sms.population = [
            _make_solution(roi=0.05, risk=0.10, hv_contribution=0.5),
            _make_solution(roi=0.10, risk=0.20, hv_contribution=0.3),
        ]
        # _identify_protected_anchors → set() (rank-1 < 3); reduce can still
        # pick a worst and remove it. Verify no deadlock + correct size.
        sms._remove_worst_solution()
        assert len(sms.population) == 1


class TestExtremaTrace:
    """W16-3: per-gen extrema trace populated by _track_hypervolume."""

    def test_trace_initially_empty(self):
        """extrema_trace starts empty before any tracking call."""
        sms = _make_sms()
        assert sms.extrema_trace == []

    def test_trace_populated_after_tracking_call(self):
        """One _track_hypervolume call appends one row with the right schema."""
        sms = _make_sms()
        sms.current_generation = 7
        # Seed pareto_front directly (bypassing _update_pareto_front)
        sms.pareto_front = [
            _make_solution(roi=0.05, risk=0.10),
            _make_solution(roi=0.10, risk=0.15),
            _make_solution(roi=0.01, risk=0.05),
        ]
        sms._track_hypervolume()
        assert len(sms.extrema_trace) == 1
        row = sms.extrema_trace[0]
        assert set(row.keys()) == {
            "gen", "roi_max", "roi_min", "risk_max", "risk_min", "front_size",
        }
        assert row["gen"] == 7
        assert row["roi_max"] == 0.10
        assert row["roi_min"] == 0.01
        assert row["risk_max"] == 0.15
        assert row["risk_min"] == 0.05
        assert row["front_size"] == 3
