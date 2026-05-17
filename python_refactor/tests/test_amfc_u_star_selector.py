"""
W17-4: regression tests for AMFC u*_{t-1} selector per thesis Eq 6.42
(W16-2-CARRY-1 + BACKLOG M5 partial closure).

Thesis grounding (verbatim):

§6.4 "Estimating the Anticipated Maximal Flexible Choice", Eq (6.42),
p. 133:
    "AMFC selects, from the available essential set of Pareto-flexible
     decision candidates Û_t^{N*}, the portfolio u_t^{(i*)*} which is
     predicted to lead to the maximum future expected hypervolume:
     u_t^{(i*)*} = argmax_{u_t^{(i)*} ∈ Û_t^{N*}}
                     E[Hypv(Ẑ_{t+1}^{(i)*})]"
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from experiments.walk_forward import _select_amfc_index
from experiments.oos_evaluator import compute_per_portfolio_efhv


class TestSelectAMFCIndex:
    """_select_amfc_index picks argmax with deterministic tie-break."""

    def test_picks_argmax(self):
        """Clear winner → its index."""
        efhv = np.array([0.001, 0.005, 0.003, 0.002])
        assert _select_amfc_index(efhv) == 1

    def test_tie_break_smallest_index(self):
        """Ties → smallest index wins (deterministic via np.argmax)."""
        efhv = np.array([0.005, 0.005, 0.003, 0.001])
        assert _select_amfc_index(efhv) == 0

    def test_all_zero_falls_back_to_zero_with_warning(self, caplog):
        """All-zero EFHV → fallback index 0 + logged warning."""
        efhv = np.zeros(4)
        with caplog.at_level(logging.WARNING):
            idx = _select_amfc_index(efhv)
        assert idx == 0
        assert any("W17-4" in r.message for r in caplog.records)

    def test_all_nan_falls_back_to_zero(self):
        """All-NaN EFHV → fallback index 0."""
        efhv = np.array([float("nan")] * 3)
        idx = _select_amfc_index(efhv)
        assert idx == 0

    def test_empty_array_returns_zero(self):
        """Empty EFHV → 0 (degenerate; caller should already have weights[0])."""
        efhv = np.zeros(0)
        assert _select_amfc_index(efhv) == 0


class TestComputePerPortfolioEFHV:
    """compute_per_portfolio_efhv returns sensible per-portfolio array."""

    def test_returns_length_matches_n_portfolios(self):
        # Synthetic 2 portfolios x 3 assets; 50 days of returns
        rng = np.random.default_rng(42)
        oos = pd.DataFrame(
            rng.normal(loc=0.001, scale=0.01, size=(50, 3)),
            columns=[f"a{i}" for i in range(3)],
        )
        weights = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.2, 0.3, 0.5]),
        ]
        efhv = compute_per_portfolio_efhv(
            pareto_weights=weights,
            oos_returns=oos,
            n_samples=20,
            rng=rng,
        )
        assert efhv.shape == (2,)
        assert all(np.isfinite(efhv))

    def test_higher_return_portfolio_wins(self):
        """Among 2 portfolios with same risk profile, higher-return wins."""
        # Construct a returns DataFrame where asset 0 has much higher
        # mean return than asset 1; both with similar variance.
        n_days = 200
        rng = np.random.default_rng(123)
        col0 = rng.normal(loc=0.005, scale=0.01, size=n_days)
        col1 = rng.normal(loc=0.0001, scale=0.01, size=n_days)
        oos = pd.DataFrame({"a0": col0, "a1": col1})
        # Two portfolios: 100% asset 0 (high return), 100% asset 1 (low return)
        weights = [
            np.array([1.0, 0.0]),  # high return
            np.array([0.0, 1.0]),  # low return
        ]
        efhv = compute_per_portfolio_efhv(
            pareto_weights=weights,
            oos_returns=oos,
            n_samples=200,
            rng=rng,
            z_ref=(0.2, 0.0),  # risk_max, return_min
        )
        # Higher-return portfolio gets larger HV (rectangle width)
        assert efhv[0] > efhv[1], (
            f"high-return portfolio should win: efhv[0]={efhv[0]}, "
            f"efhv[1]={efhv[1]}"
        )

    def test_empty_weights_returns_empty(self):
        oos = pd.DataFrame({"a0": [0.01, 0.02, 0.03], "a1": [0.0, 0.01, -0.01]})
        efhv = compute_per_portfolio_efhv(
            pareto_weights=[],
            oos_returns=oos,
            n_samples=10,
        )
        assert efhv.shape == (0,)
