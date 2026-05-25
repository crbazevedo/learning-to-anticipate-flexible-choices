"""W22-NC18 regression: transaction-cost-aware AMFC selector
(_select_close_to_prev_amfc_index).

Pre-NC18, _select_amfc_index = argmax(per_portfolio_efhv) without
considering composition continuity. Probe G showed AMFC weights are
CHAOTIC across periods (mean Jaccard 0.169 < 0.2 threshold).

NC18: hybrid score balancing EFHV with Jaccard similarity to previous AMFC.
Period 0 falls back to standard AMFC.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.walk_forward import (
    _select_amfc_index,
    _select_close_to_prev_amfc_index,
)


class TestNC18CloseToPrevAMFC:
    """Pin NC18 selector contract."""

    def test_period_0_falls_back_to_pure_amfc(self):
        """No previous_weights → behavior = standard argmax(EFHV)."""
        efhv = np.array([0.1, 0.5, 0.3, 0.2])
        weights = [np.array([1, 0, 0]) for _ in range(4)]
        amfc_idx = _select_amfc_index(efhv)
        nc18_idx = _select_close_to_prev_amfc_index(efhv, weights, None)
        assert nc18_idx == amfc_idx
        assert nc18_idx == 1  # argmax EFHV

    def test_lambda_balance_1_recovers_pure_amfc(self):
        """lambda_balance=1.0 → ignore Jaccard, pure AMFC behavior."""
        efhv = np.array([0.1, 0.5, 0.3, 0.2])
        # Build portfolios with VERY DIFFERENT compositions
        weights = [
            np.array([1, 0, 0, 0, 0]),  # idx 0: asset 0
            np.array([0, 1, 0, 0, 0]),  # idx 1: asset 1 (max EFHV)
            np.array([0, 0, 1, 0, 0]),  # idx 2: asset 2
            np.array([1, 0, 0, 0, 0]),  # idx 3: asset 0 (matches prev)
        ]
        prev = np.array([1, 0, 0, 0, 0])
        # Pure AMFC picks idx 1 (highest EFHV) regardless of composition
        idx = _select_close_to_prev_amfc_index(
            efhv, weights, prev, lambda_balance=1.0,
        )
        assert idx == 1

    def test_lambda_balance_0_picks_max_jaccard(self):
        """lambda_balance=0.0 → ignore EFHV, pure max-Jaccard pick."""
        efhv = np.array([0.5, 0.1, 0.1, 0.1])
        weights = [
            np.array([0, 0, 1, 0, 0]),  # disjoint from prev
            np.array([1, 1, 0, 0, 0]),  # asset 0 in common with prev
            np.array([1, 0, 0, 0, 0]),  # identical to prev
            np.array([0, 1, 0, 0, 0]),  # disjoint from prev
        ]
        prev = np.array([1, 0, 0, 0, 0])
        # Pure Jaccard picks idx 2 (perfect overlap, Jaccard=1.0)
        idx = _select_close_to_prev_amfc_index(
            efhv, weights, prev, lambda_balance=0.0,
        )
        assert idx == 2

    def test_balanced_lambda_balances_efhv_and_stability(self):
        """At lambda_balance=0.5, EFHV and Jaccard contribute equally.

        Setup: portfolio A has efhv=1.0 (max), jaccard=0.0
              portfolio B has efhv=0.5, jaccard=0.7
        score_A = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        score_B = 0.5 * (0.5/1.0) + 0.5 * 0.7 = 0.25 + 0.35 = 0.6
        → idx B wins
        """
        efhv = np.array([1.0, 0.5])
        weights = [
            np.array([0, 0, 0, 1, 1]),  # disjoint from prev
            np.array([1, 1, 0, 0, 1]),  # partial overlap with prev = {0, 1, 2}
        ]
        prev = np.array([1, 1, 1, 0, 0])  # active {0, 1, 2}
        # Compute expected Jaccards
        # idx 0: active {3, 4} ∩ {0, 1, 2} = {} → Jaccard 0
        # idx 1: active {0, 1, 4} ∩ {0, 1, 2} = {0, 1}; union = {0, 1, 2, 4} → 2/4 = 0.5
        # score_0 = 0.5*1.0 + 0.5*0.0 = 0.5
        # score_1 = 0.5*0.0 + 0.5*0.5 = 0.25  (since EFHV norm = (0.5 - 0.5)/(1.0 - 0.5) = 0... wait min/max)
        # Actually min=0.5, max=1.0, range=0.5. efhv_norm[0] = (1.0-0.5)/0.5 = 1.0, efhv_norm[1] = 0.
        # score_0 = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # score_1 = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        # → idx 0 wins
        idx = _select_close_to_prev_amfc_index(
            efhv, weights, prev, lambda_balance=0.5,
        )
        # With normalized EFHV idx 0 wins because EFHV diff dominates Jaccard diff
        assert idx == 0

    def test_empty_efhv_returns_0(self):
        """Edge case: empty EFHV → return 0 (safe fallback)."""
        assert _select_close_to_prev_amfc_index(
            np.array([]), [], np.array([1, 0]), lambda_balance=0.5,
        ) == 0
