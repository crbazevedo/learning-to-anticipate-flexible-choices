"""
W16-4: regression tests for λ + TIP per-call trace CSV emit
(BACKLOG M3 + M4 partial; trace-export scope only).

Thesis grounding (verbatim):

§7.3.1 "Estimated Confidence Over the Stochastic Pareto Frontiers
(SPFs)", p. 148:
    "We recall that, according to the OAL rules proposed in chapter 6,
     the available KF predictive objective distributions of portfolios
     associated with higher predictive confidence are more intensely
     incorporated into the resulting anticipatory distributions...
     Moreover, recall that rank 1 portfolios correspond to the highest
     risk/return, whereas a rank 20 one corresponds to the lowest
     risk/return in the population."

§6.1.3 Eq (6.4), p. 116 — TIP definition.

§7.2.3 Eq (7.16), p. 146 — λ = (1/2)(λ^H + λ^K). The trace records
all 3 scalars per call so W16-5 can confirm both arms fire after W16-1.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.algorithms.anticipatory_learning import AnticipatoryLearning
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    """Reset Portfolio class variables between tests (W16-1 hygiene)."""
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


def _make_solution(num_assets: int = 5) -> Solution:
    sol = Solution(num_assets=num_assets)
    sol.P.ROI = 0.05
    sol.P.risk = 0.10
    sol.prediction_error = 0.5
    sol.alpha = 0.5
    return sol


class TestLambdaTraceCSVHeader:
    """The trace CSV header matches the documented schema."""

    def test_header_schema_matches_spec(self):
        """LAMBDA_TRACE_CSV_HEADER includes the W16-4 + W17-2 fields."""
        # W16-4 baseline: 7 fields. W17-2 added 'lambda_k_branch'.
        expected = [
            "period", "generation", "solution_rank",
            "lambda_h", "lambda_k", "lambda", "tip",
            "lambda_k_branch",  # W17-2 addition
        ]
        assert AnticipatoryLearning.LAMBDA_TRACE_CSV_HEADER == expected


class TestLambdaTraceCSVEmit:
    """Per-call trace rows are flushed to CSV with correct schema."""

    def test_flush_empty_trace_is_noop(self, tmp_path: Path):
        """Empty trace + flush → returns 0, file not created."""
        al = AnticipatoryLearning(window_size=2)
        path = tmp_path / "lambda_tip_trace.csv"
        written = al.flush_lambda_trace_csv(str(path))
        assert written == 0
        assert not path.exists()

    def test_flush_creates_csv_with_header_and_rows(self, tmp_path: Path):
        """Single flush → header + rows written; trace buffer cleared."""
        al = AnticipatoryLearning(window_size=2)
        sol = _make_solution()
        # 3 calls
        for gen in range(3):
            al.compute_anticipatory_learning_rate(
                sol, min_error=0.0, max_error=1.0,
                min_alpha=0.0, max_alpha=1.0,
                current_time=5, generation=gen, solution_rank=gen,
            )
        path = tmp_path / "lambda_tip_trace.csv"
        written = al.flush_lambda_trace_csv(str(path), append=False)
        assert written == 3
        assert path.exists()

        # Read back the CSV
        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        assert list(rows[0].keys()) == AnticipatoryLearning.LAMBDA_TRACE_CSV_HEADER
        # Verify generation 0/1/2 in order
        gens = [int(r["generation"]) for r in rows]
        assert gens == [0, 1, 2]

        # Buffer should be cleared post-flush
        assert al.get_lambda_trace() == []

    def test_flush_append_mode_concatenates(self, tmp_path: Path):
        """append=True (default) extends existing CSV without re-emitting header."""
        al = AnticipatoryLearning(window_size=2)
        sol = _make_solution()
        path = tmp_path / "lambda_tip_trace.csv"

        # Period 1: 2 rows
        for gen in range(2):
            al.compute_anticipatory_learning_rate(
                sol, 0.0, 1.0, 0.0, 1.0,
                current_time=1, generation=gen, solution_rank=0,
            )
        n1 = al.flush_lambda_trace_csv(str(path), append=True)
        assert n1 == 2

        # Period 2: 3 rows
        for gen in range(3):
            al.compute_anticipatory_learning_rate(
                sol, 0.0, 1.0, 0.0, 1.0,
                current_time=2, generation=gen, solution_rank=1,
            )
        n2 = al.flush_lambda_trace_csv(str(path), append=True)
        assert n2 == 3

        # Read back: should have 5 rows total + 1 header line
        with open(path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 5
        # Header appears exactly once → DictReader sees no extra "period"
        # entries in body rows
        periods = [int(r["period"]) for r in rows]
        assert periods == [1, 1, 2, 2, 2]
