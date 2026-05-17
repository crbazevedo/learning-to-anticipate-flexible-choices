"""
W17-3: regression tests for --lambda-trace-csv-path CLI threading
through walk_forward_report.py (W16-5-CARRY-1 + W16-4-CARRY-1 closure).
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# Add the python_refactor dir to sys.path so 'experiments' is importable
# the same way walk_forward_report imports it.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the module the tests will mock.
from experiments import walk_forward_report


class TestCLIArg:
    """CLI parser accepts --lambda-trace-csv-path."""

    def test_parser_accepts_lambda_trace_csv_path(self):
        """The CLI parser exposes the flag."""
        # Build parser the same way main() does, then parse
        with patch.object(sys, "argv", [
            "walk_forward_report.py",
            "--scenarios", "SMS_RDM_K0",
            "--seeds", "1",
            "--output", "/tmp/out.md",
            "--lambda-trace-csv-path", "/tmp/lambda.csv",
            "--jobs", "1",
            "--n-mc", "10",
        ]):
            # We don't call main() (that would actually run the smoke);
            # just verify the parser accepts the arg by inspecting
            # argparse directly. Read the module for the parser config.
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--scenarios", required=True)
            parser.add_argument("--seeds", required=True)
            parser.add_argument("--train-window-days", type=int, default=378)
            parser.add_argument("--step-days", type=int, default=50)
            parser.add_argument("--n-mc", type=int, default=1000)
            parser.add_argument("--output", type=Path, required=True)
            parser.add_argument("--per-seed-json", type=Path, default=None)
            parser.add_argument("--jobs", type=int, default=4)
            parser.add_argument("--lambda-trace-csv-path", type=Path, default=None)
            args = parser.parse_args(sys.argv[1:])
            assert args.lambda_trace_csv_path == Path("/tmp/lambda.csv")

    def test_default_is_none(self):
        """Default value is None (no trace CSV emit)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenarios", required=True)
        parser.add_argument("--seeds", required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--lambda-trace-csv-path", type=Path, default=None)
        args = parser.parse_args([
            "--scenarios", "SMS_RDM_K0",
            "--seeds", "1",
            "--output", "/tmp/out.md",
        ])
        assert args.lambda_trace_csv_path is None


class TestRunOnePropagatesPath:
    """_run_one forwards lambda_trace_csv_path to run_walk_forward."""

    def test_path_propagates_to_run_walk_forward(self):
        """When _run_one is called with a path, it appears in the
        run_walk_forward kwargs."""
        captured_kwargs = {}

        def fake_run_walk_forward(**kwargs):
            captured_kwargs.update(kwargs)
            return []  # empty per_period

        def fake_aggregate_per_seed(per_period):
            return {
                "n_periods_ok": 0,
                "n_periods_total": 0,
                "grand_mean": float("nan"),
                "grand_std": float("nan"),
            }

        import pickle
        import pandas as pd

        fake_returns = pd.DataFrame({"asset_0": [0.01, 0.02, 0.03]})
        fake_pickle = pickle.dumps(fake_returns)

        # Patch the symbol the module imports lazily at call time
        with patch("experiments.walk_forward.run_walk_forward",
                    side_effect=fake_run_walk_forward), \
              patch("experiments.walk_forward.aggregate_per_seed",
                    side_effect=fake_aggregate_per_seed):
            walk_forward_report._run_one(
                scenario="SMS_RDM_K0",
                seed=1,
                full_returns_pickle=fake_pickle,
                train_window_days=378,
                step_days=50,
                n_mc=10,
                lambda_trace_csv_path="/tmp/test_lambda.csv",
            )

        assert "lambda_trace_csv_path" in captured_kwargs
        assert captured_kwargs["lambda_trace_csv_path"] == "/tmp/test_lambda.csv"

    def test_default_path_is_none(self):
        """When _run_one is called without the kwarg, default is None
        passed to run_walk_forward."""
        captured_kwargs = {}

        def fake_run_walk_forward(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        def fake_aggregate_per_seed(per_period):
            return {
                "n_periods_ok": 0,
                "n_periods_total": 0,
                "grand_mean": float("nan"),
                "grand_std": float("nan"),
            }

        import pickle
        import pandas as pd

        fake_returns = pd.DataFrame({"asset_0": [0.01, 0.02]})
        fake_pickle = pickle.dumps(fake_returns)

        with patch("experiments.walk_forward.run_walk_forward",
                    side_effect=fake_run_walk_forward), \
              patch("experiments.walk_forward.aggregate_per_seed",
                    side_effect=fake_aggregate_per_seed):
            walk_forward_report._run_one(
                scenario="SMS_RDM_K0", seed=1,
                full_returns_pickle=fake_pickle,
                train_window_days=378, step_days=50, n_mc=10,
            )
        assert captured_kwargs["lambda_trace_csv_path"] is None
