"""
W11-3 scenario-differentiation regression gate.

CLASS-RETIRING per directive #5. Before W11, all 4 learning-enabled
scenarios (S1/S2/S3/S4) produced IDENTICAL metrics to 15 decimal
places (W10-CARRY-2). The root cause was found in W11-1 (SMS-EMOA
dispatch) and the data sanitation gap in W11-2.

This gate locks the differentiation property in: any future src/
change that silently breaks scenario dispatch will fail this test.
Tests are slow (each runs the full smoke-test in-process) and marked
accordingly — operators can skip via `pytest -k 'not differentiation'`
in tight inner loops but the wave-close gate must run them green.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).parents[2]
PYTHON_REFACTOR = REPO_ROOT / "python_refactor"


def _run_smoke(scenario: str, output_dir: Path) -> dict:
    """Run validation_matrix smoke-test and parse metrics into dict."""
    cmd = [
        sys.executable, "-m", "experiments.validation_matrix",
        "--scenario", scenario, "--window", "paper", "--seed", "1",
        "--output", str(output_dir),
    ]
    result = subprocess.run(
        cmd, cwd=str(PYTHON_REFACTOR),
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, (
        f"smoke-test {scenario} exit {result.returncode}: {result.stderr[-500:]}"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest.get("status") == "ok", f"{scenario} status={manifest.get('status')}"
    metrics: dict[str, float] = {}
    with (output_dir / "metrics.csv").open() as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 5:
                try:
                    metrics[parts[3]] = float(parts[4])
                except ValueError:
                    metrics[parts[3]] = float("nan")
    return metrics


class TestScenarioDifferentiation(unittest.TestCase):
    """Run S0-S4 once each; assert mutual differentiation."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = TemporaryDirectory(prefix="w11-3-")
        cls.metrics: dict[str, dict[str, float]] = {}
        for scenario in ("S0", "S1", "S2", "S3", "S4"):
            out = Path(cls.tmpdir.name) / scenario
            cls.metrics[scenario] = _run_smoke(scenario, out)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_all_scenarios_produce_metrics(self):
        for scenario, m in self.metrics.items():
            self.assertGreaterEqual(len(m), 5,
                                      f"{scenario} has only {len(m)} metric rows")

    def test_no_inf_or_nan_in_portfolio_metrics(self):
        """W10-CARRY-1 regression gate."""
        import math
        for scenario, m in self.metrics.items():
            val = m.get("portfolio.portfolio_value")
            if val is None:
                continue
            self.assertFalse(math.isinf(val),
                              f"{scenario}: portfolio.portfolio_value is Inf")
            self.assertFalse(math.isnan(val),
                              f"{scenario}: portfolio.portfolio_value is NaN")

    def test_scenarios_differentiate_in_diversity_metric(self):
        """W10-CARRY-2 class-retiring gate. The 4 learning-enabled
        scenarios must produce DIFFERENT algorithm.diversity_metric
        values; otherwise SMS-EMOA dispatch is broken again."""
        diversity = {
            s: self.metrics[s].get("algorithm.diversity_metric")
            for s in ("S0", "S1", "S2", "S3", "S4")
        }
        # All five must be defined.
        for s, d in diversity.items():
            self.assertIsNotNone(d, f"{s}.algorithm.diversity_metric missing")
        # No two of the 4 learning-enabled (S1-S4) may match exactly.
        learning_enabled = ["S1", "S2", "S3", "S4"]
        for i, a in enumerate(learning_enabled):
            for b in learning_enabled[i + 1:]:
                self.assertNotAlmostEqual(
                    diversity[a], diversity[b], places=10,
                    msg=f"{a} and {b} have identical diversity_metric "
                        f"({diversity[a]!r}) — scenario dispatch may be broken "
                        f"(W10-CARRY-2 recurrence; check SMS-EMOA "
                        f"_apply_anticipatory_learning)"
                )

    def test_scenarios_differentiate_in_stochastic_quality(self):
        """Same gate, second metric (robustness check)."""
        sq = {
            s: self.metrics[s].get("algorithm.stochastic_quality")
            for s in ("S0", "S1", "S2", "S3", "S4")
        }
        learning_enabled = ["S1", "S2", "S3", "S4"]
        diffs_found = 0
        for i, a in enumerate(learning_enabled):
            for b in learning_enabled[i + 1:]:
                if abs(sq[a] - sq[b]) > 1e-10:
                    diffs_found += 1
        # 4 choose 2 = 6 pairs; need at least 4 to differ (allow
        # occasional coincidence on small population).
        self.assertGreaterEqual(diffs_found, 4,
                                  f"only {diffs_found}/6 stochastic_quality pairs "
                                  f"differ — scenario dispatch suspect")


if __name__ == "__main__":
    unittest.main()
