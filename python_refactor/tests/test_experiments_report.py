"""
W8-4 tests for the report orchestrator.

Synthetic-fixture tests: build a minimal summary CSV + per-run tree
+ template, run `populate`, and assert that placeholders are
replaced where data exists and preserved where it doesn't.
"""

import csv
import datetime as _dt
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments import report


def _write_summary(path: Path, rows: list[dict]) -> None:
    fieldnames = ["scenario", "window", "metric", "n_seeds", "seeds_used",
                  "mean", "std", "median", "min", "max"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_run(run_root: Path, scenario: str, window: str, seed: int,
                metrics: dict[str, float]) -> None:
    run_dir = run_root / f"{scenario}_{window}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "ok"}))
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "window", "seed", "metric", "value"])
        for metric, val in metrics.items():
            w.writerow([scenario, window, seed, metric, val])


# ---------------------------------------------------------------------------
# Receipt block
# ---------------------------------------------------------------------------

class TestReceiptBlock(unittest.TestCase):
    def test_receipt_block_populates_all_fields(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            summary = [
                {"scenario": "S0", "window": "paper", "metric": "final_wealth",
                  "n_seeds": 3, "seeds_used": "1,2,3", "mean": 1.0, "std": 0.0,
                  "median": 1.0, "min": 1.0, "max": 1.0},
                {"scenario": "S2", "window": "paper", "metric": "final_wealth",
                  "n_seeds": 3, "seeds_used": "1,2,3", "mean": 1.1, "std": 0.01,
                  "median": 1.1, "min": 1.0, "max": 1.2},
            ]
            now = _dt.datetime(2026, 5, 17, 3, 45, 0, tzinfo=_dt.timezone.utc)
            block = report.build_receipt_block(summary, tmpdir, command="X", now=now)
            self.assertIn("git_sha:", block)
            self.assertIn("uv_lock:", block)
            self.assertIn("2026-05-17T03:45:00Z", block)
            self.assertIn("n_runs:     6", block)  # 3 + 3
            self.assertIn("[1, 2, 3]", block)
            self.assertIn("X", block)  # command echoed

    def test_receipt_block_handles_empty_summary(self):
        with TemporaryDirectory() as tmp:
            block = report.build_receipt_block([], Path(tmp))
            self.assertIn("n_runs:     0", block)
            self.assertIn("seeds:      []", block)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

class TestBuilders(unittest.TestCase):
    def test_build_table_a_passes_summary_through(self):
        summary = [{"scenario": "S0", "window": "paper", "metric": "final_wealth",
                     "mean": 1.0, "std": 0.0, "n_seeds": 3}]
        rows = report.build_table_a_rows(summary)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario"], "S0")

    def test_build_table_b_runs_pairwise_with_holm(self):
        per_run = {
            ("S0", "paper", "final_wealth"): [(s, 1.0 + 0.001 * s) for s in range(1, 11)],
            ("S2", "paper", "final_wealth"): [(s, 1.1 + 0.001 * s) for s in range(1, 11)],
        }
        rows = report.build_table_b_rows(per_run)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair"], "S0 vs S2")
        self.assertIn("p_raw", rows[0])
        self.assertIn("p_holm", rows[0])
        # Single test → Holm == raw
        self.assertAlmostEqual(rows[0]["p_holm"], rows[0]["p_raw"], places=10)
        # CLES ∈ [0,1]
        self.assertGreaterEqual(rows[0]["cles"], 0.0)
        self.assertLessEqual(rows[0]["cles"], 1.0)

    def test_build_table_d_synergy_with_clean_signal(self):
        # Construct data with positive synergy: S2 − 2*S1 + S0 > 0.
        per_run = {
            ("S0", "paper", "final_wealth"): [(s, 1.00) for s in range(1, 11)],
            ("S1", "paper", "final_wealth"): [(s, 1.05) for s in range(1, 11)],
            ("S2", "paper", "final_wealth"): [(s, 1.20) for s in range(1, 11)],
        }
        rows = report.build_table_d_rows(per_run, n_bootstrap=200)
        self.assertEqual(len(rows), 1)
        # Δ = 1.20 - 2*1.05 + 1.00 = 0.10 > 0 → synergy
        self.assertAlmostEqual(rows[0]["delta_interaction"], 0.10, places=4)
        self.assertEqual(rows[0]["interpretation"], "synergy")

    def test_build_table_d_skips_when_triplet_missing(self):
        per_run = {
            ("S0", "paper", "final_wealth"): [(s, 1.0) for s in range(1, 5)],
            # No S1, no S2.
        }
        rows = report.build_table_d_rows(per_run, n_bootstrap=100)
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# End-to-end populate
# ---------------------------------------------------------------------------

class TestPopulate(unittest.TestCase):
    TEMPLATE = """# Validation results

## 0. Reproducibility receipt

```
git_sha:    🚧
uv_lock:    🚧
generated:  🚧
n_runs:     🚧
seeds:      🚧
command:    python -m experiments.report
```

## 2. Descriptive statistics (Table A)

| scenario | window | metric | mean | std | median | IQR | n |
|---|---|---|---|---|---|---|---|
| S0 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |
| S2 | paper | final_wealth | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

## 9. Honest scars

🚧 nothing yet.
"""

    def test_populate_replaces_receipt_and_table_a(self):
        summary = [
            {"scenario": "S0", "window": "paper", "metric": "final_wealth",
              "mean": 1.0, "std": 0.0, "median": 1.0, "min": 1.0, "max": 1.0,
              "n_seeds": 3, "seeds_used": "1,2,3"},
            {"scenario": "S2", "window": "paper", "metric": "final_wealth",
              "mean": 1.12, "std": 0.05, "median": 1.10, "min": 1.05, "max": 1.20,
              "n_seeds": 3, "seeds_used": "1,2,3"},
        ]
        receipt = report.build_receipt_block(summary, Path("."))
        out = report.populate(self.TEMPLATE, summary, {}, receipt)

        # Receipt populated — placeholders gone in §0.
        self.assertNotIn("git_sha:    🚧", out)
        self.assertIn("n_runs:     6", out)

        # Table A populated — has S0 + S2 cells with concrete numbers.
        self.assertIn("| S0 | paper | final_wealth", out)
        self.assertIn("1.12", out)
        # Table A header rendered by generator.
        self.assertIn("| scenario | window | metric | mean | std | median | min | max | n |", out)

        # Section §9 untouched (graceful incompleteness).
        self.assertIn("🚧 nothing yet.", out)

    def test_populate_preserves_template_when_no_data(self):
        # Empty summary + empty per_run → template structure preserved, only
        # the receipt block is touched.
        receipt = report.build_receipt_block([], Path("."))
        out = report.populate(self.TEMPLATE, [], {}, receipt)
        # Table A placeholders intact.
        self.assertIn("| S0 | paper | final_wealth | 🚧", out)
        self.assertIn("| S2 | paper | final_wealth | 🚧", out)
        # Receipt populated even when summary is empty.
        self.assertNotIn("git_sha:    🚧", out)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------

class TestCLIRoundtrip(unittest.TestCase):
    def test_main_reads_writes_end_to_end(self):
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # Build fixtures.
            summary_path = tmpdir / "summary.csv"
            _write_summary(summary_path, [
                {"scenario": "S0", "window": "paper", "metric": "final_wealth",
                  "n_seeds": 2, "seeds_used": "1,2", "mean": 1.00, "std": 0.01,
                  "median": 1.00, "min": 0.99, "max": 1.01},
            ])
            runs_dir = tmpdir / "results"
            _write_run(runs_dir, "S0", "paper", 1, {"final_wealth": 0.99})
            _write_run(runs_dir, "S0", "paper", 2, {"final_wealth": 1.01})
            template_path = tmpdir / "template.md"
            template_path.write_text(TestPopulate.TEMPLATE)
            output_path = tmpdir / "out.md"
            rc = report.main([
                "--summary", str(summary_path),
                "--runs", str(runs_dir),
                "--template", str(template_path),
                "--output", str(output_path),
                "--repo-root", str(tmpdir),
            ])
            self.assertEqual(rc, 0)
            out = output_path.read_text()
            self.assertIn("n_runs:     2", out)
            self.assertIn("| S0 | paper | final_wealth", out)


if __name__ == "__main__":
    unittest.main()
