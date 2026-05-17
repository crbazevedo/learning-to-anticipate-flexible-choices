"""
W12-1 tests for the multi-seed sweep driver.

Covers seed parsing, tuple enumeration, and one real-dispatch
end-to-end smoke (S0 × paper × 2 seeds).
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments import sweep


class TestSeedParsing(unittest.TestCase):
    def test_range(self):
        self.assertEqual(sweep.parse_seeds("1-3"), [1, 2, 3])
        self.assertEqual(sweep.parse_seeds("1-30"),
                          list(range(1, 31)))

    def test_list(self):
        self.assertEqual(sweep.parse_seeds("1,5,10"), [1, 5, 10])
        self.assertEqual(sweep.parse_seeds("7,42"), [7, 42])

    def test_single(self):
        self.assertEqual(sweep.parse_seeds("7"), [7])

    def test_handles_whitespace_in_list(self):
        self.assertEqual(sweep.parse_seeds("1, 5, 10"), [1, 5, 10])


class TestTupleEnumeration(unittest.TestCase):
    def test_cartesian_product(self):
        out = sweep.enumerate_tuples(["S0", "S1"], ["paper"], [1, 2])
        self.assertEqual(set(out), {
            ("S0", "paper", 1), ("S0", "paper", 2),
            ("S1", "paper", 1), ("S1", "paper", 2),
        })

    def test_single_dim_collapses(self):
        out = sweep.enumerate_tuples(["S0"], ["paper"], [1])
        self.assertEqual(out, [("S0", "paper", 1)])

    def test_multi_window(self):
        out = sweep.enumerate_tuples(["S0"], ["paper", "extended"], [1])
        self.assertEqual(len(out), 2)


class TestRealDispatchSmoke(unittest.TestCase):
    """End-to-end: spawn validation_matrix subprocesses for 2 seeds.
    Slow (~5s); only validates orchestration, not metric quality."""

    def test_s0_paper_two_seeds(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "sweep-results"
            rc = sweep.run_sweep(
                scenarios=["S0"],
                windows=["paper"],
                seeds=[1, 2],
                output_root=output,
                jobs=2,
            )
            self.assertEqual(rc, 0, "sweep returncode should be 0")
            # Each tuple produced its own output dir.
            for seed in (1, 2):
                d = output / f"S0_paper_seed{seed}"
                self.assertTrue(d.exists(), f"missing {d}")
                manifest = json.loads((d / "manifest.json").read_text())
                self.assertEqual(manifest.get("status"), "ok",
                                  f"seed{seed} status={manifest.get('status')}")
                # metrics.csv has > 1 line (header + ≥ 1 metric row).
                lines = (d / "metrics.csv").read_text().splitlines()
                self.assertGreater(len(lines), 1,
                                    f"seed{seed} metrics.csv empty")


if __name__ == "__main__":
    unittest.main()
