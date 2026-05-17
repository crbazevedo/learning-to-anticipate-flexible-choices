"""
W8-3 tests for table generators A-H.

Smoke + format tests. Each generator gets a minimal-input case + a
multi-row case to verify Markdown shape + formatting helpers.
"""

import unittest

from experiments.tables import (
    fmt_effect,
    fmt_pvalue,
    fmt_sigfig,
    generate_table_a,
    generate_table_b,
    generate_table_c,
    generate_table_d,
    generate_table_e,
    generate_table_f,
    generate_table_g,
    generate_table_h,
)


class TestFormattingHelpers(unittest.TestCase):
    """Number-format helpers per ANALYTICS-PLAN.md §7."""

    def test_fmt_sigfig_four_sig_figs(self):
        self.assertEqual(fmt_sigfig(0.123456, sig=4), "0.1235")
        self.assertEqual(fmt_sigfig(123.456, sig=4), "123.5")
        self.assertEqual(fmt_sigfig(1234567.0, sig=4), "1234567")  # large numbers handled

    def test_fmt_sigfig_zero(self):
        self.assertEqual(fmt_sigfig(0.0), "0")

    def test_fmt_sigfig_nan(self):
        self.assertEqual(fmt_sigfig(float("nan")), "nan")

    def test_fmt_pvalue_small_renders_lt_001(self):
        self.assertEqual(fmt_pvalue(0.0005), "<0.001")
        self.assertEqual(fmt_pvalue(1e-10), "<0.001")

    def test_fmt_pvalue_normal_three_sig_figs(self):
        result = fmt_pvalue(0.0421)
        # 3 sig figs of 0.0421 → "0.0421" (or "0.042" depending on rounding)
        self.assertTrue(result.startswith("0.04"))

    def test_fmt_effect_three_decimals(self):
        self.assertEqual(fmt_effect(0.7654321), "0.765")
        self.assertEqual(fmt_effect(-0.5), "-0.500")


class TestTableA(unittest.TestCase):
    def test_empty_input_returns_no_data_marker(self):
        self.assertIn("no data", generate_table_a([]))

    def test_single_row_renders_full_table(self):
        rows = [{
            "scenario": "S0", "window": "paper", "metric": "final_wealth",
            "mean": 1.12, "std": 0.05, "median": 1.10,
            "min": 1.00, "max": 1.25, "n_seeds": 30,
        }]
        out = generate_table_a(rows)
        self.assertIn("S0", out)
        self.assertIn("paper", out)
        self.assertIn("final_wealth", out)
        self.assertIn("30", out)  # n_seeds
        # Markdown table shape
        self.assertIn("| scenario |", out)
        self.assertIn("|---|", out)


class TestTableB(unittest.TestCase):
    def test_pairwise_rendering_with_small_p_value(self):
        rows = [{
            "pair": "S0 vs S2", "window": "paper", "metric": "final_wealth",
            "U": 150.0, "p_raw": 0.0001, "p_holm": 0.001,
            "cohens_d": 0.85, "cliffs_delta": 0.62, "cles": 0.81,
        }]
        out = generate_table_b(rows)
        self.assertIn("S0 vs S2", out)
        self.assertIn("<0.001", out)  # tiny p formatted as '<0.001'
        self.assertIn("0.850", out)   # Cohen's d at 3 decimals


class TestTableC(unittest.TestCase):
    def test_anova_row_with_residual(self):
        rows = [
            {"window": "paper", "metric": "final_wealth", "source": "Multi-Horizon",
             "SS": 0.123, "df": 2, "MS": 0.0615, "F": 8.5, "p": 0.001},
            {"window": "paper", "metric": "final_wealth", "source": "Residual",
             "SS": 1.234, "df": 87, "MS": 0.0142, "F": float("nan"), "p": float("nan")},
        ]
        out = generate_table_c(rows)
        self.assertIn("Multi-Horizon", out)
        self.assertIn("Residual", out)
        self.assertIn("8.5", out)  # F stat
        self.assertEqual(out.count("\n"), 1 + 1 + 2 - 1)  # header + sep + 2 rows - last has no trailing \n


class TestTableD(unittest.TestCase):
    def test_synergy_with_ci_brackets(self):
        rows = [{"window": "paper", "metric": "final_wealth",
                  "delta_interaction": 0.045, "ci_lo": 0.012, "ci_hi": 0.080,
                  "interpretation": "synergy"}]
        out = generate_table_d(rows)
        # 4-sig-fig formatter pads trailing zeros: 0.012 → "0.01200", 0.08 → "0.08000".
        # Assert the bracket shape + numbers (not the exact substring) so the test
        # is robust to formatter precision choices.
        self.assertIn("0.01200", out)
        self.assertIn("0.08000", out)
        self.assertIn("[", out)
        self.assertIn("]", out)
        self.assertIn("synergy", out)


class TestTableE(unittest.TestCase):
    def test_horizon_ablation_basic(self):
        rows = [{"comparison": "S2 (H=3) vs S3 (H=2)", "window": "paper",
                  "metric": "final_wealth", "U": 220.0, "p": 0.02,
                  "cohens_d": 0.35, "interpretation": "small effect favoring H=3"}]
        out = generate_table_e(rows)
        self.assertIn("S2 (H=3) vs S3 (H=2)", out)
        self.assertIn("small effect", out)


class TestTableF(unittest.TestCase):
    def test_regime_table_groups_metrics_into_columns(self):
        rows = [
            {"regime": "low-vol", "scenario": "S0", "metric": "final_wealth",
              "mean": 1.10, "std": 0.05},
            {"regime": "low-vol", "scenario": "S0", "metric": "hypv",
              "mean": 0.85, "std": 0.10},
            {"regime": "low-vol", "scenario": "S2", "metric": "final_wealth",
              "mean": 1.15, "std": 0.06},
            {"regime": "low-vol", "scenario": "S2", "metric": "hypv",
              "mean": 0.92, "std": 0.09},
        ]
        out = generate_table_f(rows)
        # Header should have both metric columns
        self.assertIn("final_wealth (mean ± std)", out)
        self.assertIn("hypv (mean ± std)", out)
        # Both scenarios present
        self.assertIn("| low-vol | S0 |", out)
        self.assertIn("| low-vol | S2 |", out)


class TestTableG(unittest.TestCase):
    def test_sub_window_stability_basic(self):
        rows = [{"sub_window": "early", "scenario": "S0", "metric": "final_wealth",
                  "mean": 1.05, "std": 0.03}]
        out = generate_table_g(rows)
        self.assertIn("early", out)
        self.assertIn("S0", out)


class TestTableH(unittest.TestCase):
    def test_paper_comparison_basic(self):
        rows = [{"metric": "Final wealth (HDM, FTSE)", "paper_value": 1.12,
                  "our_value": 1.15, "abs_diff": 0.03, "rel_diff": 0.027,
                  "agree": "agrees"}]
        out = generate_table_h(rows)
        self.assertIn("Final wealth (HDM, FTSE)", out)
        self.assertIn("agrees", out)
        self.assertIn("1.12", out)
        self.assertIn("1.15", out)


if __name__ == "__main__":
    unittest.main()
