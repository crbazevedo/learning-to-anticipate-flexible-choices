"""
W15-3 regression tests for BACKLOG B2+H3+H5.

Grounding (verbatim):
  - Thesis §7.1.1 p. 140: "The anticipation factor is controlled by
    four levels of window size (K): K ∈ {0, 1, 2, 3}."
  - Thesis §7.2.3 p. 146 Eq (7.16): "λ_{t+h} = (1/2)(λ_{t+h}^(H) +
    λ_{t+h}^(K)), for which the anticipation horizon is H = 2
    (one-step-ahead prediction)."
  - Thesis §7.2.3 p. 145: "The real-world scenarios are composed of
    daily adjusted close prices collected between 20/11/2006 –
    31/12/2012."
"""

import unittest

from experiments.validation_matrix import SCENARIOS, WINDOWS


class TestScenariosFactorial(unittest.TestCase):
    """W15-3 (BACKLOG B2): SCENARIOS is the thesis factorial."""

    def test_thesis_factorial_present(self):
        # 4 K levels × 2 DMs = 8 anticipatory variants + legacy aliases
        thesis_keys = {
            "SMS_RDM_K0", "SMS_mHDM_K0",
            "ASMS_RDM_K1", "ASMS_RDM_K2", "ASMS_RDM_K3",
            "ASMS_mHDM_K1", "ASMS_mHDM_K2", "ASMS_mHDM_K3",
        }
        self.assertTrue(thesis_keys.issubset(SCENARIOS.keys()),
                          f"missing: {thesis_keys - set(SCENARIOS.keys())}")

    def test_k_appears_in_window_size_for_asms(self):
        # ASMS variants must have K = window_size in their learning params
        for K in (1, 2, 3):
            for dm in ("RDM", "mHDM"):
                key = f"ASMS_{dm}_K{K}"
                scn = SCENARIOS[key]
                self.assertEqual(scn["learning"]["enabled"], True)
                params = scn["learning"]["parameters"]
                self.assertEqual(params["window_size"], K,
                                  f"{key} window_size != {K}")

    def test_sms_baseline_disabled_learning(self):
        # SMS_* (K=0) variants have learning disabled (≡ myopic baseline)
        for dm in ("RDM", "mHDM"):
            key = f"SMS_{dm}_K0"
            self.assertEqual(SCENARIOS[key]["learning"], {"enabled": False})

    def test_legacy_aliases_preserved(self):
        # S0..S4 still exist for W12-W14 report reproducibility
        for legacy in ("S0", "S1", "S2", "S3", "S4"):
            self.assertIn(legacy, SCENARIOS)


class TestHorizonH2Fixed(unittest.TestCase):
    """W15-3 (BACKLOG H5): H = 2 fixed across anticipatory variants."""

    def test_all_anticipatory_have_h_equal_2(self):
        for key, scn in SCENARIOS.items():
            if scn["learning"].get("enabled"):
                params = scn["learning"].get("parameters", {})
                max_horizon = params.get("max_horizon")
                # H=2 fixed per thesis §7.2.3 Eq 7.16
                # Legacy aliases that don't have max_horizon (use_tip-only,
                # no multi-horizon) are exempt.
                if max_horizon is not None:
                    self.assertEqual(max_horizon, 2,
                                      f"{key} max_horizon != 2 (thesis H=2 fixed)")


class TestPaperWindowDateRange(unittest.TestCase):
    """W15-3 (BACKLOG H3): paper-window 2006-11-20 → 2012-12-31."""

    def test_paper_window_date_range_matches_thesis(self):
        paper = WINDOWS["paper"]
        # Verbatim thesis §7.2.3 p. 145: "20/11/2006 – 31/12/2012"
        self.assertEqual(paper["date_start"], "2006-11-20")
        self.assertEqual(paper["date_end"], "2012-12-31")

    def test_legacy_window_preserved(self):
        # Pre-W15-3 range kept as alias for W12-W14 reproducibility
        self.assertIn("legacy_paper_2003_2012", WINDOWS)
        legacy = WINDOWS["legacy_paper_2003_2012"]
        self.assertEqual(legacy["date_start"], "2003-01-01")

    def test_paper_window_glob_unchanged(self):
        # Asset universe (98-CSV FTSE archive) is unchanged; only the
        # date_range narrows.
        paper = WINDOWS["paper"]
        self.assertIn("table*.csv", paper["asset_files_glob"])


if __name__ == "__main__":
    unittest.main()
