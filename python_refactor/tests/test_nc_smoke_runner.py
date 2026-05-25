"""W22 NC smoke runner tests — verify config + aggregation + markdown rendering.

Does NOT actually run ASMS (that's the operator's compute budget). Tests
the orchestration layer: config validation, env-var translation, aggregation,
Wilcoxon test computation, markdown formatting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.nc_smoke_runner import (
    NC_CONFIGS,
    _aggregate_by_config,
    _format_markdown,
    _wilcoxon_vs_baseline,
)


def test_nc_configs_baseline_is_empty():
    """BASELINE config has no env-var overrides."""
    assert "BASELINE" in NC_CONFIGS
    # BASELINE should be empty (no opt-ins activated)
    baseline = {k: v for k, v in NC_CONFIGS["BASELINE"].items() if not k.startswith("_")}
    assert baseline == {}


def test_nc36_config_has_one_env_var():
    assert NC_CONFIGS["NC36"]["W22_NC36_TIP_ANALYTICAL"] == "1"


def test_tip_cleanup_has_three_env_vars():
    cfg = NC_CONFIGS["TIP_CLEANUP"]
    active = {k: v for k, v in cfg.items() if not k.startswith("_")}
    assert len(active) == 3
    assert "W22_NC36_TIP_ANALYTICAL" in active
    assert "W22_NC13B_SMOOTH_CLAMP" in active
    assert "W22_NC31_TIP_CONDITIONAL" in active


def test_full_stack_includes_nc27_deep():
    cfg = NC_CONFIGS["FULL_STACK"]
    assert cfg.get("W22_NC27_PREDICTOR") == "dirichlet_posterior"


def test_anticipatory_ops_requires_integration():
    """ANTICIPATORY_OPS is documented but not executable until SMSEMOA wiring."""
    cfg = NC_CONFIGS["ANTICIPATORY_OPS"]
    assert cfg.get("_status") == "REQUIRES_INTEGRATION"


def test_aggregate_by_config_returns_per_config_stats():
    """Aggregation: collect grand_mean per (config × scenario × seed)."""
    raw = [
        {"config": "BASELINE", "scenario": "ASMS", "seed": 1, "grand_mean": 0.001},
        {"config": "BASELINE", "scenario": "ASMS", "seed": 2, "grand_mean": 0.0012},
        {"config": "BASELINE", "scenario": "ASMS", "seed": 3, "grand_mean": 0.0011},
        {"config": "NC36", "scenario": "ASMS", "seed": 1, "grand_mean": 0.0015},
        {"config": "NC36", "scenario": "ASMS", "seed": 2, "grand_mean": 0.0014},
        {"config": "NC36", "scenario": "ASMS", "seed": 3, "grand_mean": 0.0016},
    ]
    summary = _aggregate_by_config(raw)
    assert "BASELINE" in summary and "NC36" in summary
    assert summary["BASELINE"]["ASMS"]["n_seeds"] == 3
    assert summary["NC36"]["ASMS"]["n_seeds"] == 3
    np.testing.assert_allclose(summary["BASELINE"]["ASMS"]["mean"], 0.0011, atol=1e-6)
    np.testing.assert_allclose(summary["NC36"]["ASMS"]["mean"], 0.0015, atol=1e-6)


def test_aggregate_handles_nan_grand_mean():
    raw = [
        {"config": "BASELINE", "scenario": "ASMS", "grand_mean": 0.001},
        {"config": "BASELINE", "scenario": "ASMS", "grand_mean": float("nan")},
    ]
    summary = _aggregate_by_config(raw)
    # NaN rows skipped
    assert summary["BASELINE"]["ASMS"]["n_seeds"] == 1


def test_wilcoxon_vs_baseline_returns_p_values():
    """Wilcoxon paired test against BASELINE for each non-baseline config."""
    summary = {
        "BASELINE": {
            "ASMS": {"raw": [0.0010, 0.0011, 0.0009, 0.0012, 0.0011],
                       "mean": 0.00106, "std": 1e-4, "n_seeds": 5,
                       "min": 0.0009, "max": 0.0012}
        },
        "NC36": {
            "ASMS": {"raw": [0.0015, 0.0014, 0.0013, 0.0016, 0.0014],
                       "mean": 0.00144, "std": 1e-4, "n_seeds": 5,
                       "min": 0.0013, "max": 0.0016}
        },
    }
    tests = _wilcoxon_vs_baseline(summary, baseline_cfg="BASELINE")
    assert ("NC36", "ASMS") in tests
    test = tests[("NC36", "ASMS")]
    assert "p_value" in test
    assert "mean_diff" in test
    assert test["mean_diff"] > 0  # NC36 mean > BASELINE mean


def test_wilcoxon_handles_unaligned_samples():
    """Different number of seeds → unaligned; report NaN p-value."""
    summary = {
        "BASELINE": {
            "ASMS": {"raw": [0.001, 0.002], "mean": 0.0015, "std": 7e-4,
                       "n_seeds": 2, "min": 0.001, "max": 0.002}
        },
        "NC36": {
            "ASMS": {"raw": [0.0015, 0.0014, 0.0016], "mean": 0.0015, "std": 1e-4,
                       "n_seeds": 3, "min": 0.0014, "max": 0.0016}
        },
    }
    tests = _wilcoxon_vs_baseline(summary, baseline_cfg="BASELINE")
    nc36_test = tests.get(("NC36", "ASMS"), {})
    # Either NaN p_value or a note about unaligned
    assert "note" in nc36_test or np.isnan(nc36_test.get("p_value", 0))


def test_format_markdown_includes_all_configs():
    """Markdown summary lists all (config, scenario) rows."""
    summary = {
        "BASELINE": {
            "ASMS": {"raw": [0.001, 0.002], "mean": 0.0015, "std": 7e-4,
                       "n_seeds": 2, "min": 0.001, "max": 0.002}
        },
        "NC36": {
            "ASMS": {"raw": [0.0015, 0.0016], "mean": 0.00155, "std": 7e-5,
                       "n_seeds": 2, "min": 0.0015, "max": 0.0016}
        },
    }
    tests = {("NC36", "ASMS"): {"p_value": 0.123, "mean_diff": 5e-5, "n": 2}}
    md = _format_markdown(summary, tests)
    assert "BASELINE" in md and "NC36" in md
    assert "ASMS" in md
    assert "Wilcoxon" in md or "p" in md.lower()


def test_format_markdown_handles_baseline_only():
    """If only BASELINE results, markdown still renders without errors."""
    summary = {
        "BASELINE": {
            "ASMS": {"raw": [0.001], "mean": 0.001, "std": float("nan"),
                       "n_seeds": 1, "min": 0.001, "max": 0.001}
        }
    }
    tests = {}
    md = _format_markdown(summary, tests)
    assert "BASELINE" in md
    assert "Generated:" in md
