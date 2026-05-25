"""W22 Probe V — tests for the opt-in fix combination ablation framework.

Pure framework tests. No shared-code imports, no subprocess spawns
(the subprocess-runner skeleton is operator-pluggable, not exercised
in CI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.probes.probe_v_combination_ablation import (
    ENV_VAR_MAP,
    OPT_IN_FIXES,
    _mock_wealth_fn,
    combination_to_dm_config,
    combination_to_env_dict,
    enumerate_combinations,
    format_ablation_table,
    run_ablation_synthetic,
)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------
def test_enumerate_full_returns_16_combinations():
    """Full enumeration over 4 binary fixes yields 2**4 = 16 distinct combos."""
    combos = enumerate_combinations("full")
    assert len(combos) == 16, f"expected 16, got {len(combos)}"
    # All distinct.
    frozensets = {frozenset(c) for c in combos}
    assert len(frozensets) == 16


def test_enumerate_pairwise_returns_baseline_plus_pairs():
    """Pairwise = baseline + C(4,2) = 1 + 6 = 7 combos, all of size 0 or 2."""
    combos = enumerate_combinations("pairwise")
    assert len(combos) == 7
    sizes = sorted(len(c) for c in combos)
    assert sizes == [0, 2, 2, 2, 2, 2, 2]


def test_enumerate_single_returns_5_combinations():
    """Single = baseline + 4 single-fix subsets = 5 combos."""
    combos = enumerate_combinations("single")
    assert len(combos) == 5
    sizes = sorted(len(c) for c in combos)
    assert sizes == [0, 1, 1, 1, 1]
    # Every single-fix subset is one of OPT_IN_FIXES.
    singletons = [next(iter(c)) for c in combos if len(c) == 1]
    assert set(singletons) == set(OPT_IN_FIXES)


def test_baseline_always_included():
    """The empty fix-set (baseline) must appear in every strategy."""
    for strategy in ("full", "pairwise", "single"):
        combos = enumerate_combinations(strategy)
        assert set() in combos, f"baseline missing from strategy={strategy!r}"


def test_enumerate_unknown_strategy_raises():
    """An unknown strategy is a hard error, not a silent baseline-only return."""
    with pytest.raises(ValueError):
        enumerate_combinations("triplets")


# ---------------------------------------------------------------------------
# Combination → env / dm_config translation
# ---------------------------------------------------------------------------
def test_combination_to_env_dict_baseline_returns_empty():
    """Baseline (no fixes) emits no env-var assignments."""
    env = combination_to_env_dict(set())
    assert env == {}


def test_combination_to_env_dict_full_set_returns_3_env_vars():
    """All four fixes active → 3 env-vars (NC30c is not env-gated)."""
    env = combination_to_env_dict(set(OPT_IN_FIXES))
    assert len(env) == 3
    assert env == {
        "W22_NC13B_SMOOTH_CLAMP": "1",
        "W22_NC27_PREDICTOR": "dirichlet_posterior",
        "W22_NC31_TIP_CONDITIONAL": "1",
    }
    # NC30c never leaks into the env dict.
    nc30c_env, _ = ENV_VAR_MAP.get("NC30c_variance_penalty", (None, None))
    assert nc30c_env is None


def test_combination_to_env_dict_ignores_unknown_keys():
    """Unknown fix names are silently ignored (caller may pass a superset)."""
    env = combination_to_env_dict({"NC13b_smooth_clamp", "definitely_not_a_fix"})
    assert env == {"W22_NC13B_SMOOTH_CLAMP": "1"}


def test_combination_to_dm_config_variance_penalty_applied_when_NC30c_in_set():
    """variance_penalty = 1.0 when NC30c is active."""
    cfg = combination_to_dm_config({"NC30c_variance_penalty"})
    assert cfg["variance_penalty"] == pytest.approx(1.0)


def test_combination_to_dm_config_variance_penalty_zero_when_NC30c_absent():
    """variance_penalty = 0.0 (the default OFF-value) when NC30c is not in set."""
    cfg = combination_to_dm_config(set())
    assert cfg["variance_penalty"] == pytest.approx(0.0)
    cfg2 = combination_to_dm_config({"NC13b_smooth_clamp"})
    assert cfg2["variance_penalty"] == pytest.approx(0.0)


def test_combination_to_dm_config_respects_base_off_value():
    """If base_dm_config provides variance_penalty, that is the OFF-value."""
    base = {"variance_penalty": 0.5, "other_key": "preserved"}
    # NC30c OFF → use the base 0.5.
    cfg_off = combination_to_dm_config(set(), base_dm_config=base)
    assert cfg_off["variance_penalty"] == pytest.approx(0.5)
    assert cfg_off["other_key"] == "preserved"
    # NC30c ON → snap to 1.0 regardless of base.
    cfg_on = combination_to_dm_config({"NC30c_variance_penalty"}, base_dm_config=base)
    assert cfg_on["variance_penalty"] == pytest.approx(1.0)
    assert cfg_on["other_key"] == "preserved"


def test_combination_to_dm_config_does_not_mutate_base():
    """The base dict must not be mutated (callers reuse it across combos)."""
    base = {"variance_penalty": 0.5}
    _ = combination_to_dm_config({"NC30c_variance_penalty"}, base_dm_config=base)
    assert base == {"variance_penalty": 0.5}


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------
def test_run_ablation_synthetic_returns_dict_keyed_by_frozenset():
    """Result dict uses frozenset keys (set is unhashable)."""
    combos = enumerate_combinations("single")
    results = run_ablation_synthetic(combos)
    assert isinstance(results, dict)
    assert all(isinstance(k, frozenset) for k in results)
    assert len(results) == len(combos)


def test_run_ablation_synthetic_baseline_in_results():
    """Every result map contains the baseline (empty frozenset)."""
    for strategy in ("full", "pairwise", "single"):
        results = run_ablation_synthetic(enumerate_combinations(strategy))
        assert frozenset() in results, (
            f"baseline missing from strategy={strategy!r}"
        )


def test_run_ablation_synthetic_uses_custom_wealth_fn():
    """The wealth_fn override is called once per combination."""
    combos = enumerate_combinations("single")
    calls: list[set[str]] = []

    def fake_wealth(fix_set: set[str]) -> float:
        calls.append(set(fix_set))
        return 42.0 + len(fix_set)

    results = run_ablation_synthetic(combos, wealth_fn=fake_wealth)
    assert len(calls) == len(combos)
    # Baseline wealth from the fake fn.
    assert results[frozenset()] == pytest.approx(42.0)


def test_run_ablation_synthetic_default_baseline_anchored_to_100():
    """The default mock fn anchors baseline at 100.0 for human-readable tables."""
    results = run_ablation_synthetic([set()])
    assert results[frozenset()] == pytest.approx(100.0)


def test_mock_wealth_fn_is_deterministic():
    """The mock wealth fn must be stable across calls (hash-based)."""
    fix_set = {"NC13b_smooth_clamp", "NC30c_variance_penalty"}
    w1 = _mock_wealth_fn(fix_set)
    w2 = _mock_wealth_fn(fix_set)
    w3 = _mock_wealth_fn(set(fix_set))  # different set instance, same content
    assert w1 == w2 == w3


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------
def test_format_ablation_table_returns_markdown_with_all_combos():
    """Table includes a header row + one row per combination."""
    combos = enumerate_combinations("full")
    results = run_ablation_synthetic(combos)
    table = format_ablation_table(results)
    assert table.startswith("| rank | combination | wealth | uplift vs baseline |")
    # Header (1) + divider (1) + 16 data rows + trailing newline ⇒ 18 lines + ''.
    lines = table.rstrip("\n").split("\n")
    assert len(lines) == 2 + 16, f"expected 18 lines, got {len(lines)}: {lines}"
    # Baseline annotation must appear exactly once.
    assert table.count("◀ baseline") == 1


def test_format_ablation_table_sorted_descending_by_wealth():
    """Rows are ordered with highest wealth at the top."""
    # Construct a controlled results dict so the ordering is unambiguous.
    results: dict[frozenset, float] = {
        frozenset(): 100.0,
        frozenset({"NC13b_smooth_clamp"}): 105.0,
        frozenset({"NC31_tip_conditional"}): 110.0,
        frozenset({"NC13b_smooth_clamp", "NC31_tip_conditional"}): 95.0,
    }
    table = format_ablation_table(results)
    lines = table.rstrip("\n").split("\n")[2:]  # drop header + divider
    # Extract the wealth column from each row.
    wealths = [float(line.split("|")[3].strip()) for line in lines]
    assert wealths == sorted(wealths, reverse=True), f"not sorted desc: {wealths}"


def test_format_ablation_table_empty_results_returns_header_only():
    """Empty input yields just the header (no rows)."""
    table = format_ablation_table({})
    assert "rank" in table
    assert "combination" in table
    # No data rows.
    assert table.count("\n") <= 2


def test_format_ablation_table_uplift_column_signed():
    """Uplift column shows signed deltas vs baseline."""
    results: dict[frozenset, float] = {
        frozenset(): 100.0,
        frozenset({"NC13b_smooth_clamp"}): 110.0,
        frozenset({"NC31_tip_conditional"}): 90.0,
    }
    table = format_ablation_table(results)
    assert "+10.00" in table
    assert "-10.00" in table
