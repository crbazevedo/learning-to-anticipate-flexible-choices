---
wave: W9
gate_type: wave-gate
verdict: PASS-WITH-CARRIES
date: 2026-05-17
units_completed: [W9-1, W9-2, W9-3, W9-4]
units_carry_forward: [W7-1-CARRY-1, W8-CARRY-3, W9-3-CARRY-1, W9-CARRY-3]
verify:
  # ─── Wave-shipped via PRs #49, #50, #51, #52 ──────────────────────────
  - "git log --oneline master | grep -q 'W9-1.*bool'"
  - "git log --oneline master | grep -q 'W9-2.*pivot.*glob'"
  - "git log --oneline master | grep -q 'W9-3.*retro-hygiene'"
  - "git log --oneline master | grep -q 'W9-4.*smoke-test'"

  # ─── W9-1: experiment_manager bool(DataFrame) fix ─────────────────────
  - "test -f .dfg/agents/W9-1-experiment-manager-bool-dataframe-fix.md"
  - "test -f .dfg/retrospectives/W9/W9-1.md"
  - "test -f python_refactor/tests/test_experiments_experiment_manager.py"
  - "grep -q 'asset_data is None or asset_data.empty' python_refactor/src/experiments/experiment_manager.py"

  # ─── W9-2: data_loader pivot + glob fix ───────────────────────────────
  - "test -f .dfg/agents/W9-2-data-loader-pivot-and-glob.md"
  - "test -f .dfg/retrospectives/W9/W9-2.md"
  - "test -f python_refactor/tests/test_experiments_data_loader.py"
  - "grep -q \"columns='asset_id'\" python_refactor/src/experiments/data_loader.py"
  - "grep -q 'drop_duplicates' python_refactor/src/experiments/data_loader.py"

  # ─── W9-3: retro-hygiene gate ─────────────────────────────────────────
  - "test -f .dfg/agents/W9-3-retro-hygiene-gate.md"
  - "test -f .dfg/retrospectives/W9/W9-3.md"
  - "test -f scripts/pre-pr-reflect-validate.sh"
  - "test -x scripts/pre-pr-reflect-validate.sh"
  - "grep -q 'what_worked' .dfg/retrospectives/W7/W7-1.md"
  - "grep -q 'what_broke' .dfg/retrospectives/W8/W8-1.md"
  - "grep -q 'what_youd_change' .dfg/retrospectives/W8/W8-4.md"
  - "grep -q 'assumption_to_challenge' .dfg/retrospectives/W9/W9-3.md"

  # ─── W9-4: smoke-test rerun + receipts ────────────────────────────────
  - "test -f .dfg/agents/W9-4-smoke-test-end-to-end-rerun.md"
  - "test -f .dfg/retrospectives/W9/W9-4.md"
  - "grep -q 'W8-1-CARRY-1 — CLOSED' .dfg/retrospectives/W9/W9-4.md"
  - "grep -q 'W8-1-CARRY-2 — CLOSED' .dfg/retrospectives/W9/W9-4.md"
  - "grep -q 'W9-CARRY-3' .dfg/retrospectives/W9/W9-4.md"

  # ─── Test suites green ────────────────────────────────────────────────
  - "cd python_refactor && uv run python -m pytest tests/test_experiments_experiment_manager.py tests/test_experiments_data_loader.py tests/test_experiments_stats.py tests/test_experiments_tables.py tests/test_experiments_report.py -q"

  # ─── Retro-hygiene gate green for W9 ──────────────────────────────────
  - "bash scripts/pre-pr-reflect-validate.sh --wave W9"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W9 closes the W8-1 src/ cascade: bool(DataFrame) (W9-1), pivot KeyError + multi-CSV glob (W9-2), and a year-end dedup hotfix (W9-4). Smoke-test now runs end-to-end with status=ok."
  - "W9-3 retires sub-papercut #23 by re-keying 7 W7+W8 retros to ADR-004 canonical (what_worked / what_broke / what_youd_change / assumption_to_challenge) and wiring `dfg reflect --validate` into scripts/pre-pr-reflect-validate.sh. After re-key + cursor reset, dfg reflect --backfill emits 7 events (was 0)."
  - "47 tests green across the analytics + integration test surface (3 W9-1 + 7 W9-2/W9-4 + 16 W8-2 + 15 W8-3 + 9 W8-4). Add 3 from each prior wave; full suite is wider."
  - "Verdict is PASS-WITH-CARRIES, not PASS. The smoke-test orchestration succeeds but metrics.csv has 0 metric rows because validation_matrix.py's row-writer filters `isinstance(value, (int, float))` against a nested final_metrics dict. Filed as W9-CARRY-3 for W10. Dead-code-coming-alive instance #7."
  - "Honest scar: W9-2 synthetic test fixtures missed the real-data year-end-duplicate quirk; W9-4 surfaced it and shipped both the fix + a real-CSV regression test (sub-papercut #15 receipt: 'synthesized fixtures must match real substrate')."
  - "Discipline win: stopped scope-creep at the metric-flattening layer. Filed as W9-CARRY-3 rather than padding W9-4 with a third inline fix. The 2-of-3 acceptance pattern is the contract working as designed."

carry_forward:
  - id: W7-1-CARRY-1
    why: "PARTIAL closure only. Smoke-test orchestration reaches algorithm + portfolio completion (was blocked at bool(DataFrame) before W9-1). Full closure needs metrics.csv populated, which needs W9-CARRY-3."
    next_action: "Closes when W10 lands metric-flattening + a re-run smoke-test populates ≥ 1 metric row"
  - id: W8-CARRY-3
    why: "Tables C/E/F/G/H builders + paired-by-seed bootstrap for D — analytics-completion. Needs per-run inputs (regime tags, sub-window splits, paper-Table-I) the scaffold doesn't yet emit."
    next_action: "Future analytics-completion wave; depends on metric-flattening landing first"
  - id: W9-3-CARRY-1
    why: "Backfill W1-W6 retros' frontmatter keys (~22 retros use non-canonical keys). `scripts/pre-pr-reflect-validate.sh --all` surfaces them; default `--wave` invocation hides them."
    next_action: "Optional mechanical cleanup wave"
  - id: W9-CARRY-3
    why: "validation_matrix.py:266-269 row-writer filters `isinstance(value, (int, float))` against final_metrics which is a nested dict — every top-level value is a sub-dict, so 0 metric rows get written. Smoke-test status=ok but metrics.csv has only the header."
    next_action: "W10-1 candidate: flatten final_metrics recursively, emit `<category>.<metric>` rows. ~10 LOC fix + a smoke-test assertion that metrics.csv has at least 1 metric row."
---

# W9-gate — WAVE 9 CLOSE: src/ integration fixup + retro-hygiene gate

## What W9 delivered

| Unit | PR | What |
|---|---|---|
| W9-1 | #50 | bool(DataFrame) fix at experiment_manager.py:190 (3/3 tests pass) |
| W9-2 | #51 | data_loader pivot KeyError + multi-CSV glob support (5/5 tests pass; paper-window ≥ 90 assets) |
| W9-3 | #49 | Retro-hygiene gate (7 retros re-keyed to ADR-004 canonical; pre-pr-reflect-validate.sh wired; sub-papercut #23 retired) |
| W9-4 | #52 | Smoke-test re-run + year-end dedup hotfix (sub-papercut #15 receipt; 7/7 data_loader tests; status=ok end-to-end) |

Plus PR #48 (W9 replan, ALSO carries W7/W8 wave-close emits + ReflectionEmit backfill for the prior 7 retros).

**Test suite:** 47 tests green across the analytics + integration surface (3 W9-1 + 7 W9-2/W9-4 + 16 W8-2 + 15 W8-3 + 9 W8-4; not including the much wider pre-existing src/ suite).

## Carries forwarded

| ID | Status | Next |
|---|---|---|
| W7-1-CARRY-1 | reissued (PARTIAL closure) | Closes when W10 lands metric-flattening |
| W8-CARRY-3 | reissued | Future analytics-completion wave |
| W9-3-CARRY-1 | new (optional) | W1-W6 retro key backfill |
| W9-3-CARRY-2 | **CLOSED** | plan.yaml path corrected in wave-close PR (replan unit-W9-3-20260517T024044Z) |
| W9-CARRY-3 | new | validation_matrix metric-flattening — W10-1 candidate |

## Verdict

**PASS-WITH-CARRIES** — src/ cascade fixed, smoke-test green at orchestration level, retro-hygiene gate live. New cascade layer (metric flattening) honestly filed as W9-CARRY-3, not laundered. Dead-code-coming-alive pattern instance #7.

## Class-retirement receipts (directive #5)

1. **Sub-papercut #23** (retro frontmatter wrong-keys silently skipping ReflectionEmit): retired via `scripts/pre-pr-reflect-validate.sh` calling existing `dfg reflect --validate` — no harness modification, just discipline to invoke the gate.
2. **Substrate "checkpoint PASS but no WaveGatePass event"** (operator-intervention scar): retired via running `dfg wave close W7 / W8` during the W9 setup. Existing gate; consistent invocation from now on.
3. **Sub-papercut #15** (synthetic fixtures don't match real substrate): caught in W9-4 — W9-2's synthetic tests missed the year-end dup quirk; the fix shipped a real-CSV regression test. Receipt documented; class-retiring pattern is "always include a real-data regression test alongside synthetic fixtures for data-loader changes."
