---
wave: W8
gate_type: wave-gate
verdict: PASS-WITH-CARRIES
date: 2026-05-17
units_completed: [W8-1, W8-2, W8-3, W8-4]
units_carry_forward: [W7-1-CARRY-1, W8-1-CARRY-1, W8-1-CARRY-2, W8-CARRY-3]
verify:
  # ─── Wave-shipped via PRs #43, #44, #45, #46 ──────────────────────────
  - "git log --oneline master | grep -q 'W8-1.*smoke-test'"
  - "git log --oneline master | grep -q 'W8-2.*stats'"
  - "git log --oneline master | grep -q 'W8-3.*table generators'"
  - "git log --oneline master | grep -q 'W8-4.*report orchestrator'"

  # ─── W8-1: smoke-test scaffold reachability ───────────────────────────
  - "test -f .dfg/agents/W8-1-validation-matrix-smoke-test.md"
  - "test -f .dfg/retrospectives/W8/W8-1.md"
  - "grep -q 'HONEST SCAR' .dfg/retrospectives/W8/W8-1.md"

  # ─── W8-2: statistical test helpers ───────────────────────────────────
  - "test -f .dfg/agents/W8-2-stats-helpers.md"
  - "test -f .dfg/retrospectives/W8/W8-2.md"
  - "test -f python_refactor/experiments/stats.py"
  - "test -f python_refactor/tests/test_experiments_stats.py"
  - "grep -q 'def mann_whitney_u' python_refactor/experiments/stats.py"
  - "grep -q 'def cohens_d' python_refactor/experiments/stats.py"
  - "grep -q 'def holm_bonferroni' python_refactor/experiments/stats.py"

  # ─── W8-3: table generators A-H ───────────────────────────────────────
  - "test -f .dfg/agents/W8-3-table-generators.md"
  - "test -f .dfg/retrospectives/W8/W8-3.md"
  - "test -f python_refactor/experiments/tables.py"
  - "test -f python_refactor/tests/test_experiments_tables.py"
  - "grep -q 'def generate_table_a' python_refactor/experiments/tables.py"
  - "grep -q 'def generate_table_h' python_refactor/experiments/tables.py"

  # ─── W8-4: report orchestrator ────────────────────────────────────────
  - "test -f .dfg/agents/W8-4-report-orchestrator.md"
  - "test -f .dfg/retrospectives/W8/W8-4.md"
  - "test -f python_refactor/experiments/report.py"
  - "test -f python_refactor/tests/test_experiments_report.py"
  - "grep -q 'def build_receipt_block' python_refactor/experiments/report.py"
  - "grep -q 'def populate' python_refactor/experiments/report.py"

  # ─── Analytics suite green ────────────────────────────────────────────
  - "cd python_refactor && uv run python -m pytest tests/test_experiments_stats.py tests/test_experiments_tables.py tests/test_experiments_report.py -q"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W8 ships the full analytics layer (stats + tables + report) for the validation results — 40/40 tests pass in 0.9s."
  - "8 statistical helpers (W8-2), 8 table generators (W8-3), 1 report orchestrator (W8-4) — all pure functions, no I/O coupling."
  - "Tables A, B, D wired end-to-end in the orchestrator. C/E/F/G/H stubbed (need per-run inputs not yet emitted by W7-1 scaffold) → W8-CARRY-3."
  - "W7-1-CARRY-1 remains OPEN — the validation_matrix real-run smoke-test still does NOT pass end-to-end. W8-1 surfaced 3+ src/ integration bugs (bool(DataFrame), pivot KeyError); fixing them was out-of-scope per directive #4 (no scope creep without replan). Honest scar — not laundered."
  - "Verdict is PASS-WITH-CARRIES, not PASS. The analytics layer is complete and correct; the data pipeline upstream of it has unresolved cascade bugs that block real-run consumption. W9 candidate: focused src/ integration fixup."

carry_forward:
  - id: W7-1-CARRY-1
    why: "Still OPEN. validation_matrix.py is scaffold-callable up to data-loading layer, but full end-to-end smoke-test blocked by W8-1-CARRY-2 src/ bug cascade."
    next_action: "Closes when W9 (src/ integration fixup) lands"
  - id: W8-1-CARRY-1
    why: "Paper-window data loader: 98 per-asset CSVs at legacy-cpp/executable/data/ftse-original/. data_loader treats asset_files as literal paths, not globs. Smoke-test temporarily uses FTSE-updated CSV for both windows."
    next_action: "Add multi-CSV per-asset loader to src/data_loader.py in W9 or W10"
  - id: W8-1-CARRY-2
    why: "Pre-existing src/ integration bugs surfaced by W8-1's smoke-test: experiment_manager.py:190 bool(DataFrame) ambiguity, data_loader → pandas.pivot() KeyError(None) cascade, likely more deeper. Dead-code-coming-alive pattern, 6th instance."
    next_action: "Focused W9 'src/ integration fixup' wave — investigate cascade, fix bugs, re-run W8-1 smoke-test to verify closure"
  - id: W8-CARRY-3
    why: "Report orchestrator wires Tables A, B, D end-to-end. C (ANOVA), E (per-horizon ablation), F (regime segmentation), G (sub-window stability), H (paper comparison) need per-run inputs the W7-1 scaffold doesn't yet emit (regime tags, sub-window splits, paper-Table-I gold values). Also: paired-by-seed bootstrap upgrade for D when seeds align."
    next_action: "Complete in a future analytics-completion wave once W9 src/ fixup + W7-1 scaffold extensions land"
---

# W8-gate — WAVE 8 CLOSE: analytics layer (stats + tables + report orchestrator)

## What W8 delivered

| Unit | PR | What |
|---|---|---|
| W8-1 | #43 | validation_matrix.py callable end-to-end up to data layer; 2 scaffold fixes; surfaced + filed src/ cascade as W8-1-CARRY-2 (honest scar). |
| W8-2 | #44 | `experiments/stats.py` (~180 lines, 8 pure functions): Mann-Whitney, Wilcoxon, Welch, ANOVA, Cohen's d, Cliff's δ, bootstrap CI, Holm-Bonferroni. 16/16 tests. |
| W8-3 | #45 | `experiments/tables.py` (~285 lines, 8 Markdown generators A-H + 3 formatters). 15/15 tests. |
| W8-4 | #46 | `experiments/report.py` (~330 lines): composes stats + tables into populated VALIDATION-RESULTS.md; receipt block per ANALYTICS-PLAN §8; graceful incompleteness for cells without data. 9/9 tests. |

Plus PR #42 (W8 replan).

**Analytics suite:** 40/40 tests pass in 0.9s.

## Carries forwarded

- **W7-1-CARRY-1** (REISSUED): validation_matrix real-run smoke-test still blocked.
- **W8-1-CARRY-1** (NEW): paper-window per-asset CSV loader.
- **W8-1-CARRY-2** (NEW): src/ integration bug cascade (3+ bugs).
- **W8-CARRY-3** (NEW): Tables C/E/F/G/H builders + paired bootstrap for D.

## Verdict

**PASS-WITH-CARRIES** — analytics layer shipped complete and correct; data pipeline upstream of it has unresolved cascade bugs (W8-1-CARRY-2) blocking real-run consumption. Honest scar — gate criterion #2 ("W7-1-CARRY-1 closed") NOT met; reissued for W9.
