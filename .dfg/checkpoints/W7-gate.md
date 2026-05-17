---
wave: W7
gate_type: wave-gate
verdict: PASS
date: 2026-05-17
units_completed: [W7-1, W7-2, W7-3]
units_carry_forward: [W7-1-CARRY-1]
verify:
  # ─── Wave-shipped via PRs #38, #39, #40 ───────────────────────────────
  - "git log --oneline master | grep -q 'W7-1.*scaffold'"
  - "git log --oneline master | grep -q 'W7-2.*analytics'"
  - "git log --oneline master | grep -q 'W7-3.*template'"

  # ─── W7-1: execution scaffold ─────────────────────────────────────────
  - "test -f .dfg/agents/W7-1-experiment-execution-scaffold.md"
  - "test -f .dfg/retrospectives/W7/W7-1.md"
  - "test -f python_refactor/experiments/validation_matrix.py"
  - "test -f python_refactor/experiments/aggregate_validation.py"
  - "test -f python_refactor/experiments/figures.py"

  # ─── W7-2: analytics plan ─────────────────────────────────────────────
  - "test -f .dfg/agents/W7-2-analytics-plan.md"
  - "test -f .dfg/retrospectives/W7/W7-2.md"
  - "test -f docs/ANALYTICS-PLAN.md"
  - "grep -q 'Descriptive statistics' docs/ANALYTICS-PLAN.md"
  - "grep -q 'Inferential tests' docs/ANALYTICS-PLAN.md"
  - "grep -q 'Multi-factor analysis' docs/ANALYTICS-PLAN.md"
  - "grep -q 'Data segmentation' docs/ANALYTICS-PLAN.md"
  - "grep -q 'Visualization catalog' docs/ANALYTICS-PLAN.md"
  - "grep -q 'Table specifications' docs/ANALYTICS-PLAN.md"
  - "grep -q 'interpretation playbook' docs/ANALYTICS-PLAN.md"

  # ─── W7-3: results template ───────────────────────────────────────────
  - "test -f .dfg/agents/W7-3-validation-results-template.md"
  - "test -f .dfg/retrospectives/W7/W7-3.md"
  - "test -f docs/VALIDATION-RESULTS.md"
  - "grep -q 'TEMPLATE — populated by W8' docs/VALIDATION-RESULTS.md"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W7 builds the live-experiment scaffold + analytics plan + results template. 3 file-disjoint units in 1 parallel group."
  - "Dry-run smoke-test PASSES end-to-end on the scaffold (S0/paper/seed=1)."
  - "ANALYTICS-PLAN.md (~350 lines) covers all 7 operator-named categories: reports, statistical tests, comparisons, multi-factor analyses, data segmentation, viz, tables."
  - "VALIDATION-RESULTS.md is template-only with 🚧 markers everywhere — no fabricated numbers."

carry_forward:
  - id: W7-1-CARRY-1
    why: "Real-run path on validation_matrix.py is unverified — only dry-run smoke-tested. W8 first action is end-to-end S0/paper/seed=1 with --no-dry-run."
    next_action: "W8 smoke-test as opening unit"
---

# W7-gate — WAVE 7 CLOSE: live experiment scaffold + analytics planning

## What W7 delivered

| Unit | PR | What |
|---|---|---|
| W7-1 | #38 | 3 NEW scaffold modules: `validation_matrix.py` (CLI driver), `aggregate_validation.py` (per-seed→summary), `figures.py` (per-run + comparison PNGs) |
| W7-2 | #39 | `docs/ANALYTICS-PLAN.md` (~350 lines, 11 sections covering all 7 operator-named analytics categories) |
| W7-3 | #40 | `docs/VALIDATION-RESULTS.md` (~200 lines template skeleton; 🚧 markers throughout) |

Plus PR #37 (replan W7-add).

## Carries forwarded

- W7-1-CARRY-1: scaffold real-run path needs W8 end-to-end smoke-test.
