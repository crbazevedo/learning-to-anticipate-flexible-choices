---
wave: W10
gate_type: wave-gate
verdict: PASS-WITH-MAJOR-CARRY
date: 2026-05-17
units_completed: [W10-1, W10-2, W10-3, W10-4, W10-5]
units_carry_forward: [W7-1-CARRY-1, W8-CARRY-3, W9-3-CARRY-1, W10-CARRY-1, W10-CARRY-2]
verify:
  # ─── Wave-shipped via PRs #55, #56 ────────────────────────────────────
  - "git log --oneline master | grep -q 'W10-1.*flatten'"
  - "git log --oneline master | grep -q 'W10-2..W10-5.*S1-S4'"

  # ─── W10-1: validation_matrix tightening ──────────────────────────────
  - "test -f .dfg/agents/W10-1-validation-matrix-tightening.md"
  - "test -f .dfg/retrospectives/W10/W10-1.md"
  - "test -f python_refactor/tests/test_experiments_validation_matrix.py"
  - "grep -q '_flatten_metrics' python_refactor/experiments/validation_matrix.py"
  - "grep -q 'legacy-cpp.*table.*csv' python_refactor/experiments/validation_matrix.py"

  # ─── W10-2..W10-5: S1-S4 smoke-tests ──────────────────────────────────
  - "test -f .dfg/agents/W10-2-s1-smoke-test.md"
  - "test -f .dfg/agents/W10-3-s2-smoke-test.md"
  - "test -f .dfg/agents/W10-4-s3-smoke-test.md"
  - "test -f .dfg/agents/W10-5-s4-smoke-test.md"
  - "test -f .dfg/retrospectives/W10/W10-2.md"
  - "test -f .dfg/retrospectives/W10/W10-3.md"
  - "test -f .dfg/retrospectives/W10/W10-4.md"
  - "test -f .dfg/retrospectives/W10/W10-5.md"

  # ─── Honest scar surfacing ────────────────────────────────────────────
  - "grep -q 'W10-CARRY-2' .dfg/retrospectives/W10/W10-2.md"
  - "grep -q 'W10-CARRY-2' .dfg/retrospectives/W10/W10-3.md"
  - "grep -q 'W10-CARRY-2' .dfg/retrospectives/W10/W10-4.md"
  - "grep -q 'W10-CARRY-2' .dfg/retrospectives/W10/W10-5.md"
  - "grep -q 'IDENTICAL\\|identical' .dfg/retrospectives/W10/W10-3.md"

  # ─── Test suite green ─────────────────────────────────────────────────
  - "cd python_refactor && uv run python -m pytest tests/test_experiments_validation_matrix.py tests/test_experiments_data_loader.py tests/test_experiments_experiment_manager.py tests/test_experiments_stats.py tests/test_experiments_tables.py tests/test_experiments_report.py -q"

  # ─── Retro-hygiene gate green for W10 ─────────────────────────────────
  - "bash scripts/pre-pr-reflect-validate.sh --wave W10"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W10 activates the pipeline: metric flattening + paper-window real-glob. metrics.csv goes from 0 rows to 37 (or 38 with S0). 5 scenarios all smoke-test green at orchestration level."
  - "CRITICAL FINDING (W10-CARRY-2): S1, S2, S3, S4 all produce IDENTICAL metrics to 15 decimal places. The `learning` config dispatch from scenario → ExperimentManager → algorithm is broken (or instrumentation is missing). This blocks meaningful experiment execution until W11-1 audits the dispatch path."
  - "Verdict is PASS-WITH-MAJOR-CARRY, not PASS-WITH-CARRIES. The smoke-tests technically meet acceptance (exit 0, status=ok, ≥ 5 metric rows) but expose a fundamental blocker for downstream analytics — W11 cannot run multi-seed scenarios meaningfully until W10-CARRY-2 is investigated and resolved."
  - "Class-retirement (directive #5): the gate that retires the 'identical-scenario-metrics' class is a per-wave acceptance criterion 'no two scenarios with different learning configs produce identical algorithm/diversity metrics'. Added to W11's plan structure (will land in next replan)."
  - "W10-CARRY-1 (portfolio NaN/Inf on real data, surfaced by W10-1) and W10-CARRY-2 (scenario dispatch broken, surfaced by W10-2..W10-5) are now the two blockers for W11. Both are instances of dead-code-coming-alive — synthetic configs hid real-data + real-dispatch issues."

carry_forward:
  - id: W7-1-CARRY-1
    why: "Reissued. Smoke-test orchestration green for all 5 scenarios but the experiments don't actually exercise their differentiating code paths (W10-CARRY-2). Real closure requires meaningful scenario differentiation."
    next_action: "W11 — scenario-dispatch audit + per-scenario differentiation gate"
  - id: W8-CARRY-3
    why: "Tables C/E/F/G/H builders — but Table E (horizon ablation) and Table D (synergy) are degenerate by construction while W10-CARRY-2 is open. Postpone builder completion until dispatch fixed."
    next_action: "Future analytics-completion wave, after W11/W12 produce real differentiated metrics"
  - id: W9-3-CARRY-1
    why: "Optional W1-W6 retro key backfill — unchanged."
    next_action: "Optional mechanical cleanup wave"
  - id: W10-CARRY-1
    why: "Portfolio-evaluator emits NaN/Infinity on real 98-asset paper-window data (max_drawdown, calmar_ratio, portfolio_value). Synthetic FTSE-updated single-CSV hid this; real data fires it. Likely division-by-zero or Inf-return propagation in src/experiments/."
    next_action: "W11-2 candidate: portfolio-evaluator investigation + NaN/Inf guards"
  - id: W10-CARRY-2
    why: "Scenarios S1, S2, S3, S4 produce identical metrics to 15 decimal places. learning config dispatch from scenario → ExperimentManager → algorithm broken OR not instrumented. BLOCKING for W11/W12 meaningful execution."
    next_action: "W11-1 (highest priority): scenario-dispatch audit. Trace scenario['learning'] → manager._run_algorithm → optimizer → final_metrics. Locate the gap; add metric-differentiation regression gate."
---

# W10-gate — WAVE 10 CLOSE: pipeline activation + critical scenario-dispatch finding

## What W10 delivered

| Unit | PR | What |
|---|---|---|
| W10-1 | #55 | validation_matrix tightening: metric flatten (W9-CARRY-3 closed) + paper-window real 98-CSV glob (W8-1-CARRY-1 closed). 9/9 tests pass. |
| W10-2 | #56 | S1/paper smoke-test: exit 0, 37 metric rows. Identical to S2/S3/S4 (W10-CARRY-2 signal). |
| W10-3 | #56 | S2/paper smoke-test (paper headline config): exit 0, 37 metric rows. Identical to S1/S3/S4. |
| W10-4 | #56 | S3/paper smoke-test (H=2 control): exit 0, 37 metric rows. Identical to S2 — horizon-ablation degenerate. |
| W10-5 | #56 | S4/paper smoke-test (1000 MC, explicit cov): exit 0, 37 metric rows. Identical to S2 — explicit-cov ablation degenerate. |

Plus PR #54 (W10 replan).

## Closes

- ✅ W9-CARRY-3 (metric flattening)
- ✅ W8-1-CARRY-1 (paper-window per-asset CSV loader)

## CRITICAL carries forward

| ID | Why | Block? |
|---|---|---|
| **W10-CARRY-2** | S1=S2=S3=S4 in metrics — learning config dispatch broken | **YES — W11 must audit before any multi-seed run** |
| W10-CARRY-1 | Portfolio NaN/Inf on real data | Yes for meaningful results (numbers are unusable) |
| W7-1-CARRY-1 (reissued) | Real closure requires meaningful differentiation | Yes |

## Verdict

**PASS-WITH-MAJOR-CARRY** — orchestration is fully wired (metrics flow end-to-end), but the experimental SCENARIOS don't actually differentiate. The pipeline is built; the science isn't yet possible. W11 must be the dispatch audit + portfolio-evaluator fix BEFORE multi-seed runs.

## What the user asked: "when do we execute the experiments?"

**Not after W11 anymore** — the W10-CARRY-2 finding pushes the answer out by one wave. Revised path:

1. **W11** — scenario-dispatch audit + portfolio-evaluator fix (Likely L-sized; this is where the science gets unblocked)
2. **W12** — single-scenario multi-seed end-to-end (proves pipeline once dispatch works)
3. **W13** — full 300-run matrix + populated analytics report

Original estimate was W10→W11→W12. The W10-CARRY-2 finding adds one wave because we can't multi-seed scenarios that produce identical numbers — that would burn compute on a null result.
