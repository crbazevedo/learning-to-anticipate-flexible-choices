---
wave: W12
gate_type: wave-gate
verdict: PASS-WITH-CRITICAL-CARRY
date: 2026-05-17
units_completed: [W12-1, W12-2]
units_carry_forward: [W8-CARRY-3, W9-3-CARRY-1, W12-CARRY-1]
verify:
  - "git log --oneline master | grep -q 'W12-1.*sweep driver'"
  - "git log --oneline master | grep -q 'W12-2.*populated'"

  # ─── W12-1: sweep driver ──────────────────────────────────────────────
  - "test -f .dfg/agents/W12-1-multi-seed-sweep-driver.md"
  - "test -f .dfg/retrospectives/W12/W12-1.md"
  - "test -f python_refactor/experiments/sweep.py"
  - "test -f python_refactor/tests/test_experiments_sweep.py"
  - "grep -q 'def run_sweep' python_refactor/experiments/sweep.py"
  - "grep -q 'ProcessPoolExecutor' python_refactor/experiments/sweep.py"

  # ─── W12-2: populated doc ─────────────────────────────────────────────
  - "test -f .dfg/agents/W12-2-first-populated-results-doc.md"
  - "test -f .dfg/retrospectives/W12/W12-2.md"
  - "test -f docs/VALIDATION-RESULTS.populated.md"
  - "grep -q 'git_sha:    28bc9062\\|git_sha:' docs/VALIDATION-RESULTS.populated.md"
  - "grep -q 'algorithm.diversity_metric' docs/VALIDATION-RESULTS.populated.md"
  - "grep -q 'S0 | paper' docs/VALIDATION-RESULTS.populated.md"
  - "grep -q 'S2 | paper' docs/VALIDATION-RESULTS.populated.md"

  # ─── Honest scar surfaced ─────────────────────────────────────────────
  - "grep -q 'W12-CARRY-1' .dfg/retrospectives/W12/W12-2.md"
  - "grep -q 'decoupled\\|DECOUPLED' .dfg/retrospectives/W12/W12-2.md"

  # ─── Test suite green ─────────────────────────────────────────────────
  - "cd python_refactor && uv run python -m pytest tests/test_experiments_sweep.py tests/test_sms_emoa_dispatch.py tests/test_experiments_portfolio_evaluator.py tests/test_experiments_validation_matrix.py -q"

  # ─── Retro-hygiene gate green for W12 ─────────────────────────────────
  - "bash scripts/pre-pr-reflect-validate.sh --wave W12"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W12 ships the first real experimental output. 60-run sweep (S0+S2 × paper × 30 seeds) completed 60/60 ok in 5:07 wall-clock. VALIDATION-RESULTS.populated.md (312 lines) exists with §0 receipt block + Table A populated for 27 metrics × 2 scenarios."
  - "Algorithm IS real: S2 takes ~37s/run vs S0's ~0.6s/run (60x — the W11-1 dispatch fix means S2 actually runs MultiHorizon compute). Algorithm metrics differentiate seed-by-seed and scenario-by-scenario."
  - "CRITICAL FINDING (W12-CARRY-1): portfolio.portfolio_value = 1.132857 ± 0 for BOTH S0 and S2 across ALL 30 seeds. Despite portfolio.concentration differing (0.41 vs 0.44 — different portfolios chosen), the EVALUATED wealth is identical. portfolio_evaluator is decoupled from the optimizer's weight output."
  - "Verdict: PASS-WITH-CRITICAL-CARRY. The infrastructure works end-to-end; the science requires W13 audit of the optimizer→evaluator data flow before the full 300-run matrix can produce meaningful paper-comparison."
  - "Dead-code-coming-alive instance #9. Each smoke-test depth-pass surfaces one more layer."

carry_forward:
  - id: W8-CARRY-3
    why: "Tables C/E/F/G/H builders — now also blocked on W12-CARRY-1 (meaningful wealth required for synergy contrast, paper comparison)."
    next_action: "Future analytics-completion wave after W13 resolves portfolio evaluator decoupling"
  - id: W9-3-CARRY-1
    why: "Optional W1-W6 retro key backfill — unchanged."
    next_action: "Optional mechanical cleanup wave"
  - id: W12-CARRY-1
    why: "portfolio.portfolio_value = 1.132857 ± 0 for S0 = S2 across 30 seeds. Algorithm picks different portfolios (concentration 0.41 vs 0.44) but evaluator computes same wealth → optimizer's weights don't reach the wealth calculation. Likely: equal-weight baseline used in evaluation, OR algorithm converges to same 3 of 98 assets every time, OR portfolio_evaluator computes a constant. Blocks paper-comparison validity."
    next_action: "W13-1 (HIGH PRIORITY): instrument PortfolioEvaluator._calculate_portfolio_performance to log the weight vector + per-asset returns + per-period portfolio returns. Probe with 4 runs (2 scenarios × 2 seeds) before scaling. Do NOT run W13 full matrix until decoupling resolved."
---

# W12-gate — WAVE 12 CLOSE: first real experimental output (with critical scar)

## What W12 delivered

| Unit | PR | What |
|---|---|---|
| W12-1 | #64 | Multi-seed sweep driver (`python -m experiments.sweep`). 8/8 tests including real 2-seed end-to-end smoke. |
| W12-2 | #65 | 60-run sweep (S0+S2 × paper × 30 seeds) + first populated VALIDATION-RESULTS.md (312 lines). Surfaces W12-CARRY-1. |

Plus PR #63 (W12 replan).

## Wins

- ✅ Pipeline works end-to-end: sweep → aggregate → report → populated doc
- ✅ Receipt block (§0) populated: git SHA + uv.lock fingerprint + 2220 per-run values
- ✅ Algorithm differentiates across scenarios (compute-time receipt: S2 60x slower than S0; W11-1 dispatch fix verified at scale)
- ✅ All 60 runs succeeded without seed-specific bugs

## Critical scar (W12-CARRY-1)

`portfolio.portfolio_value = 1.133 ± 0` for BOTH S0 and S2 across all 30 seeds. Despite optimizer choosing different portfolios (concentration 0.41 vs 0.44), evaluator returns same wealth. Decoupling. Blocks paper-comparison.

## Verdict

**PASS-WITH-CRITICAL-CARRY** — infrastructure complete, experimental science blocked on W13-1 portfolio-evaluator audit.

## What this means for "execute the experiments"

The 300-run matrix (W13 original scope) is NOT the next move. Per directive #5 inverse: don't scale the experiment until the treatment actually affects the outcome.

| Wave | Was | Now | Why |
|---|---|---|---|
| W13 | Full 300-run matrix | **portfolio_evaluator decoupling audit + fix** | W12-CARRY-1 |
| W14 | populated paper-comparison | **Full 300-run matrix** | depends on W13 |
| W15 | — | **populated paper-comparison + analytics report** | depends on W14 |

Each wave so far has surfaced one more layer (instance #9 here). The W13 probe will be small + targeted: instrument 4 runs, find the decoupling, fix.
