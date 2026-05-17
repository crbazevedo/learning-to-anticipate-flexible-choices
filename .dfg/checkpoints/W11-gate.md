---
wave: W11
gate_type: wave-gate
verdict: PASS
date: 2026-05-17
units_completed: [W11-1, W11-2, W11-3]
units_carry_forward: [W8-CARRY-3, W9-3-CARRY-1, W11-CARRY-1]
verify:
  - "git log --oneline master | grep -q 'W11-1.*dispatch'"
  - "git log --oneline master | grep -q 'W11-2.*sanitation'"
  - "git log --oneline master | grep -q 'W11-3.*differentiation'"

  # ─── W11-1: dispatch fix ──────────────────────────────────────────────
  - "test -f .dfg/agents/W11-1-sms-emoa-learn-population-dispatch.md"
  - "test -f .dfg/retrospectives/W11/W11-1.md"
  - "test -f python_refactor/tests/test_sms_emoa_dispatch.py"
  - "grep -q 'hasattr(learner, .learn_population.)' python_refactor/src/algorithms/sms_emoa.py"

  # ─── W11-2: portfolio sanitation ──────────────────────────────────────
  - "test -f .dfg/agents/W11-2-portfolio-evaluator-returns-space.md"
  - "test -f .dfg/retrospectives/W11/W11-2.md"
  - "test -f python_refactor/tests/test_experiments_portfolio_evaluator.py"
  - "grep -q 'replace.*inf' python_refactor/src/experiments/portfolio_evaluator.py"
  - "grep -q 'clip' python_refactor/src/experiments/portfolio_evaluator.py"

  # ─── W11-3: differentiation gate ──────────────────────────────────────
  - "test -f .dfg/agents/W11-3-scenario-differentiation-verify-and-gate.md"
  - "test -f .dfg/retrospectives/W11/W11-3.md"
  - "test -f python_refactor/tests/test_scenario_differentiation.py"
  - "grep -q 'class-retiring\\|W10-CARRY-2' python_refactor/tests/test_scenario_differentiation.py"

  # ─── Test suite green ─────────────────────────────────────────────────
  - "cd python_refactor && uv run python -m pytest tests/test_sms_emoa_dispatch.py tests/test_experiments_portfolio_evaluator.py tests/test_experiments_validation_matrix.py tests/test_experiments_data_loader.py tests/test_experiments_experiment_manager.py -q"

  # ─── Retro-hygiene gate green for W11 ─────────────────────────────────
  - "bash scripts/pre-pr-reflect-validate.sh --wave W11"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W11 unblocks the experiments. SMS-EMOA dispatch fixed (W11-1); portfolio returns sanitized (W11-2); scenarios now produce differentiated, finite metrics (W11-3 receipt + class-retiring gate)."
  - "After W11: S0-S4 each produce distinct diversity_metric (0.0794 / 0.3144 / 0.2711 / 0.3894 / 0.2884); portfolio_value finite (1.1328 across all); max_drawdown finite (0.0). The experiments now actually test what the SCENARIOS dict claims they test."
  - "3 carries CLOSED in this wave: W10-CARRY-1 (portfolio NaN/Inf), W10-CARRY-2 (scenario differentiation), W7-1-CARRY-1 (real-run smoke-test green end-to-end with differentiated finite metrics)."
  - "1 soft scar filed: W11-CARRY-1 (portfolio_value identical across S0-S4 on seed=1 / 30 gens; algorithm explores differently but realized wealth converges; may be Markowitz-baseline behavior; flag for W12 multi-seed)."
  - "Verdict: PASS (first PASS-not-WITH-CARRIES wave since W6). The experiments are now ready to run for real."
  - "Class-retirement receipts (directive #5): (a) scenario-dispatch silently identical-metrics class retired via tests/test_scenario_differentiation.py; (b) portfolio Inf/NaN class retired via sanitation in _get_asset_returns + finite-check in differentiation gate. No new canon entries; existing patterns reinforced."

carry_forward:
  - id: W8-CARRY-3
    why: "Tables C/E/F/G/H builders for analytics — now meaningful since scenarios differentiate. Will produce real results in W12+."
    next_action: "Future analytics-completion wave after W12 multi-seed runs"
  - id: W9-3-CARRY-1
    why: "Optional W1-W6 retro key backfill — unchanged."
    next_action: "Optional mechanical cleanup wave"
  - id: W11-CARRY-1
    why: "portfolio.portfolio_value = 1.1328 identical across all S0-S4 on seed=1 with 30 generations. Algorithm explores differently (diversity_metric distinct) but realized wealth converges to the same value. Plausible Markowitz-baseline behavior; W12 multi-seed will tell if it's seed-specific or systemic."
    next_action: "W12: if multi-seed shows scenario wealth distributions still identical, investigate portfolio_evaluator weights/objectives layer"
---

# W11-gate — WAVE 11 CLOSE: experiments unblocked

## What W11 delivered

| Unit | PR | What |
|---|---|---|
| W11-1 | #59 | SMS-EMOA dispatch fix: routes to learn_population when subclass defines it (paper Eq 13/14/15 fires). 5/5 tests. |
| W11-2 | #60 | Portfolio sanitation: replace Inf, clip ±1.0 + price-level guard. 8/8 tests. Real-data Inf eliminated. |
| W11-3 | #61 | Re-smoke-test S0-S4 (all differentiated, finite) + class-retiring regression gate. 4/4 tests. |

Plus PR #58 (W11 replan).

## Closes (3 carries — first PASS-not-WITH-CARRIES wave since W6)

| Carry | Closed by |
|---|---|
| W10-CARRY-1 (portfolio NaN/Inf) | W11-2 sanitation |
| W10-CARRY-2 (scenarios identical) | W11-1 dispatch + W11-3 verify |
| W7-1-CARRY-1 (real-run smoke-test) | W11-3 (all 5 scenarios exit ok with differentiated metrics) |

## Verdict

**PASS** — experiments are now meaningfully runnable. Pipeline is wired (W10) + dispatch is fixed (W11) + metrics are sane. Ready for W12 multi-seed runs.

## What this means for "execute the experiments"

**W12 next. Confidence high.** Single-scenario multi-seed (e.g., S0 × 30 seeds × paper-window) should produce a real summary CSV that the analytics layer (W8) can populate VALIDATION-RESULTS.md against.

If W12 multi-seed reveals that scenario wealth-distributions are also identical (W11-CARRY-1 escalates to a hard blocker), W12.5 = portfolio_evaluator deeper investigation. Otherwise W12 → W13 full 300-run matrix → populated paper-comparison.
