---
wave: W13
gate_type: wave-gate
verdict: PASS-WITH-CRITICAL-METHODOLOGY-SCAR
date: 2026-05-17
units_completed: [W13-1, W13-2, W13-3]
units_carry_forward: [W8-CARRY-3, W9-3-CARRY-1, W12-CARRY-1, W13-CARRY-1, W13-CARRY-2]
verify:
  - "git log --oneline master | grep -q 'W13-1.*EFHV'"
  - "git log --oneline master | grep -q 'W13-2.*OOS.*hypervolume'"
  - "git log --oneline master | grep -q 'W13-3.*mean OOS EFHV'"

  # ─── W13-1: EFHV threading + hypervolume fix ──────────────────────────
  - "test -f .dfg/agents/W13-1-hypervolume-and-efhv-threading.md"
  - "test -f .dfg/retrospectives/W13/W13-1.md"
  - "test -f python_refactor/tests/test_hypervolume_threading.py"
  - "grep -q '_flatten_metrics\\|Portfolio.mean_ROI' python_refactor/src/algorithms/sms_emoa.py"

  # ─── W13-2: OOS evaluator ─────────────────────────────────────────────
  - "test -f .dfg/agents/W13-2-oos-evaluator.md"
  - "test -f .dfg/retrospectives/W13/W13-2.md"
  - "test -f python_refactor/experiments/oos_evaluator.py"
  - "test -f python_refactor/tests/test_experiments_oos_evaluator.py"
  - "grep -q 'def compute_oos_efhv' python_refactor/experiments/oos_evaluator.py"
  - "grep -q 'def hypervolume_2d' python_refactor/experiments/oos_evaluator.py"

  # ─── W13-3: report orchestrator + real receipts ───────────────────────
  - "test -f .dfg/agents/W13-3-oos-efhv-report.md"
  - "test -f .dfg/retrospectives/W13/W13-3.md"
  - "test -f python_refactor/experiments/oos_report.py"
  - "test -f python_refactor/tests/test_experiments_oos_report.py"
  - "test -f docs/OOS-EFHV-REPORT.md"
  - "grep -q 'mean OOS EFHV' docs/OOS-EFHV-REPORT.md"
  - "grep -q 'W13-CARRY-1\\|W13-CARRY-2' .dfg/retrospectives/W13/W13-3.md"

  # ─── Methodology caveat surfaced ──────────────────────────────────────
  - "grep -q 'methodology' docs/VALIDATION-RESULTS.populated.md"
  - "grep -q 'walk-forward' docs/VALIDATION-RESULTS.populated.md"

  # ─── Thesis provenance + supporting docs ──────────────────────────────
  - "test -f docs/Azevedo_CarlosRenatoBelo_D.pdf"
  - "test -f docs/THESIS-INDEX.md"
  - "grep -q 'Eqs (7.10)+(7.11)\\|Eq 7.10\\|Eq 7.11\\|Eqs 7.10' docs/THESIS-INDEX.md"

  # ─── Test suite green ─────────────────────────────────────────────────
  - "cd python_refactor && uv run python -m pytest tests/test_experiments_oos_evaluator.py tests/test_experiments_oos_report.py tests/test_hypervolume_threading.py -q"

  # ─── Retro-hygiene gate green for W13 ─────────────────────────────────
  - "bash scripts/pre-pr-reflect-validate.sh --wave W13"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W13 delivered the full OOS evaluator + report pipeline + first real receipts at 30 seeds × 1000 MC scenarios."
  - "Empirical receipt: S0=5.12e-05, S2=4.22e-05, Δ=-17.59% (Welch t≈9, p<0.0001) — direction OPPOSITE to paper Table 7.2."
  - "CRITICAL METHODOLOGY SCAR (W13-CARRY-2): The W13-3 single-shot 80/20 split defeats the thesis's online walk-forward anticipation property. The thesis aggregates over (T-1) × 30 seeds rolling-window observations (~720); we produced 1 × 30 = 30. The S2 < S0 finding therefore says 'pipeline + 80/20 methodology doesn't reproduce paper' — NOT 'algorithm is broken'. Disambiguation requires W14 walk-forward."
  - "Secondary finding (W13-CARRY-1): SCENARIOS dict maps max_horizon (paper Eq 14 H) but NOT K∈{0,1,2,3} (OAL historical-window from thesis §7.2.3). Even with proper walk-forward, S2 may still degrade to K=0-equivalent if SCENARIOS isn't re-keyed."
  - "5 PRs shipped in W13: #68 (replan), #69 (W13-1), #70 (thesis provenance + THESIS-INDEX), #71 (W13-2), #72 (W13-3). Plus the wave-close PR."
  - "Class-retirement receipt (directive #5): no new canon entry; the methodology scar is filed as W13-CARRY-2 and the existing 'always run against real data + walk-forward when claiming paper replication' discipline is reinforced."

carry_forward:
  - id: W8-CARRY-3
    why: "Tables C/E/F/G/H analytics builders — blocked until W14 disambiguates the methodology and gives meaningful comparisons. The current direction-reversed receipts would only make these tables show the wrong story."
    next_action: "Future analytics-completion wave AFTER walk-forward establishes the true direction"
  - id: W9-3-CARRY-1
    why: "Optional W1-W6 retro key backfill — unchanged."
    next_action: "Optional mechanical cleanup wave"
  - id: W12-CARRY-1
    why: "portfolio_value identical across scenarios on seed=1 — same family as W13-CARRY-1 and partly explained by it (S0 and S2 produce same wealth because S2's anticipation isn't actually firing in the way the thesis intends). Re-evaluate post-W14."
    next_action: "Re-check after W14 walk-forward implementation"
  - id: W13-CARRY-1
    why: "SCENARIOS dict maps max_horizon (paper Eq 14 H) but NOT K∈{0,1,2,3} from thesis §7.2.3 OAL historical-window. Likely candidate for why anticipation doesn't activate in our S2."
    next_action: "W14 second probe: re-key SCENARIOS to expose K explicitly; verify mean OOS EFHV monotone in K (per Fig 7.15 thesis)"
  - id: W13-CARRY-2
    why: "CRITICAL methodology scar. W13-3 used single-shot 80/20 train/test split; thesis uses online walk-forward across T=25 rolling 50-day periods. Aggregation differs ~24× in power AND defeats the algorithm's anticipation property (no temporal stepping during training)."
    next_action: "W14 keystone (HIGH PRIORITY): implement walk-forward OOS protocol matching thesis §7.2.2. For each rolling period t in T: train SMS-EMOA on data up to t → extract EMFC m̂_{u*_t} → compute Ŝ_{t+1} via Eq 7.11 against MLE-fit of next-period observed returns → aggregate across (T-1) × 30 seeds × scenarios. Reuses W13-2 compute_oos_efhv per period; the new code is the rolling-period orchestrator."
---

# W13-gate — WAVE 13 CLOSE: OOS EFHV pipeline ships; methodology scar caught

## What W13 delivered

| Unit | PR | What |
|---|---|---|
| W13 replan (OOS pivot) | #68 | Restructured scope per operator caveat "out-of-sample EFHV" |
| W13-1 | #69 | Thread EFHV kwarg + fix hypervolume=0 (3 root causes: kwarg omitted, Portfolio stats not set, data_loader Inf leak) |
| Thesis provenance + THESIS-INDEX | #70 | Anchor codebase on Azevedo PhD 2014; cite Eqs 7.10+7.11; map chapters; list future research directions |
| W13-2 | #71 | OOS evaluator (~180 LOC, 16/16 tests) — pure-function implementation of thesis Eqs 7.10+7.11 |
| W13-3 | #72 | Report orchestrator + 30-seed × 1000-MC run (16:33 wall-clock); first real receipts; populated VALIDATION-RESULTS.md §1 |

## Real receipts

| Scenario | Mean OOS EFHV | Std (n=30) |
|---|---|---|
| S0 (myopic baseline) | **5.12e-05** | 4.06e-06 |
| S2 (anticipatory) | **4.22e-05** | 3.68e-06 |

Delta = **−17.59%** (Welch t ≈ 9, p < 0.0001) — direction OPPOSITE
paper Table 7.2 claim.

## Verdict

**PASS-WITH-CRITICAL-METHODOLOGY-SCAR.**

Infrastructure ships: 8 PRs landed, 30+ regression tests pass, real
multi-seed receipts produced. But the negative empirical finding
is contaminated by a methodology choice (single-shot 80/20 vs the
thesis's online walk-forward). **The result says "pipeline + this
methodology doesn't reproduce paper claim" — NOT "algorithm is
broken."** Disambiguation requires W14 walk-forward.

## Two W14 candidate keystones (in priority order)

1. **W13-CARRY-2: walk-forward OOS protocol** (HIGH PRIORITY).
   Implement thesis §7.2.2 rolling-period evaluation. Should be
   re-run on both S0 and S2 to see whether the direction reverses
   under proper online evaluation.

2. **W13-CARRY-1: K window-size hypothesis** (second probe).
   Re-key SCENARIOS to expose K∈{0,1,2,3} explicitly so the OAL
   self-adjustment loop runs per Eq 7.16. Verify mean OOS EFHV
   monotone in K (per Fig 7.15).

W14 may unify both into one wave: implement walk-forward + add
K-aware scenarios + re-run. If S2 > S0 under proper methodology,
the W7→W13 chain is structurally complete. If S2 ≤ S0 even there,
we have a deeper algorithmic bug to investigate in W15+.
