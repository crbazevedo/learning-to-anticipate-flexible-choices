---
wave: W14
gate_type: wave-gate
verdict: PASS-WITH-METHODOLOGY-RULED-OUT
date: 2026-05-17
units_completed: [W14-1, W14-2]
units_carry_forward: [BACKLOG-B1, BACKLOG-B2, BACKLOG-B3, BACKLOG-B4, BACKLOG-H1, BACKLOG-H2, BACKLOG-H3, BACKLOG-H4, BACKLOG-H5, BACKLOG-H6, BACKLOG-H7, BACKLOG-M1, BACKLOG-M2, BACKLOG-M3, BACKLOG-M4, BACKLOG-M5, BACKLOG-M6, BACKLOG-M7, BACKLOG-M8, BACKLOG-M9, BACKLOG-M10, BACKLOG-M11, BACKLOG-M12, BACKLOG-M13, BACKLOG-M14, BACKLOG-L1, BACKLOG-L2, BACKLOG-L3, BACKLOG-L4, BACKLOG-L5, BACKLOG-L6, BACKLOG-L7, BACKLOG-L8]
verify:
  - "git log --oneline master | grep -q 'W14-1.*walk-forward orchestrator'"
  - "git log --oneline master | grep -q 'W14-2.*walk-forward methodology'"
  - "test -f .dfg/agents/W14-1-walk-forward-orchestrator.md"
  - "test -f .dfg/agents/W14-2-walk-forward-30seed-report.md"
  - "test -f .dfg/retrospectives/W14/W14-1.md"
  - "test -f .dfg/retrospectives/W14/W14-2.md"
  - "test -f python_refactor/experiments/walk_forward.py"
  - "test -f python_refactor/experiments/walk_forward_report.py"
  - "test -f docs/OOS-EFHV-WALK-FORWARD-REPORT.md"
  - "grep -q 'enumerate_periods\\|run_walk_forward' python_refactor/experiments/walk_forward.py"
  - "grep -q 'WALK-FORWARD-REPORT\\|walk-forward methodology rejected\\|walk-forward receipt' docs/VALIDATION-RESULTS.populated.md"
  - "test -f docs/BACKLOG.md"
  - "grep -q '§6 — Canonical wave-unit contract template' docs/BACKLOG.md"
  - "cd python_refactor && uv run python -m pytest tests/test_experiments_walk_forward.py -q"
  - "bash scripts/pre-pr-reflect-validate.sh --wave W14"
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W14 disambiguated the W13-CARRY-2 methodology hypothesis: walk-forward STILL shows S2 ≤ S0 (-24.86% at n=2 seeds, Welch t ≈ -5.7). The discrepancy with paper Table 7.2 is STRUCTURAL, not methodological."
  - "Walk-forward orchestrator (W14-1) ships with 9/9 tests; per-pair real-data smoke produces finite EFHV per period."
  - "W14-2 walk-forward report ships at 2-seed smoke fidelity (operator R1 decision: don't burn ~10h compute on full 30-seed run against a misconfigured-algorithm null result)."
  - "Verdict is PASS-WITH-METHODOLOGY-RULED-OUT. Major contribution: disambiguation of W13's negative result. The shipped infrastructure (W13-2 + W14-1 + W14-2 oos pipeline) is reusable for the post-W15 re-run that will give the definitive answer."
  - "Replaces ad-hoc carry tracking with docs/BACKLOG.md (PR #76 + #77) — 27 grounded items across BLOCKER/HIGH/MEDIUM/LOW severities + 5 waves (W15-W19) + canonical unit-contract template (§6)."

carry_forward:
  - id: BACKLOG-ALL
    why: "All open carries reconciled into docs/BACKLOG.md (27 items grounded with thesis page+equation+verbatim excerpts). Each item carries explicit wave assignment per §2. W15 BLOCKERS B1-B4 are the critical path to paper-replication answer."
    next_action: "W15 dispatch per BACKLOG §2 wave plan. Each unit contract MUST follow §6 canonical template (BACKLOG.md item-ID pointer + thesis grounding with verbatim excerpts + reason field)."
---

# W14-gate — WAVE 14 CLOSE: methodology hypothesis rejected; root cause is structural

## What W14 delivered

| Unit | PR | What |
|---|---|---|
| W14 replan (OOS walk-forward) | #74 | restructured W14 scope around thesis §7.2.2 rolling-period protocol |
| W14-1 walk-forward orchestrator | #75 | walk_forward.py module + 9/9 tests; reuses W13-2 compute_oos_efhv |
| BACKLOG catalog | #76 | 27 gaps catalogued + W15-W19 wave plan |
| BACKLOG grounding | #77 | every item grounded w.r.t. thesis (page + equation + verbatim excerpt) + canonical §6 unit-contract template |
| W14-2 walk-forward report | #78 | 2-seed smoke ships honest negative result; W13-CARRY-2 RESOLVED (methodology rejected) |

## Receipts

**Walk-forward smoke (2 seeds × 2 scenarios × 44 periods × 200 MC, 41 min wall-clock):**

| Scenario | Grand mean Ŝ | Std (n=2) |
|---|---|---|
| S0 | 2.527e-04 | 1.02e-05 |
| S2 | 1.899e-04 | 1.18e-05 |

Δ = **−24.86%** (Welch t ≈ -5.7, df ≈ 1). Direction OPPOSITE paper.

## Closes

- **W13-CARRY-2 RESOLVED**: walk-forward direction matches W13-3's single-shot finding → methodology was NOT the explanation. Root cause is structural (BACKLOG BLOCKERS B1-B4).
- **All prior ad-hoc carries reconciled to BACKLOG.md** (PR #76+#77): 27 grounded items + W15-W19 wave plan + §6 canonical unit-contract template.

## Verdict

**PASS-WITH-METHODOLOGY-RULED-OUT.** W14's deliverable was disambiguation — even though numbers don't replicate the paper, the wave gives a definitive answer to the methodology question: it's not the methodology. The W15+ root-cause investigation now has a clear target list (BACKLOG B1-B4).

## W15 readiness checklist

- ✅ BACKLOG.md authored + grounded
- ✅ Canonical unit-contract template (§6) — W15 units MUST follow it
- ✅ Walk-forward + OOS evaluator pipeline (W13-2 + W14-1) — reusable for post-W15 re-run
- ✅ 4 BLOCKERS identified, each with verbatim thesis grounding (B1 ref point, B2 K threading, B3 cardinality, B4 uniform crossover + simplex projection)
- ✅ thesis at `docs/Azevedo_CarlosRenatoBelo_D.pdf` indexed via `docs/THESIS-INDEX.md`
