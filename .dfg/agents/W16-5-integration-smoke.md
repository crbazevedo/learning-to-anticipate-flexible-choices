---
id: W16-5
role: verifier
name: Integration smoke with W16 fixes (λ^K + txn costs + extrema)
purpose: "Verify W16-1+W16-2+W16-3+W16-4 combined effect: re-run 2-seed walk-forward smoke on {SMS_RDM_K0, ASMS_mHDM_K3}; report Δ(S2 vs S0); commit empirical receipt + λ^K + λ^H trace assertions."
wave: W16
unit: W16-5
depends_on: [W16-1, W16-2, W16-3, W16-4]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (reasons) in contract body below per BACKLOG §6
    # (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/OOS-EFHV-W15-INTEGRATION-SMOKE.md
    - python_refactor/experiments/walk_forward_report.py
output_contract:
  files:
    - docs/OOS-EFHV-W16-INTEGRATION-SMOKE.md
    - .dfg/retrospectives/W16/W16-5.md
  branch_name: feat/w16-5-integration-smoke
  acceptance: >
    Re-run W15-5 walk-forward smoke with SMS_RDM_K0 + ASMS_mHDM_K3
    after W16-1..W16-4 land. Empirical Δ(S2 − S0) reported honestly.
    Assertions over lambda_tip_trace.csv: both λ^H AND λ^K nonzero for
    K>0 scenarios; λ^K=0 (or trivially small) for K=0 scenario.
    Retro qualitative inspection.
dispatch_instructions: |
  Run smoke (2 seeds × 2 scenarios × paper-window date range), same
  shape as W15-5.

  Expected wall-clock: similar to W15-5 (~15 min). Use foreground
  invocation (NOT walk_forward_report.py with ProcessPoolExecutor —
  it silently masks per-pair errors per W15-5-CARRY-1).

  Report direction honestly:
    - If S2 > S0: W7→W16 chain structurally complete; paper claim
      direction replicates. WAVE-CLOSE VERDICT = PASS-DIRECTION-REPLICATED.
    - If S2 ≤ S0 but gap < W15-5's -8.75%: incremental improvement;
      remaining gap is in M-items (instrumentation) or H4 (asset
      universe). WAVE-CLOSE VERDICT = PASS-FURTHER-GAP-CLOSURE.
    - If S2 ≤ S0 and gap ≥ W15-5: regression. Investigate λ^K + txn
      cost interaction; possible that lambda_tip_trace.csv shows λ^K
      always 0 or always 1 (saturation). WAVE-CLOSE VERDICT = HOLD-FOR-DEBUG.

  Trace assertions (load lambda_tip_trace.csv from one of the
  ASMS_mHDM_K3 runs; assert):
    - λ^H ∈ [0,1] for all rows
    - λ^K ∈ [0,1] for all rows
    - For K=3 runs: mean(λ^K) > 0.01 (truly fires, not just 0)
    - For K=0 runs: mean(λ^K) < 1e-6 (effectively zero by construction)
    - λ ≈ 0.5*(λ^H + λ^K) within 1e-9 (formula correctness)

  What NOT to do:
    - Don't ship the analytics (W18).
    - Don't refactor walk_forward_report.py (file the per-pair error
      visibility as W16-CARRY if surfaced).
    - Don't run the full 30-seed grid (still a smoke; full grid is W18).
---

# W16-5 — Integration smoke with W16 fixes

Closes BACKLOG.md items: integration receipt + trace verification of
W16-1..W16-4.

## Background

W15-5 smoke result (post-BLOCKERS, pre-W16):
  SMS_RDM_K0    = 4.041e-04 ± 2.66e-05
  ASMS_mHDM_K3  = 3.687e-04 ± 1.17e-05
  Δ(S2 − S0)    = -8.75%

W15 closed 65% of the gap from W14-2's -24.86%. The remaining 9% is
hypothesized (per W15-5 retro analysis) to come from:
  1. λ^K not consumed (W16-1)
  2. Transaction costs not in HV (W16-2)
  3. Extrema oscillation (W16-3)

W16-5 measures the COMBINED effect.

## Trace assertions

W16-4 emits `lambda_tip_trace.csv`. The verifier loads it and runs
the following assertions:

- λ^H ∈ [0,1] for all rows
- λ^K ∈ [0,1] for all rows
- mean(λ^K) > 0.01 for K=3 scenarios (proof of life)
- mean(λ^K) < 1e-6 for K=0 scenarios (correct by construction)
- λ ≈ 0.5*(λ^H + λ^K) within 1e-9 (Eq 7.16 verbatim)

If any assertion fails: W16-1's λ^K consumption is mis-wired even if
the smoke shows direction reversal.

## Wave-close verdict matrix

| Outcome | Verdict |
|---|---|
| S2 > S0 | PASS-DIRECTION-REPLICATED |
| S2 ≤ S0, gap < -8.75% (W15-5 baseline) | PASS-FURTHER-GAP-CLOSURE |
| S2 ≤ S0, gap ≥ -8.75% | HOLD-FOR-DEBUG |

## Files

- `docs/OOS-EFHV-W16-INTEGRATION-SMOKE.md` — smoke report with
  receipts + trace-assertion outcomes
- `.dfg/retrospectives/W16/W16-5.md` — ADR-004 four-question retro
