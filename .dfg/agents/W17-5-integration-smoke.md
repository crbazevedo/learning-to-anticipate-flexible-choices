---
id: W17-5
role: verifier
name: Integration smoke with 87-asset universe + λ trace assertions + AMFC selector
purpose: "Verify W17-1..W17-4 combined effect: re-run walk-forward smoke on 87-asset universe with λ trace assertions enabled + AMFC selector; report Δ(S2 vs S0); commit empirical receipt."
wave: W17
unit: W17-5
depends_on: [W17-1, W17-2, W17-3, W17-4]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/OOS-EFHV-W16-INTEGRATION-SMOKE.md
    - python_refactor/experiments/walk_forward_report.py
output_contract:
  files:
    - docs/OOS-EFHV-W17-INTEGRATION-SMOKE.md
    - .dfg/retrospectives/W17/W17-5.md
  branch_name: feat/w17-5-integration-smoke
  acceptance: >
    Re-run W16-5 smoke with W17 fixes (87-asset universe, λ trace
    CSV emitted, λ^K firing in production per W17-2, AMFC u*_{t-1}
    selector). Empirical Δ direction reported honestly. Trace
    assertions: mean(λ^K) > 0 for K=3; lambda_k_branch == 'kperiod_sum'
    dominant. Retro qualitative inspection.
dispatch_instructions: |
  Run smoke (2 seeds × 2 scenarios × paper-window) with all W17
  fixes enabled:
    - 87-asset universe (set enforce_thesis_continuous_trades=True)
    - λ trace CSV emit (via --lambda-trace-csv-path)
    - λ^K consumption firing in production (W17-2)
    - AMFC u*_{t-1} selector (W17-4)

  Verdict matrix:
    - S2 > S0: PASS-DIRECTION-REPLICATED
    - S2 ≤ S0, gap < -5.53% (W16-5 baseline): PASS-FURTHER-GAP-CLOSURE
    - S2 ≤ S0, gap ≥ -5.53%: HOLD-FOR-DEBUG

  Trace assertions on lambda_tip_trace.csv:
    - λ^H ∈ [0, 1]
    - λ^K ∈ [0, 1]
    - mean(λ^K) > 0.01 for K=3 rows (PROOF of λ^K firing in
      production; previously 0 per W16-1-CARRY-1)
    - lambda_k_branch column shows 'kperiod_sum' > warm-up after
      enough periods
    - λ ≈ 0.5*(λ^H + λ^K) within 1e-9 (Eq 7.16 verbatim)

  What NOT to do:
    - Don't run full 30-seed grid (still smoke).
    - Don't refactor walk_forward_report (W17-3 owns).
    - Don't ship analytics (W18).
---

# W17-5 — Integration smoke with W17 fixes
