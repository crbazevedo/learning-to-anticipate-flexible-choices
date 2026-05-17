---
id: W11-3
role: verifier
name: Re-smoke-test S0-S4 + scenario-differentiation regression gate
purpose: "Verifies W11-1+W11-2 unblock real experiments. Ships the class-retiring regression gate per directive #5 (no two scenarios with different learning configs produce identical algorithm/diversity metrics)."
wave: W11
unit: W11-3
depends_on: [W11-1, W11-2]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/validation_matrix.py
output_contract:
  files:
    - python_refactor/tests/test_scenario_differentiation.py
    - .dfg/retrospectives/W11/W11-3.md
  branch_name: feat/w11-3-scenario-differentiation-verify-and-gate
  acceptance: >
    Re-running S0-S4 smoke-tests: at least one algorithm/diversity
    metric differs across each pair. No metric value is NaN or Inf.
    test_scenario_differentiation.py asserts pairwise inequality
    on diversity_metric across the 4 learning-enabled scenarios.
    Retro documents the regression gate as class-retirement per #5.
dispatch_instructions: |
  Pure verifier + regression-gate ship. Runs in-process smoke-tests
  against all 5 scenarios; pins their differentiation properties as
  a regression test that future src/ changes can't silently break.

  What NOT to do:
    - Don't fabricate receipts.
    - Don't fix new bugs found; file as carries.
---

# W11-3 — Scenario differentiation verify + class-retiring regression gate
