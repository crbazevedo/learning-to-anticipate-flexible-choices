---
id: W3-1
role: bug-fixer
name: Fix multi-horizon lambda-rates returns [0.0] for H<2 instead of []
purpose: "Closes W2-2-CARRY-1. Root cause: calculate_multi_horizon_lambda_rates returns [0.0] for H<2 (test expects []). 1-line fix."
wave: W3
unit: W3-1
depends_on: []
blocks: ['W3-5']
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/tests/test_multi_horizon_anticipatory.py
output_contract:
  files:
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/tests/test_multi_horizon_anticipatory.py
  branch_name: feat/w3-1-multi-horizon-lambda-rates-bug
  acceptance: >
    calculate_multi_horizon_lambda_rates returns [] (empty list) for
    H<2, NOT [0.0]. test_calculate_multi_horizon_lambda_rates PASSES
    in .venv-w1. No regression in other multi_horizon tests.
dispatch_instructions: |
  Single-line fix in multi_horizon_anticipatory.py: change
  `if prediction_horizon < 2: return [0.0]` to `return []`.
  Semantic: H<2 means no future horizons, so the list of per-horizon
  lambda rates should be EMPTY (consistent with
  apply_anticipatory_learning_rule's len(predicted) == len(lambdas)
  invariant — both 0 when H<2).
---

# W3-1 — Fix multi-horizon lambda-rates [0.0] vs [] bug

Closes W2-2-CARRY-1.
