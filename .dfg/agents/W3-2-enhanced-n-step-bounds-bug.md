---
id: W3-2
role: bug-fixer
name: Fix enhanced_n_step integration setUp missing step_2/3
purpose: "Closes W2-2-CARRY-2. Root cause: TestEnhancedNStepPredictionIntegration.setUp only declares step_1 in kalman_predictions+dirichlet_predictions; bounds test loops H=1..3."
wave: W3
unit: W3-2
depends_on: []
blocks: ['W3-5']
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/tests/test_enhanced_n_step_prediction.py
output_contract:
  files:
    - python_refactor/src/algorithms/enhanced_n_step_prediction.py
    - python_refactor/tests/test_enhanced_n_step_prediction.py
  branch_name: feat/w3-2-enhanced-n-step-bounds-bug
  acceptance: >
    test_enhanced_prediction_bounds_validation PASSES in .venv-w1.
    Test setUp extended with step_2 + step_3 in both kalman_predictions
    and dirichlet_predictions (mirroring the unit-test class's setUp).
    Source file may NOT need changes; if it does, document in commit.
dispatch_instructions: |
  Add step_2 + step_3 entries to TestEnhancedNStepPredictionIntegration.setUp's
  kalman_predictions + dirichlet_predictions dicts (copy-shape from
  TestEnhancedNStepPredictor.setUp lines 33-67). Verify all tests in
  test_enhanced_n_step_prediction.py pass.
---

# W3-2 — enhanced_n_step integration test setUp completeness

Closes W2-2-CARRY-2.
