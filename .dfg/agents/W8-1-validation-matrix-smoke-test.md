---
id: W8-1
role: bug-fixer
name: Smoke-test validation_matrix real-run path
purpose: "Closes W7-1-CARRY-1 partial. End-to-end real-run + integration fixes in validation_matrix.py."
wave: W8
unit: W8-1
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/run_experiments.py
output_contract:
  files:
    - python_refactor/experiments/validation_matrix.py
  branch_name: feat/w8-1-validation-matrix-smoke-test
  acceptance: >
    Scaffold's real-run path verified callable up to data-loading layer.
    Pre-existing src/ integration bugs surfaced are filed as carries
    for a future wave — NOT fixed in W8-1 (scope-discipline per
    directive #4).
dispatch_instructions: |
  Run `python -m experiments.validation_matrix --scenario S0
  --window paper --seed 1 --output /tmp/w8-smoke-S0/`. Fix any
  integration bugs INSIDE validation_matrix.py. Surface any
  src/-side bugs as honest carries — do NOT scope-creep into
  src/ fixes without replan.
---

# W8-1 — Smoke-test validation_matrix real-run

Closes W7-1-CARRY-1 partial.
