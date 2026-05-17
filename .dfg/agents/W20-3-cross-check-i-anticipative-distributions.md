---
id: W20-3
role: experimenter
name: Cross-check I — anticipative distributions resulting from OAL application
purpose: "Closes operator check I. Cross-validate the output of Online Anticipatory Learning (OAL) applied to a portfolio: anticipative ROI/risk + covariance after KF+TIP+λ+Dirichlet chain on identical fixture."
wave: W20
unit: W20-3
depends_on: []
blocks: [W20-6]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/CROSS-VALIDATION-V2-REASSESSMENT.md
    - legacy-cpp-v2/source/asms_emoa.cpp
    - python_refactor/src/algorithms/anticipatory_learning.py
output_contract:
  files:
    - legacy-cpp-v2/build/drivers/anticipative_distribution_driver.cpp
    - python_refactor/scripts/cross_validation/run_anticipative_distribution.py
    - python_refactor/tests/test_cross_check_anticipative_distribution.py
    - docs/CROSS-VALIDATION-I-ANTICIPATIVE-DISTRIBUTIONS.md
  branch_name: feat/w20-3-cross-check-i-anticipative-distributions
  acceptance: >
    Cross-validation receipt comparing v2's anticipatory_learning_obj_space
    output to Python's anticipatory_learning_obj_space output on identical
    (KF state, predicted state, alpha) fixture. Verdict per W18 matrix.
dispatch_instructions: |
  Pattern per W18-1. v2's anticipatory_learning_obj_space is at
  asms_emoa.cpp:639+. Python's equivalent is in anticipatory_learning.py.

  Honest scar candidates:
   - alpha formula (already W19-4 finding); will compound with anticipative-distribution divergence
   - transaction-cost integration timing (Python W16-2 vs v2)
   - covariance update path
---

# W20-3 — Cross-check I (anticipative distributions from OAL)
