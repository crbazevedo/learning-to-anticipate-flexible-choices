---
id: W20-4
role: experimenter
name: Cross-check J — crowding distance for NDS
purpose: "Closes operator check J. Cross-validate crowding distance computation between v2 and Python NDS."
wave: W20
unit: W20-4
depends_on: []
blocks: [W20-6]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - legacy-cpp-v2/source/asms_emoa.cpp
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - legacy-cpp-v2/build/drivers/crowding_distance_driver.cpp
    - python_refactor/scripts/cross_validation/run_crowding_distance.py
    - python_refactor/tests/test_cross_check_crowding_distance.py
    - docs/CROSS-VALIDATION-J-CROWDING-DISTANCE.md
  branch_name: feat/w20-4-cross-check-j-crowding-distance
  acceptance: >
    Cross-validation receipt comparing v2's assigns_crowding_distance_obj
    output to Python's crowding distance computation on identical
    Pareto-front fixture. Verdict per W18 matrix.
dispatch_instructions: |
  Pattern per W18-1. v2's CD lives in asms_emoa.cpp:82+. Python has it
  in sms_emoa.py. Likely AGREE (deterministic algorithm); independent
  verification.
---

# W20-4 — Cross-check J (crowding distance)
