---
id: W5-1
role: bug-fixer
name: Seed RNG in test_correspondence_integration setUp
purpose: "Closes W3-5-CARRY-1. Add `np.random.seed(42)` in setUp so the cosine-similarity correspondence test is deterministic."
wave: W5
unit: W5-1
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/tests/test_correspondence_integration.py
output_contract:
  files:
    - python_refactor/tests/test_correspondence_integration.py
  branch_name: feat/w5-1-correspondence-test-rng-seed
  acceptance: >
    setUp seeds np.random with a fixed value. test_correspondence_mapping_with_multiple_time_steps PASSES deterministically 5/5 runs in .venv-w1.
dispatch_instructions: |
  Add `np.random.seed(42)` at the start of setUp. Verify by running
  the previously-flaky test 5 times in a row — must pass all 5.
---

# W5-1 — RNG seed in correspondence test setUp

Closes W3-5-CARRY-1.
