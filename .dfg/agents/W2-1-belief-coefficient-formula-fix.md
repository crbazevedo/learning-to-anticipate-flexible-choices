---
id: W2-1
role: bug-fixer
name: Fix inverted _calculate_confidence formula; un-skip test
purpose: "1-line fix at belief_coefficient.py:142 + un-skip test_calculate_confidence. Closes W1-4-CARRY-1."
wave: W2
unit: W2-1
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/belief_coefficient.py  # line 142 — inverted formula
    - python_refactor/tests/test_belief_coefficient.py  # the test currently skipped with W1-4-CARRY-1 marker
output_contract:
  files:
    - python_refactor/src/algorithms/belief_coefficient.py
    - python_refactor/tests/test_belief_coefficient.py
  branch_name: feat/w2-1-belief-formula-fix
  acceptance: >
    Line 142 of belief_coefficient.py reads `tip_confidence = abs(tip - 0.5) * 2`
    (removes the inverting `1.0 -` prefix). The docstring intent
    ("higher when TIP is closer to 0 or 1") now matches the formula.
    test_calculate_confidence is un-skipped (decorator removed,
    W1-4-CARRY-1 marker comment removed). All 12 existing belief
    tests + 2 W1-4 tests + the un-skipped test PASS (88 → 91 in
    `.venv-w1`).
dispatch_instructions: |
  Two edits, both small:
  1. belief_coefficient.py:142 — change
     `tip_confidence = 1.0 - abs(tip - 0.5) * 2`
     to
     `tip_confidence = abs(tip - 0.5) * 2`
  2. test_belief_coefficient.py — remove the @unittest.skip decorator
     + the W1-4-CARRY-1 comment block.

  Run pytest on test_belief_coefficient.py + a sample of dependent
  tests to confirm 91/91 PASS.
---

# W2-1 — Fix inverted _calculate_confidence formula

Closes W1-4-CARRY-1. The formula was always wrong; the test that
caught it was always right; no-one knew because the test never ran
until W1-4 fixed the test file's imports.
