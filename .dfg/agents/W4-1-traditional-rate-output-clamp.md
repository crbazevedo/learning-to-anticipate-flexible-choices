---
id: W4-1
role: bug-fixer
name: Clamp _compute_traditional_learning_rate output to [rate_lwb, rate_upb]
purpose: "Closes W3-3-CARRY-1. 1-line clamp on the returned rate + revert W3-3's test-mock pinning."
wave: W4
unit: W4-1
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/tests/test_correspondence_integration.py
output_contract:
  files:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/tests/test_correspondence_integration.py
  branch_name: feat/w4-1-traditional-rate-output-clamp
  acceptance: >
    _compute_traditional_learning_rate clamps the returned rate to
    [rate_lwb=0, rate_upb=0.5] per the C++ comment intent. W3-3's
    pinned test mock prediction_error reverted to random; test
    PASSES under any random input because the clamp guarantees
    valid rates.

## What this contract does NOT authorise

- Touching any file outside output_contract.
- Adding new tests (the W3-3 test was the regression case; reverting
  its pinning is the verification).

dispatch_instructions: |
  1. anticipatory_learning.py: in _compute_traditional_learning_rate,
     change `return anticipation_rate` to
     `return max(rate_lwb, min(rate_upb, anticipation_rate))`.
  2. test_correspondence_integration.py: revert the W3-3 mock
     prediction_error pin (back to `np.random.random() * 0.1`).
  3. Run correspondence_integration tests in venv to confirm.
---

# W4-1 — clamp traditional rate

Closes W3-3-CARRY-1.
