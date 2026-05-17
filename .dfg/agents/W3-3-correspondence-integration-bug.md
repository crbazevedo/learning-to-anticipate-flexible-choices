---
id: W3-3
role: bug-fixer
name: Fix correspondence_with_anticipatory_learning test non-determinism
purpose: "Closes W2-2-CARRY-3 from test side. Algorithm bug (anticipatory_learning.py:328 doesn't clamp negative rates) filed as W3-3-CARRY."
wave: W3
unit: W3-3
depends_on: []
blocks: ['W3-5']
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/tests/test_correspondence_integration.py
    - python_refactor/src/algorithms/correspondence_mapping.py
output_contract:
  files:
    - python_refactor/src/algorithms/correspondence_mapping.py
    - python_refactor/tests/test_correspondence_integration.py
  branch_name: feat/w3-3-correspondence-integration-bug
  acceptance: >
    test_correspondence_with_anticipatory_learning PASSES deterministically
    in .venv-w1. Test mock uses bounded prediction_error within
    [min_error, max_error]. Real algorithm bug at anticipatory_learning.py
    (negative rates when prediction_error > max_error; no output clamp on
    _compute_traditional_learning_rate) documented as W3-3-CARRY for a
    follow-up unit that touches anticipatory_learning.py.
dispatch_instructions: |
  Root cause: MockSolution's prediction_error = np.random.random() * 0.1
  is in [0, 0.1] but test sets min_error=0.01, max_error=0.05. ~50% of
  runs have prediction_error > max_error → accuracy_factor goes negative
  → traditional_rate < 0 → combined_rate < 0 → assertion fails.

  Real algorithm bug: `_compute_traditional_learning_rate` should clamp
  its output to [rate_lwb=0, rate_upb=0.5] per the C++ comment intent
  but doesn't. NOT FIXED in W3-3 (anticipatory_learning.py NOT in scope).

  W3-3 fix: seed RNG + bound the mock's prediction_error explicitly.
  Or simpler: set prediction_error = 0.03 (mid-bounds) deterministically.
---

# W3-3 — Fix correspondence_with_anticipatory_learning non-determinism

Closes W2-2-CARRY-3 from the test side. Files an honest carry for the
real algorithm bug.
