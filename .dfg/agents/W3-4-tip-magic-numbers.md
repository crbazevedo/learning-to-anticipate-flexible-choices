---
id: W3-4
role: bug-fixer
name: Replace TIP _calculate_tip_simple magic numbers with named constants
purpose: "Closes W1-4-CARRY-2. Replaces inline 0.5 / 0.3 magic numbers with documented module-level constants flagged as heuristic placeholders."
wave: W3
unit: W3-4
depends_on: []
blocks: ['W3-5']
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/temporal_incomparability_probability.py
output_contract:
  files:
    - python_refactor/src/algorithms/temporal_incomparability_probability.py
    - python_refactor/tests/test_temporal_incomparability_probability.py
  branch_name: feat/w3-4-tip-magic-numbers
  acceptance: >
    `_calculate_tip_simple` uses two named module-level constants
    `_TIP_FALLBACK_MAX_ROI_DIFF` + `_TIP_FALLBACK_MAX_RISK_DIFF` (or
    similar) instead of inline `0.5` and `0.3`. Constants are
    documented with a comment naming them as HEURISTIC placeholders
    (not paper-grounded). All existing 19 TIP tests + 7 W1-4 tests
    continue to pass in .venv-w1.
dispatch_instructions: |
  1. Add module-level constants near the top of
     temporal_incomparability_probability.py:
       # W3-4 / W1-4-CARRY-2: heuristic placeholders for the simple-
       # fallback TIP path (when KF covariance is unavailable). NOT
       # paper-grounded; chosen to span the typical FTSE-100 range
       # observed during development. Replace with empirically-fitted
       # values when a real calibration unit lands.
       _TIP_FALLBACK_MAX_ROI_DIFF: float = 0.5
       _TIP_FALLBACK_MAX_RISK_DIFF: float = 0.3
  2. In _calculate_tip_simple, replace the inline 0.5 and 0.3 with
     the new constants.
  3. No test additions required — the constants behave identically.
---

# W3-4 — TIP magic numbers → named constants

Closes W1-4-CARRY-2.
