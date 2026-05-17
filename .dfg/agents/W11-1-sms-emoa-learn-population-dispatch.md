---
id: W11-1
role: code-fixer
name: Fix SMS-EMOA learning dispatch (prefer learn_population)
purpose: "Closes W10-CARRY-2 half. _apply_anticipatory_learning prefers learn_population when subclass overrides it; falls back to per-solution loop for base AnticipatoryLearning."
wave: W11
unit: W11-1
depends_on: []
blocks: [W11-3]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/tests/test_sms_emoa_dispatch.py
  branch_name: feat/w11-1-sms-emoa-learn-population-dispatch
  acceptance: >
    _apply_anticipatory_learning detects whether the learner overrides
    learn_population (not inherited from base) and routes to it; falls
    back to per-solution loop otherwise. ≥ 2 regression tests covering
    base + override paths.
dispatch_instructions: |
  Surgical fix at sms_emoa.py:200-208. Change the loop body to
  first check whether learn_population is overridden on the learner's
  type (vs the base AnticipatoryLearning class), and call it once
  for the unlearned subset if so. Per-solution loop becomes fallback.

  What NOT to do:
    - Don't touch portfolio_evaluator.py (W11-2).
    - Don't refactor _run_generation or other methods.
    - Don't import learner subclasses directly (use override detection
      via type.__mro__ or `learn_population.__qualname__` check).
---

# W11-1 — SMS-EMOA learn_population dispatch fix
