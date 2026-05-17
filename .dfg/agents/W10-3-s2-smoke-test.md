---
id: W10-3
role: verifier
name: S2 (TIP + Multi-Horizon H=3) end-to-end smoke-test
purpose: "Verifies S2 — the paper's headline configuration. Surfaces Multi-Horizon-wiring src/ bugs."
wave: W10
unit: W10-3
depends_on: [W10-1]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/validation_matrix.py
output_contract:
  files:
    - .dfg/retrospectives/W10/W10-3.md
  branch_name: feat/w10-2-thru-w10-5-s1234-smoke-tests
  acceptance: >
    `python -m experiments.validation_matrix --scenario S2 --window paper
    --seed 1 --output /tmp/w10-s2` exits 0. manifest.json status=ok.
    metrics.csv has >= 5 metric rows. Retro documents green-path
    receipt OR honest scar.
dispatch_instructions: |
  Verification-only. Capture whether S2's Multi-Horizon H=3 path
  produces metrics distinguishable from S0/S1.

  What NOT to do:
    - Don't fix bugs surfaced; file as carries.
---

# W10-3 — S2 smoke-test (paper headline config)
