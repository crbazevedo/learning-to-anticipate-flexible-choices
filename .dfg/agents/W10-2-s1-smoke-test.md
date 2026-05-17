---
id: W10-2
role: verifier
name: S1 (TIP integrated, H=2) end-to-end smoke-test
purpose: "Verifies S1 scenario runs end-to-end with metrics populated. Surfaces any TIP-wiring src/ bugs."
wave: W10
unit: W10-2
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
    - .dfg/retrospectives/W10/W10-2.md
  branch_name: feat/w10-2-thru-w10-5-s1234-smoke-tests
  acceptance: >
    `python -m experiments.validation_matrix --scenario S1 --window paper
    --seed 1 --output /tmp/w10-s1` exits 0. manifest.json status=ok.
    metrics.csv has >= 5 metric rows. Retro documents green-path
    receipt OR honest scar.
dispatch_instructions: |
  Verification-only. Run the command; capture output; write retro
  noting any anomalies (especially whether S1's TIP-specific code
  paths produced any distinguishing metrics vs S0 baseline).

  What NOT to do:
    - Don't fix bugs surfaced; file as carries.
    - Don't fabricate receipts.
---

# W10-2 — S1 smoke-test
