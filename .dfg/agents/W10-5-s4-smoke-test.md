---
id: W10-5
role: verifier
name: S4 (TIP + Multi-Horizon H=3 explicit covariance) end-to-end smoke-test
purpose: "Verifies S4 — the explicit-covariance variant. Surfaces covariance-threading src/ bugs."
wave: W10
unit: W10-5
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
    - .dfg/retrospectives/W10/W10-5.md
  branch_name: feat/w10-2-thru-w10-5-s1234-smoke-tests
  acceptance: >
    `python -m experiments.validation_matrix --scenario S4 --window paper
    --seed 1 --output /tmp/w10-s4` exits 0. manifest.json status=ok.
    metrics.csv has >= 5 metric rows. Retro documents green-path
    receipt OR honest scar.
dispatch_instructions: |
  Verification-only. S4 = S2 + monte_carlo_samples=1000 (vs 500).
  Check whether the metric output reflects explicit-covariance use.

  What NOT to do:
    - Don't fix bugs surfaced; file as carries.
---

# W10-5 — S4 smoke-test (explicit covariance)
