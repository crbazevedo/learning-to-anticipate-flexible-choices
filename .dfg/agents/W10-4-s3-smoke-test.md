---
id: W10-4
role: verifier
name: S3 (TIP + Multi-Horizon H=2 control) end-to-end smoke-test
purpose: "Verifies S3 — horizon-ablation control. Should reuse most of S2's code paths."
wave: W10
unit: W10-4
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
    - .dfg/retrospectives/W10/W10-4.md
  branch_name: feat/w10-2-thru-w10-5-s1234-smoke-tests
  acceptance: >
    `python -m experiments.validation_matrix --scenario S3 --window paper
    --seed 1 --output /tmp/w10-s3` exits 0. manifest.json status=ok.
    metrics.csv has >= 5 metric rows. Retro documents green-path
    receipt OR honest scar.
dispatch_instructions: |
  Verification-only. S3 differs from S2 only in max_horizon (2 vs 3).
  Check whether the metric output reflects that difference.

  What NOT to do:
    - Don't fix bugs surfaced; file as carries.
---

# W10-4 — S3 smoke-test (horizon-ablation control)
