---
id: W9-4
role: verifier
name: Re-run W8-1 smoke-test end-to-end (closes W7-1-CARRY-1)
purpose: "After W9-1 + W9-2 land, re-runs the W8-1 smoke-test on master. Closes W7-1-CARRY-1 + W8-1-CARRY-2."
wave: W9
unit: W9-4
depends_on: [W9-1, W9-2]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/src/experiments/data_loader.py
output_contract:
  files:
    - .dfg/retrospectives/W9/W9-4.md
  branch_name: feat/w9-4-smoke-test-end-to-end-rerun
  acceptance: >
    `python -m experiments.validation_matrix --scenario S0 --window paper
    --seed 1 --output /tmp/w9-smoke` exits 0 without --dry-run.
    /tmp/w9-smoke/manifest.json shows status=ok. /tmp/w9-smoke/metrics.csv
    has >= 1 metric row. Retro documents closure of W7-1-CARRY-1 +
    W8-1-CARRY-2 OR honestly reports new src/ cascade bugs as new
    carries (no laundering).
dispatch_instructions: |
  Verification-only unit. Re-runs the W8-1 smoke-test command. If
  it now passes end-to-end, retro reports closure of W7-1-CARRY-1
  and W8-1-CARRY-2. If new src/ bugs surface (real-data quirks that
  the W9-2 synthetic-fixture tests missed), they become new carries
  — sub-papercut #15 (synthesized fixtures must match real substrate)
  pattern.

  What NOT to do:
    - Don't fabricate a passing manifest.
    - Don't extend scope to fix arbitrary bugs unless they're a
      direct W9-1/W9-2 regression.
---

# W9-4 — Re-run smoke-test end-to-end

Verifies W9-1 + W9-2 fix the cascade. Closes W7-1-CARRY-1 if the
real-run smoke-test now passes.
