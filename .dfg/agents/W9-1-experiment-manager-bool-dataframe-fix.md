---
id: W9-1
role: code-fixer
name: Fix bool(DataFrame) ambiguity at experiment_manager.py:190
purpose: "Closes W8-1-CARRY-2 partial. `if asset_data:` on a DataFrame raises ValueError; replace with explicit emptiness check."
wave: W9
unit: W9-1
depends_on: []
blocks: [W9-4]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/experiments/experiment_manager.py
output_contract:
  files:
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/tests/test_experiments_experiment_manager.py
  branch_name: feat/w9-1-experiment-manager-bool-dataframe-fix
  acceptance: >
    Line 190 no longer triggers ValueError. num_assets log field
    reports the column count for an asset DataFrame; 0 when None or empty.
    >= 1 regression test that exercises the path without raising.
dispatch_instructions: |
  Surgical fix: line 190 changes from
    'num_assets': len(asset_data) if asset_data else 0
  to
    'num_assets': 0 if asset_data is None or asset_data.empty else len(asset_data.columns)

  What NOT to do:
    - Don't refactor the surrounding _load_experiment_data method.
    - Don't add type hints unrelated to the fix.
    - Don't touch data_loader.py (that's W9-2).
---

# W9-1 — bool(DataFrame) fix

One-line fix at experiment_manager.py:190 + regression test.
