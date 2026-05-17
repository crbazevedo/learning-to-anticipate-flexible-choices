---
id: W9-2
role: code-fixer
name: Fix data_loader pivot KeyError + add multi-CSV glob support
purpose: "Closes W8-1-CARRY-1 + W8-1-CARRY-2 (partial). Bugs 2+3 converge on the same pivot site in src/data_loader.py: glob-expand asset_files, tag rows with asset_id, pivot on asset_id."
wave: W9
unit: W9-2
depends_on: []
blocks: [W9-4]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/experiments/data_loader.py
output_contract:
  files:
    - python_refactor/src/experiments/data_loader.py
    - python_refactor/tests/test_experiments_data_loader.py
  branch_name: feat/w9-2-data-loader-pivot-and-glob
  acceptance: >
    load_asset_data: each asset_files entry is glob-expanded (via
    pathlib.Path or glob.glob); each loaded row tagged with asset_id;
    pivot uses columns='asset_id'. No KeyError(None). Paper-window
    multi-CSV path loads ≥ 90 assets. FTSE single-CSV path still works
    (backward compat). ≥ 2 regression tests.
dispatch_instructions: |
  Two bugs converge:
   - Bug 2: pivot(columns=None) → KeyError(None). Need columns=<col>.
   - Bug 3: asset_files entries are literal paths, not globs. Paper-
     window data is 98 per-asset CSVs.

  Fix shape:
   1. For each entry in asset_files, if it has glob chars (* ? [),
      Path(entry).parent.glob(Path(entry).name) → list of paths.
      Otherwise treat as literal.
   2. For each loaded CSV, set asset_id = Path(file).stem.
   3. Keep ['Date', 'asset_id', 'Return'] in the appended frame.
   4. After concat: pivot(index='Date', columns='asset_id', values='Return').
   5. If `assets` filter is non-empty, restrict columns to that list.

  What NOT to do:
    - Don't refactor load_market_data (different code path).
    - Don't add caching, async, or progress bars.
    - Don't change the function signature.
---

# W9-2 — data_loader pivot + glob fix

Bugs 2 and 3 converge. ~15 LOC change in load_asset_data + tests.
