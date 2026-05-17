---
id: W8-3
role: tooling-author
name: Table generators A-H from summary CSV
purpose: "Reads VALIDATION-SUMMARY.csv + per-run results, emits Markdown table strings for A-H per ANALYTICS-PLAN.md §7."
wave: W8
unit: W8-3
depends_on: []
blocks: ['W8-4']
governance_tier: VT2
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/ANALYTICS-PLAN.md
    - docs/VALIDATION-RESULTS.md
output_contract:
  files:
    - python_refactor/experiments/tables.py
    - python_refactor/tests/test_experiments_tables.py
  branch_name: feat/w8-3-table-generators
  acceptance: >
    tables.py exposes generate_table_a..generate_table_h. Each
    returns a Markdown-formatted string. Numerical formatting per
    ANALYTICS-PLAN §7. test_experiments_tables.py has ≥ 4 tests.
dispatch_instructions: |
  Pure-function module. Reads list-of-dicts (one row per
  scenario × window × metric); writes Markdown table strings.
  Number formatting: 4 sig figs default; p < 0.001 → '<0.001';
  effect sizes to 3 decimals.
---

# W8-3 — Table generators A-H

Per ANALYTICS-PLAN.md §7.
