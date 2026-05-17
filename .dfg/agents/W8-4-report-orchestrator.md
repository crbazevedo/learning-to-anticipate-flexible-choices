---
id: W8-4
role: tooling-author
name: Report orchestrator (CSVs → populated VALIDATION-RESULTS.md)
purpose: "Composes stats + tables; reads summary CSV + per-run metric files; emits a populated copy of VALIDATION-RESULTS.md with 🚧 markers replaced where data exists, preserved where it doesn't."
wave: W8
unit: W8-4
depends_on: ['W8-2', 'W8-3']
blocks: []
governance_tier: VT2
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/ANALYTICS-PLAN.md
    - docs/VALIDATION-RESULTS.md
    - python_refactor/experiments/stats.py
    - python_refactor/experiments/tables.py
    - python_refactor/experiments/aggregate_validation.py
output_contract:
  files:
    - python_refactor/experiments/report.py
    - python_refactor/tests/test_experiments_report.py
  branch_name: feat/w8-4-report-orchestrator
  acceptance: >
    report.py exposes a CLI accepting --summary, --runs, --template,
    --output. Reads VALIDATION-SUMMARY.csv + per-run files; computes
    Tables A-H using experiments.stats + experiments.tables; emits a
    copy of the template with 🚧 markers replaced where data exists,
    preserved where it doesn't (graceful incompleteness). Embeds
    reproducibility receipt block per ANALYTICS-PLAN.md §8 (git SHA,
    uv lock hash, generated timestamp, n_runs, seeds, command). At
    least 4 tests covering: end-to-end render with synthetic summary,
    graceful 🚧-preservation when data is partial, receipt-block
    population, table substitution.
dispatch_instructions: |
  Pure-orchestration layer — no new analytics. Imports from
  experiments.stats + experiments.tables. Reads the W8-3 table
  output strings and substitutes them into the template by section
  header. Receipt block uses subprocess to capture git SHA + hashlib
  to fingerprint uv.lock.
---

# W8-4 — Report orchestrator

Composes stats + tables into a populated `VALIDATION-RESULTS.md`.

## Substitution strategy

For each Table A–H section in `VALIDATION-RESULTS.md`:
1. Locate the section header (e.g. `## 2. Descriptive statistics (Table A)`).
2. Filter summary rows by the section's expected (scenario, window, metric) tuples.
3. Call the matching `generate_table_*` from `experiments.tables`.
4. Replace the placeholder table-of-🚧 in the template with the generated string.

If no rows exist for a section, leave the placeholder intact.

## Receipt block

Populates section §0:
- `git_sha`: first 8 chars of `git rev-parse HEAD`
- `uv_lock`: first 16 chars of `sha256(uv.lock)`
- `generated`: ISO-8601 UTC timestamp
- `n_runs`: total run count from summary
- `seeds`: deduplicated seed list from summary
- `command`: literal CLI string for reproducibility
