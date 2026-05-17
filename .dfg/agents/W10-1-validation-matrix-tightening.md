---
id: W10-1
role: tooling-author
name: Metric flatten + paper-window real-glob switch
purpose: "Closes W9-CARRY-3 (metrics.csv has 0 rows because final_metrics is nested dict and writer filters scalars) + W8-1-CARRY-1 fully (switches WINDOWS paper to legacy-cpp/.../table*.csv real 98-CSV glob)."
wave: W10
unit: W10-1
depends_on: []
blocks: [W10-2, W10-3, W10-4, W10-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/experiments/aggregate_validation.py
output_contract:
  files:
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/tests/test_experiments_validation_matrix.py
  branch_name: feat/w10-1-validation-matrix-tightening
  acceptance: >
    validation_matrix.py: final_metrics recursed into '<category>.<metric>'
    rows in metrics.csv. WINDOWS['paper']['asset_files_glob'] points to
    legacy-cpp/executable/data/ftse-original/table*.csv. ≥ 2 regression
    tests (flatten round-trip + WINDOWS paper-window loads ≥ 90 assets).
    S0/paper/seed=1 smoke-test → metrics.csv has ≥ 5 metric rows.
dispatch_instructions: |
  Two surgical edits in one file:

  1) Replace the metric-writer loop at line ~266-269 with a recursive
     flatten:
       def _flatten(d: dict, prefix: str = "") -> Iterator[tuple[str, float]]:
           for k, v in d.items():
               key = f"{prefix}.{k}" if prefix else k
               if isinstance(v, dict):
                   yield from _flatten(v, key)
               elif isinstance(v, (int, float)) and not isinstance(v, bool):
                   yield key, float(v)
     Then writer.writerow([scenario, window, seed, key, value])
     for each (key, value) in _flatten(final_metrics).

  2) WINDOWS['paper']['asset_files_glob'] →
       'legacy-cpp/executable/data/ftse-original/table*.csv'
     Update the note + date_start / date_end if the paper-window dates
     differ from the FTSE-updated dates (paper §V-A says 2006-2012;
     verify via the actual CSV first row, fall back to "loose" dates
     if data spans differently). Existing 'extended' window stays
     pointing at FTSE-updated.

  What NOT to do:
    - Don't touch experiment_manager.py, data_loader.py (W9-1/W9-2 territory).
    - Don't refactor build_experiment_config — just feed WINDOWS through.
    - Don't add caching, async, progress bars, or extend SCENARIOS.
---

# W10-1 — validation_matrix tightening

Keystone for W10: closes W9-CARRY-3 + W8-1-CARRY-1, unblocks
W10-2..W10-5 smoke-tests.
