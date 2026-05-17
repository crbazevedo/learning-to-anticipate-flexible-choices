---
id: W12-2
role: experimenter
name: S0+S2 × paper × 30 seeds → first populated VALIDATION-RESULTS.md
purpose: "Runs the first real multi-seed sweep (60 runs); aggregates; populates docs/VALIDATION-RESULTS.populated.md. First W7→W11 chain receipt with experimental output."
wave: W12
unit: W12-2
depends_on: [W12-1]
blocks: []
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/sweep.py
    - python_refactor/experiments/aggregate_validation.py
    - python_refactor/experiments/report.py
    - docs/VALIDATION-RESULTS.md
output_contract:
  files:
    - docs/VALIDATION-RESULTS.populated.md
    - .dfg/retrospectives/W12/W12-2.md
  branch_name: feat/w12-2-first-populated-results-doc
  acceptance: >
    results/W12-paper/ has 60 run dirs (≥ 55 status=ok).
    VALIDATION-SUMMARY.csv has rows for both S0 and S2 on paper window.
    docs/VALIDATION-RESULTS.populated.md exists with Table A populated.
    Retro documents whether S0 vs S2 wealth distributions differ at
    multi-seed (resolves W11-CARRY-1).
dispatch_instructions: |
  Pure execution + reporting. Steps:
    1. `python -m experiments.sweep --scenarios S0,S2 --windows paper
        --seeds 1-30 --output /tmp/W12-paper --jobs 4`
    2. `python -m experiments.aggregate_validation --input /tmp/W12-paper
        --output docs/VALIDATION-SUMMARY.csv`
    3. `python -m experiments.report --summary docs/VALIDATION-SUMMARY.csv
        --runs /tmp/W12-paper --template docs/VALIDATION-RESULTS.md
        --output docs/VALIDATION-RESULTS.populated.md`
    4. Inspect the populated doc; verify S0/S2 means differ at multi-seed
       (resolves W11-CARRY-1).

  What NOT to do:
    - Don't fix new src/ bugs surfaced; file as carries.
    - Don't ship VALIDATION-SUMMARY.csv to docs/ if results are
      meaningless; only ship the populated MD with whatever numbers
      come out (honestly).
---

# W12-2 — First populated VALIDATION-RESULTS.md
