---
id: W14-2
role: experimenter
name: 30-seed walk-forward run + populate VALIDATION-RESULTS
purpose: "Run S0+S2 × 30 seeds × ~24 periods via W14-1; aggregate per-scenario grand mean; update VALIDATION-RESULTS.md §1 + write walk-forward report."
wave: W14
unit: W14-2
depends_on: [W14-1]
blocks: []
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/walk_forward.py
    - docs/VALIDATION-RESULTS.populated.md
    - docs/THESIS-INDEX.md
output_contract:
  files:
    - python_refactor/experiments/walk_forward_report.py
    - docs/OOS-EFHV-WALK-FORWARD-REPORT.md
    - docs/VALIDATION-RESULTS.populated.md
    - .dfg/retrospectives/W14/W14-2.md
  branch_name: feat/w14-2-walk-forward-30seed-report
  acceptance: >
    Full S0+S2 × 30 seeds walk-forward run completes; aggregated
    per-scenario grand mean reported. VALIDATION-RESULTS.md §1
    updated with walk-forward receipt replacing the W13-3
    single-shot headline. Retro documents direction (S2 vs S0)
    + next-step recommendation.
dispatch_instructions: |
  CLI:
    python -m experiments.walk_forward_report \
      --scenarios S0,S2 --seeds 1-30 \
      --train-window-days 378 --step-days 50 --n-mc 1000 \
      --output docs/OOS-EFHV-WALK-FORWARD-REPORT.md \
      --per-seed-json /tmp/w14-2-per-seed.json \
      --jobs 4

  Parallelize (scenario, seed) pairs via ProcessPoolExecutor
  (same pattern as W12-1 sweep). Each pair calls run_walk_forward
  → aggregate_per_seed → list of grand_mean values.

  Aggregate across seeds: grand_mean ± std per scenario, plus
  Welch t-test S2 vs S0.

  Folds into VALIDATION-RESULTS.populated.md §1 to REPLACE the
  W13-3 single-shot headline (which becomes legacy reference).

  What NOT to do:
    - Don't re-implement walk_forward (use W14-1 module).
    - Don't change SCENARIOS (K-aware re-keying is W15).
    - Don't ship if the run errors out for > 10% of (scenario, seed)
      pairs — file as scar instead.
---

# W14-2 — 30-seed walk-forward report
