---
id: W13-3
role: experimenter
name: Re-run sweep + OOS-evaluate + report mean OOS EFHV
purpose: "Runs S0+S2 × paper × 30 seeds; for each seed extracts trained Pareto weights, computes OOS EFHV against extended-window via W13-2; aggregates; populates VALIDATION-RESULTS.md §1 with mean OOS EFHV S0 vs S2."
wave: W13
unit: W13-3
depends_on: [W13-2]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/experiments/oos_evaluator.py
    - python_refactor/experiments/sweep.py
    - python_refactor/experiments/validation_matrix.py
    - docs/THESIS-INDEX.md
    - docs/VALIDATION-RESULTS.populated.md
output_contract:
  files:
    - python_refactor/experiments/oos_report.py
    - python_refactor/tests/test_experiments_oos_report.py
    - docs/OOS-EFHV-REPORT.md
    - docs/VALIDATION-RESULTS.populated.md
    - .dfg/retrospectives/W13/W13-3.md
  branch_name: feat/w13-3-oos-efhv-report
  acceptance: >
    oos_report.py CLI: --in-sample-glob ... --oos-glob ... --scenarios S0,S2
    --seeds 1-30 --output ... .  Per (scenario, seed): trained Pareto
    weights extracted from a re-run; OOS EFHV computed via W13-2;
    aggregated mean ± std per scenario reported in
    VALIDATION-RESULTS.populated.md §1 headline. Retro documents
    empirically whether S2 mean OOS EFHV > S0 (paper claim direction).
dispatch_instructions: |
  The Pareto weights aren't currently persisted by sweep.py — only
  metrics.csv + manifest.json. Two options:
    (a) re-run a small sweep in-process inside oos_report.py to get
        the Pareto weights live
    (b) extend sweep.py / validation_matrix.py to persist Pareto
        weights to disk per run (paretofront.csv)

  Pick (a) for W13-3 (smaller scope; in-process avoids cross-process
  data handoff). Iterate over (S0,S2) × 30 seeds; for each:
    - build_experiment_config + ExperimentManager.run_experiment
    - extract pareto = algorithm.get_pareto_front()
    - extract weights = [s.P.investment for s in pareto]
    - load OOS returns (extended-window FTSE-updated CSV via data_loader)
    - efhv = compute_oos_efhv(weights, oos_returns, n_samples=1000)
    - record efhv['efhv_mean']

  Aggregate: per-scenario list of seed-level efhv_means → grand mean ± std.

  What NOT to do:
    - Don't burn 60 full sweep re-runs if sweep.py can be extended in
      a future unit to persist weights. For W13-3 we run in-process.
    - Don't ship updates to VALIDATION-RESULTS.md template (only
      VALIDATION-RESULTS.populated.md).
---

# W13-3 — Report mean OOS EFHV S0 vs S2
