---
id: W12-1
role: tooling-author
name: Multi-seed sweep driver
purpose: "Adds sweep CLI to dispatch scenarios × windows × seeds in parallel via subprocess + concurrent.futures; writes a results tree compatible with aggregate_validation.py."
wave: W12
unit: W12-1
depends_on: []
blocks: [W12-2]
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
    - python_refactor/experiments/sweep.py
    - python_refactor/tests/test_experiments_sweep.py
  branch_name: feat/w12-1-multi-seed-sweep-driver
  acceptance: >
    sweep.py CLI: --scenarios S0,S2 --windows paper --seeds 1-30
    --output results/W12 --jobs N. Each tuple writes
    results/W12/<scenario>_<window>_seed<N>/{metrics.csv, manifest.json}.
    Uses subprocess + concurrent.futures.ProcessPoolExecutor. ≥ 3
    regression tests including 1 real-dispatch end-to-end (S0 × paper
    × 2 seeds smoke).
dispatch_instructions: |
  Pure-orchestration script. Spawns validation_matrix.py subprocess
  per (scenario, window, seed) tuple; collects exit codes; reports
  summary at end (N total / N ok / N error).

  Seed parsing: accept "1-30" (range) or "1,2,5,10" (list) or
  "30" (single).

  What NOT to do:
    - Don't reimplement run_one (that's validation_matrix's job).
    - Don't add custom output schema; just dispatch to validation_matrix.
    - Don't add resume/checkpoint logic (keep simple for W12).
---

# W12-1 — Multi-seed sweep driver
