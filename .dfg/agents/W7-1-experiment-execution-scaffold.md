---
id: W7-1
role: tooling-author
name: Build validation_matrix + aggregator + figures scaffold
purpose: "Scaffold to run the W6-3 experiment matrix. Three NEW modules in python_refactor/experiments/. Driver-level seed plumbing. Scaffold-only — no production-algorithm code modified."
wave: W7
unit: W7-1
depends_on: []
blocks: []
governance_tier: VT2
sized: M
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - docs/EXPERIMENT-VALIDATION-PLAN.md
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/run_experiments.py
output_contract:
  files:
    - python_refactor/experiments/validation_matrix.py
    - python_refactor/experiments/aggregate_validation.py
    - python_refactor/experiments/figures.py
  branch_name: feat/w7-1-experiment-execution-scaffold
  acceptance: >
    Three new files in python_refactor/experiments/, scaffold-only,
    no production code modified. validation_matrix.py exposes CLI
    --scenario / --window / --seed / --output and produces
    metrics.csv + manifest.json. aggregate_validation.py reads
    results/ tree → summary CSV. figures.py generates per-run PNGs.
    Smoke-test invocation `python -m experiments.validation_matrix
    --scenario S0 --window paper --seed 1 --dry-run` exits 0.
dispatch_instructions: |
  ## What this contract authorises

  - Creating 3 new files in python_refactor/experiments/.
  - Importing from python_refactor/src/ for ExperimentManager + config.

  ## What this contract does NOT authorise

  - Modifying any file in python_refactor/src/ (production-algorithm
    code stays untouched; only the driver layer changes).
  - Running the actual experiment matrix (smoke-test only).
  - Touching docs/ (W7-2 + W7-3 own those).

  ## Implementation

  1. validation_matrix.py:
     - argparse CLI: --scenario {S0,S1,S2,S3,S4}, --window {paper,extended}, --seed int, --output path, --dry-run flag
     - Map (scenario, window) → experiment_config dict (load via run_experiments.py patterns)
     - np.random.seed(seed) at entry (driver-level)
     - On --dry-run: validate config, print summary, exit 0 without running
     - On real run: invoke ExperimentManager.run_experiment(config), write metrics.csv + manifest.json to --output

  2. aggregate_validation.py:
     - argparse CLI: --input results/ root, --output summary.csv
     - Walk results/<scenario>/<seed>/metrics.csv
     - Produce summary table: (scenario, window, metric) → (mean, std, n, seeds)
     - Output as CSV + JSON (machine-readable)

  3. figures.py:
     - argparse CLI: --input metrics.csv, --output figures/ dir
     - Generate per-run PNGs: Pareto frontier evolution, wealth trajectory, λ trajectory, v trajectory
     - matplotlib Agg backend (headless)
     - Each figure has a self-explanatory title + axis labels matching paper notation
---

# W7-1 — Experiment execution scaffold

Closes W6-3's "execution scaffold" gap (per W6-3 §8). Scaffold-only —
no production code modified.
