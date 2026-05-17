---
id: W3-5
role: tooling-author
name: Wider ruff F401 sweep across src/
purpose: "Closes W2-4-CARRY-1. Runs ruff check --select F401 --fix across all src/ files. No semantic changes."
wave: W3
unit: W3-5
depends_on: ['W3-1', 'W3-2', 'W3-3', 'W3-4']
blocks: ['W3-6']
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/algorithms/correspondence_mapping.py
    - python_refactor/src/algorithms/temporal_incomparability_probability.py
output_contract:
  files:
    - python_refactor/src/algorithms/__init__.py
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/algorithms/belief_coefficient.py
    - python_refactor/src/algorithms/correspondence_mapping.py
    - python_refactor/src/algorithms/enhanced_n_step_prediction.py
    - python_refactor/src/algorithms/kalman_filter.py
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
    - python_refactor/src/algorithms/n_step_prediction.py
    - python_refactor/src/algorithms/nsga2.py
    - python_refactor/src/algorithms/operators.py
    - python_refactor/src/algorithms/sliding_window_dirichlet.py
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/src/algorithms/solution.py
    - python_refactor/src/algorithms/statistics.py
    - python_refactor/src/algorithms/temporal_incomparability_probability.py
    - python_refactor/src/config/experiment_config.py
    - python_refactor/src/config/thesis_parameters.py
    - python_refactor/src/experiments/data_loader.py
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/src/experiments/logger.py
    - python_refactor/src/experiments/metrics_collector.py
    - python_refactor/src/experiments/portfolio_evaluator.py
    - python_refactor/src/experiments/thesis_aligned_experiment.py
  branch_name: feat/w3-5-ruff-f401-wider-sweep
  acceptance: >
    `ruff check --select F401 python_refactor/src/` returns 0 errors.
    All in-scope tests continue to pass in .venv-w1 (test collection
    count + pass count unchanged from W3-4 baseline).
dispatch_instructions: |
  Run `ruff check --select F401 --fix python_refactor/src/`. Verify 0
  F401 errors remain. Run pytest on all currently-collecting test
  files to confirm no regression.
---

# W3-5 — Wider ruff F401 sweep

Closes W2-4-CARRY-1.
