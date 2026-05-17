---
id: W14-1
role: tooling-author
name: Walk-forward rolling-period orchestrator (thesis §7.2.2)
purpose: "New module: at each rolling period t, train SMS-EMOA on 1.5-year window → extract Pareto → compute Ŝ_{t+1} via W13-2 against next-period MLE Gaussian → return list per seed."
wave: W14
unit: W14-1
depends_on: []
blocks: [W14-2]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/THESIS-INDEX.md
    - python_refactor/experiments/oos_evaluator.py
    - python_refactor/experiments/oos_report.py
    - python_refactor/experiments/validation_matrix.py
output_contract:
  files:
    - python_refactor/experiments/walk_forward.py
    - python_refactor/tests/test_experiments_walk_forward.py
  branch_name: feat/w14-1-walk-forward-orchestrator
  acceptance: >
    walk_forward.py exposes run_walk_forward(scenario, seed,
    full_returns, train_window_days≈378, step_days=50, n_mc=1000, rng)
    → list of Ŝ_{t+1}. Defaults match thesis §7.2.3. Per-period output
    includes trained Pareto weights + Ŝ_{t+1}. ≥ 5 regression tests.
dispatch_instructions: |
  Rolling protocol per thesis §7.2.2:
    For t in 1..T-1 where T = floor((len(full_returns) - train_window_days) / step_days) + 1:
      train_start = (t-1) * step_days
      train_end   = train_start + train_window_days
      oos_start   = train_end
      oos_end     = oos_start + step_days  (the next period)
      if oos_end > len(full_returns): break
      train       = full_returns.iloc[train_start:train_end]
      oos         = full_returns.iloc[oos_start:oos_end]

      # train SMS-EMOA on `train` (in-process; same pattern as
      # oos_report._run_one_scenario_seed)
      # extract Pareto weights
      # call compute_oos_efhv(weights, oos, n_samples=n_mc, rng) → Ŝ_{t+1}

  Per-period results returned as a list of dicts:
    {'period': t, 'train_range': (train_start, train_end),
     'oos_range': (oos_start, oos_end), 'n_pareto': N,
     'efhv_mean': Ŝ_{t+1}, 'efhv_std': MC std}

  What NOT to do:
    - Don't re-implement compute_oos_efhv (reuse W13-2).
    - Don't change the algorithm — only the periodization.
    - Don't ship the runner CLI (that's W14-2).
---

# W14-1 — Walk-forward orchestrator
