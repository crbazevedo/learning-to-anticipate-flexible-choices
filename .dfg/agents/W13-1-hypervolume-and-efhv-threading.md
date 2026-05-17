---
id: W13-1
role: code-fixer
name: Thread expected_future_hypervolume + audit hypervolume=0
purpose: "Thread EFHV kwarg through experiment_manager.py:269 + audit why hypervolume=0 + minimum-viable fix."
wave: W13
unit: W13-1
depends_on: []
blocks: [W13-2]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/src/experiments/metrics_collector.py
output_contract:
  files:
    - python_refactor/src/experiments/experiment_manager.py
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/tests/test_hypervolume_threading.py
  branch_name: feat/w13-1-hypervolume-and-efhv-threading
  acceptance: >
    experiment_manager.py:269 passes
    expected_future_hypervolume=algorithm.get_expected_future_hypervolume().
    Smoke S2/paper/seed=1: algorithm.expected_future_hypervolume NOT null;
    algorithm.hypervolume > 0 (or precise documented reason). ≥ 3 tests.
dispatch_instructions: |
  Two surgical edits + audit:

  1. experiment_manager.py:269 — add kwarg:
     expected_future_hypervolume=algorithm.get_expected_future_hypervolume()
     (only when algorithm has the method; guard with hasattr)

  2. Audit hypervolume=0: print/log self.pareto_front objectives in
     sms_emoa._compute_hypervolume on first call. Likely causes:
       - ROI values all = 0 (W12-CARRY-1 family — same root)
       - R1/R2 reference points mis-set
     Fix smallest cause. If structural (ROI=0 from upstream),
     document precisely and minimum-viable defer to W14.

  What NOT to do:
    - Don't refactor SMSEMOA or PortfolioEvaluator end-to-end.
    - Don't fix W12-CARRY-1 unrelated portfolio_value decoupling.
---

# W13-1 — Thread EFHV + audit hypervolume=0
