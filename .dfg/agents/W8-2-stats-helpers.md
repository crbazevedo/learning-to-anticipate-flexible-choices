---
id: W8-2
role: tooling-author
name: Statistical test helpers
purpose: "Pure functions: Mann-Whitney U, Wilcoxon, Welch t, one-way ANOVA, Cohen d, Cliff delta, bootstrap CI, Holm-Bonferroni."
wave: W8
unit: W8-2
depends_on: []
blocks: ['W8-4']
governance_tier: VT2
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/ANALYTICS-PLAN.md
output_contract:
  files:
    - python_refactor/experiments/stats.py
    - python_refactor/tests/test_experiments_stats.py
  branch_name: feat/w8-2-stats-helpers
  acceptance: >
    stats.py exposes the 8 named helpers, each with docstring citing
    methodology source. test_experiments_stats.py has ≥ 8 equation-level
    tests against known-analytical inputs.
dispatch_instructions: |
  Pure-function module. No I/O, no global state. Functions:
    - mann_whitney_u(x, y, alternative='two-sided') → (U, p)
    - wilcoxon_signed_rank(x, y) → (W, p) (paired)
    - welch_t(x, y) → (t, p)
    - one_way_anova(*groups) → (F, p)
    - cohens_d(x, y) → float
    - cliffs_delta(x, y) → float in [-1, 1]
    - bootstrap_ci(values, statistic=mean, n_resamples=10000, alpha=0.05) → (lo, hi)
    - holm_bonferroni(p_values) → list[float]
  
  Use scipy.stats where available; otherwise implement from definition.
  Tests: each function asserted against a known-analytical case.
---

# W8-2 — Statistical test helpers

Per ANALYTICS-PLAN.md §3. Pure functions.
