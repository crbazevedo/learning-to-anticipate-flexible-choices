---
id: W11-2
role: code-fixer
name: Fix portfolio_evaluator price-vs-returns input
purpose: "Closes W10-CARRY-1. _get_asset_returns detects price-level input (median |value| > 1.5) and converts via pct_change before downstream cumprod."
wave: W11
unit: W11-2
depends_on: []
blocks: [W11-3]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - python_refactor/src/experiments/portfolio_evaluator.py
output_contract:
  files:
    - python_refactor/src/experiments/portfolio_evaluator.py
    - python_refactor/tests/test_experiments_portfolio_evaluator.py
  branch_name: feat/w11-2-portfolio-evaluator-returns-space
  acceptance: >
    _get_asset_returns auto-detects price-level input (median |value| > 1.5)
    and converts to returns via pct_change(); returns-shaped input passes
    through. No more Infinity in portfolio_value on real paper-window data.
    ≥ 2 regression tests.
dispatch_instructions: |
  Surgical fix at portfolio_evaluator.py:87-103. Wrap the
  `if isinstance(...) ... return data['assets']` with a
  price-level detector + pct_change conversion. Threshold 1.5 is
  conservative: real daily equity returns sit in [-0.2, +0.2];
  index/price levels are typically 100s-1000s.

  What NOT to do:
    - Don't touch sms_emoa.py (W11-1).
    - Don't refactor _calculate_portfolio_performance.
    - Don't change the function signature.
---

# W11-2 — portfolio_evaluator returns-space fix
