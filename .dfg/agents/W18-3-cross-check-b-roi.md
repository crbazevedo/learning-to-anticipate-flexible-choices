---
id: W18-3
role: experimenter
name: Cross-check B — ROI computation parity (C++ vs Python)
purpose: "Closes operator check B: is ROI being computed correctly + scaled correctly? Cross-validate Portfolio::compute_ROI between C++ legacy and Python port on identical fixtures."
wave: W18
unit: W18-3
depends_on: [W18-1]
blocks: [W18-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - legacy-cpp/source/portfolio.cpp
    - python_refactor/src/portfolio/portfolio.py
output_contract:
  files:
    - legacy-cpp/build/drivers/roi_driver.cpp
    - python_refactor/scripts/cross_validation/run_roi.py
    - python_refactor/tests/test_cross_check_roi.py
    - docs/CROSS-VALIDATION-B-ROI.md
  branch_name: feat/w18-3-cross-check-b-roi
  acceptance: >
    Cross-validation receipt for Portfolio::compute_ROI. Both sides on
    identical fixture; ROI vector compared; markdown diff documents
    agreement / discrepancy with thesis Eq (7.4) cross-ref.
dispatch_instructions: |
  Closes operator check B: ROI computation parity.

  Same pattern as W18-2 risk check.

  Thesis §7.2 Eq (7.4) verbatim:
    g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T
  ROI component = μ̂^T u (linear combination of mean returns).

  Honest scar candidates:
   - Sign convention: μ might be in % vs decimal
   - Time-scaling: daily vs annualized
   - Robust vs non-robust mean estimator
   - μ may be EXCLUDING the risk-free rate (Sharpe-style) or not

  PR body MUST include fixture + raw outputs + diff + thesis cross-ref.
---

# W18-3 — Cross-check B: ROI computation parity

Closes operator check B.

## Thesis grounding

**§7.2 Eq (7.4), p. 142** — verbatim:
> "g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T"

ROI = μ̂^T u (linear combination of mean asset returns weighted by
portfolio composition). Decimal returns (NOT %) per thesis §7.2.3
"daily adjusted close prices" → returns are O(0.001) scale.

## Receipt format

Same as W18-2.
