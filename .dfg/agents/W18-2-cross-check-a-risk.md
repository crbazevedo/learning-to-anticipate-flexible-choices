---
id: W18-2
role: experimenter
name: Cross-check A — risk computation parity (C++ vs Python)
purpose: "Closes operator check A: is risk being computed correctly + scaled correctly? Cross-validate Portfolio::compute_risk between C++ legacy and Python port on identical fixtures."
wave: W18
unit: W18-2
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
    - legacy-cpp/build/drivers/risk_driver.cpp
    - python_refactor/scripts/cross_validation/run_risk.py
    - python_refactor/tests/test_cross_check_risk.py
    - docs/CROSS-VALIDATION-A-RISK.md
  branch_name: feat/w18-2-cross-check-a-risk
  acceptance: >
    Cross-validation receipt for Portfolio::compute_risk. Both sides run
    on identical fixture (N portfolios × M assets × seed); risk vector
    reported; comparison emits markdown diff with abs+rel tolerance.
    Documented honestly: agreement / discrepancy / SCALE issue.
dispatch_instructions: |
  Closes operator check A: risk computation parity.

  Workflow per W18-1 harness pattern:
   1. Generate fixture: 20 random portfolios × 87 assets × seed=42;
      synthetic returns matrix 50 days × 87 assets.
   2. C++ driver: read fixture; for each portfolio compute risk via
      portfolio::compute_risk(P, covariance); emit risk vector CSV.
   3. Python driver: same fixture; for each portfolio compute risk via
      Portfolio.compute_risk; emit risk vector CSV.
   4. Compare: abs+rel tolerance 1e-9 by default; relax progressively
      if discrepancy found.
   5. Document outcome:
      - Agreement → "no scale issue; risk parity confirmed"
      - Disagreement → diff numbers + hypothesis (covariance scale?
        formula variant? ddof?)
      - Identify which side is wrong via thesis §7.2 Eq (7.4) verbatim

  Thesis §7.2 Eq (7.4) verbatim:
    g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T
  Risk component = u^T Σ̂ u (variance, NOT std dev).

  Honest scar candidates:
   - C++ may use sqrt(variance) (std dev) while Python uses variance
   - ddof=0 vs ddof=1 sample-vs-population covariance
   - Sign convention on returns

  What NOT to do:
   - Don't fix any discrepancy found (just report it)
   - Don't touch other checks (B/F are own units)
   - Don't decide A vs B side is wrong (mutual-skepticism: both sides
     reported; pick winner only after thesis cross-ref)

  PR body MUST include the fixture spec + both sides' raw output +
  diff + thesis cross-ref per BACKLOG §6.
---

# W18-2 — Cross-check A: risk computation parity

Closes operator check A.

## Thesis grounding

**§7.2 Eq (7.4), p. 142** — verbatim:
> "g(u_t, χ_t) = (u_t^T Σ̂_{r,t} u_t, μ̂_{r,t}^T u_t)^T"

The risk component is the QUADRATIC FORM u^T Σ̂ u (variance, not std-dev).
Some implementations use std-dev (sqrt of variance); the thesis text is
ambiguous on whether the optimizer sees variance or std-dev. Cross-check
to determine.

## Receipt format

`docs/CROSS-VALIDATION-A-RISK.md` MUST include:
1. Fixture spec (N portfolios, M assets, seed)
2. C++ raw risk vector (head + summary)
3. Python raw risk vector (head + summary)
4. Diff: abs-max, rel-max, agreement-fraction at each tolerance level
5. Thesis cross-ref: which side matches Eq (7.4) verbatim?
6. Verdict: AGREE / DISAGREE-PYTHON-WRONG / DISAGREE-CPP-WRONG /
   DISAGREE-AMBIGUOUS (latter when thesis text doesn't disambiguate)
