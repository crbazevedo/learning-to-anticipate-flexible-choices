---
id: W19-1
role: experimenter
name: Cross-check C — bivariate Gaussian inputs (mean, covariance) feeding KF
purpose: "Closes operator check C. Cross-validate the (mean, covariance) construction that feeds Kalman filter on identical fixtures."
wave: W19
unit: W19-1
depends_on: []
blocks: [W19-2, W19-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/W18-CROSS-VALIDATION-SYNTHESIS.md
    - legacy-cpp/source/portfolio.cpp
    - python_refactor/src/portfolio/portfolio.py
output_contract:
  files:
    - legacy-cpp/build/drivers/bivariate_gaussian_driver.cpp
    - python_refactor/scripts/cross_validation/run_bivariate_gaussian.py
    - python_refactor/tests/test_cross_check_bivariate_gaussian.py
    - docs/CROSS-VALIDATION-C-BIVARIATE-GAUSSIAN.md
  branch_name: feat/w19-1-cross-check-c-bivariate-inputs
  acceptance: >
    Cross-validation receipt for (mean, covariance) construction. Both
    sides on identical fixture; estimate_assets_mean_ROI +
    estimate_covariance outputs compared; markdown diff documents
    agreement / discrepancy. Upstream of cross-check D (W19-2).
dispatch_instructions: |
  Closes operator check C. Pattern per W18-1 harness.

  Two functions to cross-validate:
   - estimate_assets_mean_ROI(returns) → vector of per-asset means
   - estimate_covariance(mean_ROI, returns) → covariance matrix

  Both are simple statistical estimators; expected to AGREE within
  machine precision. The check matters because the KF inputs in W19-2
  derive from these (mean, cov) constructions.

  Receipt: docs/CROSS-VALIDATION-C-BIVARIATE-GAUSSIAN.md per W18 template.
---

# W19-1 — Cross-check C: bivariate Gaussian inputs to KF

Closes operator check C.

## Thesis grounding

§7.2 Eq (7.4) p.142 — defines (μ̂_{r,t}, Σ̂_{r,t}) as the inputs to the
ASMS-EMOA objective. These same estimates feed the Kalman filter
in §6.1.

## Hypothesis

Both sides compute MLE estimates of mean + sample covariance from
returns matrix. Expected: AGREE within machine precision.
