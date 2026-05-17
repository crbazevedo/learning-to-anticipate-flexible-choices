---
id: W20-5
role: experimenter
name: Cross-check K — expected HV contribution for ASMS-EMOA
purpose: "Closes operator check K. Cross-validate stochastic Δ_S (expected hypervolume contribution) between v2 (compute_stochastic_Delta_S in asms_emoa.cpp:315+) and Python's expected_future_hypervolume."
wave: W20
unit: W20-5
depends_on: []
blocks: [W20-6]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - legacy-cpp-v2/source/asms_emoa.cpp
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - legacy-cpp-v2/build/drivers/expected_hv_driver.cpp
    - python_refactor/scripts/cross_validation/run_expected_hv.py
    - python_refactor/tests/test_cross_check_expected_hv.py
    - docs/CROSS-VALIDATION-K-EXPECTED-HV.md
  branch_name: feat/w20-5-cross-check-k-expected-hv
  acceptance: >
    Cross-validation receipt comparing v2's Δ_S formula (using
    stochastic_params with conditional mean/var per portfolio) to
    Python's expected_future_hypervolume. Verdict per W18 matrix.
dispatch_instructions: |
  Pattern per W18-1. v2's stochastic Δ_S uses:
   delta_S = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI)
              / (var_delta_ROI + var_delta_risk)

  per asms_emoa.cpp:312/330/351. This is a CLOSED-FORM expected HV
  contribution under Gaussian assumption — different from Python's
  Monte Carlo approach.

  Likely DISAGREE structurally. Investigate whether the closed-form
  estimator matches the MC estimator in expectation.
---

# W20-5 — Cross-check K (expected HV contribution)
