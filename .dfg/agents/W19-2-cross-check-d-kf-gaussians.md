---
id: W19-2
role: experimenter
name: Cross-check D — KF predictive Gaussians (Reading-C KEYSTONE)
purpose: "Closes operator check D. Cross-validate Kalman filter predictive (mean, covariance) outputs on identical fixtures. KEYSTONE: if C++/Python KFs agree, structural-uncertainty (Reading C from W18-4) is confirmed."
wave: W19
unit: W19-2
depends_on: [W19-1]
blocks: [W19-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/W18-CROSS-VALIDATION-SYNTHESIS.md
    - docs/CROSS-VALIDATION-F-TIP.md
    - legacy-cpp/source/kalman_filter.cpp
    - legacy-cpp/headers/kalman_filter.h
    - python_refactor/src/algorithms/kalman_filter.py
output_contract:
  files:
    - legacy-cpp/build/drivers/kf_driver.cpp
    - python_refactor/scripts/cross_validation/run_kf.py
    - python_refactor/tests/test_cross_check_kf.py
    - docs/CROSS-VALIDATION-D-KF-GAUSSIANS.md
  branch_name: feat/w19-2-cross-check-d-kf-gaussians
  acceptance: >
    Cross-validation receipt for Kalman filter prediction + update on
    identical state + measurement fixtures. Both sides report
    posterior (x, P) after N steps; compared with abs+rel tolerance.
    THE Reading-C critical test: if KFs agree, structural-uncertainty
    finding (W18-4) is confirmed; if disagree, drift to fix.
dispatch_instructions: |
  Closes operator check D + W18-CARRY-2 (Reading C test) + the
  W17-5 saturation diagnostic chain.

  Fixture: (F, H, R, x0, P0, measurements) — random small KF state
  with constant-velocity dynamics on (ROI, risk) per thesis Eq (11).
  Run N=5 predict-update cycles; compare posterior (x, P) at each
  step.

  Per W18-4 analysis: if C++ and Python KFs produce IDENTICAL
  predictive distributions (mean, cov), the W17-5 TIP saturation
  IS structural — the algorithm correctly reports max-uncertainty
  on this data + parameterization. Reading C confirmed.

  If KFs DISAGREE — find the drift (initialization? state ordering?
  matrix algebra path?). Likely candidates: process noise Q (not
  explicit in C++ code — implicit identity?), F structure (Python
  W1-1 settled this; verify C++ matches).

  Receipt: docs/CROSS-VALIDATION-D-KF-GAUSSIANS.md.
---

# W19-2 — Cross-check D: KF Gaussians (Reading-C KEYSTONE)

Closes operator check D.

## Why this is the W19 keystone

W18-4 found TIP saturation is reproduced across 3 implementations on
synthetic. W19-2 tests whether the SAME predictive distributions
(KF outputs) feed both implementations in production. If yes →
saturation is genuinely structural (Reading C confirmed) → operator
strategic decision (KF re-calibration / multi-period metric / publish
replication-failure) becomes next-level question.

## Thesis grounding

§6.1 + Eq (11) — Kalman filter with constant-velocity dynamics on
(ROI, risk) state vector.
