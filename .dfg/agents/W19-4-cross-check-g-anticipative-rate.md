---
id: W19-4
role: experimenter
name: Cross-check G — anticipative learning rate λ end-to-end (Eq 7.16 chain)
purpose: "Closes operator check G. Cross-validate the full Eq 7.16 λ = 0.5*(λ^H + λ^K) computation chain end-to-end against C++ on identical fixtures. Integrates findings from W18-4 (TIP) + W19-2 (KF)."
wave: W19
unit: W19-4
depends_on: []
blocks: [W19-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - docs/CROSS-VALIDATION-F-TIP.md
    - legacy-cpp/source/nsga2.cpp
    - python_refactor/src/algorithms/anticipatory_learning.py
    - python_refactor/src/algorithms/multi_horizon_anticipatory.py
output_contract:
  files:
    - legacy-cpp/build/drivers/anticipative_rate_driver.cpp
    - python_refactor/scripts/cross_validation/run_anticipative_rate.py
    - python_refactor/tests/test_cross_check_anticipative_rate.py
    - docs/CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md
  branch_name: feat/w19-4-cross-check-g-anticipative-rate
  acceptance: >
    Cross-validation receipt for the full Eq 7.16 chain end-to-end.
    Both sides compute λ on identical (KF state, predictions, K, H)
    inputs; compared with abs+rel tolerance. Integrates W18-4 (TIP)
    + W19-2 (KF) findings.
dispatch_instructions: |
  Closes operator check G. Pattern per W18-1.

  Cross-validate the full chain: KF state → TIP → λ^H, λ^K → combined λ.

  C++: alpha computation in nsga2.cpp:565 = 1 - linear_entropy(nd_probability)
  (Eq 6.6 form; H-only arm; no λ^K).
  Python: compute_anticipatory_learning_rate per W16-1 = 0.5*(λ^H + λ^K)
  (Eq 7.16 verbatim).

  EXPECTED FINDING: structural difference (C++ uses Eq 6.6, Python
  uses Eq 7.16). The thesis is unambiguous about Eq 7.16 (verbatim
  including 1/2 factor). So either:
   - C++ implements an earlier draft of the thesis
   - Python's W16-1 fix is the correct thesis-faithful reading
   - The "1/2" averaging is the difference that makes Python's
     anticipation HALF-strength compared to C++

  This is a major finding either way. Report honestly.

  Receipt: docs/CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md.
---

# W19-4 — Cross-check G: anticipative rate end-to-end
