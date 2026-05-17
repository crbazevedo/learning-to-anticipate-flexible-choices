---
id: W19-3
role: experimenter
name: Cross-check E — Dirichlet distributions from Dirichlet filter
purpose: "Closes operator check E. Cross-validate the Dirichlet predictive filter (decision-space tracking) on identical fixtures."
wave: W19
unit: W19-3
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
    - python_refactor/src/algorithms/anticipatory_learning.py
    - legacy-cpp/source/nsga2.cpp
output_contract:
  files:
    - legacy-cpp/build/drivers/dirichlet_driver.cpp
    - python_refactor/scripts/cross_validation/run_dirichlet.py
    - python_refactor/tests/test_cross_check_dirichlet.py
    - docs/CROSS-VALIDATION-E-DIRICHLET.md
  branch_name: feat/w19-3-cross-check-e-dirichlet-filter
  acceptance: >
    Cross-validation receipt for DirichletPredictor.dirichlet_mean_prediction_vec
    + dirichlet_mean_map_update vs C++ equivalent. Identical fixture;
    posterior Dirichlet concentration parameters compared.
dispatch_instructions: |
  Closes operator check E. Pattern per W18-1.

  Python: DirichletPredictor in anticipatory_learning.py has
  dirichlet_mean_prediction_vec + dirichlet_mean_map_update.
  C++ equivalent likely lives in nsga2.cpp or aprendizado_operadores.cpp.

  If C++ lacks the Dirichlet filter (the thesis presents it as a
  contribution; may not be in the legacy code) → document this as a
  scar; the C++ may use a simpler decision-space tracker.

  Receipt: docs/CROSS-VALIDATION-E-DIRICHLET.md.
---

# W19-3 — Cross-check E: Dirichlet filter
