---
id: W6-3
role: doc-author
name: Author live experiment validation plan
purpose: "Doc-only — plan the live experiments that would empirically validate the wired algorithm against the paper's reported results. Execution deferred."
wave: W6
unit: W6-3
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf
    - docs/VISION.md
    - python_refactor/run_experiments.py
    - python_refactor/src/experiments/experiment_manager.py
output_contract:
  files:
    - docs/EXPERIMENT-VALIDATION-PLAN.md
  branch_name: feat/w6-3-experiment-validation-plan
  acceptance: >
    docs/EXPERIMENT-VALIDATION-PLAN.md exists with the structure
    from the W6-3 acceptance criteria in plan.yaml: dataset,
    scenarios (with/without TIP, with/without multi-horizon),
    expected outputs, acceptance criteria, reproducibility recipe,
    comparison-to-paper anchor.
dispatch_instructions: |
  ## Implementation

  Structure (sections):
  1. Purpose + scope (plan, not execution)
  2. Dataset — FTSE-100 from python_refactor/data/ftse-updated/
  3. Experiment matrix — 2×2 (TIP on/off × MultiHorizon on/off) + Markowitz baseline
  4. Expected outputs — for each scenario, which figure / metric to compare
  5. Acceptance criteria — quantitative thresholds vs paper Table I
  6. Reproducibility — seeds, command lines, dataset paths
  7. Comparison-to-paper anchor — explicit Eq + Table mapping
  8. Execution scaffold — what a future wave will need to add
  9. Honest scars
---

# W6-3 — Experiment validation plan

Doc-only.
