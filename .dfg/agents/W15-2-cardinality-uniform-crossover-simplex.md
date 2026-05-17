---
id: W15-2
role: code-fixer
name: Cardinality + uniform crossover + simplex projection (BACKLOG B3+B4)
purpose: "Closes BACKLOG B3 (c_l=5, c_u=15 enforced) + B4 (replace SBX with uniform crossover; dual-mode mutation; non-negative simplex projection) per thesis §7.2.3 Search Operators + Constraint Handling (p.146-147)."
wave: W15
unit: W15-2
depends_on: []
blocks: [W15-5]
governance_tier: VT1
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/src/algorithms/operators.py
    - python_refactor/src/portfolio/portfolio.py
output_contract:
  files:
    - python_refactor/src/algorithms/operators.py
    - python_refactor/src/portfolio/portfolio.py
    - python_refactor/tests/test_operators_thesis_spec.py
  branch_name: feat/w15-2-cardinality-uniform-crossover-simplex
  acceptance: >
    uniform_crossover function implements thesis §7.2.3 (uniform per
    asset over mean DD vectors). dual_mode_mutation implements modify/
    add-remove per §7.2.3 verbatim. Cardinality enforced post-operator
    with c_l=5, c_u=15. Simplex projection clips negative weights then
    renormalizes. ≥ 5 regression tests.
dispatch_instructions: |
  Closes BACKLOG: B3 + B4.

  New / replaced functions in operators.py:
    1. uniform_crossover(parent1, parent2, p=0.5) — per asset
       independently swap weight with prob p; renormalize; project to
       simplex; enforce cardinality
    2. dual_mode_mutation(solution, p_add_remove=0.5,
                          p_factor_lo=0.10, p_factor_hi=0.50,
                          add_jitter=0.10) — mode 1: modify a non-zero
       weight by factor; mode 2: add or remove an asset. Verbatim
       thesis spec.
    3. project_to_simplex(weights, c_l, c_u) — helper: clip negatives;
       if cardinality > c_u, zero out smallest |weights| until cardinality
       == c_u; if cardinality < c_l, randomly add small weights to
       inactive assets until cardinality == c_l; renormalize.

  Update Portfolio: set max_cardinality default if used elsewhere.
  Don't break existing tests (SBX kept available for non-thesis paths,
  marked deprecated).

  What NOT to do:
    - Don't touch sms_emoa.py (W15-1).
    - Don't touch SCENARIOS dict (W15-3).
    - Don't reimplement the algorithm — only operators + cardinality.

  PR body must echo the thesis excerpts per BACKLOG §6.
---

# W15-2 — Uniform crossover + dual-mode mutation + cardinality + simplex

Closes BACKLOG.md items: **B3** + **B4**.
