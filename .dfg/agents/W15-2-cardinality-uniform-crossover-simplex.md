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
    - path: docs/BACKLOG.md
      sections: ["§1.1 B3 — Cardinality constraint not enforced", "§1.1 B4 — Non-simplex weights AND wrong crossover operator"]
      reason: "Catalog entries — read FIRST; full grounding for this unit"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [146]
      sections: ["§7.2.3 Constraint Handling"]
      excerpt: "We considered minimum and maximum cardinality of c_l = 5 and c_u = 15 assets."
      reason: "Source of truth for cardinality bounds (B3)"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [142]
      sections: ["§7.2 Eq (7.3)"]
      excerpt: "s.t. c_l ≤ c(u_t) ≤ c_u, where c(u_t) computes the number of assets in u_t with non-zero weight (u_t > 0)."
      reason: "Formal cardinality constraint (B3)"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [147]
      sections: ["§7.2.3 Search Operators"]
      excerpt: "We utilized uniform crossover over the mean DD vectors. For mutation, we randomly choose between (1) modifying the non-zero weights; or (2) adding/removing assets. If operator (1) is selected, then, with probability 1/2, we either increase or decrease the investment on a randomly chosen asset by a uniformly drawn factor from 10 to 50%. If (2) is selected, then, with probability 1/2, we either add or remove a randomly chosen asset. If it is removed, we simply set its weight to zero. If it is added, we randomly set its weight within a ±10% range from an equally-balanced allocation. All modified DD vectors are renormalized."
      reason: "Verbatim spec for the operators (B4): UNIFORM crossover (NOT SBX) + dual-mode mutation"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [141]
      sections: ["§7.2 Solving Portfolio Selection with AS-MOO"]
      excerpt: "u ∈ S^{N-1} denote the proportions of wealth to be invested"
      reason: "Simplex constraint: u_i ≥ 0, sum(u_i) = 1 (B4)"
    - path: python_refactor/src/algorithms/operators.py
      reason: "Module being edited — current crossover/mutation/normalize"
    - path: python_refactor/src/portfolio/portfolio.py
      reason: "Cardinality computation site (line ~298 cls.card) + Portfolio.max_cardinality default"
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
