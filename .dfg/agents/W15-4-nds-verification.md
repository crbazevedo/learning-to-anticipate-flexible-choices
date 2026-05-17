---
id: W15-4
role: verifier
name: Verify NDS basis matches thesis §6.5.1 (BACKLOG M15)
purpose: "Closes BACKLOG M15. Audit non-dominated sorting basis: deterministic means per thesis §6.5.1 p.134."
wave: W15
unit: W15-4
depends_on: []
blocks: [W15-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - path: docs/BACKLOG.md
      sections: ["§1.3 M15 — Non-dominated sorting basis verification"]
      reason: "Catalog entry"
    - path: docs/Azevedo_CarlosRenatoBelo_D.pdf
      pages: [134]
      sections: ["§6.5.1 Sorting the Population of Random Objective Vectors"]
      excerpt: "we argue that, since we are already able to handle the estimated uncertainty (see section 6.1) and to combine the learned predictive correlation into the computation of E[Δ_S] (see section 6.3), there is little need to incorporate uncertainty awareness directly into the dominance relation. Therefore, our proposed non-dominated sorting procedures are executed in terms of the deterministic Pareto Dominance over the estimated means of the random objective vectors"
      reason: "Source of truth: NDS over means, not over stochastic distributions"
    - path: python_refactor/src/algorithms/sms_emoa.py
      reason: "Audit target: _evaluate_solution line ~195 + _fast_non_dominated_sort"
output_contract:
  files:
    - docs/NDS-VERIFICATION-W15-4.md
  branch_name: feat/w15-4-nds-verification
  acceptance: >
    Audit doc cites sms_emoa.py:_evaluate_solution line 195
    (solution.objectives = [P.ROI, P.risk]) — deterministic means.
    Cross-ref thesis §6.5.1 + Theorem 6.5.1.
    Verdict: matches thesis (no code change) OR documents deviation.
dispatch_instructions: |
  Doc-only unit. No code change unless audit reveals a deviation.
---

# W15-4 — NDS verification
