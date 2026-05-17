---
id: W20-2
role: experimenter
name: Cross-check H — Dirichlet variant (confirm dedupes with E)
purpose: "Closes operator check H. Investigates whether check H is a distinct concern from check E (already closed in W19-3) or refers to a different Dirichlet-related computation."
wave: W20
unit: W20-2
depends_on: []
blocks: [W20-6]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/CROSS-VALIDATION-V2-REASSESSMENT.md
    - legacy-cpp-v2/source/dirichlet.cpp
    - python_refactor/src/algorithms/anticipatory_learning.py
output_contract:
  files:
    - docs/CROSS-VALIDATION-H-DIRICHLET-VARIANT.md
  branch_name: feat/w20-2-cross-check-h-dirichlet-variant
  acceptance: >
    Determine if operator check H is distinct from check E or a
    duplicate. If distinct: full cross-check against the variant. If
    duplicate: documented dedupe + cite W19-3 receipt.
dispatch_instructions: |
  Operator listed H separately from E in the original letter:
   - E: "Are the Dirichlet distributions resulting from Dirichlet
        filter application computed correctly?"
   - H: "Are the Dirichlet distributions resulting from Dirichlet
        filter application computed correctly?" (verbatim same prompt)

  H appears to be a verbatim duplicate of E. If so, document the
  dedupe + cite W19-3 receipt + close.

  If a meaningful distinct interpretation exists (e.g., H is about
  posterior Dirichlet AFTER multiple update cycles, vs E which is
  about a single update), build the appropriate cross-check using
  the W19-3 driver pattern.

  Receipt: docs/CROSS-VALIDATION-H-DIRICHLET-VARIANT.md.
---

# W20-2 — Cross-check H (Dirichlet variant)
