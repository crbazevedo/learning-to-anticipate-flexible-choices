---
id: W20-6
role: verifier
name: W20 synthesis + W21 roadmap (L/M/N + Reading-E verdict consolidation)
purpose: "Aggregate W20-1..W20-5 findings + consolidate Reading-E experimental verdict + pre-stub W21 for the final 3 cross-checks (L/M/N)."
wave: W20
unit: W20-6
depends_on: [W20-1, W20-2, W20-3, W20-4, W20-5]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/READING-E-EXPERIMENTAL-TEST.md
    - docs/CROSS-VALIDATION-H-DIRICHLET-VARIANT.md
    - docs/CROSS-VALIDATION-I-ANTICIPATIVE-DISTRIBUTIONS.md
    - docs/CROSS-VALIDATION-J-CROWDING-DISTANCE.md
    - docs/CROSS-VALIDATION-K-EXPECTED-HV.md
output_contract:
  files:
    - docs/W20-CROSS-VALIDATION-SYNTHESIS.md
    - .dfg/retrospectives/W20/W20-6.md
  branch_name: feat/w20-6-synthesis
  acceptance: >
    W20 synthesis doc + Reading framework update with W20 evidence +
    W21 roadmap for L/M/N (the final 3 cross-checks).
---

# W20-6 — Synthesis
