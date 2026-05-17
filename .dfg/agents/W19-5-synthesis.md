---
id: W19-5
role: verifier
name: W19 synthesis — C+D+E+G findings + W20-W21 roadmap (H,I,J,K,L,M,N)
purpose: "Aggregate W19-1..4 cross-check verdicts; produce wave-close receipt + W20-W21 roadmap for the remaining 7 checks."
wave: W19
unit: W19-5
depends_on: [W19-1, W19-2, W19-3, W19-4]
blocks: []
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/BACKLOG.md
    - docs/W18-CROSS-VALIDATION-SYNTHESIS.md
    - docs/CROSS-VALIDATION-C-BIVARIATE-GAUSSIAN.md
    - docs/CROSS-VALIDATION-D-KF-GAUSSIANS.md
    - docs/CROSS-VALIDATION-E-DIRICHLET.md
    - docs/CROSS-VALIDATION-G-ANTICIPATIVE-RATE.md
output_contract:
  files:
    - docs/W19-CROSS-VALIDATION-SYNTHESIS.md
    - .dfg/retrospectives/W19/W19-5.md
  branch_name: feat/w19-5-synthesis
  acceptance: >
    W19 synthesis doc + Reading A/B/C verdict update with W19 evidence.
    W20-W21 roadmap for the remaining 7 checks (H, I, J, K, L, M, N).
dispatch_instructions: |
  Aggregate W19-1..W19-4 verdicts. Update Reading A/B/C framework
  with the KF + λ end-to-end results. Set W20 keystone.
---

# W19-5 — Synthesis
