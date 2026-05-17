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
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/src/algorithms/sms_emoa.py
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
