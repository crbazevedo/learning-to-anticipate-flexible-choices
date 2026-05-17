---
id: W7-3
role: doc-author
name: VALIDATION-RESULTS.md skeleton template
purpose: "Empty results doc with the section structure W8 will populate. Every numeric cell is a 🚧 marker, never a fabricated value."
wave: W7
unit: W7-3
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/ANALYTICS-PLAN.md
    - docs/EXPERIMENT-VALIDATION-PLAN.md
output_contract:
  files:
    - docs/VALIDATION-RESULTS.md
  branch_name: feat/w7-3-validation-results-template
  acceptance: >
    docs/VALIDATION-RESULTS.md exists. Section headers match the
    8 tables (A-H) + 12 figures (F1-F12) from ANALYTICS-PLAN.md.
    Every numeric cell is a 🚧 marker (placeholder, NOT a number).
    Document explains its own status (template; W8 populates).
dispatch_instructions: |
  Structure mirrors ANALYTICS-PLAN.md sections so a reader can
  cross-reference. Every number is 🚧 — populated by W8 only.
---

# W7-3 — VALIDATION-RESULTS.md skeleton

Empty template. Numbers come in W8.
