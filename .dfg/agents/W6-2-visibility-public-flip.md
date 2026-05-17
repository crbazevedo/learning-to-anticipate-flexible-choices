---
id: W6-2
role: tooling-author
name: Flip repo visibility to public
purpose: "One-way door. `gh repo edit --visibility public`. Depends on W6-1 (README must already be public-ready)."
wave: W6
unit: W6-2
depends_on: ['W6-1']
blocks: []
governance_tier: VT3
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - README.md
output_contract:
  files:
    - .dfg/checkpoints/W6-2-visibility-receipt.md
  branch_name: feat/w6-2-visibility-public-flip
  acceptance: >
    `gh repo view --json visibility` returns PUBLIC. Receipt at
    .dfg/checkpoints/W6-2-visibility-receipt.md records the timestamp
    + the gh command output.
dispatch_instructions: |
  ## VT3 ceremony

  This is a one-way door. Pre-conditions verified by W6-1 merge:
    - README.md is publication-quality (no stub, no TODOs)
    - No secrets in tracked files (verify via `git ls-files | xargs grep -l "ELEVENLABS\|API_KEY\|SECRET"` returns nothing)
    - Repo is private at execution time (otherwise no-op)

  Execute:
    gh repo edit crbazevedo/learning-to-anticipate-flexible-choices \
        --visibility public

  Write receipt + verify.
---

# W6-2 — Flip repo visibility to public

VT3 one-way door. Operator-approved 2026-05-17.
