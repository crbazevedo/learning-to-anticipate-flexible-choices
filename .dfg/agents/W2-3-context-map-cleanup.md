---
id: W2-3
role: doc-author
name: Remove W1-5 dangling must_read from context-map.yaml
purpose: Closes W1-5-CARRY-1 — W1-5 must_read in .dfg/context-map.yaml still references the deleted repository_analysis.md.
wave: W2
unit: W2-3
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - .dfg/context-map.yaml  # W1-5 entry to remove
output_contract:
  files:
    - .dfg/context-map.yaml
  branch_name: feat/w2-3-context-map-cleanup
  acceptance: >
    .dfg/context-map.yaml W1-5 entry is removed entirely (W1-5 is
    COMPLETED; per-unit must_read no longer needed). No remaining
    references to the deleted repository_analysis.md anywhere in
    .dfg/. dfg validate continues to pass.
dispatch_instructions: |
  Single-file change. Remove the entire W1-5 block from per_unit
  in .dfg/context-map.yaml. Verify via `grep repository_analysis
  .dfg/context-map.yaml` returns zero hits.
---

# W2-3 — Remove W1-5 dangling must_read from context-map.yaml

W1-5 is COMPLETED. Its per_unit must_read entry still references
`repository_analysis.md` which W1-5 itself deleted. No future
SessionStart hook will try to read it; but leaving the dangling
reference is honest-scar pollution. Clean removal.
