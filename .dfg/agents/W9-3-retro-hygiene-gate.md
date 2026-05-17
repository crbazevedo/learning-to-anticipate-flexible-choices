---
id: W9-3
role: tooling-author
name: Retro-hygiene — fix W7/W8 retro frontmatter + wire dfg reflect --validate
purpose: "Retires sub-papercut #23 (retro frontmatter wrong-keys silently skips ReflectionEmit, observed 7x). Re-keys 7 retros to ADR-004 canonical + wires dfg reflect --validate into pre-pr."
wave: W9
unit: W9-3
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - .dfg/retrospectives/W7/W7-1.md
    - .dfg/retrospectives/W7/W7-2.md
    - .dfg/retrospectives/W7/W7-3.md
    - .dfg/retrospectives/W8/W8-1.md
    - .dfg/retrospectives/W8/W8-2.md
    - .dfg/retrospectives/W8/W8-3.md
    - .dfg/retrospectives/W8/W8-4.md
output_contract:
  files:
    - .dfg/retrospectives/W7/W7-1.md
    - .dfg/retrospectives/W7/W7-2.md
    - .dfg/retrospectives/W7/W7-3.md
    - .dfg/retrospectives/W8/W8-1.md
    - .dfg/retrospectives/W8/W8-2.md
    - .dfg/retrospectives/W8/W8-3.md
    - .dfg/retrospectives/W8/W8-4.md
    - kit/scripts/pre-pr-reflect-validate.sh
  branch_name: feat/w9-3-retro-hygiene-gate
  acceptance: >
    All 7 retros re-keyed to ADR-004 canonical (what_worked, what_broke,
    what_youd_change, assumption_to_challenge). dfg reflect --backfill
    emits >= 7 ReflectionEmit events. kit/scripts/pre-pr-reflect-validate.sh
    runs `dfg reflect --validate` for each merged W*-* unit on current
    branch and surfaces failures.
dispatch_instructions: |
  Pure mechanical re-key. Semantic content preserved across the rename:
    what_didnt          → what_broke
    what_to_change      → what_youd_change
    new_carries         → fold into assumption_to_challenge body (or
                           append to what_youd_change). Use the field
                           that best fits the content.
    scars_carried_forward → fold into assumption_to_challenge body.

  What NOT to do:
    - Do not rewrite retro semantics (preserve every fact).
    - Do not modify dfg-harness itself.
    - Do not add the canon entry for sub-papercut #23 (the scar IS the
      gate; the gate is dfg reflect --validate, already in dfg).
---

# W9-3 — Retro-hygiene gate

Re-keys W7+W8 retros to ADR-004 canonical and wires `dfg reflect --validate`
into pre-pr so this scar can't recur silently.
