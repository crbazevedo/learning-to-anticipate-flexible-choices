# .dfg/ — runtime state directory

Created by `dfg-harness instrument` (v0.5.10). See
[DFG-AGENT-LCM-SPEC.md §3](../docs/DFG-AGENT-LCM-SPEC.md) for the full
canonical-vs-derived contract. Key invariants:

- `events.jsonl` is the single canonical source of truth.
- `agents/`, `handoffs/`, `checkpoints/` are append-canonical.
- `state.json`, `master-index.md`, `modifications.md`, `PROVENANCE_INDEX.md`
  are derived projections; do not hand-edit.
