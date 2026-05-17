---
id: W3-6
role: tooling-author
name: Add uv.lock for reproducible installs
purpose: "Closes W2-4-CARRY-2. Generates uv.lock from pyproject.toml; removes uv.lock from .gitignore."
wave: W3
unit: W3-6
depends_on: ['W3-5']
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - pyproject.toml
    - .gitignore
output_contract:
  files:
    - uv.lock
    - .gitignore
  branch_name: feat/w3-6-uv-lock-add
  acceptance: >
    uv.lock exists at repo root, generated from current pyproject.toml.
    .gitignore no longer lists uv.lock as ignored. uv sync --frozen
    reproduces a venv from the lock.
dispatch_instructions: |
  1. Run `uv lock` at repo root.
  2. Remove the `uv.lock` line from .gitignore.
  3. Verify uv.lock is now a tracked file.
---

# W3-6 — uv.lock for reproducible installs

Closes W2-4-CARRY-2.
