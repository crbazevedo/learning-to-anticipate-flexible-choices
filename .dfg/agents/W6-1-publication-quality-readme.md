---
id: W6-1
role: doc-author
name: Rewrite README.md for public release
purpose: "Replace bootstrap README stub with publication-quality content: paper anchor, quickstart, algorithm map, project layout, reproducibility, citation."
wave: W6
unit: W6-1
depends_on: []
blocks: ['W6-2']
governance_tier: VT2
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf
    - docs/VISION.md
    - pyproject.toml
    - README.md
output_contract:
  files:
    - README.md
  branch_name: feat/w6-1-publication-quality-readme
  acceptance: >
    README.md is the publication-quality content per W6-1 acceptance
    criteria in plan.yaml. The reader can: (a) understand what the
    repo implements (paper anchor + identity), (b) install + test in
    under 5 commands, (c) navigate the project layout, (d) cite the
    work properly. Length ~150-250 lines; no marketing copy.
dispatch_instructions: |
  ## Implementation

  Sections (in order):
  1. Title + 1-paragraph identity + paper citation block (with DOI URL)
  2. Status / visibility line (private→public transitioning W6-2)
  3. Quick start: `uv sync` + `uv run pytest` example
  4. Algorithm overview — short prose with explicit paper-Eq references:
     - Eq (11) state vector
     - Eq (12)–(13) TIP + λ
     - Eq (14)–(15) OAL convex combo (mean + covariance)
     - Eq (17)–(18) Sliding-window Dirichlet
  5. Project layout — table of top-level dirs
  6. Reproducibility recipe — short pointer to docs/EXPERIMENT-VALIDATION-PLAN.md (W6-3)
  7. Testing + curated test set
  8. Governance note — mention dfg-harness briefly with link
  9. Citation block — BibTeX
  10. License pointer
  11. Acknowledgments + author contact

  Style: direct, technical, no marketing fluff. Match the paper's tone.
---

# W6-1 — Rewrite README.md for public release

Replaces the bootstrap stub. Anchored on docs/paper.pdf + docs/VISION.md.
