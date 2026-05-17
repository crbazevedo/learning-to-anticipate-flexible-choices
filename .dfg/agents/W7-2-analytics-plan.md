---
id: W7-2
role: doc-author
name: Comprehensive analytics framework doc
purpose: "docs/ANALYTICS-PLAN.md covering reports, stats tests, comparisons, multi-factor, segmentation, viz, tables per operator directive 2026-05-17."
wave: W7
unit: W7-2
depends_on: []
blocks: []
governance_tier: VT2
sized: M
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    - docs/EXPERIMENT-VALIDATION-PLAN.md
    - docs/paper.pdf
output_contract:
  files:
    - docs/ANALYTICS-PLAN.md
  branch_name: feat/w7-2-analytics-plan
  acceptance: >
    docs/ANALYTICS-PLAN.md exists with the 8 sections from the W7-2
    acceptance criteria. Each statistical test cited to a methodology
    source. Each visualization linked to a paper figure when applicable.
    Plan is W8-execution-ready (no further design needed).
dispatch_instructions: |
  Sections (in order):
  1. Purpose + scope + status framing
  2. Descriptive statistics (per-scenario × per-window summaries)
  3. Inferential tests (Mann-Whitney U per paper §V-D; Wilcoxon; ANOVA for 2×2)
  4. Effect size (Cohen's d; Cliff's delta as non-parametric alternative)
  5. Multi-factor analysis (2×2 factorial: TIP × MultiHorizon)
  6. Data segmentation (by year, by market regime, by sub-window)
  7. Visualization catalog (10+ figure types, each linked to use case)
  8. Table specifications (paper-Table-I-comparable + multi-factor table)
  9. Reproducibility receipts (seed + commit + lock per result)
  10. Honest scars (what could go wrong + interpretation playbook)
---

# W7-2 — Comprehensive analytics plan

Doc-only.
