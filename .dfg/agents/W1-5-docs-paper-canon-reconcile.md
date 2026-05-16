---
id: W1-5
role: doc-author
name: Reconcile docs to paper canon; strike thesis-only equation numbering
purpose: >
  Reframe the 3 existing analysis docs so every equation citation maps
  to the canonical IEEE paper (docs/paper.pdf), not the thesis chapter
  numbering (6.x / 7.x). Add an explicit Thesis-Paper reconciliation
  table at the top of thesis_codebase_analysis.md. Annotate every
  thesis-eq citation across thesis_codebase_analysis.md and
  100_percent_adherence_backlog.md with its paper-equation equivalent
  (or flag as thesis-only extension when not in the paper). In
  HVDM_OPTIMIZATION_PLAN.md, flag any concepts not in the published
  paper as "thesis/post-paper extension proposals" so they cannot be
  mistaken for paper-canonical constructs. Delete repository_analysis.md
  (stale per the 2026-05-16 audit; superseded by docs/VISION.md).
wave: W1
unit: W1-5
depends_on: []
blocks: []
governance_tier: VT2
sized: S
hardening_max_cycles: 2
prompt_version: 1
read_contract:
  must_read:
    - docs/paper.pdf  # §IV-A/B Eqs (11)-(25) — the canonical equation set
    - thesis_codebase_analysis.md  # 30+ thesis-eq citations to reconcile
    - 100_percent_adherence_backlog.md  # 20+ thesis-eq citations + PT narrative
    - HVDM_OPTIMIZATION_PLAN.md  # results plan; flag extension proposals
    - repository_analysis.md  # candidate for deletion (verify superseded by VISION.md)
  may_read:
    - docs/VISION.md
output_contract:
  files:
    - thesis_codebase_analysis.md
    - 100_percent_adherence_backlog.md
    - HVDM_OPTIMIZATION_PLAN.md
    - repository_analysis.md
  branch_name: feat/w1-5-docs-paper-canon-reconcile
  acceptance: >
    thesis_codebase_analysis.md: opens with a Thesis-Paper Equation
    Reconciliation table mapping every thesis equation cited in the file
    to its paper equivalent (or to "thesis-only extension" when not in
    the paper). Every body citation of a thesis equation is annotated
    inline with its paper-equation counterpart.

    100_percent_adherence_backlog.md: every thesis-eq citation in the
    Portuguese narrative is annotated inline with its paper-equation
    counterpart. Full PT-to-EN translation is deferred to a later unit;
    this unit only adds paper-canonical anchoring.

    HVDM_OPTIMIZATION_PLAN.md: a new top-of-file note flags any
    concepts that go beyond the published paper (e.g. UncertaintyAware
    Predictor, AMFCTrajectoryPredictor if present) as "post-paper
    extension proposals," so the reader cannot mistake them for
    constructs grounded in the IEEE paper.

    repository_analysis.md: DELETED. The audit found it stale on at
    least three load-bearing claims (test count 68 vs actual 287;
    "100% functional equivalence" against legacy C++ now stored at
    legacy-cpp/ which is not buildable on modern macOS; no mention of
    the thesis-adherence layer added in 2025-09). Its replacement role
    is filled by docs/VISION.md.
dispatch_instructions: |
  ## What this contract authorises

  Touching exactly the 4 files in output_contract.files. No other
  files; no other surfaces.

  ## What this contract does NOT authorise

  - Translating Portuguese narrative to English (queue this as a
    later unit; do NOT do it in this PR).
  - Refactoring or rewriting any code in python_refactor/ (that's
    W1-2/W1-3 territory).
  - Renumbering or modifying VISION.md (it's already paper-canonical;
    leave it alone).
  - Re-writing the analysis docs' substantive findings — only the
    equation-citation annotation + the reconciliation preamble +
    the HVDM extensions flag + the repository_analysis deletion.

  ## Required reading sequence (per operator directive 2026-05-16
  sub-papercut #16 — "must_read means BOTH emit Read events AND
  actually open + scan the files")

  1. docs/paper.pdf — verify the canonical equation set (already
     read pages 1-9 in W1-1 + W1-5 prep; the reconciliation table
     below was built from that reading).
  2. thesis_codebase_analysis.md — full file scan to enumerate
     every thesis-eq citation (52 across all 3 docs per grep).
  3. 100_percent_adherence_backlog.md — full file scan.
  4. HVDM_OPTIMIZATION_PLAN.md — scan to find any non-paper-grounded
     concepts (UncertaintyAwarePredictor, AMFCTrajectoryPredictor,
     etc.) that need the extension-proposal flag.
  5. repository_analysis.md — full scan to verify deletion is safe
     (nothing load-bearing remains that isn't in VISION.md).

  ## The Thesis ↔ Paper equation reconciliation (canonical)

  This table is the source of truth this unit installs into
  thesis_codebase_analysis.md:

  | Thesis Eq | Paper Eq | Concept |
  |---|---|---|
  | 6.5 / Definition 6.1 | (12) | TIP — Time Incomparability Probability |
  | 6.6 | (13) | λ^(H) anticipation rate from binary entropy of TIP |
  | 6.7 | (6) | Normalisation constraint on the anticipation-rate set |
  | 6.9 | §IV-A.1 narrative | λ^(K) from KF residuals (not explicitly numbered in the paper) |
  | 6.10 | (14) | OAL anticipatory learning rule — multi-horizon convex combination |
  | 6.11 | — | Alternate algebraic form of (14); not separately numbered in the paper |
  | 6.16 | (15) | Linear combination of Gaussians stays Gaussian |
  | 6.17 | (16) | H=2 special case of OAL Gaussian form |
  | 6.24–6.27 | (17) + (18) | Sliding-Window Dirichlet recursive concentration update — paper consolidates the t<K, t=K, t>K cases into the recurrence in (17) and the explicit forms in (18) |
  | 6.28 | (19) | Belief-weighted mean dynamics |
  | 6.30 | (20) | Belief coefficient v_{t+1} from binary entropy of TIP |
  | 6.31 | (21) | Unconditional mean approximation E[Û^{N⋆}_{t+1}] |
  | 6.33 | (22) | MAP correction for predicted mean decision vectors |
  | 6.35 | (24) | Δ_S — hypervolume contribution definition |
  | 7.16 | — (thesis-only) | Combined λ that averages λ^(H) and λ^(K). The IEEE paper's λ in Eq (5) is only λ^(H); the (1/2)(λ^(H) + λ^(K)) blend is a thesis extension not in the published paper. |

  ## Implementation order

  1. Read all 4 files fully (or re-confirm reading via grep + tail).
  2. Author thesis_codebase_analysis.md changes:
     - INSERT a new section "## 0. Thesis ↔ Paper Equation
       Reconciliation" at the top, containing (a) a provenance note
       citing the paper PDF and (b) the table above.
     - For each thesis-eq citation in the body, append a "(= paper
       Eq N)" parenthetical immediately after the thesis citation.
       Mass-annotate via grep-driven edit; verify by re-grep.
  3. Author 100_percent_adherence_backlog.md changes:
     - INSERT a top-of-file English-language note (the PT narrative
       stays — translation is a later unit) pointing readers to
       docs/paper.pdf as canon + linking to the reconciliation table
       in thesis_codebase_analysis.md.
     - For each thesis-eq citation in the file, append a "(= paper
       Eq N)" parenthetical.
  4. Author HVDM_OPTIMIZATION_PLAN.md changes:
     - Scan for non-paper-grounded concepts. If present
       (UncertaintyAwarePredictor, AMFCTrajectoryPredictor, etc.),
       add a top-of-file "Note: this document includes extension
       proposals beyond the published IEEE paper. Concepts X, Y, Z
       are post-paper extensions. See docs/paper.pdf for the
       canonical algorithmic spec."
  5. DELETE repository_analysis.md (`git rm`). Verify nothing
     references it from the rest of the repo via grep.

  ## Verification before commit

  - Re-grep for thesis-eq citations (6.x / 7.x patterns); verify
    every body citation has its "(= paper Eq N)" annotation OR is
    explicitly flagged as thesis-only extension.
  - Verify the reconciliation table at the top of
    thesis_codebase_analysis.md renders cleanly (Markdown table
    syntax sane).
  - Verify repository_analysis.md is gone and no other doc still
    points to it.
  - `dfg validate` still passes.

  ## Commit discipline (per operator directive 2026-05-16 #3)

  First commit on the feat/w1-5-docs-paper-canon-reconcile branch =
  THIS contract file alone (P3). Implementation commits come AFTER
  the contract is on the branch.

  ## Honest scars to surface in retro

  - The Portuguese narrative in 100_percent_adherence_backlog.md is
    untranslated as part of this unit; flag as a future translation
    unit so the scope is explicit, not silently deferred.
  - Equation citations in PYTHON code/tests are NOT reconciled in
    this unit (those will be touched by W1-2/W1-3/W1-4 which
    rewrite the affected modules); flag so we don't double-touch.
  - The thesis equations 6.11 and 6.9 have no exact paper number;
    the table marks them as "narrative" or "alternate form" — this
    is a honest gap, not a fudge.
---

# W1-5 — Reconcile docs to paper canon

See YAML frontmatter for the structured contract. Free-form prose below
is operator-readable context.

## The single sentence

Three of the four root-level analysis docs cite equations using the
thesis chapter-numbered convention (6.x / 7.x). The published IEEE
paper uses its own numbering ((11)–(25)). Going forward, the paper is
canon. This unit installs a reconciliation table and inline annotations
so the docs are paper-anchored without rewriting their substance.

## Why this is a single unit, not several

- The reconciliation TABLE is a single artifact; it belongs in one
  place (thesis_codebase_analysis.md).
- The inline annotations are mechanical edits driven by the table.
- The repository_analysis.md deletion is one git rm.
- The HVDM flag is one paragraph at the top.

Combining these into one unit avoids a 4-PR cascade for a coherent
piece of housekeeping.

## What this unit deliberately does NOT do

- Translate Portuguese narrative.
- Touch any Python code or tests.
- Rewrite the analysis docs' substantive technical findings.

Those are explicitly out of scope and queued as later units.
