---
wave: W6
gate_type: wave-gate
verdict: PASS
date: 2026-05-17
units_completed: [W6-1, W6-3, W6-2]
units_carry_forward: [W6-2-CARRY-1]
verify:
  # ─── Wave-shipped via PRs #33, #34, #35 ───────────────────────────────
  - "git log --oneline master | grep -q 'W6-1.*README'"
  - "git log --oneline master | grep -q 'W6-3.*experiment validation'"
  - "git log --oneline master | grep -q 'W6-2.*PUBLIC'"

  # ─── W6-1: publication-quality README ─────────────────────────────────
  - "test -f .dfg/agents/W6-1-publication-quality-readme.md"
  - "test -f .dfg/retrospectives/W6/W6-1.md"
  - "grep -q '10.1109/TCYB.2015.2415732' README.md"
  - "grep -q 'Quick start' README.md"
  - "grep -q 'Algorithm overview' README.md"
  - "grep -q '@article{azevedo2015learning' README.md"

  # ─── W6-3: experiment validation plan ─────────────────────────────────
  - "test -f .dfg/agents/W6-3-experiment-validation-plan.md"
  - "test -f .dfg/retrospectives/W6/W6-3.md"
  - "test -f docs/EXPERIMENT-VALIDATION-PLAN.md"
  - "grep -q 'Experiment matrix' docs/EXPERIMENT-VALIDATION-PLAN.md"
  - "grep -q 'Honest scars' docs/EXPERIMENT-VALIDATION-PLAN.md"

  # ─── W6-2: visibility flip ────────────────────────────────────────────
  - "test -f .dfg/agents/W6-2-visibility-public-flip.md"
  - "test -f .dfg/retrospectives/W6/W6-2.md"
  - "test -f .dfg/checkpoints/W6-2-visibility-receipt.md"
  - "gh repo view crbazevedo/learning-to-anticipate-flexible-choices --json visibility --jq .visibility | grep -q PUBLIC"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W6 ships public-ready state. README publication-quality + validation plan documented + visibility flipped."
  - "Repo is now publicly accessible at https://github.com/crbazevedo/learning-to-anticipate-flexible-choices"
  - "Live experiment execution deferred to W7 per plan in docs/EXPERIMENT-VALIDATION-PLAN.md §8"

carry_forward:
  - id: W6-2-CARRY-1
    why: "FTSE-original CSV filenames in legacy-cpp/ contain parens (e.g. `table (0).csv`) which break naive shell expansion. Trivia-tier."
    next_action: "Document the null-delimited shell pattern OR rename CSVs in a future cleanup wave"
---

# W6-gate — WAVE 6 CLOSE: public-ready state

Repo is **PUBLIC** at https://github.com/crbazevedo/learning-to-anticipate-flexible-choices.

## What W6 delivered

| Unit | PR | What |
|---|---|---|
| W6-1 | #33 | Publication-quality README (~200 lines; paper anchor + quickstart + algorithm-Eq map + reproducibility + citation BibTeX + acknowledgments) |
| W6-3 | #34 | `docs/EXPERIMENT-VALIDATION-PLAN.md` — 9-section plan for live experiment validation (W7 scope) |
| W6-2 | #35 | Visibility flipped PRIVATE → PUBLIC via `gh repo edit`; receipt + retro committed |

Plus PR #32 (replan W6-add).

## Carries forwarded

- W6-2-CARRY-1 (trivia): FTSE-original CSV filenames break naive shell-grep.
