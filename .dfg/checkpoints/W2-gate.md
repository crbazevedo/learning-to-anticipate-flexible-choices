---
wave: W2
gate_type: wave-gate
verdict: PASS
date: 2026-05-16
units_completed: [W2-3, W2-1, W2-2, W2-4]
units_carry_forward: [W2-2-CARRY-1, W2-2-CARRY-2, W2-2-CARRY-3, W2-4-CARRY-1, W2-4-CARRY-2]
verify:
  # ─── Wave-shipped via PRs #10, #11, #12, #13 ──────────────────────────
  - "git log --oneline master | grep -q 'W2-3.*context-map'"
  - "git log --oneline master | grep -q 'W2-1.*_calculate_confidence'"
  - "git log --oneline master | grep -q 'W2-2.*import-style'"
  - "git log --oneline master | grep -q 'W2-4.*pyproject'"

  # ─── W2-3: context-map.yaml cleanup ───────────────────────────────────
  - "test -f .dfg/agents/W2-3-context-map-cleanup.md"
  - "test -f .dfg/retrospectives/W2/W2-3.md"
  - "! grep -E '^  W1-5:' .dfg/context-map.yaml"

  # ─── W2-1: belief_coefficient.py:142 formula fix ──────────────────────
  - "test -f .dfg/agents/W2-1-belief-coefficient-formula-fix.md"
  - "test -f .dfg/retrospectives/W2/W2-1.md"
  # Positive grep on the fixed line is sufficient — the W2-1 fix
  # comment retains the historical `1.0 - abs(...)` pattern for
  # provenance, so a negative grep would false-positive on the comment.
  - "grep -q '^        tip_confidence = abs(tip - 0.5) \\* 2' python_refactor/src/algorithms/belief_coefficient.py"

  # ─── W2-2: import-style sweep ─────────────────────────────────────────
  - "test -f .dfg/agents/W2-2-import-style-sweep.md"
  - "test -f .dfg/retrospectives/W2/W2-2.md"
  - "test -z \"$(grep -lE '^from algorithms\\.' python_refactor/tests/*.py python_refactor/src/experiments/*.py 2>/dev/null)\""

  # ─── W2-4: pyproject.toml + ruff sweep ────────────────────────────────
  - "test -f .dfg/agents/W2-4-packaging-and-ruff.md"
  - "test -f .dfg/retrospectives/W2/W2-4.md"
  - "test -f pyproject.toml"
  - "grep -q 'name = \"learning-to-anticipate-flexible-choices\"' pyproject.toml"
  - "ruff check --select F401 python_refactor/data_analysis.py python_refactor/ftse_data_downloader.py python_refactor/ftse_data_downloader_v2.py"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W2 = W1 cleanup sweep. All 4 file-disjoint units shipped in 4 PRs after the W2 replan ceremony (PR #9)."
  - "Closed 4 of 5 W1 carries: W1-4-CARRY-1 (W2-1), W1-3-CARRY-1+2 (W2-2), W1-5-CARRY-1 (W2-3), packaging gap (W2-4). W1-4-CARRY-2 (TIP magic numbers) deferred."
  - "Surfaced 3 newly-visible pre-existing bugs (W2-2-CARRY-1/2/3) — same dead-code-coming-alive pattern. Each ≤ 5-line fix when triaged."
  - "Test collection jumped from ~50 to 156 in .venv-w1 after W2-2 import-style fix. 153 pass."

carry_forward:
  - id: W2-2-CARRY-1
    why: "test_calculate_multi_horizon_lambda_rates surfaces AssertionError 1 != 0 (likely off-by-one in horizon iteration)"
    next_action: "Triage in W3"
  - id: W2-2-CARRY-2
    why: "test_enhanced_prediction_bounds_validation: ValueError 'No prediction available for horizon 2' (setup issue)"
    next_action: "Triage in W3"
  - id: W2-2-CARRY-3
    why: "test_correspondence_with_anticipatory_learning: negative value where positive expected (sign-flip likely)"
    next_action: "Triage in W3"
  - id: W2-4-CARRY-1
    why: "~144 F401 errors remain in src/ files (W2-4 scope was the 3 legacy files only)"
    next_action: "Wider ruff sweep unit in W3"
  - id: W2-4-CARRY-2
    why: "No uv.lock — packaging unit deliberately deferred locked installs"
    next_action: "Lockfile unit in W3 (~5 lines)"
---

# W2-gate — WAVE 2 CLOSE: W1 cleanup sweep

W1's follow-up wave. 4 file-disjoint units shipped in 4 PRs after the
W2 replan ceremony (PR #9). Closes 4 of 5 W1 carries + 1 packaging gap.

## What W2 delivered

| Unit | PR | Closes |
|---|---|---|
| W2-3 | #10 | W1-5-CARRY-1 (context-map dangling reference) |
| W2-1 | #11 | W1-4-CARRY-1 (belief_coefficient.py:142 formula bug) |
| W2-2 | #12 | W1-3-CARRY-1+2 (5 test files + thesis_aligned_experiment imports) |
| W2-4 | #13 | Packaging gap (pyproject.toml + 11 ruff F401 fixes) |

## Carries forwarded to W3

5 carries — 3 newly-visible test failures (W2-2-CARRY-1/2/3) +
2 wider-scope items (W2-4-CARRY-1/2). All ≤ small follow-up units.
