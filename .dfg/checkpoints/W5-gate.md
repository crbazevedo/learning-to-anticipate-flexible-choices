---
wave: W5
gate_type: wave-gate
verdict: PASS
date: 2026-05-17
units_completed: [W5-1, W5-2]
units_carry_forward: []
verify:
  # ─── Wave-shipped via PRs #29, #30 ────────────────────────────────────
  - "git log --oneline master | grep -q 'W5-1.*determinism'"
  - "git log --oneline master | grep -q 'W5-2.*covariance'"

  # ─── W5-1: correspondence test RNG seed ───────────────────────────────
  - "test -f .dfg/agents/W5-1-correspondence-test-rng-seed.md"
  - "test -f .dfg/retrospectives/W5/W5-1.md"
  - "grep -q 'np.random.seed' python_refactor/tests/test_correspondence_integration.py"
  - "grep -q 'similarity_threshold=0.5' python_refactor/tests/test_correspondence_integration.py"

  # ─── W5-2: covariance threading per paper Eq (15) ─────────────────────
  - "test -f .dfg/agents/W5-2-multihorizon-covariance-threading.md"
  - "test -f .dfg/retrospectives/W5/W5-2.md"
  - "grep -q 'paper Eq (15)' python_refactor/src/algorithms/multi_horizon_anticipatory.py"
  - "grep -q 'combined_cov' python_refactor/src/algorithms/multi_horizon_anticipatory.py"
  - "grep -q TestW5_2_CovarianceThreading python_refactor/tests/test_multi_horizon_anticipatory.py"

  # ─── Substrate health ─────────────────────────────────────────────────
  - "uv run --project /Users/crbazevedo/Documents/Korza/repos/dfg-harness dfg validate"

notes:
  - "W5 = test determinism + covariance threading. 2 file-disjoint units, 1 parallel group."
  - "Closes W3-5-CARRY-1 (test non-determinism) + W4-2-CARRY-1 (mean-only override). MultiHorizon.learn_population now drives FULL Gaussian Eq (14)+(15) convex combo."
  - "W5-2 surfaced + fixed a pre-existing dead-code-coming-alive bug: _generate_predicted_solution:250 had Q=kalman_state.Q referencing a non-existent attribute. 5th instance of this pattern across W1-W5. The class IS now codified in the test surface (any new test that constructs a KF-bearing solution will catch latent KF-API drift)."
  - "All W1-W5 carries are now CLOSED. No open carry-forwards from W5."

carry_forward: []
---

# W5-gate — WAVE 5 CLOSE: test determinism + covariance threading

## What W5 delivered

| Unit | PR | Closed |
|---|---|---|
| W5-1 | #29 | W3-5-CARRY-1 (correspondence test now deterministic 5/5 runs) |
| W5-2 | #30 | W4-2-CARRY-1 (paper Eq 15 covariance threaded) + pre-existing Q-kwarg dead-code bug |

Plus PR #28 (replan W5-add) at the start of the wave.

## Carries forwarded to W6

**NONE.** All W1-W5 carries are CLOSED.
