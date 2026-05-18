---
unit: W21-1-revision
wave: W21
template: ADR-004-four-question
schema_version: 1.0.0
what_worked: |
  Mid-wave honest revision of the W21-1 smoke results based on
  Phase B's 10-seed data. The discipline that "n=2 is preliminary"
  was upheld by extending to n=10 and surfacing the magnitude shift
  transparently rather than letting the n=2 conclusion stand.

  Revision document at docs/W21-1-HONEST-REVISION-N10-RESULTS.md
  preserves the W21-1 receipt as documented history while
  superseding the magnitude claims.
what_broke: |
  The W21-1 receipt's "+6.91pp Reading-E recovery DOMINANT" claim
  was OVERSTATED ~3.5×. The actual n=10 recovery is +1.98pp.

  Mechanically: n=2 (seeds 1+2) captured a downside outlier for
  default ASMS AND a relatively favorable v2_rate, compounding to
  inflate the apparent delta. Both biases were silent — neither
  was flagged at W21-1 commit time because the n=2 std (1.33e-06
  for default, 1.92e-05 for v2_rate) wasn't framed as "insufficient
  for verdict" but as "high variance noted".

  The W21-1 reading framework / publication-track-decision flow
  was built around the n=2 magnitudes and now requires re-baseline.
what_youd_change: |
  Going forward: ANY smoke at n<5 should be labeled "DIRECTIONAL
  only" and a higher-n confirmation flagged REQUIRED before the
  result feeds the publication-track decision. The W21-1 receipt
  did label its result as "PARTIAL" (acknowledging the gap wasn't
  fully closed), but it did NOT flag the magnitude estimate as
  insufficient — implying readers could trust the +6.91pp number.

  Concrete change for W21-5: Phase B's n=10 verdict will be the
  baseline for any "what's the residual gap?" reasoning. W21-5
  Phase C (30 seeds) becomes MORE important — the Reading-E effect
  at n=10 is small enough that combined-Reading effects from V5/V6/V7
  may dominate.

  W21-6 synthesis will need updating with the revised Reading-E
  magnitude when Phase B completes.
assumption_to_challenge: |
  Assumed "n=2 with both seeds independently dispatched is good
  enough to see directional effects, full statistical confidence
  comes from n=30 later". CHALLENGED: n=2 doesn't even reliably
  indicate the SIGN of the effect, let alone the magnitude. The
  W21-1 n=2 sample size was inappropriate for the magnitudes we
  were trying to detect (~5-10pp effects against ~15-20% noise).

  Deeper assumption: that high MC noise (n=200 MC samples per
  period) is the only thing that needs averaging. Phase B reveals
  per-seed variance is also material — the SMS_RDM_K0 baseline
  has ~16% range across 10 seeds, much wider than n=2 captured.

  Closes (operator directive 2026-05-17): "Following harness
  discipline + max parallelism", this is the documented honest
  revision before any publication-track decision is made.

  Carries:
  - W21-1-revision-CARRY-1: Update W21-6 synthesis with n=10
    magnitudes when Phase B completes
  - W21-1-revision-CARRY-2: ALL future smokes should specify
    minimum-meaningful-n in their protocol (e.g. n=10 for direction,
    n=30 for magnitude)
---

# W21-1-revision — n=2 overstated Reading-E recovery by 3.5×

Mid-wave honest revision. Phase B's 10-seed data shows Reading-E
recovery is +1.98pp (not +6.91pp as W21-1's n=2 reported).
Direction preserved; magnitude refuted. W21-1 receipt + retro
preserved as documented history; this revision supersedes the
headline claims.
