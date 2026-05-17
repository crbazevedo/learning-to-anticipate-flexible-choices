---
id: W16-3
role: code-fixer
name: Preserve rank-1 extrema (anchor) points in SMS-EMOA reduce step (BACKLOG H6)
purpose: "Closes BACKLOG H6. Make rank-1 best-ROI and best-risk anchor solutions immune to SMS-EMOA's reduce step so the HV bounding box doesn't shrink over generations per standard SMS-EMOA / NSGA-II practice."
wave: W16
unit: W16-3
depends_on: []
blocks: [W16-5]
governance_tier: VT1
sized: S
hardening_max_cycles: 1
prompt_version: 1
read_contract:
  must_read:
    # Grounding details (pages, excerpts, reasons) in contract body
    # below per BACKLOG §6 (schema requires plain-string list here).
    - docs/BACKLOG.md
    - docs/Azevedo_CarlosRenatoBelo_D.pdf
    - python_refactor/src/algorithms/sms_emoa.py
output_contract:
  files:
    - python_refactor/src/algorithms/sms_emoa.py
    - python_refactor/tests/test_extrema_preservation.py
  branch_name: feat/w16-3-extrema-preservation
  acceptance: >
    SMS-EMOA reduce step identifies rank-1 anchor solutions (max-ROI
    and min-risk) and protects them from removal. Per-generation
    [ROI_max, ROI_min, risk_max, risk_min] trace exported. ≥ 3
    regression tests covering the protection invariant and the trace.
dispatch_instructions: |
  Closes BACKLOG: H6.

  Background. SMS-EMOA's standard reduce step picks the worst-HV
  contributor in the worst non-dominated rank for removal. This can
  prune extrema if they happen to be in a crowded rank or if their
  HV contribution is locally small. Over many generations the Pareto
  front extent shrinks; the HV indicator (which is most sensitive to
  bounding-box corners) becomes unstable.

  Surgical changes:

  1. In sms_emoa._reduce_population (or wherever the worst-contributor
     selection lives), identify the rank-1 anchor solutions as the
     argmax(ROI) and argmin(risk) within the current rank-1 set. Mark
     them as protected (e.g., via a boolean flag or via a "skip-these-
     indices" set passed to the worst-contributor selector).

  2. If reduce wants to evict an anchor, evict the next-worst non-anchor
     instead. Handle the edge case where rank-1 has < 3 solutions
     (rare but possible).

  3. Add a per-generation [ROI_max, ROI_min, risk_max, risk_min] trace
     to metrics for downstream diagnosis.

  4. Tests:
     - Synthetic 10-portfolio rank-1 set with known extrema; verify
       reduce never evicts the argmax(ROI) or argmin(risk).
     - Stress: 100 reduce steps with anchor at borderline HV contribution;
       verify anchor survives all 100.
     - Trace: verify per-gen extrema-trace populated and monotone in
       expected direction (front extent non-decreasing in best-case).

  What NOT to do:
    - Don't touch the operators (W15-2).
    - Don't touch anticipatory_learning.py (W16-1).
    - Don't touch txn costs (W16-2).
    - Don't replace the SMS-EMOA reduce step algorithm — only add
      anchor protection on top of the existing logic.

  PR body MUST echo the relevant thesis + REF excerpts per BACKLOG §6.
---

# W16-3 — Preserve rank-1 extrema in SMS-EMOA reduce

Closes BACKLOG.md items: **H6**.

## Thesis grounding

**§3.4.2 "S-Metric Selection Evolutionary Algorithm (SMS-EMOA)",
pp. 69-70** — describes the base SMS-EMOA reduce step (Beume et al.
2007 [REF 32]).

**REF [Beume, Naujoks, Emmerich 2007] EJOR 181(3):1653-1669** —
original SMS-EMOA paper; standard practice preserves extrema because
the HV indicator is most sensitive to anchor solutions which define
the bounding box.

**STD** — standard SMS-EMOA / NSGA-II implementations explicitly
preserve rank-1 anchors. The thesis inherits this convention
implicitly.

## Why this contributes to closing the 9% gap

If extrema get pruned mid-generation, the HV indicator becomes
non-monotone over time and the optimizer's selection pressure
oscillates. This shows up empirically as:
- Front extent shrinking over generations
- HV trajectory non-monotone (sawtooth in per-gen HV plot)
- Anticipatory rule trying to anticipate a moving target whose
  bounding box is itself collapsing

For the OOS-Future-HV metric, an unstable bounding box means we are
predicting a moving target, which biases the predicted-vs-realized
delta in unpredictable ways.

## Files to touch

- `python_refactor/src/algorithms/sms_emoa.py` — augment _reduce_population
  with anchor protection + per-gen extrema trace.
- `python_refactor/tests/test_extrema_preservation.py` — NEW; ≥ 3
  regression tests.

## Acceptance

- argmax(ROI) and argmin(risk) within rank-1 are protected from reduce
- Per-generation [ROI_max, ROI_min, risk_max, risk_min] trace
- ≥ 3 regression tests
- Edge case: rank-1 with < 3 solutions handled
