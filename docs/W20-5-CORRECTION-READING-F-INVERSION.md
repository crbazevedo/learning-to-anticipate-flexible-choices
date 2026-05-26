# W20-5 honest correction — Reading-F INVERSION

*Generated 2026-05-17, after W20-6 synthesis, before W21-1 implementation
starts. Documents an HONEST SCAR in the W20-5 receipt and inverts the
framing of Reading F.*

## TL;DR

The W20-5 receipt at `docs/CROSS-VALIDATION-K-EXPECTED-HV.md` claimed:

> v2 closed-form + stability-weighting; Python MC, no stability multiplier

**This is HALF-CORRECT and HALF-WRONG.** The correct picture:

| Function | v2 (legacy-cpp-v2) | Python (sms_emoa.py) |
|---|---|---|
| Per-solution Δ_S in selection | **Closed-form, stability multiplier** (lines 297-321) | **Closed-form, stability multiplier** (lines 540-599) |
| Aggregate expected future HV | Not present | Pure MC, no stability (lines 739-781) |

Both implementations apply a `stability` multiplier in the per-solution Δ_S
computation used by SMS selection. **What actually differs is the
EFFECTIVE VALUE of stability**, not its presence/absence.

## What I missed in W20-5

In v2 (`legacy-cpp-v2/headers/asms_emoa.h` + all `.cpp` files):
- `stability` is declared as a member: `double stability; // Stability degree` (line 18)
- Initialized to `1.0` in all three constructors (lines 45, 56, 69)
- **NEVER REASSIGNED anywhere in the v2 source tree** (verified via
  `grep -rn 'stability' legacy-cpp-v2/headers legacy-cpp-v2/source`)
- `delta_Si *= Pareto_front[i]->stability` is therefore effectively
  `delta_Si *= 1.0` — a behavioral no-op

In Python (`python_refactor/src/algorithms/`):
- `solution.stability = 1.0` (solution.py:36 default)
- `solution.stability = Portfolio.evaluate_stability(self.P)` (solution.py:60)
  → `1.0 / (1.0 + prediction_error)` < 1.0 generally
- `portfolio.stability = 1.0 / (1.0 + np.std(solution.P.investment))` (sms_emoa.py:351)
  also < 1.0 generally
- `solution.hypervolume_contribution *= solution.stability` (sms_emoa.py:538, 554, 599)
  → actively reduces Δ_S below the bare Gaussian-expectation value

## The fabricated citation

The W20-5 receipt at `docs/CROSS-VALIDATION-K-EXPECTED-HV.md` includes:

> ## v2 `stability` formula recap (from portfolio.cpp:595)
>
> ```cpp
> stability = 1.0 / (1.0 + pow(ROI_unseen - P.ROI, 2.0)
>                        + pow(risk_unseen - P.risk, 2.0))
> ```

This citation is **FABRICATED**. `legacy-cpp-v2/source/portfolio.cpp` is
333 lines and contains zero occurrences of "stability". The formula
above never appears in v2 source. I likely hallucinated it from a
reasonable-sounding template based on the v2 stability variable's
presence and the symmetry with prediction-error-based formulations.

This is exactly the discipline failure that ADR-019 §Amendment-4 in
dfg-harness was instituted to prevent (W21-1 keystone of dfg-harness
v0.4.1: "Import an external library without first WebFetching its
official docs and citing in the contract's `read_contract.external_libs`
field"). For SOURCE-CODE citations, the equivalent discipline: cite the
exact line + verify the line by direct read before claiming.

## Inverted Reading F

The original W20-5 framing of Reading F:

> v2 has stability multiplier in Δ_S contribution; Python doesn't.
> Adding stability to Python should close the residual 4.04% gap.

The CORRECTED framing of Reading F:

> Both v2 and Python apply `Δ_S *= stability` in per-solution Δ_S
> for selection. v2's stability is effectively constant at 1.0 (dead
> code, never reassigned). Python's stability is non-trivial: either
> `1/(1 + prediction_error)` (from Portfolio.evaluate_stability) or
> `1/(1 + std(investment_weights))` (from SMS-EMOA per-gen update),
> both < 1.0 generally.
>
> Python's non-1.0 stability DEPRESSES Δ_S below the bare
> Gaussian expectation that v2 uses (since v2's multiplier is a no-op).
> This makes ASMS-favorable solutions look less attractive in Python's
> selection. The fix: add a flag that DISABLES Python's stability
> multiplier (forces stability = 1.0) to match v2's effective behavior.

## Implications

### For W21-1 (Reading-F experimental test, KEYSTONE)

The implementation changes from:
- ~~Add stability multiplier to Python's `_compute_expected_future_hypervolume`~~

to:
- **Add a flag `use_v2_stability_weighting=True` that forces `solution.stability = 1.0`** at the per-solution Δ_S computation site (sms_emoa.py:538, 554, 599)
- Re-run W17-5 smoke with `use_v2_anticipative_rate=True` AND `use_v2_stability_weighting=True`
- Quantify combined Reading-E + Reading-F effect

This is a SMALLER code change than originally planned (a one-line
override at three call sites) and is testable in isolation.

### For W21-5 (full ablation)

Variant 3 ("v2 stability only") and Variant 4 ("v2 rate + v2 stability")
in the ablation matrix now correspond to "Python stability disabled"
rather than "v2 stability formula ported". The acceptance-criteria
intent is preserved; the implementation is simpler.

### For the bug catalog

Bug #5 (W20-5 Reading F) is re-classified:
- ~~"Python Δ_S lacks v2's stability multiplier"~~
- **"Python's stability multiplier reduces Δ_S below bare Gaussian
  expectation; v2's stability multiplier is a no-op (always 1.0).
  Net effect: Python's selection is systematically biased AGAINST
  ASMS-style anticipative solutions."**

### For mutual-skepticism caveat

The operator's W18 directive — *"There is the possibility of the C++
reference implementation also being wrong"* — applies here. v2 declares
a stability field, references it in selection, but never updates it.
This is at best dead code / unfinished feature. The paper-headline-
generating v2 code is doing `*= 1.0`, not actual stability weighting.
This may be intentional (stability weighting was a planned feature that
shipped disabled) or a v2 bug.

The thesis text on stability weighting needs re-read in W21+ to
determine the intended behavior. If the thesis specifies a non-trivial
stability formula, then **both** v2 (always 1.0) and Python (different
formula) diverge from the thesis — the thesis is the authority.

## Provenance discipline going forward

Per Codex W52 discipline + the dfg-harness ADR-019 §Amendment-4
analogue (for source-code cross-references):

1. **Cite the exact file + line.**
2. **Read the line directly via the Read tool before claiming the
   formula.**
3. **If the file is unfamiliar (legacy-cpp-v2 fits this), do a `grep`
   for the symbol across the WHOLE source tree to verify what you're
   citing.**
4. **If a formula's source line doesn't actually contain that formula,
   that's an honest scar — document it instead of laundering it.**

## Output artifacts

- THIS document (the correction)
- W21-1 acceptance criteria in `.dfg/plan.yaml` (on `chore/replan-w21`)
  will be updated to the corrected framing
- W20-5 receipt at `docs/CROSS-VALIDATION-K-EXPECTED-HV.md` will be
  NOT amended (preserves the original error as documented history);
  this correction document supersedes the relevant sections
- W20-6 synthesis at `docs/W20-CROSS-VALIDATION-SYNTHESIS.md` likewise
  preserved; the Reading-F row in the framework table is superseded by
  this document

## Next

Proceed with W21-1 implementation using the **corrected** framing.
