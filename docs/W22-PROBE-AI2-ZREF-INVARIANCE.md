# W22 Probe AI-2 — z_ref data-derivation × boundary bias interaction

**Status:** SHIPPED (8/8 tests)
**Linked:** Probe AI (commit `97bb397`) honest scar + NC30 b (commit `03a3956`)
**Hypothesis (initial):** NC30 b's data-derived z_ref (R1=min ROI, R2=max risk)
should eliminate the boundary-bias degeneracy that Probe AI found at 91-100%
boundary picks.

**ACTUAL FINDING:** NC30 b's effect is **ASYMMETRIC**. It kills the LEFT
boundary's contribution but the RIGHT boundary STILL wins. Boundary
degeneracy is reduced from "both boundaries" to "only the high-ROI
boundary."

---

## Synthetic result (n_runs=300 per size)

```
| |P_t| | fixed-zref boundary % | derived-zref boundary % | argmax disagreement |
|---|---|---|---|
| 3  | 100.0% | 100.0% | 0.0% |
| 5  | 100.0% | 100.0% | 0.0% |
| 10 | 100.0% | 100.0% | 0.0% |
| 20 | 99.3%  | 99.3%  | 0.3% |
```

**Verdict:** NC30 b has ASYMMETRIC effect:

- **LEFT boundary (pos 0)**: fixed N% → derived 0% (R1 = min ROI → leftmost
  contribution = 0 → left can never win)
- **RIGHT boundary (pos -1)**: fixed N% → derived ~N% (R2 = max risk
  remains; (R2 - smallest_risk) = full risk range → right still dominates)
- **Total boundary %**: unchanged (~100%)
- **argmax disagreement**: minimal (~0-1%)

---

## Why NC30 b doesn't fix the whole problem

For position 0 (leftmost) under derived z_ref:
  contribution = (ROI_0 - R1) * (R2 - risk_0) = 0 * (R2 - risk_0) = 0
Because R1 = min ROI = ROI_0 by construction.

For position n-1 (rightmost) under derived z_ref:
  contribution = (ROI_{n-1} - ROI_{n-2}) * (R2 - risk_{n-1})
With R2 = max risk = risk_0 (highest-risk solution), and risk_{n-1} = min risk:
  (R2 - risk_{n-1}) = max_risk - min_risk = FULL RISK RANGE

So the rightmost boundary's contribution is `small ROI gap × full risk range`,
which is still a LARGE value compared to middle solutions whose
contributions involve small differences in both axes.

---

## Implication for production AMFC

Probe AI showed AMFC picks the high-ROI boundary 91-100% on synthetic.
Probe AI-2 shows that NC30 b's data-derived z_ref does NOT fix this.

**To eliminate the high-ROI boundary bias too**, one would need:
- R2 = some tighter risk reference (e.g., median risk, not max)
- But that breaks the geometric meaning of R2 ("risk above which we don't care")
- Alternative: scale the contribution by some factor that down-weights
  large risk ranges (e.g., divide by full risk range)

**Alternative interpretation:** maybe boundary preference is CORRECT —
the high-ROI solution IS the most economically attractive. AMFC's job is
to find that and pick it. If that's the design intent, then Probe AI's
"boundary bias" is a feature, not a bug.

The question for the operator: **is high-ROI boundary preference desirable?**
- If YES: NC30 b is fine; current behavior is intended
- If NO: need a different z_ref scheme (e.g., quantile-based) or a different
  contribution formula

---

## Honest scars

- **Synthetic uniform-spacing**: real Pareto fronts may have clusters in
  the middle that change the dynamics. Need empirical evidence.
- **Did NOT test**: combinations with NC30 c (variance penalty α > 0) which
  might shift the equilibrium away from boundary preference if boundary
  solutions have high forecast variance.
- **The R2 margin parameter** could tighten R2 if positive (R2 = max_risk -
  margin · risk_range). Untested; possibly mitigates right-boundary bias.

---

## Test coverage

8/8 tests in `tests/test_probe_ai2_zref_boundary_invariance.py`:
- derive_zref returns (min ROI, max risk)
- derive_zref margin widens correctly
- winners_under_zref_modes returns all required keys
- counts sum to n_runs
- **derived_zref_nulls_left_boundary** (key finding: LEFT win count → 0)
- **derived_zref_does_not_reduce_total_boundary_bias** (HONEST SCAR pinned:
  RIGHT boundary still wins ≥80%)
- sweep returns dict keyed by size
- analyze returns markdown

---

## Linkage

- **Probe AI** (`97bb397`): original boundary-bias finding (this probe's premise)
- **NC30 b** (`03a3956`): data-derived z_ref implementation
- **Probe U** (`1871489`): NC30 c α sensitivity; could potentially counter
  right-boundary preference if α high AND boundary solutions are uncertain
- **Inspection 6**: original AMFC soundness inspection
