# W22 Probe T — NC29a γ sensitivity sweep

**Branch:** `feat/w22-inspection-backlog`
**Code:** `python_refactor/src/probes/probe_t_gamma_sensitivity.py`
**Tests:** `python_refactor/tests/test_probe_t_gamma_sensitivity.py` (6/6 pass)
**Dependencies:** NC29a (commit `f42cc9d`) — geometric γ^h discount in λ^H.

---

## Hypothesis

NC29a replaced the flat `1/(H-1)` prefactor in `λ^H` with a geometric
`γ^h` discount, tunable via `W22_NC29A_GAMMA` (default 0.9). The fix
matches the RL/DP standard discount and gives near-term horizons more
weight (where KF predictions are most reliable). What it does *not* fix
is the question: **is γ=0.9 actually the right setting?**

> **Hypothesis (H-T):** The optimal γ is regime-dependent. In high-confidence
> regimes (low TIP entropy) a larger γ extends the effective look-ahead
> horizon and increases cumulative anticipation weight; in low-confidence
> regimes (TIP saturated near 0.5) every γ collapses to zero weight
> regardless. There is a sweet spot γ* per regime that balances effective
> horizon against total weight.

## Methodology

Pure synthetic, closed-form. **No ASMS run required.** We replicate the
NC29a formula directly

    λ^H_h = clamp(γ^h · (1 - H(TIP_h)),  0,  0.5)        h = 1..H

where `H(p)` is the binary entropy in bits and the `0.5` cap is the same
safety clamp that ships in `calculate_multi_horizon_lambda_rates`. The γ
input itself is clamped to `(0.01, 0.999)` to match production bounds.

Four functions are exposed:

| function | purpose |
|---|---|
| `compute_lambda_h_profile(γ, H, tip)` | per-horizon λ^H for one γ + fixed TIP |
| `sweep_gamma(γs, H, tip_schedule=None)` | dict mapping γ → profile (per-h TIP optional) |
| `effective_horizon(profile, threshold=0.01)` | last h where λ^H_h > threshold |
| `cumulative_anticipation_weight(profile)` | Σ λ^H_h (total weight to predictions) |
| `analyze_gamma_tradeoff(γs, H)` | markdown summary across all γs |

## Success criteria

Pick γ* that maximizes a balance metric. Candidate (proposed in this
probe):

> **`balance(γ) := effective_horizon(γ) × cumulative_anticipation_weight(γ)`**

Rationale: a γ with high cumulative weight but `eh = 1` is operating like
a single-step lookahead (wasting the H-horizon machinery); a γ with long
`eh` but each weight near zero is contributing no signal. The product
penalizes both degeneracies.

A future probe (Probe T-extension) could (a) replace the product with an
asymmetric utility (penalize over-weighting far horizons more than
under-weighting near ones), (b) add a per-regime sweep using realized
TIP schedules from `.dfg/trace` instead of a constant TIP, and (c) score
γ* on a downstream metric (Hv on a synthetic problem) rather than this
proxy.

## Synthetic-run summary (TIP=0.5 default, max entropy)

Generated at write-time by

```python
from src.probes.probe_t_gamma_sensitivity import analyze_gamma_tradeoff
print(analyze_gamma_tradeoff([0.5, 0.7, 0.9, 0.99], H=5))
```

## γ sensitivity sweep (H=5, TIP=0.5 constant)

| γ | effective_horizon (λ^H>0.01) | cumulative_weight (Σ λ^H) | balance (eh × Σ) |
|---|---|---|---|
| 0.5 | 0 | 0.0000 | 0.0000 ← γ* |
| 0.7 | 0 | 0.0000 | 0.0000 |
| 0.9 | 0 | 0.0000 | 0.0000 |
| 0.99 | 0 | 0.0000 | 0.0000 |

**Candidate γ* under balance = effective_horizon × cumulative_weight: γ=0.5** (eh=0, Σ=0.0000, balance=0.0000)

### Per-horizon λ^H profiles

| h | γ=0.5 | γ=0.7 | γ=0.9 | γ=0.99 |
|---|---|---|---|---|
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Honest finding from the default-TIP sweep

**The contract's default `tip_value=0.5` is the maximum-entropy case** —
`H(0.5) = 1.0 bit` → `(1 - H) = 0` → λ^H_h ≡ 0 for every γ and every h.
The "γ* = 0.5" tie-break above is meaningless (all four γs tie at 0).
This is itself the strongest validation receipt for Inspection 5's
original finding: **when TIP saturates near 0.5, the discount doesn't
discount because there is nothing to discount.** γ choice is irrelevant
in this regime; what dominates is TIP shape.

## Realistic-regime sweep (TIP=0.1 constant, post-NC13a clamp)

NC13a clamps TIP into `[0.05, 0.95]`. After clamping, a "confident"
posterior typically sits near 0.1 (or 0.9). Re-running with `TIP=0.1`
(via `sweep_gamma(γs, H, tip_schedule=[0.1] * H)`):

| γ | profile (h=1..5)                                  | eh | Σ λ^H |
|---|---------------------------------------------------|----|-------|
| 0.5  | [0.2655, 0.1328, 0.0664, 0.0332, 0.0166]    | 5 | 0.5144 |
| 0.7  | [0.3717, 0.2602, 0.1821, 0.1275, 0.0892]    | 5 | 1.0308 |
| 0.9  | [0.4779, 0.4301, 0.3871, 0.3484, 0.3136]    | 5 | 1.9571 |
| 0.99 | [0.5000, 0.5000, 0.5000, 0.5000, 0.5000]    | 5 | 2.5000 |

balance(γ=0.5) = 5 × 0.5144 = **2.5720**
balance(γ=0.7) = 5 × 1.0308 = **5.1540**
balance(γ=0.9) = 5 × 1.9571 = **9.7855**
balance(γ=0.99) = 5 × 2.5000 = **12.5000**

Under the proxy metric, **γ=0.99 wins** in the confident regime — but
the entire profile is at the 0.5 clamp, which means the γ=0.99 case is
*indistinguishable from constant λ^H_h = 0.5 for every h*. The clamp
floors out the γ signal, exactly the failure mode that motivated NC29a
in the first place. γ=0.9 gives a visible decay shape (0.48 → 0.31 over
5 horizons) while still preserving substantial near-horizon weight.

> **Operational read of the synthetic sweep:**
> - **TIP saturated (≈ 0.5):** γ is irrelevant; everything is zero. Fix
>   is upstream — sharpen TIP, not tune γ.
> - **TIP confident + γ ≥ 0.99:** clamp dominates; γ signal vanishes. Not
>   a useful operating point.
> - **TIP confident + γ ∈ [0.7, 0.9]:** discount shape is visible and
>   preserves cumulative weight. **The default γ=0.9 sits in the right
>   band of this proxy analysis.**

## Next steps

1. Replace constant-TIP synthetic sweep with a TIP schedule extracted
   from a real ASMS run (`.dfg/trace/lambda_h_trace.csv`).
2. Score γ* on a downstream Hv proxy (Probe T-extension).
3. If the realized-TIP sweep confirms γ=0.9 lies in the right band,
   close the question and pin γ=0.9 as canon in the NC29a contract.
4. If a regime-dependent optimum appears, draft a contract for an
   adaptive γ(TIP) scheduler (γ shrinks as TIP approaches 0.5).
