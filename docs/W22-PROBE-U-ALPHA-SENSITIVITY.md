# W22 Probe U — NC30 c variance_penalty α sensitivity

**Status:** SHIPPED (standalone analyzer + tests + this doc)
**Linked NC:** NC30 c (commit `300fedc`) — continuous variance-aware AMFC contribution
**Hypothesis:** optimal α (variance penalty coefficient) varies across regimes;
characterize how AMFC's argmax changes as α sweeps from 0 (no penalty) to
large values (variance dominates).

---

## Why this matters

NC30 c added a continuous variance penalty: `effective_contribution[i] =
expected_contribution[i] - α · trace(Σ_h[i])`. With α = 0 (default), behavior
is unchanged. With α > 0, candidates with high forecast variance are
penalized continuously across all argmax decisions, not just at ties.

The question Probe U answers: **how does the AMFC pick change as α sweeps,
and where are the flip points?** This characterizes the trade-off
between forecast quality and forecast certainty.

---

## Methodology

Pure-synthetic analyzer in `src/probes/probe_u_alpha_sensitivity.py`:

- `compute_effective_contribution(expected, variances, alpha)` — replicates
  NC30 c formula
- `argmax_under_alpha(expected, variances, alpha)` — returns the winner
- `sweep_alpha(...)` — maps each α to its winner
- `find_flip_points(...)` — uses a logspace grid α ∈ [10⁻³, 10³] to detect
  where the argmax changes
- `analyze_alpha_tradeoff(...)` — generates the markdown table below

---

## Synthetic example

5 candidates with varied (E[Δ_S], trace(Σ)) profiles, illustrating the
trade-off:

```
expected   = [0.0003, 0.0005, 0.0008, 0.0010, 0.0015]
variances  = [0.0010, 0.0050, 0.0001, 0.0100, 0.0005]
```

Candidate 4 has the highest expected contribution (1.5e-3) but moderate
variance (5e-4). Candidate 2 has slightly lower contribution (8e-4) with
much lower variance (1e-4).

| α | argmax idx | E[Δ_S] of pick | trace(Σ) of pick | effective_contrib |
|---|---|---|---|---|
| 0.0 | 4 | 1.5000e-03 | 5.0000e-04 | 1.5000e-03 |
| 0.001 | 4 | 1.5000e-03 | 5.0000e-04 | 1.4995e-03 |
| 0.01 | 4 | 1.5000e-03 | 5.0000e-04 | 1.4950e-03 |
| 0.1 | 4 | 1.5000e-03 | 5.0000e-04 | 1.4500e-03 |
| 1.0 | 4 | 1.5000e-03 | 5.0000e-04 | 1.0000e-03 |
| 10.0 | 2 | 8.0000e-04 | 1.0000e-04 | -2.0000e-04 |
| 100.0 | 2 | 8.0000e-04 | 1.0000e-04 | -9.2000e-03 |

**Flip point:** α ≈ 1.80 → argmax switches from candidate 4 to candidate 2.

For α below the flip, the high-mean-moderate-σ candidate wins. Above, the
moderate-mean-low-σ candidate wins. The flip α is a function of the
candidate-pair's (Δmean, Δvariance) ratio.

---

## Success criteria

A meaningful α for production AMFC must:
1. Be small enough that high-expected-contribution candidates aren't
   prematurely overruled
2. Be large enough that high-variance candidates get DOWNWEIGHTED noticeably
3. Be stable across regimes (FTSE / NASDAQ / HangSeng)

**Operator action items** (out of probe scope):
- Run AMFC with α ∈ {0, 0.01, 0.1, 1.0, 10.0} on FTSE n=10
- Measure final wealth + Sharpe per α
- Pick the α that maximizes wealth gain over Hv-DM baseline

---

## Honest scars

- This is a PURE SYNTHETIC analyzer — does not run ASMS. The flip point
  on real FTSE data may differ because the (expected, variance) joint
  distribution of solutions is regime-specific.
- The variance penalty competes with the natural variance-conditioning
  from NC30-v2's analytical mode. If analytical=True (default), the
  expected_contributions already factor in mean positions, and the
  variance penalty acts as an INDEPENDENT regularizer rather than as a
  forecast-aware adjustment.
- Large α (≥10) drives effective_contribution NEGATIVE for most candidates.
  This is fine for argmax (still picks the least-negative) but signals
  that the penalty is over-dominating; suggests α ∈ [0.01, 1] is the
  meaningful operating range.

---

## Test coverage

9/9 tests passing in `tests/test_probe_u_alpha_sensitivity.py`:
- Zero-α identity
- Argmax at zero α matches argmax(expected)
- Argmax at large α picks low-variance candidate
- Sweep returns dict keyed by α
- Flip-point detection on a clear-flip scenario
- No-flip detection when one candidate dominates both dimensions
- Markdown formatting + custom α list

---

## Linkage

- **NC30 c** (`300fedc`): implementation
- **Probe T**: γ^h discount sensitivity — same analyzer pattern as U, but for NC29a
- **Probe AA**: AMFC telemetry harness — can feed AMFC selector telemetry into
  this analyzer for per-period α-flip characterization on real data
