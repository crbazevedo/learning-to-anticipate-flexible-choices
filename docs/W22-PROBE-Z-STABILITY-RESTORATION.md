# W22 Probe Z — stability factor restoration (Reading-F revisited)

**Status:** SHIPPED (standalone analyzer + 12/12 tests + this doc)
**Linked:** SMS-EMOA `use_v2_stability_weighting` kwarg (default False; legacy stability factor active by default)
**Hypothesis:** Reading-F's flip from per-solution `stability_factor = 1/(1 + trace(P))`
to `stability_factor = 1.0` lost a load-bearing signal. Quantify the disagreement
rate between the two modes on synthetic populations.

---

## Why this matters

The Reading-F W21-1 change introduced `use_v2_stability_weighting=True` mode
which forces stability_factor = 1.0, ignoring per-solution KF uncertainty.
Inspection 6 NC30 c flagged this as a potential degeneracy: high-KF-trace
solutions should be DOWN-weighted, but v2 mode skips that.

In production, the DEFAULT is `use_v2_stability_weighting=False` (legacy
stability factor IS applied). v2 is opt-in.

This probe characterizes: **how often does the two modes' argmax disagree
on synthetic populations?**

---

## Synthetic result

```
## Stability factor (Reading-F) sensitivity

n_runs per spread: 300
population size: 8

| trace(P) spread | legacy ≠ v2 disagreement rate |
|---|---|
| 0.00 | 0.00% |
| 0.10 | 16.33% |
| 0.50 | 46.67% |
| 1.00 | 51.00% |
| 2.00 | 60.33% |
| 5.00 | 68.33% |

**Verdict:** stability factor is LOAD-BEARING in high-trace-spread regimes.
```

At realistic trace spreads (≥ 0.5), the two modes pick DIFFERENT solutions
in ~50% of synthetic populations. This is a STRUCTURAL difference, not noise:
the stability factor IS load-bearing.

The verdict for production:
- **Legacy mode** (default) DOES use stability — and it does materially affect picks
- **v2 mode** (opt-in) loses this signal
- Reading-F's experimental claim was that v2 produced better results despite
  the lost signal. This probe confirms the signal isn't trivial; if Reading-F
  improvements were real, they must have COMPENSATED for losing this.

---

## Success criteria

A meaningful operator decision around stability_factor needs to:
1. Run FTSE 2015 n≥10 with `use_v2_stability_weighting=False` (default)
2. Run the same n≥10 with `use_v2_stability_weighting=True`
3. Compare final wealth, Sharpe, max-DD
4. Ratify the better mode as production default

This probe provides the **a priori reason to expect non-trivial difference**
(50%+ argmax disagreement at realistic trace spread). The empirical
production answer requires the FTSE benchmark run.

---

## Honest scars

- **Synthetic ≠ production**: 50% argmax disagreement on synthetic doesn't
  mean 50% wealth difference. Wealth depends on which specific picks differ
  and what their realized returns are.
- **Trace(P) distribution unknown for production**: the spread of trace(P) on
  FTSE/NASDAQ depends on KF tuning. Probe AC (KF NIS/NEES diagnostics) is the
  natural companion — characterize production trace(P) first, then re-run
  Probe Z with that empirical spread.
- **Reading-F may have been right**: it's possible v2 mode wins despite (or
  because of) ignoring stability. Probe Z doesn't claim legacy is better —
  only that the modes are NOT equivalent.
- **NC30 c is the smoother fix**: instead of binary on/off stability_factor,
  NC30 c provides continuous variance_penalty α. Probe U sweeps α; Probe Z
  is the legacy-mode equivalent.

---

## Test coverage

12/12 tests passing in `tests/test_probe_z_stability_restoration.py`:
- stability_legacy zero-trace identity + monotonic decreasing
- stability_v2 constant 1.0
- apply_stability_legacy down-weights high-trace
- apply_stability_v2 is no-op
- argmax disagreement on engineered cases
- argmax agreement when all traces equal
- disagreement_rate sweep against trace spread
- markdown summary format

---

## Linkage

- **Reading-F (W21-1)**: original change introducing v2 mode
- **Inspection 6**: NC30 c suggestion to restore stability or replace with NIS gate
- **NC30 c** (`300fedc`): smoother continuous variance penalty for AMFC (analogous fix)
- **Probe U**: NC30 c α sensitivity — operator action on AMFC variance penalty
- **Probe AC** (agent-shipped): KF NIS/NEES — measures production trace(P) distribution
