# W22 Probe AI — Pareto front size sensitivity for AMFC

**Status:** SHIPPED (11/11 tests)
**Linked:** W22 Inspection 6 TEST 4 (AMFC selects 'least crowded' / boundary)
**Hypothesis:** AMFC pick behavior depends on |P_t|. Boundary candidates have
structural advantage in the contribution formula; on large fronts AMFC
degenerates to boundary selection.

---

## Why this matters

Inspection 6 TEST 2 found that on synthetic Pareto fronts of size 7,
AMFC picks the rightmost (highest-ROI) boundary 100% of the time —
because the boundary contribution `(ROI_i − ROI_{i−1}) · (R2 − risk_i)`
benefits from the large (R2 − smallest_risk) gap.

Probe AI generalizes: **how does this boundary bias scale with |P_t|?**

---

## Synthetic result (n_runs=500 per size)

```
| |P_t| | most-common winner | boundary % | middle % |
|---|---|---|---|
| 2  | pos 1  | 100.0% | 0.0%  |
| 3  | pos 2  | 100.0% | 0.0%  |
| 5  | pos 4  | 100.0% | 0.0%  |
| 7  | pos 6  | 100.0% | 0.0%  |
| 10 | pos 9  | 100.0% | 0.0%  |
| 20 | pos 19 | 99.4%  | 0.6%  |
| 50 | pos 49 | 91.6%  | 8.4%  |

**Verdict:** at |P_t|=50, 92% of picks at boundaries → STRONG boundary bias.
```

**Findings:**
1. For |P_t| ≤ 10, the highest-ROI boundary wins 100% of the time on these
   synthetic fronts. Middle candidates are NEVER picked.
2. For |P_t|=20, 99.4% boundary; for |P_t|=50, 91.6% boundary. Middle picks
   only become non-trivial at very large front sizes.
3. This **confirms Inspection 6's boundary-bias finding** at scale: AMFC's
   current-Δ_S argmax structurally favors boundaries, regardless of how
   "rich" the middle of the front is.

---

## What this means for production AMFC

The NC30-v2 analytical mode (DEFAULT, commit 03a3956) uses fixed
mean-sorted positions. The fixed-position contribution formula has the
same boundary advantage as the deterministic Hv-DM. So:

- On real FTSE data with |P_t| ≈ 5-15, AMFC will likely concentrate
  picks at the high-ROI boundary (per Inspection 6 + Probe AI).
- The forecast-variance discount (NC30 c, opt-in) might partially counter
  this if boundary candidates have high forecast uncertainty.
- The tie-break by variance (NC30 d) only fires on ties, which don't
  happen here (boundary wins by a wide margin in expectation).

**Operator action items**:
1. Run AMFC on FTSE n=10 with telemetry (Probe AA) enabled.
2. Measure the empirical pick-position distribution.
3. If boundary % ≥ 90%, the AMFC selection is essentially "highest-ROI
   solution on the current Pareto front" — which is already what Hv-DM
   does with current-Δ_S. AMFC's value is then mostly from the **forward
   forecast** changing the mean ROI values, not from the position argmax.

---

## Honest scars

- **Synthetic fronts have uniform spacing**: real Pareto fronts may have
  non-uniform density (clusters in middle). This could mitigate the
  boundary bias on real data.
- **Fixed R1/R2** in this analysis: under data-derived z_ref (NC30 b
  default), R1/R2 ARE the boundary positions, so boundary contribution
  → 0. This significantly changes the picture: with data-derived z_ref,
  the boundary advantage may VANISH because R1 = min ROI → boundary
  contribution = (boundary_ROI - boundary_ROI) * (R2 - risk) = 0.
- This probe uses HARDCODED R1=0, R2=0.05. A follow-up probe should
  characterize the joint effect of NC30 b (data-derived z_ref) + front
  size.

---

## Test coverage

11/11 in `tests/test_probe_ai_front_size_sensitivity.py`:
- synthetic_front shape + sorted-by-ROI
- front_contribution boundary formulas (single, 2-element)
- pick_distribution shape + sum invariant
- boundary_concentration in {all, none, half} regimes
- front_size_sweep returns dict
- large-front (n=20) boundary concentration > 0.8
- analyze returns markdown with verdict

---

## Linkage

- **Inspection 6 TEST 2**: original 100% boundary finding at |P_t|=7
- **NC30 b** (`03a3956`): data-derived z_ref may invalidate the boundary
  advantage on real data; warrants follow-up probe
- **Probe AA**: collect production pick-position distribution to compare
  with synthetic predictions
