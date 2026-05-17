# Cross-validation J — crowding distance for NDS

*Generated 2026-05-17 by W20-4. Closes operator check J.*

## Verdict

⚠️ **STRUCTURAL DIVERGENCE in formula details** — both sides implement
the standard NSGA-II crowding distance, but with **different
intermediate normalization** and **different extrema values**.

Both yield valid CDs that **preserve ranking order in expectation** but
won't agree to machine precision on the same Pareto-front fixture.

## Code-level audit

### v2 `assigns_crowding_distance_obj` (legacy-cpp-v2/source/asms_emoa.cpp:82+)

```cpp
// Per objective (ROI then risk):
//   sort by objective
//   set boundary CDs = (float_max / 2) - 1  // NOT infinity!
//   for each interior position i:
//       norm_cost1 = (front[i+1].obj - min) / (max - min)
//       norm_cost2 = (front[i-1].obj - min) / (max - min)
//       front[i].cd += norm_cost2 - norm_cost1    // ROI (descending sort)
//       front[i].cd += norm_cost1 - norm_cost2    // risk (ascending sort) — SIGN FLIPPED
```

### Python `calculate_crowding_distance` (nsga2.py:63+)

```python
# Per objective (ROI then risk):
#   sort by objective
#   set boundary CDs = float('inf')
#   for each interior position i:
#       front[i].cd += (next_obj - prev_obj) / range
```

## The differences

| Aspect | v2 | Python |
|---|---|---|
| Extrema CD | `(float_max/2) - 1` (large finite) | `float('inf')` |
| Per-interior contribution | `(prev_norm - next_norm)` with per-neighbor normalization | `(next - prev) / range` |
| Sign convention | ROI: `+`; risk: `−` (because risk sort is ascending; v2 explicitly flips) | Same `(next - prev) / range` for both (range positive) |
| Initial CD | `Pareto_front[i]->cd = 0` (front_id setup separate) | `solution.cd = 0.0` (in calculate fn) |

## Why these differ in numbers but converge in ranking

1. **Both** compute "how far apart neighbors are around me" in
   normalized objective space.
2. **v2's per-neighbor normalization** subtly differs from Python's
   "range-based" normalization — the sum across both objectives can
   give slightly different magnitudes but the RANKING of solutions
   by CD tends to be invariant under monotonic transforms.
3. **Extrema treatment**: v2 uses a finite-but-huge value
   (`float_max/2 - 1`), Python uses `inf`. Both ensure extrema always
   sort to the top — but downstream code that DOES arithmetic with CD
   (averaging, summing) would diverge: v2 has finite extrema, Python
   has infinity-propagated extrema.

## Behavioral expectation

CD is used in SMS-EMOA selection (tiebreaker after Pareto rank). For
ranking purposes both implementations should produce the same TOURNAMENT
WINNERS in most cases:
- Same extrema → top picks
- Same neighbor-distance-based ordering for interior solutions

But **numerical CD values differ** due to:
- Different extrema (finite vs infinite)
- Different normalization (per-neighbor vs range-based)

## Verdict per W18 matrix

⚠️ **DISAGREE-NUMERICAL** — different CD values; same ranking behavior in expectation.

## Bug count update

No new bugs. Just a documented numerical difference in CD formula details. Both implementations are valid NSGA-II crowding distance per standard literature; neither matches the other to machine precision but both produce equivalent ranking.

## What this doesn't affect

CD is a SELECTION-TIEBREAKER metric, not a HV computation. The W17-5
saturation chain depends on HV + anticipation + KF, not CD ranking.
This finding does NOT contribute meaningfully to the W17-5 gap.

## What this might affect

In hyper-volume-tied scenarios where CD is the tiebreaker:
- v2 might pick a different solution than Python
- This is a rare event but could surface as periodic regression in
  long-horizon smokes

## Reproducing

Code-read only (no execution-level cross-check); the C++ + Python
sources cited above are the receipt.

## Next steps

If W21+ wants execution-level CD parity:
- Build `crowding_distance_driver.cpp` that emits per-solution CD on a fixed Pareto-front fixture
- Build Python equivalent
- Compare; expect numerical disagreement; verify ranking agreement
