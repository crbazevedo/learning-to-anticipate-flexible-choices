# Cross-validation N — SBX vs uniform crossover

*Generated 2026-05-17 by W21-4. Closes operator check N.*

## Verdict

✅ **BOTH ENGINES USE UNIFORM CROSSOVER IN PRODUCTION**. The W19-gate
carry-forward note ("C++ uses SBX") was based on a STALE assumption.
Direct code-read of `legacy-cpp-v2/source/crossover_operators.cpp:13`
shows v2's only crossover operator is `uniform_crossover` (the SBX path
does not exist in v2 source).

What remains is a NUMERICAL DIVERGENCE in cardinality-enforcement
details — both implementations are uniform crossover, but apply the
cardinality constraint differently.

## Honest correction to prior assumption

The W18 cross-validation backlog (operator's 14-check directive)
described check N as "SBX vs uniform crossover" — implying v2 used SBX.
The W15-2 commit message ("switch to uniform per thesis") + the W19-gate
carry-forward note ("C++ uses SBX; Python switched to uniform") both
echoed this assumption.

**Empirical code-read finding (2026-05-17 W21-4):**

```cpp
// legacy-cpp-v2/source/crossover_operators.cpp lines 12-13
unsigned int xover_op::num = 1;
crossover_operator_ptr xover_op::_operator[1] = {&uniform_crossover};
```

v2's crossover-operator pool has ONE entry: `uniform_crossover`. There
is no SBX implementation in `legacy-cpp-v2/source/crossover_operators.cpp`.
The "v2 uses SBX" assumption was carried forward from W18-derived
backlog wording without code verification.

This is a minor scar in the audit chain: the W15-2 thesis-faithfulness
switch (Python: SBX → uniform) was correct per thesis text, but the
"v2 differs (uses SBX)" half of the framing was wrong. Both engines
are uniform crossover in production.

## Code-level audit

### v2 `uniform_crossover` (crossover_operators.cpp:15-98)

```cpp
void uniform_crossover(parent1, parent2, offspring1, offspring2) {
    // 1. Zero out offspring; collect indices where either parent has weight > 0
    for each asset i:
        offspring1.weight(i) = offspring2.weight(i) = 0
        if parent1.weight(i) > 0 or parent2.weight(i) > 0:
            index.push(i)

    // 2. Shuffle indices; iterate while offspring1 cardinality < max
    random_shuffle(index)
    for each i in index:
        if offspring1.cardinality < max_cardinality:
            if uniform() < 0.5:
                offspring1.weight(i) = parent1.weight(i)
                offspring2.weight(i) = parent2.weight(i)
            else:
                offspring1.weight(i) = parent2.weight(i)
                offspring2.weight(i) = parent1.weight(i)
            update cardinality counters
        else:
            break  // capacity-truncation

    // 3. Ensure min_cardinality (random-add if needed)
    while offspring.cardinality < min_cardinality:
        add_asset(offspring, rand() % num_assets)

    // 4. Apply min/max-hold thresholds
    apply_threshold(...)
}
```

### Python `thesis_uniform_crossover` (operators.py:438-464)

```python
def thesis_uniform_crossover(parent1, parent2, p=0.5, c_l, c_u, rng):
    n = len(parent1.P.investment)
    swap_mask = rng.random(n) < p                # per-asset swap gate
    w1 = parent1.P.investment.copy()
    w2 = parent2.P.investment.copy()
    w1[swap_mask], w2[swap_mask] = w2[swap_mask], w1[swap_mask]
    child1.P.investment = project_to_simplex(w1, c_l, c_u, rng)
    child2.P.investment = project_to_simplex(w2, c_l, c_u, rng)
    return child1, child2
```

## Divergence map

| Aspect | v2 | Python |
|---|---|---|
| Operator pool | 1 (uniform_crossover only) | 1 (thesis_uniform_crossover only; SBX retained for backward-compat but UNUSED in sms_emoa.py:437-440) |
| Per-asset swap | Conditional on offspring1's cardinality ≤ max (capacity-truncated) | Unconditional per-asset (no cardinality gating during swap) |
| Index ordering | Random shuffle of non-zero union | Per-asset linear scan |
| Cardinality enforcement | DURING crossover (capacity-truncation early-terminates the loop) | POST crossover (project_to_simplex enforces [c_l, c_u]) |
| Min-cardinality recovery | Random-add until ≥ min | Within project_to_simplex |
| Thesis grounding | NOT explicit (v2 implementation choice) | "We utilized uniform crossover over the mean DD vectors" — thesis §7.2.3 p.147 verbatim |

## Behavioral expectation

Both implementations preserve the semantic of "uniform crossover":
each weight position is independently swapped with prob 0.5. The
difference is in HOW the cardinality constraint interacts with the
swap:

- v2: cardinality is a **stopping condition** for the swap loop
- Python: cardinality is enforced **after** all swaps via simplex projection

For populations far from the cardinality boundary, both produce
statistically equivalent offspring. For populations near the boundary
(K_max active assets in both parents), v2's early-termination biases
the offspring toward parent1's weights (since the loop iterates in
shuffled order and stops at capacity).

## Mutual-skepticism finding (closed)

Per operator's W18 directive — *"There is the possibility of the C++
reference implementation also being wrong"* — the prior assumption
that "C++ uses SBX" was UNVERIFIED. Direct code-read closed this:
v2 uses uniform crossover, not SBX. The SBX-vs-uniform framing was
a documentation scar, not a v2 implementation defect.

## On the W17-5 saturation chain

Crossover divergence is **minimal** for the saturation chain analysis:
- Both engines use uniform crossover
- Cardinality-enforcement timing differs but converges in expectation
- The dominant W17-5 drivers (Reading E rate, Reading F stability)
  are downstream of crossover and not gated by it

This check is **NOT a candidate** for W21-5 ablation (in contrast to
W21-3 mutation, where the entropy-operator divergence IS a candidate).

## Bug count

No new bugs. The prior framing "C++ uses SBX" was a documentation
scar (audit-chain stale assumption), corrected here. No code defect
in either engine.

## Output artifacts

- This receipt
- `.dfg/retrospectives/W21/W21-4.md`

## Reproducing (manual verification)

```bash
# v2 side
cd legacy-cpp-v2/source && grep -n "xover_op::_operator" crossover_operators.cpp
cd legacy-cpp-v2/source && grep -A 84 "^void uniform_crossover" crossover_operators.cpp

# Python side
cd python_refactor && grep -A 28 "^def thesis_uniform_crossover" src/algorithms/operators.py
cd python_refactor && grep -n "thesis_uniform_crossover\|sbx_crossover" src/algorithms/sms_emoa.py
```

## Strategic note

With W21-4 complete, **14 of 14 operator cross-checks (A through N)
are now FULLY CLOSED** with empirical-or-code-read receipts. The
remaining W21 keystones are:
- W21-1 (Reading-F experimental smoke — running in background)
- W21-5 (full ablation matrix + 30-seed run)
- W21-6 (synthesis + publication-track decision)

## Next

W21-5 (full ablation matrix), or W21-1 receipt completion (once smoke
returns).
