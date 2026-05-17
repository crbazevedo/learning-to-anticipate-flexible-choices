# Cross-validation M — mutation operator

*Generated 2026-05-17 by W21-3. Closes operator check M.*

## Verdict

⚠️ **STRUCTURAL DIVERGENCE — operator suite differs**. Python
implements the THESIS-PRESCRIBED dual-mode mutation verbatim
(thesis §7.2.3 p.147). v2 implements FOUR mutation operators
selected via roulette wheel: modify_investment + modify_portfolio
(matching thesis dual modes) + raise_entropy + lower_entropy
(NOT in the thesis text).

Python is **thesis-faithful**; v2 has **two additional operators
that don't appear in the thesis**. The entropy operators may
contribute to v2's paper-headline saturation behavior.

## Code-level audit

### v2 mutation suite (legacy-cpp-v2/source/mutation_operators.cpp)

```cpp
// 4 mutation operators chosen by roulette_wheel_selection_mutation()
mutation_operator_ptr mutation_op::_operator[4] = {
    &modify_investment,   // (1) per-asset multiplicative perturbation
    &modify_portfolio,    // (2) add/remove assets
    &raise_entropy,       // (3) flatten the weight distribution
    &lower_entropy        // (4) concentrate the weight distribution
};
```

#### `modify_investment(offspring, p)` (lines 48-63)
```cpp
for each i where investment(i) > 0:
    if uniform(0,1) < p:                    // per-asset gate
        if uniform(0,1) < 0.5:
            raise:  investment(i) *= 1 + uniform(0.05, 0.25)
        else:
            reduce: investment(i) *= uniform(0.05, 0.25)
```

#### `modify_portfolio(offspring, p)` (lines 86-122)
Add or remove assets stochastically; final cardinality clamped to
`[min_cardinality, max_cardinality]`.

#### `raise_entropy(offspring, dummy)` (lines 145+)
Reduce the maximum-weight asset's allocation; redistribute → entropy ↑

#### `lower_entropy(offspring, dummy)` (lines 165+)
Increase the maximum-weight asset's allocation; concentrate → entropy ↓

### Python mutation (operators.py:467+ — added W15-2)

```python
def thesis_dual_mode_mutation(solution, ...):
    """Dual-mode mutation per thesis §7.2.3 p.147 (verbatim spec)."""
    if rng.random() < 0.5:
        # Mode 1: modify a non-zero weight (1 random asset)
        asset = choose_random_active_asset()
        factor = uniform(0.10, 0.50)
        if rng.random() < 0.5:
            w[asset] *= (1.0 + factor)  # increase
        else:
            w[asset] *= (1.0 - factor)  # decrease
    else:
        # Mode 2: add or remove an asset
        if remove_branch and cardinality > min:
            w[active_asset] = 0.0  # remove
        else:
            w[inactive_asset] = equal_alloc + jitter  # add
    return project_to_simplex(w, c_l, c_u)
```

## Divergence map

| Dimension | v2 | Python |
|---|---|---|
| # Operators | 4 (roulette-wheel) | 2 (50/50 coin flip) |
| Operator 1 | modify_investment (per-asset multiplicative, factor [0.05, 0.25]) | Mode 1: modify ONE non-zero weight (factor [0.10, 0.50]) |
| Operator 2 | modify_portfolio (add/remove, multiple per call) | Mode 2: add OR remove ONE asset |
| Operator 3 | raise_entropy | ❌ NOT PRESENT |
| Operator 4 | lower_entropy | ❌ NOT PRESENT |
| Per-asset gate | Yes (rate p per asset) | Mode 1: only ONE asset per call (no per-asset gate) |
| Mutation factor range | [0.05, 0.25] | [0.10, 0.50] |
| Cardinality enforcement | apply_mutation() post-call (drop lowest-weight until ≤ max) | project_to_simplex (per thesis §7.2 Eq 7.3) |
| Source | Paper-companion impl (v2 release) | Thesis §7.2.3 p.147 verbatim |

## Verbatim thesis quote (§7.2.3 p.147)

> "For mutation, we randomly choose between (1) modifying the
> non-zero weights; or (2) adding/removing assets..."

The thesis describes 2 modes. Python implements 2 modes. v2 implements
4 operators (the 2 thesis-prescribed + 2 entropy operators).

## Mutual-skepticism finding

Per operator's W18 directive — *"There is the possibility of the C++
reference implementation also being wrong"* — v2's `raise_entropy` and
`lower_entropy` operators **do NOT correspond to anything in the thesis
text**. They are paper-companion additions.

Two possible explanations:
1. **Intentional v2 extension**: the paper added entropy operators to
   maintain diversity in the saturation regime; thesis is the older
   spec, paper is the updated one.
2. **Unintentional v2 inclusion**: the paper-companion code may have
   accumulated operators from earlier exploratory phases that were
   left enabled by oversight.

**Without thesis re-read of mutation semantics, we cannot adjudicate.**
The receipts default to thesis-faithfulness for Python.

## On the W17-5 saturation chain

The mutation operator divergence is **plausibly relevant** to the
saturation chain:
- v2's entropy operators actively manipulate the weight-distribution
  shape, which may interact differently with the anticipative-rate
  formula (Reading E) and stability multiplier (Reading F).
- Python's thesis-faithful dual-mode does NOT have entropy mutation,
  so the per-generation weight-vector trajectory may diverge from v2's.

This is a **CANDIDATE FOR W21-5 ABLATION** — adding v2's entropy
operators to Python as an ablation variant could quantify their
contribution to the residual gap. Deferred to W21-5 as part of the
full ablation matrix.

## Bug count

No new bugs identified. The divergence is structural (different
operator suites), not a defect. Both implementations are internally
consistent with their respective sources (thesis for Python; paper
for v2).

## Output artifacts

- This receipt
- `.dfg/retrospectives/W21/W21-3.md`

## Reproducing (manual verification)

```bash
# v2 side
cd legacy-cpp-v2/source && grep -n "mutation_op::_operator\[4\]" mutation_operators.cpp
cd legacy-cpp-v2/source && grep -A 16 "^void modify_investment" mutation_operators.cpp

# Python side
cd python_refactor && grep -A 60 "^def thesis_dual_mode_mutation" src/algorithms/operators.py
```

## Next

W21-4 (cross-check N — SBX vs uniform crossover semantic divergence).
