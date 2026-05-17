# Cross-validation L — non-dominated sorting (NDS) algorithm

*Generated 2026-05-17 by W21-2. Closes operator check L.*

## Verdict

✅ **CORE NDS AGREES (algorithm + Pareto dominance) — diverges only in
constraint type**. Both implement the standard Deb-2002 fast-non-dominated
sort with identical structure. Bare Pareto dominance is bit-identical.
The constraint-handling layer differs:

- **v2:** `epsilon_feasibility` (continuous-tolerance feasibility per
  the original paper)
- **Python:** `max_cardinality` (NSGA-II constraint-domination per
  the thesis cardinality framing)

Both are valid; they answer different questions ("is the solution
epsilon-feasible?" vs "does the solution respect the cardinality
budget?"). On unconstrained inputs the two **AGREE to bit-equality**.

## Code-level audit

### v2 `fast_nondominated_sort` (legacy-cpp-v2/source/asms_emoa.cpp:168+)

```cpp
unsigned int fast_nondominated_sort(std::vector<sol*> &P) {
    std::vector<std::vector<unsigned int>> S(P.size());  // dominated-by-p
    std::vector<unsigned int> n(P.size());               // n[p] = dominators
    std::vector<unsigned int> H;                          // current-rank holders

    for (unsigned int p = 0; p < P.size(); ++p) {
        for (unsigned int q = 0; q < P.size(); ++q) {
            if (p != q) {
                if (constrained_dominance(P[p], P[q]))
                    S[p].push_back(q);
                else if (constrained_dominance(P[q], P[p]))
                    ++n[p];
            }
        }
        if (n[p] == 0) {
            H.push_back(p);
            P[H.back()]->Pareto_front = 0;
        }
    }
    F.push_back(H);
    while (!F[i].empty()) {
        std::vector<unsigned int> H;
        for (each p in F[i]) for (each q in S[p]) {
            if (--n[q] == 0) {
                P[q]->Pareto_front = i + 1;
                H.push_back(q);
            }
        }
        F.push_back(H); ++i;
    }
    std::sort(P.begin(), P.end(), cmp_front_ptr);  // sort population by front
    return num_fronts - 1;
}
```

### Python `fast_non_dominated_sort` (nsga2.py:14+)

```python
def fast_non_dominated_sort(population: List[Solution]) -> List[List[Solution]]:
    fronts = [[]]
    domination_count = {}      # n[p] = dominators
    dominated_solutions = {}    # S[p] = dominated by p

    for p in population:
        domination_count[p] = 0
        dominated_solutions[p] = []
        for q in population:
            if p != q:
                if p.dominates_with_constraints(q):
                    dominated_solutions[p].append(q)
                elif q.dominates_with_constraints(p):
                    domination_count[p] += 1
        if domination_count[p] == 0:
            p.Pareto_rank = 0
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    q.Pareto_rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts
```

## Algorithmic structure

| Aspect | v2 | Python |
|---|---|---|
| Per-pair domination test | `constrained_dominance(p, q)` | `p.dominates_with_constraints(q)` |
| First front | `{p : n[p] == 0}` | `{p : domination_count[p] == 0}` |
| Front iteration | Decrement n[q] when p in S[p]; q joins H when n[q] hits 0 | Same logic, dict-based |
| Output | `Pareto_front` attribute (0-indexed) + sort by front | `Pareto_rank` attribute (0-indexed) + nested fronts |
| Complexity | O(MN²) where M=2 objectives, N=pop size | O(MN²) — identical |
| Worst-case fronts | N (degenerate case) | N (degenerate case) |

**Both are textbook Deb-2002 NSGA-II fast non-dominated sort. AGREE.**

## Bare Pareto dominance

Both implementations use IDENTICAL bare Pareto dominance (ROI-maximize,
risk-minimize):

### v2 `unconstrained_dominance` (asms_emoa.h:148-156)
```cpp
if (p->P.ROI < q->P.ROI || p->P.risk > q->P.risk)
    return false;
else if (p->P.ROI > q->P.ROI || p->P.risk < q->P.risk)
    return true;
else
    return false;
```

### Python `dominates_without_constraints` (solution.py:72-77)
```python
if (self.P.ROI < other.P.ROI or self.P.risk > other.P.risk):
    return False
elif (self.P.ROI > other.P.ROI or self.P.risk < other.P.risk):
    return True
else:
    return False
```

**Bit-identical short-circuit logic.** Both implement standard
`weak-dominance + strict-improvement-in-one` semantics for the
(ROI ↑, risk ↓) bi-objective problem.

## Constraint layer divergence

| Aspect | v2 `constrained_dominance` | Python `dominates_with_constraints` |
|---|---|---|
| Constraint type | `epsilon_feasibility` (continuous-tolerance) | `max_cardinality` (active-asset budget) |
| Both infeasible | Sub-domination on `epsilon_feasibility` pair | Sub-domination on cardinality value |
| Feasible vs infeasible | Feasible wins | Feasible wins (cardinality ≤ max) |
| Both feasible | Delegate to bare Pareto dominance | Delegate to bare Pareto dominance (lines 101-106) |

This is a **semantic divergence**, NOT a bug:
- The IEEE TCYB 2015 paper formulates constrained optimization with
  epsilon-feasibility (paper Eq 7).
- The thesis adopts cardinality constraint per §7.2.3 p.146 ("a maximum
  cardinality K_max is enforced").
- Python's `max_cardinality` reflects the thesis formulation; v2's
  `epsilon_feasibility` reflects the paper's continuous-tolerance form.

Both are thesis/paper-faithful for their respective scopes.

## On the W17-5 saturation chain

Neither implementation's NDS contributes to the saturation chain.
The dominance test, the front construction, and the rank assignment
are all deterministic and bit-identical on unconstrained populations.

The W17-5 production gap is NOT in NDS. Cross-check L is a **CLEAN
AGREE** on the load-bearing dimension (ranking by dominance) and a
**documented intentional divergence** on the orthogonal constraint
dimension.

## Mutual-skepticism caveat

Per operator's W18 directive — *"There is the possibility of the C++
reference implementation also being wrong"* — both implementations
were spot-checked for:

- ✅ Correct handling of N=1 population (both produce single-front output)
- ✅ Correct front iteration termination (both halt when no new front)
- ✅ Correct rank assignment (0-indexed)
- ✅ Stable behavior on duplicate-objective inputs (both correctly
  classify as same rank since `dominates(p, q)` returns false when
  ROI(p)==ROI(q) AND risk(p)==risk(q))

No bugs surfaced on this check.

## Output artifacts

- This receipt
- `.dfg/retrospectives/W21/W21-2.md`

## Reproducing (manual verification)

```bash
# v2 side
cd legacy-cpp-v2/source && grep -A 60 "^unsigned int fast_nondominated_sort" asms_emoa.cpp
cd legacy-cpp-v2/headers && grep -A 8 "^inline bool unconstrained_dominance" asms_emoa.h

# Python side
cd python_refactor && grep -A 50 "def fast_non_dominated_sort" src/algorithms/nsga2.py
cd python_refactor && grep -A 25 "def dominates_without_constraints" src/algorithms/solution.py
```

## Next

W21-3 (cross-check M — mutation operator).
