# W22-NC34 — Anticipatory Mutation Scorer

**Module:** `python_refactor/src/algorithms/anticipatory_mutation_scorer.py`
**Tests:** `python_refactor/tests/test_nc34_anticipatory_mutation_scorer.py` (13 tests, all passing)
**Spec:** `docs/W22-NEXT-STEPS-NC32-36.md` §B3
**Status:** Standalone module — no modifications to shared code paths.

---

## 1. Hypothesis and falsifiable criterion

**Operator flag (W22):** *"operators that preserve anticipatory signals or that
anticipate children or mutants performance/rank in the FNDS class or in the
Pareto frontier."*

**Hypothesis (H-NC34-anticipatory-beats-random):** Standard mutation perturbs
portfolio weights randomly; an anticipatory scorer that ranks offspring by
their PREDICTED FUTURE Δ_S contribution beats blind perturbation. Selecting
mutants whose forecast positions yield higher Δ_S avoids wasted evaluations
on dominated offspring.

**Falsifiable criterion:** On synthetic data (10 mutants spanning a clear
ROI/risk gradient, small reference front), the anticipatory-picked mutant's
predicted Δ_S must be **≥ 20% higher than the mean of all mutants' Δ_S**
(the random-pick baseline).

Verification: `test_anticipatory_beats_random_on_synthetic`. The test
constructs 10 mutants with linearly increasing ROI and decreasing risk, scores
them, and asserts `max(scores) ≥ 1.2 * mean(scores)`. Currently the observed
ratio is **far above** the 1.20 threshold (the highest-ROI/lowest-risk mutant
dominates the rectangle contribution by ~10×).

---

## 2. Math

For each of K candidate mutants `{m_1, ..., m_K}`:

1. **Forecast** mutant `m_k`'s `(ROI, risk)` at horizon `h` via its existing
   KF state: `μ_k_h = F^h · x_k` (means only for score 1).
2. **Forecast** each `current_front` solution at horizon `h` analogously.
3. **Predicted Δ_S:** insert `m_k`'s forecast into the forecast front (sorted
   ascending by ROI), then compute its `_front_contribution` at the resulting
   position:
   - First position: `(ROI − R1) · (R2 − risk)`
   - Middle position: `(ROI_i − ROI_{i+1}) · (risk_{i−1} − risk_i)`
   - Last position: `(ROI_i − ROI_{i−1}) · (R2 − risk_i)`
4. **Selection (deterministic):** `k* = argmax_k predicted_Δ_S(m_k)`
5. **Selection (softmax):** `P(m_k) ∝ exp(β · predicted_Δ_S(m_k))`
   - β = 0 → uniform random across mutants
   - β → ∞ → equivalent to argmax

For **probability-of-non-dominance** scoring (Defn 6.1 / TIP-style logic
inlined since each Solution already carries its KF state):

- Sample `n_mc` draws of `(ROI, risk)` from the mutant's bivariate Gaussian
  `(μ_k_h, Σ_k_h)` and from each current_front member's Gaussian.
- A front sample `(r', k')` dominates a mutant sample `(r, k)` iff
  `r' ≥ r AND k' ≤ k AND (r' > r OR k' < k)`.
- `P_ND(m_k) = (count of MC draws in which m_k is dominated by NONE) / n_mc`.

Empty `current_front` → `P_ND = 1.0` (vacuously non-dominated).

---

## 3. Module surface

```python
from src.algorithms.anticipatory_mutation_scorer import (
    score_mutants_by_predicted_delta_s,  # (parent, mutants, front, h, R1, R2) -> np.ndarray
    score_mutants_by_non_dominance,      # (parent, mutants, front, h, n_mc, rng) -> np.ndarray
    select_best_mutant_argmax,           # (parent, mutants, front, h, R1, R2) -> int
    select_mutant_softmax,               # (parent, mutants, front, h, beta, R1, R2, rng) -> int
)
```

All four functions raise `ValueError` on `mutants == []`. All four reuse
`_forecast_solution_at_horizon` and `_front_contribution` from
`src/algorithms/amfc_selector.py` — no math duplication.

---

## 4. Tests (13 cases, all passing)

| Test | What it checks |
|---|---|
| `test_score_mutants_returns_correct_shape` | Returns `np.ndarray` of shape `(K,)` |
| `test_score_mutants_higher_for_better_position` | Higher-ROI mutant scores above lower-ROI |
| `test_score_mutants_handles_empty_front` | Empty `current_front` → single-rectangle fallback |
| `test_select_best_mutant_returns_valid_index` | argmax returns `int` in `[0, K)` |
| `test_select_softmax_returns_valid_index` | softmax returns `int` in `[0, K)` |
| `test_select_softmax_beta_zero_uniform_random` | β=0 → uniform across 4 buckets (400 draws) |
| `test_select_softmax_high_beta_concentrates_on_argmax` | β=1e6 → every draw hits argmax |
| `test_score_non_dominance_returns_probability_in_zero_one` | P_ND ∈ [0, 1] |
| `test_anticipatory_beats_random_on_synthetic` | **H-NC34 falsifiable: best ≥ 1.2 × mean** |
| `test_handles_single_mutant` | K=1 → score shape (1,), argmax=0, softmax=0 |
| `test_handles_zero_mutants_raises_or_returns_none` | K=0 → all 4 fns raise `ValueError` |
| `test_non_dominance_empty_front_returns_one` | No front → P_ND = 1.0 for all |
| `test_non_dominance_dominated_lower_than_dominating` | Dominated < 0.05; dominating > 0.95 |

Run: `python3 -m pytest tests/test_nc34_anticipatory_mutation_scorer.py -v`

---

## 5. Honest scars

1. **Cost is O(K · n_mc) per call.** Tractable for K ≤ 20, n_mc ≤ 100. The
   default `n_mc=50` for non-dominance gives 500 draws per call at K=10. For
   pure mean-based scoring (`score_mutants_by_predicted_delta_s`), cost is
   O(K) — fast.

2. **Forecast accuracy depends entirely on KF tuning.** Mutants without KF
   state degenerate to `Σ_h = 0`; non-dominance scoring then collapses to
   a deterministic check. This couples directly with NC32 (Logistic-Normal-KF)
   — better-tuned KF posteriors yield more discriminating non-dominance
   probabilities.

3. **"Best by predicted Δ_S" is a heuristic.** Multi-objective alternatives are
   plausible:
   - Score by combined Pareto-rank + Δ_S
   - Score by **rank in FNDS class** (the operator's "rank in the FNDS class"
     phrasing) — this would slot in as a new scoring function alongside the
     two provided here. **Deferred** until experimental evidence justifies the
     added surface area.

4. **Empty `current_front` degenerates.** The fallback returns
   `(mutant.ROI − R1) · (R2 − mutant.risk)`, which makes argmax meaningful but
   does not anchor against the "Pareto frontier" semantics. Callers should
   prefer a non-empty front in production.

5. **Single-rectangle insertion is greedy.** Each mutant's contribution is
   computed in isolation (one mutant inserted into the front at a time).
   Scoring K mutants jointly — accounting for how multiple mutants would
   simultaneously change the frontier — is a strictly stronger formulation,
   deferred as a research extension.

6. **Reference point is data-derived when not supplied.** Default is
   `(min ROI, max risk)` across the union of mutants + front. Caller should
   pass explicit `R1, R2` to match the hypervolume reference used downstream
   (e.g., the same one fed to `select_amfc`).

---

## 6. Integration sketch (operator decision)

Inside `SMSEMOA._mutation`, after generating K candidate mutants for one
parent (or per generation), the integration call would be::

    from src.algorithms.anticipatory_mutation_scorer import (
        select_best_mutant_argmax,
    )

    # ... after self.mutation_operator produces `candidate_mutants` ...
    best_idx = select_best_mutant_argmax(
        parent_solution=parent,
        mutants=candidate_mutants,
        current_front=[s for s in population if s.Pareto_rank == 0],
        horizon=1,
        R1=self.R1, R2=self.R2,
    )
    chosen_mutant = candidate_mutants[best_idx]

Or, for stochastic-exploitation behavior::

    best_idx = select_mutant_softmax(
        parent, candidate_mutants, current_front,
        horizon=1, beta=2.0, R1=self.R1, R2=self.R2, rng=self.rng,
    )

**Wiring is NOT done in this push.** This module is standalone (no
modifications to `sms_emoa.py` or any shared code path) per W22 discipline
("shipped standalone, integration is operator's decision").

---

## 7. Connection to the operator's "rank in FNDS class" question

The W22 operator brief mentions *"operators that... anticipate children or
mutants performance/rank in the FNDS class or in the Pareto frontier."*

This probe scores by **predicted Δ_S** and **predicted non-dominance
probability** — both are "Pareto frontier" semantics. The complementary
**rank-in-FNDS-class** scoring is a natural extension:

```python
# Sketch (NOT implemented in this push):
def score_mutants_by_fnds_rank(parent, mutants, full_population, horizon=1):
    """For each mutant, forecast its position; insert into the
    forecasted full population; run FNDS; return that mutant's rank.
    Lower rank = better."""
    ...
```

Deferred because: (a) FNDS over forecasted positions is O(N²) per mutant,
making the K·N² cost much higher than the Δ_S/non-dominance pair already
shipped; (b) the discrete rank signal is less informative than the continuous
Δ_S for argmax-style decisions; (c) the operator's H-NC34 falsifiable
criterion targets Δ_S, not rank.

If experimental evidence on real benchmarks shows the Δ_S-based scorer
mispicks mutants that an FNDS-rank scorer would catch, the rank scorer
becomes a follow-up unit.

---

## 8. Where this slots into the W22 next-steps catalog

- **§B3 (this unit):** anticipatory mutation scorer — DONE (standalone)
- **§B1 NC32:** Logistic-Normal-KF — separate parallel unit
- **§B2 NC33:** Dirichlet→KF Q coupling — deferred
- **§B4 NC35:** Accumulated future Δ_S — separate parallel unit
- **NC36:** analytical TIP — foreground
- **NC-AD-fix:** rectangle alignment — foreground

NC34's `score_mutants_by_non_dominance` will benefit from NC36 (analytical
TIP replaces the inlined MC dominance check); however, the current MC version
is correct and self-contained, and the conversion is mechanical when NC36
lands.
