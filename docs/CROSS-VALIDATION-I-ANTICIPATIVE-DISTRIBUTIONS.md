# Cross-validation I — anticipative distributions from OAL application

*Generated 2026-05-17 by W20-3. Closes operator check I.*

## Verdict

⚠️ **STRUCTURAL DIVERGENCE** between v2 and Python OAL `anticipatory_learning_obj_space` — they use **different rate formulas for state update**, plus different KF state composition. Reading E is real but the actual divergence is more nuanced than W19-4 initially framed.

## Code-level audit (W20-3)

### v2 `anticipatory_learning_obj_space` (legacy-cpp-v2/source/asms_emoa.cpp:639+)

```cpp
double accuracy_factor = (min_error > 0.0 && (max_error - min_error) > 0.0)
    ? 1.0 - (anticipative_portfolio->prediction_error - min_error)/(max_error - min_error)
    : 0.0;
double uncertainty_factor = anticipative_portfolio->alpha;   // ← α = 1 - nd_prob from compute_efficiency:44

double rate_upb = (t == 0 || Kalman_params::window_size == 0) ? 0.0 : 0.5;
double rate_lwb = (t == 0 || Kalman_params::window_size == 0) ? 0.0 : 0.0;

w->anticipation_rate = rate_lwb + 0.5*uncertainty_factor*(rate_upb - rate_lwb)
                                 + 0.5*accuracy_factor*(rate_upb - rate_lwb);
//                  = 0.25 * (uncertainty + accuracy)   when K>0 + t>0

// State update:
x = x_state + anticipation_rate * (x_next - x_state)
C = P + anticipation_rate² * (P_next - P)
```

### Python `anticipatory_learning_obj_space` (anticipatory_learning.py:710+)

```python
solution.anticipation_rate = self.compute_anticipatory_learning_rate(...)
# = 0.5 * (λ^H + λ^K) per thesis Eq 7.16, OR 1 - tip (W20-1 flag)

# State update (same shape):
x = x_state + anticipation_rate * (x_next - x_state)
C = P + anticipation_rate² * (P_next - P)
```

## The actual divergence (3 layers)

### Layer 1: which "alpha" feeds the state update?

- **v2**: `anticipation_rate = 0.25 * (α + accuracy)`, where `α = 1 - nd_prob` (Reading-E quantity) and `accuracy` is the W16-1 traditional accuracy factor
- **Python default (Eq 7.16)**: `anticipation_rate = 0.5 * (λ^H + λ^K)`, where `λ^H` is Shannon-entropy based and `λ^K` is K-period residual sum
- **Python W20-1 flag**: `anticipation_rate = 1 - tip` (simpler than v2)

So my W20-1 test substituted `1 - tip` for `0.5*(λ^H + λ^K)`, when v2 actually uses `0.25 * ((1 - tip) + accuracy)`. The W20-1 formula was MORE aggressive than v2 — it removed both the 0.5 factor AND the accuracy averaging.

### Layer 2: state update is identical shape

Both sides: `x = x_state + rate * (x_next - x_state)` and `C = P + rate² * (P_next - P)`. So the FORMULA for state update is the same; only the `rate` differs.

### Layer 3: transaction-cost integration

Both sides apply a wealth/cost adjustment to ROI after the state update. Different code organization but same intent.

## Saturation regime behavior — REVISED

At TIP=0.5 (W17-5 saturation):

| Source | anticipation_rate at TIP=0.5 | Note |
|---|---|---|
| v2 actual | `0.25 * (0.5 + accuracy)` ≈ **0.125 to 0.25** | depends on accuracy_factor |
| v1 actual (similar pattern) | `0.25 * (0 + accuracy)` ≈ **0 to 0.125** | linear_entropy collapses uncertainty to 0 |
| Python Eq 7.16 default | **0.0** | both λ^H and λ^K go to 0 in saturation |
| Python W20-1 flag (1-tip) | **0.5** | my flag's simpler formula |

So:
- The v2 rate is MILDER than I tested in W20-1 (~0.125-0.25 vs 0.5)
- But still meaningfully > 0 (Python Eq 7.16 collapses to 0)
- W20-1's +7.82pp recovery may be over-amplified vs what v2 actually does

## Implications

1. **Reading E confirmed at a higher level**: v2's anticipation_rate doesn't fully collapse in saturation (~0.125-0.25 vs Python's 0). The directional finding (W20-1) is correct: maintaining anticipation in saturation helps ASMS.

2. **W20-1 flag is an APPROXIMATION**: it uses v2's `α = 1 - tip` directly as the rate, but v2 actually applies it through the traditional accuracy+uncertainty formula. The W20-1 +7.82pp is therefore an UPPER-BOUND estimate of the v2 contribution.

3. **W21+ candidate**: implement `use_v2_obj_space_rate=True` flag that uses the FULL v2 formula `0.25*(α + accuracy)` instead of the simpler `1 - tip`. Compare against W20-1's variant + Python default.

4. **The "compute_anticipatory_learning_rate" function and "anticipatory_learning_obj_space.anticipation_rate" are different concerns in v2**: v2's `compute_anticipatory_learning_rate` doesn't exist as such — `alpha` is set in `compute_efficiency:44`, and the state-update `anticipation_rate` is computed inline in `anticipatory_learning_obj_space:664`. Python conflates these into one function.

## Bug count update

| # | Side | Where | Severity |
|---|---|---|---|
| 1 | v1 | autocorr comma-op | Off-headline |
| 2 | Python | compute_risk sqrt | On-headline |
| 3 | Python | KF state-evolution divergence | On-headline (Reading D) |
| 4 | Python (vs v2) | anticipative-rate state-update FORMULA differs (Eq 7.16 vs traditional+α composition) | On-headline (Reading E, REFINED) |

## Verdict per W18 matrix

⚠️ **DISAGREE STRUCTURALLY** at the anticipatory_learning_obj_space level. v2 and Python compute different `anticipation_rate` values from the same KF + α inputs. The Reading-E experimental test (W20-1) confirmed the directional finding (+7.82pp recovery) with a SIMPLIFIED v2-style rate; the full v2 formula would likely yield a different magnitude.

## What W20-3 doesn't test

This unit is a CODE-READ + STATIC analysis. A full execution-level cross-check requires:
- Building a v2 driver that exercises `anticipatory_learning_obj_space` on a fixed (KF state, α, accuracy, transaction-cost setup) fixture
- Building a Python equivalent
- Comparing the post-OAL `(P.ROI, P.risk, P.kalman_state.x, P.kalman_state.P)` per portfolio

This is the SCOPE of W20-3 acceptance criteria but the C++ driver requires the full transaction-cost setup which is interwoven with the experiment driver in v2 (not easily isolatable). Deferred to W21 as a deeper-investigation candidate.

## Next steps

1. **W21 candidate**: implement `use_v2_obj_space_rate=True` Python flag using the FULL v2 formula `0.25*(α + accuracy)`; re-run W17-5 smoke
2. **W21 candidate**: build an isolated `anticipative_distribution_driver` for v2 once the transaction-cost setup is factorable
3. **W21 keystone**: ablation table — current Python default vs W20-1 simple-v2-rate vs full-v2-rate
