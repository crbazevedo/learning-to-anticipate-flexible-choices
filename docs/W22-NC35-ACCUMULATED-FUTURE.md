# W22-NC35 — Accumulated Future Δ_S over H Periods (AMFC multi-period)

**Spec ref:** `docs/W22-NEXT-STEPS-NC32-36.md` §B4
**Operator-flagged:** "future expected hypervolume contribution w.r.t. whole
accumulated future period."

---

## Hypothesis

AMFC's pre-NC35 implementation (NC30-v1) used a **single-period** forward
forecast: the per-candidate score was `E[Δ_S^(s)_{t+h}]` at a single chosen h
(typically h=1). The operator's stated intent — *"anticipative maximal
flexible choice"* over multiple future periods — calls for the **accumulated
future** version:

```
Q^A(s) = Σ_{h=1}^H γ^h · E[Δ_S^(s)_{t+h}]
```

where:
- `γ` is the same geometric discount NC29a uses (env var `W22_NC29A_GAMMA`,
  default `0.9`).
- `E[Δ_S^(s)_{t+h}]` is the per-candidate expected hypervolume contribution
  if `s` were chosen at time `t+h`, evaluated within the **forecast front**
  at `t+h`.

The selection is `s* = argmax_s Q^A(s)`.

---

## Falsifiable criteria

| ID | Hypothesis | Where verified |
|----|------------|----------------|
| **H-NC35-H1-identity** | At `horizon_accumulated=1`, NC35 ≡ NC30-v1 byte-for-byte (same argmax). | `test_horizon_accumulated_1_matches_pre_nc35_behavior`, `test_horizon_accumulated_default_is_1` |
| **H-NC35-multi-period-different** | At H=3, NC35 argmax can differ from NC30-v1 when one candidate's contribution PERSISTS across periods. | `test_horizon_accumulated_changes_argmax_on_persistent_solutions` |
| **H-NC35-no-kf-invariant** | When candidates have no KF state, multi-period reduces to a uniform scaling of single-period scores → argmax preserved. | `test_horizon_accumulated_invariant_under_no_kf` |
| **H-NC35-gamma-decay** | With γ<1, deeper horizons contribute less. With γ→1, all horizons weight equally. With γ=0, the sum degenerates. | `test_horizon_accumulated_3_higher_h_contributes_less`, `test_horizon_accumulated_high_gamma_weights_all_horizons_equally`, `test_horizon_accumulated_zero_gamma_uses_only_h1` |
| **H-NC35-cost-bound** | At `|P|=20, H=3, n_mc=200`, the call completes in under 500 ms (O(|P|·H·|P|) bound). | `test_horizon_accumulated_cost_under_500ms` |
| **H-NC35-dispatcher-wiring** | `dm_config['horizon_accumulated']` is forwarded end-to-end through `ThesisAlignedExperiment`. | `test_dispatcher_passes_horizon_accumulated_kwarg` |

---

## Math

For each candidate `s ∈ P_t` on the current Pareto frontier:

1. For each `h ∈ {1, ..., H}`:
   - Forecast `s`'s `(ROI, risk)` at `t+h` via the existing
     `_forecast_solution_at_horizon` helper (KF `F^h`-projection of the
     state mean; covariance projection through `F · P · Fᵀ` h times).
   - Forecast **all other** current-front solutions at `t+h` via the same KF
     mechanism.
   - Sort the resulting `n` forecast means by ROI; compute the per-candidate
     analytical contribution at the sorted position using `_front_contribution`
     (same Δ_S formula as deterministic SMS-EMOA per Eq 11; tied-mean groups
     get averaged contributions per the NC30-v2 symmetry fix).
   - Call the resulting `n`-vector `predicted_Δ_S(·, t+h)`.

2. Accumulate per candidate:
   ```
   Q^A[i] = Σ_{h=1}^H γ^h · predicted_Δ_S[i, t+h]
   ```

3. Apply NC30 c (`variance_penalty`) and NC30 d (`tie_break_by_variance`) to
   the accumulated `Q^A` vector. Variance trace is taken at the **deepest
   horizon** `t+H` (where KF uncertainty is largest) — the operator's
   "low variance wins" intuition still bites under accumulation.

4. `s* = candidates[argmax(Q^A_effective)]`.

When `horizon_accumulated == 1`, the code path skips the accumulation block
entirely and runs the original single-period analytical/MC path —
**byte-identical** to pre-NC35 behavior (regression invariant
H-NC35-H1-identity verified explicitly).

---

## Module additions

| File | Change |
|------|--------|
| `python_refactor/src/algorithms/amfc_selector.py` | + `horizon_accumulated: int = 1` kwarg on `select_amfc`. + helper `_compute_expected_contributions_analytical` (factored from analytical block for reuse). + multi-period accumulation block (guarded by `horizon_accumulated > 1`). + telemetry field `horizon_accumulated`. |
| `python_refactor/src/experiments/thesis_aligned_experiment.py` | + `dm_config.get('horizon_accumulated', 1)` parse + forward through `select_amfc` kwargs in the `Hv-DM-AMFC` branch. |
| `python_refactor/tests/test_nc35_accumulated_future_delta_s.py` | New file. 12 tests covering H=1 identity, no-KF invariant, γ-decay (3 tests: standard, γ=0, γ→1), persistent-grower argmax flip, sanity, dispatcher wiring, cost guard, telemetry, NC30 c composition, default-kwarg backward-compat. |
| `docs/W22-NC35-ACCUMULATED-FUTURE.md` | This file. |

**Shared-code touch surface:** 1 source file (`amfc_selector.py`) + 1
dispatcher file (`thesis_aligned_experiment.py`). Both changes are
**additive with backward-compat default** (`horizon_accumulated=1`).
NC30 regression test suite (28 tests) **unchanged** — all pass post-NC35.

---

## Tests

```bash
cd python_refactor
python3 -m pytest tests/test_nc35_accumulated_future_delta_s.py tests/test_nc30_amfc_selector.py -q
# 40 passed (28 NC30 unchanged + 12 NC35 new)
```

---

## HONEST SCARS

1. **KF forecast variance grows with `h`.** `Σ_h = F^h · P_0 · (F^h)ᵀ`
   compounds at each step. Deeper horizons are intrinsically noisier; the
   accumulated sum is biased toward near-term forecasts where the KF is
   tightest. *Mitigation:* the `γ^h` decay already down-weights deeper
   horizons. At `γ=0.9`, h=5 weight is `0.9^5 ≈ 0.59`; at h=10 it's `0.35`.
   For very long H, the operator can lower γ (e.g., `γ=0.7`) to truncate
   effective contribution from noisy deep forecasts.

2. **Compute cost scales `O(|P| · H · |P|)` per call.** Each horizon
   re-forecasts ALL `|P|` candidates and re-sorts (O(|P|²)). For `|P|=20,
   H=3`, that's 1200 elementary ops — well under the 500 ms budget guarded
   by `test_horizon_accumulated_cost_under_500ms`. For `|P|=100, H=10`,
   expect ~100k ops per call (~5-50 ms range, still tractable). Operator
   should profile if extending to `H > 20`.

3. **Composes with NC30 family but with caveats:**
   - **NC30 b (`derive_zref`)**: the SAME `z_ref` is used across all H
     horizons. This is intentional (uniform reference makes per-horizon
     contributions comparable for accumulation), but it means the
     forecast frontier's `extremes` at deep `h` may sit OUTSIDE the
     initial `z_ref` — some contributions can go negative. The argmax
     still works (the relative ranking is preserved), but operators
     interpreting telemetry should know negative contributions are
     possible under accumulation.
   - **NC30 c (`variance_penalty`)**: applied to the ACCUMULATED `Q^A`
     using the DEEPEST-horizon variance trace. This is the strictest
     penalty (deepest h has largest Σ). An alternative would be to apply
     the penalty per-horizon (`Σ_h γ^h · (E[Δ_S_h] - α · trace(Σ_h))`),
     which would be smoother — deferred to NC35-v2 if the operator wants
     it.
   - **NC30 d (`tie_break_by_variance`)**: fires on the accumulated
     scores. Tie-detection epsilon is relative to the accumulated
     magnitude, so tie thresholds may need adjustment when H grows.

4. **Does NOT add forecast-uncertainty penalty to the accumulation itself.**
   Operator could extend to `Q^A_penalized(s) = Q^A(s) - β · Σ_h γ^h ·
   trace(Σ_h^{(s)})` (per-horizon uncertainty discount). Currently
   `variance_penalty` only penalizes the deepest-horizon uncertainty.
   Filed as candidate for NC35-v2.

5. **γ=0 is pathological.** Every horizon h ≥ 1 contributes
   `0^h · Δ_S = 0` to the sum. All candidates score 0; argmax is unstable
   (ties). The selector returns a real candidate (no crash), but the choice
   is arbitrary. Operator should not set `W22_NC29A_GAMMA=0` in practice.
   Test `test_horizon_accumulated_zero_gamma_uses_only_h1` documents the
   no-crash guard.

6. **MC mode (`analytical=False`) is NOT supported at H > 1.** The
   multi-period block uses the analytical per-horizon contribution
   regardless of the `analytical` kwarg. This is intentional: MC at H>1
   would multiply cost by `n_mc` per horizon and inject n_mc · H sort-order
   noise per call. Deferred to NC35-v2 if the operator wants stochastic
   accumulation.

---

## Operator action items for FTSE H-sweep validation

1. **Sweep H ∈ {1, 2, 3, 5, 10}** on FTSE monthly data with default γ=0.9,
   `derive_zref=True`, `variance_penalty=0.0`. Compare:
   - Final-wealth-curve area-under-curve vs the H=1 baseline.
   - Per-period AMFC↔Hv-DM agreement rate from telemetry
     (`amfc_agrees_with_hv_dm`).
   - Pick churn rate (how often the AMFC pick changes period-to-period —
     persistent winners should reduce churn).

2. **Compare H=1 to H=3** under three γ regimes: `0.7`, `0.9`, `0.99`.
   The H-NC35-multi-period-different hypothesis predicts the argmax
   disagreement rate vs H=1 should be ≥ 15% in at least one regime
   (per spec §B4).

3. **Stress-test the cost guard at |P|=50, H=10, n_mc=200**: expected
   under 2 s wall-clock per call. If exceeded, profile and consider
   numpy-vectorizing the per-horizon contribution loop.

4. **Compose with NC30 c**: rerun FTSE sweep with
   `variance_penalty ∈ {0.0, 0.1, 1.0}` × H ∈ {1, 3} grid. The NC30
   variance-penalty was tuned for single-period; under H=3 the deepest-
   horizon variance is much larger (compounded), so the same α may be
   too aggressive. Recommend tuning α down by a factor of `1/H` as a
   starting point.

5. **Telemetry-driven post-hoc analysis**: with `collect_telemetry=True`,
   the per-call dict now records `horizon_accumulated` so multi-mode
   experiments can be disambiguated in the same run. Use the telemetry
   to plot accumulated-score distributions per H.

---

*W22-NC35 — central session, 2026-05-20. Tests: 12 NC35 + 28 NC30 unchanged = 40 pass. Branch: feat/w22-inspection-backlog.*
