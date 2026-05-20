# W22 Probes Synthesis — parallel autonomous push

**Status:** active; 9 probes shipped, 3 in flight
**Trigger:** operator directive 2026-05-20 — "more parallel probes; go"
**Branch:** `feat/w22-inspection-backlog`
**Pattern:** each probe is a CONTAINED standalone module (`python_refactor/src/probes/probe_*.py`) + tests + design doc. NO modifications to shared code paths. Operator can integrate independently.

---

## Shipped probes (9)

| Probe | Topic | Commit | Tests | Key empirical finding |
|---|---|---|---|---|
| Q-v1 | Per-asset AR(1) prediction | `b835dcb` | 18/18 | AR(1) recovers ρ within 0.05 of true ρ=0.7; beats no-change on AR(1) data |
| S-v1 | Asset network centrality | `e8403cb` | 14/14 | Eigenvector centrality on star graph correctly identifies hub; hub_tilt API verified |
| T | NC29a γ sensitivity | `087a96d` | 6/6 | γ=0.99 saturates at clamp (signal vanishes); γ=0.9 default is in sweet spot |
| U | NC30 c variance_penalty α sensitivity | `1871489` | 9/9 | Synthetic flip point at α≈1.80 (5-candidate scenario); meaningful range α ∈ [0.01, 1] |
| Y | NC13b smooth clamp impact | `c6022c6` | 13/13 | At 40% TIP saturation: 100% signal recovery, mean output diff 4.7e-2; verdict HIGH-saturation matters |
| AA | AMFC vs Hv-DM telemetry harness | `eb4753e` | 10/10 | Production-ready analyzer for telemetry summarization + side-by-side comparison |
| AD | Stochastic vs deterministic Δ_S | `881d0f2` | 10/10 | **CRITICAL BUG**: stochastic correction OVERSHOOTS in multi-solution regime due to rectangle mismatch in sms_emoa.py:772-781 (already documented scar; now SCAR-pinned by regression test) |
| Z | Stability factor restoration | `50ced01` | 12/12 | At trace(P) spread ≥0.5: legacy/v2 disagreement 50%+; stability factor is LOAD-BEARING |
| AC | KF NIS/NEES consistency diagnostics | `6aed91d` | 17/17 | Bar-Shalom χ² tests verified; correctly detects under-tuned R/Q; documented spec-correction (scipy.stats.chi2 truth over example values) and NEES-MC methodology |

**Cumulative tests added this push**: 109 (across 9 probes); all passing.

## In-flight probes (3 — agents may complete by next status check)

| Probe | Topic | Status |
|---|---|---|
| AF | Mehra Q-noise estimation from innovation autocorrelation | Dispatched |
| AB | Per-portfolio λ^K differentiation analyzer | Dispatched |
| V | NC opt-in combination ablation framework (2^4 = 16 combos) | Dispatched |

---

## What the shipped probes ENABLE (operator action items)

### Already-actionable (probe results suffice without further work)

1. **Probe AD's rectangle-mismatch finding** is a STRUCTURAL BUG, not a research question. The deterministic middle-branch uses `(ROI_i - ROI_{i+1})(risk_{i-1} - risk_i)` while Eq 6.41 specifies `(ROI_i - ROI_{i-1})(risk_{i+1} - risk_i)`. The stochastic `- Cov` correction is derived against the Eq 6.41 rectangle, so subtracting from the legacy rectangle overshoots. Operator can decide:
   - Realign deterministic to Eq 6.41 rectangle (will trip Probe AD's SCAR test)
   - OR redo the stochastic correction to match the legacy rectangle
   - Either way, the current implementation has a known overshoot in multi-solution regimes

2. **Probe T's γ=0.99 saturation finding** confirms NC29a's structural premise: γ=0.99 means λ^H stays at the [0, 0.5] clamp boundary across horizons (clamp signal vanishes). γ=0.9 (default) is the sweet spot per the balance metric.

3. **Probe Z's 50%+ disagreement finding** at realistic trace spread confirms that Reading-F's v2 mode (stability_factor=1.0) is NOT equivalent to the legacy mode. The two pick DIFFERENT solutions in half of synthetic populations. Operator must validate Reading-F empirically.

### Needs FTSE run to act

4. **Probe Y verdict pending real TIP distribution**: synthetic HIGH-saturation regime shows 100% signal recovery, but production TIPs may not be saturated. Probe AA (telemetry harness) can collect real TIPs; pipe into Probe Y for the verdict.

5. **Probe U α flip-point pending real (E[Δ_S], variance) joint distribution**: synthetic flip at α≈1.80; real flip depends on production statistics.

### Compose with each other

6. **Probe AA telemetry + Probe Y saturation summary**: AA collects raw TIPs from production; Y classifies the regime; combined verdict on whether NC13b matters in production.

7. **Probe T γ-sensitivity + Probe AA lambda_h trace**: AA's telemetry hook can emit lambda_h per period; T's analyzer can characterize the effective γ in production (back-derive γ from the observed decay).

---

## What we LEARNED algebraically (key insights independent of empirical work)

1. **Stability factor is load-bearing** (Probe Z): synthetic 50%+ disagreement between legacy and v2 at realistic trace spread. Reading-F's claim of "v2 better despite ignoring stability" requires the gain to COMPENSATE for losing a non-trivial signal.

2. **γ=0.99 is a NC29a edge case** (Probe T): high γ saturates the [0, 0.5] λ^H clamp; signal vanishes. γ=0.9 default is principled.

3. **AMFC's variance penalty α has a meaningful flip regime** (Probe U): α ∈ [0.01, 1] is where flip transitions happen on synthetic data; α=10+ over-dominates.

4. **NC13b smooth clamp matters proportionally to saturation rate** (Probe Y): if TIPs saturate (≥20% in tails), smooth clamp recovers 100% of tail gradient.

5. **Eq 6.41 stochastic correction overshoots** (Probe AD): documented scar in sms_emoa.py:772-781 is REAL and produces wrong-direction Δ_S errors in multi-solution regimes. Pinned by SCAR regression test in Probe AD.

6. **AR(1) correctly recovers known ρ** (Probe Q-v1): on synthetic AR(1) with ρ=0.7, estimated ρ within 0.05 in 4000-step trajectory. Confirms the estimator is sound; production deployment requires Q-v1 → portfolio optimizer wiring (out of probe scope).

7. **Eigenvector centrality works as advertised** (Probe S-v1): star graph correctly identifies hub; Gini increases with concentration. Production deployment requires hub_tilt → portfolio optimizer wiring (out of probe scope).

---

## File index (probes shipped this push)

```
python_refactor/src/probes/
  __init__.py
  probe_q_ar1_predictor.py             # Q-v1
  probe_s_centrality.py                # S-v1
  probe_t_gamma_sensitivity.py         # T
  probe_u_alpha_sensitivity.py         # U
  probe_y_smooth_clamp_impact.py       # Y
  probe_aa_amfc_telemetry_analyzer.py  # AA
  probe_ad_delta_s_comparison.py       # AD
  probe_z_stability_restoration.py     # Z
  (probe_ab, ac, af, v pending agent completion)

python_refactor/tests/
  test_probe_q_ar1.py                  # 18 tests
  test_probe_s_centrality.py           # 14 tests
  test_probe_t_gamma_sensitivity.py    # 6 tests
  test_probe_u_alpha_sensitivity.py    # 9 tests
  test_probe_y_smooth_clamp_impact.py  # 13 tests
  test_probe_aa_telemetry_analyzer.py  # 10 tests
  test_probe_ad_delta_s_comparison.py  # 10 tests
  test_probe_z_stability_restoration.py # 12 tests

python_refactor/scripts/
  probe_aa_run_telemetry_smoke.py      # CLI for synthetic AA smoke

docs/
  W22-PROBE-Q-V1-AR1.md
  W22-PROBE-S-V1-CENTRALITY.md
  W22-PROBE-T-GAMMA-SENSITIVITY.md
  W22-PROBE-U-ALPHA-SENSITIVITY.md
  W22-PROBE-Y-SMOOTH-CLAMP-IMPACT.md
  W22-PROBE-AA-AMFC-TELEMETRY.md
  W22-PROBE-AD-DELTA-S-COMPARISON.md
  W22-PROBE-Z-STABILITY-RESTORATION.md
  (W22-PROBE-AB/AC/AF/V-*.md pending agent completion)
```

---

## Backlog (next probes to dispatch)

Per docs/W22-MASTER-BACKLOG.md Section D, remaining probes that could
ship in further parallel pushes:

| Probe | Topic | Already designed? |
|---|---|---|
| AE | Asset universe scaling — ASMS uplift vs K (30/81/31) | docs designed |
| AG | F-matrix selection (constant-velocity vs random-walk) | docs designed |
| AH | Aggregated AMFC vs Hv-DM weighting | docs designed |
| Q-v2 | GARCH(1,1) extension | docs designed |
| Q-v3 | VAR(1) cross-asset | docs designed |
| R | Granger causality (high risk per literature) | docs designed |
| S-v2 | Centrality-augmented KF state | docs designed |

Additional probe ideas (not yet designed in detail):
- AI: Pareto front size sensitivity (does AMFC depend on \|P_t\|?)
- AJ: Cardinality K sensitivity (10, 15, 30, 87 assets)
- AK: Window K (Eq 6.9) sensitivity (K=0, 3, 5, 10)
- AL: Transaction cost sensitivity
- AM: Initial wealth sensitivity
- AN: NIS-bin histogram per period (KF tuning telemetry)
- AO: Cross-Probe-correlation matrix (do Q/R/S signals correlate?)
- AP: Random-baseline floor (n=100 random portfolios vs ASMS)
- AQ: Bootstrap CI on ASMS-vs-baseline gain
- AR: Calibration plot for predicted vs realized risk
- AS: Concordance index between Δ_S ranking and realized gain ranking
- AT: AMFC vs MEDIAN-DM crossover

---

## Discipline preserved across the push

- **Containment**: every probe is a NEW file under `src/probes/`, `tests/`, `scripts/`, `docs/`. NO modifications to shared code paths (sms_emoa.py, anticipatory_learning.py, multi_horizon_anticipatory.py, temporal_incomparability_probability.py, amfc_selector.py).
- **Test coverage**: 92 new tests across 8 probes; all passing.
- **Honest scars**: each probe doc lists 3-6 expected scars (regime dependence, synthetic ≠ production, composition with other probes).
- **Operator action separation**: probe SHIPS the analyzer; integration into ASMS / FTSE-run is operator's call. Probe results are diagnostic, not prescriptive.
