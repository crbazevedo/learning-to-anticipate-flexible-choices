# W22 Probes Synthesis — parallel autonomous push (FINAL)

**Status:** complete; 13 probes shipped, 177 tests passing, 0 in flight
**Trigger:** operator directive 2026-05-20 — "more parallel probes; go"
**Branch:** `feat/w22-inspection-backlog`
**Pattern:** each probe is a CONTAINED standalone module (`python_refactor/src/probes/probe_*.py`) + tests + design doc. NO modifications to shared code paths. Operator can integrate independently.

---

## Shipped probes (13)

| Probe | Topic | Commit | Tests | Key empirical finding |
|---|---|---|---|---|
| Q-v1 | Per-asset AR(1) prediction | `b835dcb` | 18/18 | AR(1) recovers ρ within 0.05 of true ρ=0.7; beats no-change on AR(1) data |
| S-v1 | Asset network centrality | `e8403cb` | 14/14 | Eigenvector centrality correctly identifies star-graph hub; pure numpy, no networkx |
| T | NC29a γ sensitivity | `087a96d` | 6/6 | γ=0.99 saturates clamp; γ=0.9 default in sweet spot (validates NC29a default) |
| U | NC30 c variance_penalty α sensitivity | `1871489` | 9/9 | Synthetic flip point at α≈1.80; meaningful operating range α ∈ [0.01, 1] |
| Y | NC13b smooth clamp impact | `c6022c6` | 13/13 | At 40% TIP saturation: 100% signal recovery, mean diff 4.7e-2 — HIGH-regime matters |
| AA | AMFC vs Hv-DM telemetry harness | `eb4753e` | 10/10 | Production-ready analyzer for telemetry summarization + side-by-side comparison |
| AD | Stochastic vs deterministic Δ_S | `881d0f2` | 10/10 | **CRITICAL BUG**: stochastic correction OVERSHOOTS in multi-solution regime due to rectangle mismatch in sms_emoa.py:772-781 (SCAR-pinned by regression test) |
| Z | Stability factor restoration | `50ced01` | 12/12 | trace(P) spread ≥0.5: legacy/v2 disagreement 50%+; stability factor is LOAD-BEARING |
| AC | KF NIS/NEES consistency diagnostics | `6aed91d` | 17/17 | Bar-Shalom χ² tests; correctly detects under-tuned R/Q; documents NEES Monte-Carlo methodology |
| AI | Pareto front size sensitivity | `97bb397` | 11/11 | 91-100% boundary picks across \|P_t\|=2-50; confirms Inspection 6 boundary-bias finding |
| AF | Mehra Q-noise estimation | `6b0f7b9` | 9/9 | Innovation-autocorrelation-based Q estimator; handles Mehra-observability edge case |
| AB | λ^K per-portfolio differentiation | `da960ce` | 19/19 | Per-portfolio λ^K computation with discrimination significance classifier (NEGLIGIBLE/MODEST/STRONG) |
| V | NC opt-in fix combination ablation | `a2ddda2` + `96ceec7` | 21/21 | Full/pairwise/single strategies (16/7/5 combos); mock wealth_fn ready for ASMS plug-in |
| AI-2 | NC30 b z_ref boundary asymmetry | `b27b549` | 8/8 | **ASYMMETRIC FIX**: NC30 b nulls LEFT boundary contribution but RIGHT still wins on full risk range |

**Cumulative this push**: 177 tests; **all passing in 5.83s**.

---

## Key algebraic insights extracted

1. **Stability factor is LOAD-BEARING** (Probe Z): synthetic 50%+ argmax disagreement between legacy and v2 at realistic trace spread. Reading-F's flip lost a non-trivial signal.

2. **γ=0.9 is the sweet spot for NC29a** (Probe T): γ=0.99 saturates the [0, 0.5] clamp (signal vanishes); γ=0.5 over-decays. Default validated.

3. **AMFC variance penalty meaningful in α ∈ [0.01, 1]** (Probe U): synthetic flip point near α≈1.80; α>10 over-dominates.

4. **NC13b matters proportionally to TIP saturation rate** (Probe Y): HIGH-saturation regimes recover 100% of tail gradient.

5. **Eq 6.41 stochastic correction OVERSHOOTS in multi-solution regimes** (Probe AD): rectangle mismatch in sms_emoa.py:772-781 documented as scar; now SCAR-pinned. Operator decision needed: realign deterministic to Eq 6.41, OR redo stochastic correction for legacy rectangle.

6. **AR(1) correctly recovers known ρ** (Probe Q-v1): estimator validated within 0.05 of ρ=0.7 on 4000-step trajectory.

7. **Eigenvector centrality is well-defined and computable** (Probe S-v1): star-graph hub correctly identified; hub_tilt API verified.

8. **KF NIS/NEES tests correctly detect under-tuned R/Q** (Probe AC): Bar-Shalom χ²-based tests verified; methodology adjusted to MC across realizations for NEES (single-realization NEES is time-correlated).

9. **AMFC boundary bias is structural** (Probe AI): 91-100% boundary picks across all tested |P_t|. The contribution formula structurally favors high-ROI extreme.

10. **NC30 b data-derived z_ref is NOT a complete fix for boundary bias** (Probe AI-2): asymmetric — nulls LEFT boundary's contribution (R1 = min ROI → leftmost = 0) but RIGHT still wins because (R2 - smallest_risk) = full risk range.

11. **Mehra Q-estimation works under observability** (Probe AF): innovation autocorrelation → Q estimate; correctly rejects unobservable cases (e.g., F=I_4).

12. **λ^K per-portfolio differentiation can be measured ex-ante** (Probe AB): discrimination significance classifier (NEGLIGIBLE / MODEST / STRONG) tells operator whether the per-portfolio shift would even matter.

13. **Combination ablation framework is plug-and-play** (Probe V): operator can plug in real ASMS wealth_fn; framework handles 2^4=16 combinations with full/pairwise/single strategies.

---

## Operator decision points surfaced

| # | Probe | Decision | Cost to act |
|---|---|---|---|
| 1 | AD | Realign deterministic rectangle OR fix stochastic correction (sms_emoa.py:772-781) | M — touches shared code; needs regression battery |
| 2 | Z | Empirically validate legacy vs Reading-F's v2 on FTSE n=10 | L — ASMS runs ~10 min × 20 seeds |
| 3 | AI / AI-2 | Is high-ROI boundary preference desirable? If no, replace z_ref scheme | M — design decision |
| 4 | AC + AF | Instrument KF predict-update loop to capture innovations; run AC + AF diagnostics on FTSE | M — KF instrumentation + telemetry plumbing |
| 5 | V | Run full ablation on FTSE n=10 using mock wealth_fn replaced with real ASMS | XL — 16 combos × 10 seeds × 10 min = 27 hrs serial |
| 6 | T / U / Y | These probes confirm structural fix defaults are sensible; no urgent decision |  |

---

## File index

```
python_refactor/src/probes/
  __init__.py
  probe_q_ar1_predictor.py              # Q-v1
  probe_s_centrality.py                 # S-v1
  probe_t_gamma_sensitivity.py          # T
  probe_u_alpha_sensitivity.py          # U
  probe_y_smooth_clamp_impact.py        # Y
  probe_aa_amfc_telemetry_analyzer.py   # AA
  probe_ad_delta_s_comparison.py        # AD
  probe_z_stability_restoration.py      # Z
  probe_ac_kf_diagnostics.py            # AC
  probe_ai_front_size_sensitivity.py    # AI
  probe_af_mehra_q_estimation.py        # AF
  probe_ab_per_portfolio_lambda_k.py    # AB
  probe_v_combination_ablation.py       # V
  probe_ai2_zref_boundary_invariance.py # AI-2

python_refactor/tests/  (177 tests, all passing in 5.83s)
  test_probe_q_ar1.py                   # 18
  test_probe_s_centrality.py            # 14
  test_probe_t_gamma_sensitivity.py     # 6
  test_probe_u_alpha_sensitivity.py     # 9
  test_probe_y_smooth_clamp_impact.py   # 13
  test_probe_aa_telemetry_analyzer.py   # 10
  test_probe_ad_delta_s_comparison.py   # 10
  test_probe_z_stability_restoration.py # 12
  test_probe_ac_kf_diagnostics.py       # 17
  test_probe_ai_front_size_sensitivity.py # 11
  test_probe_af_mehra_q_estimation.py   # 9
  test_probe_ab_per_portfolio_lambda_k.py # 19
  test_probe_v_combination_ablation.py  # 21
  test_probe_ai2_zref_boundary_invariance.py # 8

python_refactor/scripts/
  probe_aa_run_telemetry_smoke.py       # CLI for synthetic AA smoke

docs/
  W22-PROBE-Q-V1-AR1.md
  W22-PROBE-S-V1-CENTRALITY.md
  W22-PROBE-T-GAMMA-SENSITIVITY.md
  W22-PROBE-U-ALPHA-SENSITIVITY.md
  W22-PROBE-Y-SMOOTH-CLAMP-IMPACT.md
  W22-PROBE-AA-AMFC-TELEMETRY.md
  W22-PROBE-AD-DELTA-S-COMPARISON.md
  W22-PROBE-Z-STABILITY-RESTORATION.md
  W22-PROBE-AC-KF-DIAGNOSTICS.md
  W22-PROBE-AI-FRONT-SIZE.md
  W22-PROBE-AF-MEHRA-Q.md
  W22-PROBE-AB-PER-PORTFOLIO-LAMBDA-K.md
  W22-PROBE-V-COMBINATION-ABLATION.md
  W22-PROBE-AI2-ZREF-INVARIANCE.md
  W22-PROBES-SYNTHESIS.md (this doc)
```

---

## Discipline preserved

- **Containment**: every probe is a NEW file under `src/probes/`, `tests/`, `scripts/`, `docs/`. NO modifications to shared code paths (sms_emoa.py, anticipatory_learning.py, multi_horizon_anticipatory.py, temporal_incomparability_probability.py, amfc_selector.py).
- **Test coverage**: 177 new tests; all passing in 5.83s. Zero regressions to pre-existing tests.
- **Honest scars**: each probe doc lists 3-6 expected scars; multiple probes (Y, U, AI, AI-2) include SCAR-pinned regression tests to prevent silent regression.
- **Operator-action separation**: probes diagnose; integration into ASMS/FTSE-run is operator's call.

---

## Branch status (final)

- 14 commits ahead of base (synthesis adds 1 more)
- 177 probe tests + 125 prior NC tests + 28 multi-horizon tests + 23 TIP tests = **353+ green tests across the W22 instrumentation**
- Zero regressions in the broader 700+ test surface
- Ready for operator review / FTSE empirical validation
