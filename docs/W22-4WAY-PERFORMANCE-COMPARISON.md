# W22 4-way performance comparison + reframing TIP saturation as benign

*Generated 2026-05-18 after NC8b (commit 27cbcd2), NC12 (commit 2154270),
NC13a (commit 3d41e91), and Probe E (commit fec6332).*

## Headline metric: ASMS−SMS Δ %

| Run | ASMS Ŝ | SMS Ŝ | Δ % | n | Wall |
|---|---|---|---|---|---|
| Baseline (post-NC7) | 0.000381 | 0.000405 | −5.92% | 2 | 1086s |
| NC8b only | 0.000422 | 0.000415 | **+1.70%** ✅ | 2 | 1135s |
| NC8b + NC12 | 0.000401 | 0.000414 | −3.09% | 2 | 1043s |
| **NC8b + NC12 + NC13a** | **0.000416** | **0.000423** | **−1.61%** | **3** | **970s** |

**Headline insight**: NC8b is the only shipped fix with empirically validated positive impact (+1.70% at n=2, but the mechanism is well-understood: fixes a W15-2 regression where offspring objectives bore no relation to weights). NC12 and NC13a are mathematically correct but appear to have either no effect or slight regression on multi-horizon ASMS scenarios.

## TIP saturation persists post-NC13a (99.80%)

| Run | TIP mean | TIP std | TIP saturation (0.45, 0.55) | λ_combined CoV |
|---|---|---|---|---|
| Pre-NC12 (post-NC7) | 0.4999 | 0.0158 | 99.86% | 0.0315 |
| Post-NC12 (NC8b+NC12) | 0.5001 | 0.0158 | 99.87% | 0.0315 |
| **Post-NC13a (NC8b+NC12+NC13a)** | 0.4999 | 0.0157 | **99.80%** | 0.0316 |

NC13a's clamp = 1.0 did NOT escape saturation. Even with predicted_cov ≤ 1.0:
- Steady-state P[:2,:2] = w_0² · current + Σ w_h² · 1.0 ≈ 0.5 (under λ ≈ 0.5)
- TIP MC samples: std = √0.5 ≈ 0.7 around means ≈ 0.001 → still pure noise → TIP ≈ 0.5

To get TIP discrimination at portfolio level, we'd need std comparable to ROI differences (~ 5e-4) → P ~ 2.5e-7. That's a 1e7 reduction from clamp = 1.0. Not practical via a simple clamp.

## REFRAMING: TIP saturation may be BENIGN for v2_anticipative_rate scenarios

The `ASMS_mHDM_K3_v2both` scenario uses `use_v2_anticipative_rate=True`, which overrides paper Eq 7.16 with v2's formula:

```
λ_combined = 1 − TIP
```

So when TIP ≈ 0.5, λ_combined ≈ 0.5 (constant across all portfolios). The anticipation effect (anticipative-mean overwrite of ROI/risk):

```
anticipative_ROI = w_0 · current_ROI + Σ w_h · predicted_ROI_h
```

If λ_h ≈ 0.5 uniformly, the anticipation blends ALL portfolios identically — a constant shift to objectives. **This does NOT change relative ranking** → selection is unaffected by TIP saturation in this regime.

Under thesis Eq 7.16 (`use_v2_anticipative_rate=False`):
```
λ_combined = 0.5 · (λ^H + λ^K) where λ^H = (1 − H(TIP)) / (H − 1)
```
When TIP = 0.5, H(TIP) = 1, λ^H = 0. So anticipation arm is silently disabled — ASMS reverts to SMS behavior. Worse than v2 in this regime.

**Implication**: NC8b's +1.70% gain comes from SELECTION QUALITY (objectives match weights), not from anticipation effect. Fixing TIP saturation may not move ASMS performance directly — only by enabling per-portfolio differentiation in the anticipation blend (which requires λ to vary per portfolio).

## Refined hill-climb priorities

Given the reframing, NC13a/c/d (TIP saturation fixes) are LOWER priority. The high-leverage remaining moves are:

| P | Item | Mechanism | Predicted ROI |
|---|---|---|---|
| **P2** | **NC8c — cross-period KF state persistence** | KF carries velocity learning across periods → predicted state actually differs from persistence → anticipation arm has signal to leverage per-portfolio | HIGH |
| P3 | Probe C (AMFC vs alternatives) | If AMFC is uninformative, replace DM with HighROI/Sharpe/Median | MEDIUM-HIGH (independent of KF) |
| P4 | NC15 (NEW) — λ-per-portfolio variation | Even with constant TIP, make λ depend on per-portfolio uncertainty → portfolios with HIGH uncertainty get LESS anticipation weight | MEDIUM |
| P5 | Wider 5-seed NC8b validation | Statistical confidence on the +1.70% | confirmation only |
| P6 | NC13c (TIP uses R, not P) | Decouples TIP from KF P drift; deferred since reframing suggests benign | LOW |
| P7 | PO(8,1.0) loader | Tests strongest-signal dataset | sidecar |

## Decision: ship NC8c next

NC8c (cross-period KF state persistence) is the highest-leverage remaining change. Mechanism:

1. Per period t > 0, load PREVIOUS period's AMFC portfolio's KF state
2. Use it to initialize the KF state of new period's portfolios (instead of fresh `[ROI, risk, 0, 0]`)
3. KF velocity now reflects period-over-period dynamics observed at the implementation level

Implementation plan:
- `walk_forward.py`: save `amfc_solution.P.kalman_state` post-AMFC selection; pass to next period via `previous_kf_state`
- `sms_emoa._initialize_population`: if `previous_kf_state` provided in `data`, use it as initial KF state for new portfolios (with their actual ROI/risk as fresh measurements via kalman_update)

This requires:
- Data plumbing: previous_kf_state through walk_forward → mgr → sms_emoa
- Initialization logic: how to apply previous KF state to N new portfolios (broadcast same state? specialize per portfolio?)
- Probe A re-run to verify KF predictions diverge from persistence

Risk: medium. Plumbing is straightforward; initialization logic needs care.

Predicted impact: if successful, KF actually predicts t+1 (not persistence), and:
- Probe A would show KF MAE < persistence MAE with Wilcoxon p < 0.025
- Anticipation arm would have signal to leverage
- ASMS−SMS Δ should improve further from NC8b alone (+1.70% → ?)
- Combined with NC13c (TIP fix), per-portfolio differentiation also unlocked

## What we have learned (lessons)

1. **Stale state was the load-bearing bug**: NC8b (one helper function) closed a 7.6 pp performance gap. The W15-2 regression silently dropped a 1-line compute_efficiency call.

2. **Mathematical correctness ≠ production impact**: NC12 fixed Eq 15 in `AnticipativeDistribution` (a real bug) but had ZERO effect on the production code path used by ASMS scenarios (multi-horizon).

3. **TIP saturation is a downstream symptom, not the root cause**: Under v2_anticipative_rate, saturation gives λ = 0.5 uniformly → benign at selection level. Under thesis Eq 7.16, saturation disables anticipation → reverts to SMS.

4. **The KF needs cross-period state to be predictive**: Within a period, all measurements are from the same training window — no temporal dynamics for velocity to learn. Per-period KF reset destroys whatever velocity signal might exist.

5. **Read-before-write discipline**: Probe E (zero compute cost; reused Probe A's predictions.jsonl) revealed the 4860× P drift between ASMS and SMS, which informed NC13a. Reading existing data first is high-leverage diagnostics.
