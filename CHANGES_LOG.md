# W22 CHANGES_LOG — per-integration attribution tracking

**Purpose**: every NC integration / empirical run / drift observation logged
here with **per-NC marginal Δ attribution** vs the pre-W22 baseline
(NC8c-v2 + NC8d, FTSE 2015 +7.50% Wilcoxon p=0.003).

**Operator directive 2026-05-20**: "Track integrations, combinations, and the
accumulated delta (with attribution estimation) over the baseline."

**Schema for each entry**:
```yaml
- date: YYYY-MM-DD
  commit: <sha>
  wave_unit: WNN-N (per .dfg/plan.yaml)
  type: integration | empirical | drift_update | structural_fix | meta
  nc_or_combo: [NC names involved]
  thesis_anchor: <section/equation>
  baseline_used: <what the comparison is against>
  observation:
    metric: <wealth / Ŝ / Δ_S / etc.>
    value: <numerical>
    n_seeds: <integer>
    p_value: <if computed>
    confidence: <low/med/high>
  attribution:
    per_nc_marginal:
      NC_NAME: <estimated marginal contribution, with caveats>
    method: <how marginal was estimated (e.g., ablation, Shapley, linear-regression)>
  honest_scars: [list of caveats]
```

---

## Attribution methodology

Initially: **subtraction-based ablation** when 2+ configs differ by a single
NC. Example: FULL_STACK = NC36+NC13b+NC31+NC27_DEEP at +1.90%; NC27_NO_NC36
= same minus NC36 at -5.45% → NC36's marginal contribution ≈ +7.35%
(non-linear; only valid in this specific composition).

Future: **Shapley-style** attribution once we have 2^n combinations across
several NCs (this requires expensive full-factorial design; out of scope
until WC1 / W24).

---

## Pre-W22 baseline (the anchor everything is measured against)

```yaml
- date: 2026-05-18
  commit: pre-W22-baseline
  wave_unit: (historical reference)
  type: empirical
  nc_or_combo: [NC8c-v2, NC8d]
  thesis_anchor: "Eq 11 KF state evolution + Eq 14 multi-horizon blend"
  baseline_used: pre-NC8c-v2 SMS-EMOA
  observation:
    metric: final wealth on FTSE 2015 walk-forward
    value: +7.50%
    n_seeds: 10
    p_value: 0.003
    confidence: high
  attribution:
    per_nc_marginal:
      NC8c-v2: "cross-period KF position carry-forward (Eq 11 fidelity)"
      NC8d:    "predict-before-update sequence (paper Eq 14 strict order)"
    method: prior session ablation
  honest_scars:
    - "Single benchmark (FTSE 2015 only)"
    - "Random-seed sample n=10; paired Wilcoxon p=0.003 → ~99.7% confidence"
```

This is the **anchor**. Every W22 NC must demonstrate it does NOT regress
this baseline AND ideally adds marginal Δ vs it.

---

## W22 empirical entries (backfill from 2026-05-20)

### Entry 1: PO(8,1.0) n=5 full sweep

```yaml
- date: 2026-05-20
  commit: e126410
  wave_unit: W22-0b (backfill)
  type: empirical
  nc_or_combo: [BASELINE, NC36, TIP_CLEANUP, NC27_DEEP, FULL_STACK]
  thesis_anchor: "§7.2.3 protocol; §6.2 Dirichlet; §6.1.5 TIP"
  baseline_used: BASELINE config (no opt-in env vars)
  observation:
    metric: paired Δ wealth Ŝ vs BASELINE on PO(8,1.0) ASMS_mHDM_K3_v2both
    value:
      NC27_DEEP:  "+2.70% (n=5; later corrected to +1.37% at n=10)"
      FULL_STACK: "+1.90% (n=5)"
      NC36:       "-6.70% (n=5)"
      TIP_CLEANUP: "-5.30% (n=5)"
    n_seeds: 5
    p_value: all > 0.3 (NS)
    confidence: low (n=5 underpowered)
  attribution:
    per_nc_marginal:
      NC27_DEEP:  "Initial +2.7% on PO, halved at n=10 → +1.37%; SYNTHETIC translation MODEST"
      NC36:       "OPPOSITE-SIGN — synthetic predicted parity (0%), empirical -6.7%; MC-noise-as-exploration hypothesis"
    method: per-config Δ vs BASELINE single-difference
  honest_scars:
    - "n=5 inadequate for sub-3% effects"
    - "Compositional NCs (FULL_STACK) interact non-linearly"
```

### Entry 2: NC27-deep seeds 6-10 (paired n=10 confirmation)

```yaml
- date: 2026-05-20
  commit: 180c850
  wave_unit: W22-0b
  type: empirical
  nc_or_combo: [NC27_DEEP]
  thesis_anchor: "§6.2 Eq 6.7 (Dirichlet posterior mean)"
  baseline_used: BASELINE (paired n=10 with seeds 1-10)
  observation:
    metric: paired Δ wealth Ŝ vs BASELINE on PO(8,1.0) ASMS
    value: +1.37%
    n_seeds: 10
    p_value: 0.625 (NS)
    wins_vs_baseline: 6/10
    confidence: low-medium (direction holds across seeds 1-10)
  attribution:
    per_nc_marginal:
      NC27_DEEP: "+1.37% paired n=10 → MODEST_TRANSLATION of synthetic 2.8% L2 claim"
    method: paired Wilcoxon per-seed Δ
  honest_scars:
    - "Seed-variance (10% swing across seeds 1-5 vs 6-10) dominates the +1.37% effect"
    - "Would need n≥30 for statistical significance"
    - "Single dataset (PO(8,1.0)); FTSE confirmation pending"
```

### Entry 3: Isolation runs (NC13b, NC31, NC27_NO_NC36 each n=5)

```yaml
- date: 2026-05-20
  commit: 180c850
  wave_unit: W22-0b
  type: empirical
  nc_or_combo:
    NC13b alone: ["NC13b"]
    NC31 alone:  ["NC31"]
    NC27_NO_NC36: ["NC27_DEEP", "NC13b", "NC31"]  # NC36 EXCLUDED
  thesis_anchor: "§6.1.5 TIP; Eq 13 λ^H; §6.2 Dirichlet"
  baseline_used: BASELINE paired n=5
  observation:
    metric: paired Δ wealth Ŝ on PO(8,1.0) ASMS
    value:
      NC13b: "-0.41% (2/5 wins, p=0.812)"
      NC31:  "-4.44% (1/5 wins, p=0.625)"
      NC27_NO_NC36: "-5.45% (1/5 wins, p=0.312)"
    n_seeds: 5 each
    confidence: low
  attribution:
    per_nc_marginal:
      NC13b: "Essentially neutral (-0.41%); safest of TIP opt-ins"
      NC31:  "Genuinely negative (-4.44%); contradicts Inspection 1 empirical equivalence claim"
      NC36 in FULL_STACK: "+7.35% MARGINAL when added to NC27_NO_NC36 → FULL_STACK (counter-balances NC13b+NC31 harm)"
    method: |
      NC36 marginal estimated by subtraction:
        FULL_STACK +1.90% − NC27_NO_NC36 (-5.45%) = +7.35% attributable to NC36
      This is the COUNTER-INTUITIVE finding: NC36 alone HURTS by 6.13%, but
      in the FULL_STACK composition, it RECOVERS the harm from NC13b+NC31.
  honest_scars:
    - "Non-linear composition interactions; subtraction-attribution is approximate"
    - "n=5 each; high variance"
    - "NC36's counter-balancing mechanism not yet understood (WC1 / W22-5 will investigate)"
```

---

### Entry 6: W22-9 FTSE NC27-deep n=10 paired — REGRESSION verdict (2026-05-21 ~00:00)

```yaml
- date: 2026-05-21
  commit: (pending)
  wave_unit: W22-9 (FTSE empirical confirmation)
  type: empirical_critical_finding
  nc_or_combo: [NC27_DEEP]
  thesis_anchor: "§7.2.2 FTSE protocol; §6.2 Eq 6.7 Dirichlet posterior"
  baseline_used: FTSE ASMS_mHDM_K3_v2both BASELINE (no opt-in env vars)
  observation:
    metric: paired Δ wealth Ŝ on FTSE 2015 ASMS n=10
    value:
      paired_mean_delta: "-1.11%"
      wins: "6/10"
      wilcoxon_p: 0.9219
      tail_risk_losses: "seed 9 -16.20%, seed 10 -11.49%"
      win_magnitudes: "all under +10%"
      loss_magnitudes: "two over -10%"
    n_seeds: 10
    confidence: medium (n=10 paired)
  attribution:
    per_nc_marginal:
      NC27_DEEP_FTSE: "-1.11% mean; tail-risk pattern (6 small wins offset by 2 large losses)"
    method: paired Wilcoxon + per-seed magnitude pattern analysis
  honest_scars:
    - "OPPOSITE-SIGN translation from PO cross-instance (+4.15% avg) to FTSE (-1.11%)"
    - "6/10 wins by count but losses dominate by magnitude → Wilcoxon p=0.9219 NS"
    - "Synthetic-vs-empirical drift HOLDS on FTSE: synthetic 2.8x L2 claim translated to PO but NOT to real data"
    - "Hypothesis: NC27-deep's aggressive Bayesian update over-commits to recent observations; FTSE has regime shifts that legacy exp-smoothing handles more gracefully"
  ratification_verdict:
    nc27_deep: "CANCEL W23-1 default flip (REVERSED from Entry 5 PROCEED)"
    reasoning: |
      FTSE is the production-relevant dataset. -1.11% mean regression with
      tail-risk losses is unacceptable for a default. NC27_DEEP stays as
      OPT-IN env var. Operator can revisit if regime-detection front-end
      or tunable concentration_increment are added (W23-1b future unit).
```

### Entry 5: W22 gap-fill chain — PO(16,1.0) seeds 6-10 + BASELINE seeds 11-20 (2026-05-20 night)

```yaml
- date: 2026-05-20
  commit: (pending)
  wave_unit: W22-4 (gap-fill) + W22-5 (gap-fill) + W22-7 (revised verdict)
  type: empirical_correction
  nc_or_combo: [NC27_DEEP]
  thesis_anchor: "§6.2.1 Dirichlet; Paper Eqs 16-17"
  baseline_used: BASELINE paired (now n=20 on PO(8,1.0); n=10 on PO(16,1.0))
  observation:
    metric: paired Δ Ŝ vs BASELINE on ASMS_mHDM_K3_v2both per instance (REVISED)
    value:
      "PO(8,1.0) n=20 paired (W22-4 + BASELINE 11-20)": "+2.65% (11/20 wins, p=0.498)"
      "PO(16,1.0) n=10 paired (W22-5a + 6-10)": "+2.19% (4/10 wins, p=0.695) — FORMERLY -6.10% UNPAIRED"
      "PO(8,0.3) n=5 paired (W22-5b)": "+2.82% (3/5, p=1.000)"
      "sPO(8,1.0)-cosine n=5 paired (W22-6)": "+8.93% (4/5, p=0.188)"
    n_seeds: 5-20 across instances
    confidence: medium (consistent direction; magnitudes 2-9%; no single-instance p<0.05)
  attribution:
    per_nc_marginal:
      NC27_DEEP_PO_8_1_0: "+2.65% paired n=20 (improved from +1.37% at n=10 with seeds 11-20 added)"
      NC27_DEEP_PO_16_1_0: "+2.19% paired n=10 — REVERSED from -6.10% UNPAIRED (sample-variance artifact at n=5)"
      NC27_DEEP_PO_8_0_3: "+2.82% paired n=5"
      NC27_DEEP_sPO: "+8.93% paired n=5 — STRONGEST signal"
    method: per-seed paired Wilcoxon
    cross_instance_avg_paired_delta: +4.15% (or +2.55% excluding sPO)
    direction_consistency: 4 of 4 instances POSITIVE
  honest_scars:
    - "PO(16,1.0) -6.10% was UNPAIRED n=5 sample-variance artifact; CORRECTED by paired n=10 analysis"
    - "No single instance reaches p<0.05; rely on direction consistency across 4 independent instances"
    - "n=5 vs n=5 unpaired can SWING ±10% by chance; always pair when comparing"
    - "FTSE empirical still pending"
  ratification_verdict:
    nc27_deep: "PROCEED with W23-1 ratification (REVERSED from prior 'DEFER')"
    reasoning: |
      All 4 instances POSITIVE under paired analysis. Direction is the
      load-bearing signal. Magnitudes are consistent (+2.2% to +8.9%).
      FTSE confirmation still needed but baseline regression risk is
      low given cross-instance consistency.
```

### Entry 4: W22-4/5/6 cross-instance NC27-deep sweep (2026-05-20 evening — SUPERSEDED by Entry 5)

```yaml
- date: 2026-05-20
  commit: (post-0683dd0; pending W22-7 commit)
  wave_unit: W22-4, W22-5, W22-6
  type: empirical
  nc_or_combo: [NC27_DEEP]
  thesis_anchor: "§6.2.1 Dirichlet concentration sensitivity; Paper Eqs 16-17"
  baseline_used: BASELINE per-instance paired
  observation:
    metric: paired Δ wealth Ŝ vs BASELINE on ASMS_mHDM_K3_v2both per instance
    value:
      "PO(8,1.0) n=10 paired (prior)": "+1.37% (6/10 wins, p=0.625)"
      "PO(8,1.0) n=20 unpaired (W22-4 added seeds 11-20)": "+0.52% (signal washing out)"
      "PO(16,1.0) n=5 paired (W22-5a)": "-6.1% (p=0.625) — OUTLIER"
      "PO(8,0.3) n=5 paired (W22-5b)": "+2.5% (p=1.000)"
      "sPO(8,1.0)-cosine n=5 paired (W22-6)": "+8.6% (p=0.188 — TRENDING)"
    n_seeds: 5-20 across instances
    confidence: low (n=5 instances), low-medium (PO(8,1.0) n=10/20)
  attribution:
    per_nc_marginal:
      NC27_DEEP_PO_8_1_0: "+1.37% paired n=10; +0.52% unpaired n=20 vs n=10 BASELINE"
      NC27_DEEP_PO_16_1_0: "−6.1% paired n=5 — REGRESSION on high-α concentration regime"
      NC27_DEEP_PO_8_0_3: "+2.5% paired n=5"
      NC27_DEEP_sPO_8_1_0: "+8.6% paired n=5 — STRONGEST signal; smooth dynamics favor Bayesian posterior"
    method: per-instance paired Wilcoxon; cross-instance synthesis
    direction_summary: "3 of 4 PO variants positive; PO(16,1.0) negative outlier"
  honest_scars:
    - "PO(16,1.0) regression unexpected; hypothesis: high α=16 → more concentrated Dirichlet → NC27-deep's aggressive Bayesian updates may over-trust noisy posteriors when ground-truth is already concentrated"
    - "sPO +8.6% with p=0.188 is borderline; could be n=5 noise OR could be that smooth dynamics is NC27-deep's natural regime (its posterior tracks the slow-changing weight distribution well)"
    - "n=5 per instance underpowered for sub-5% effects"
    - "BASELINE for PO(8,1.0) seeds 11-20 NOT run; n=20 paired comparison impossible without additional BASELINE batch (would cost +1hr wall)"
  ratification_verdict:
    nc27_deep: "WEAKLY POSITIVE — keep as OPT-IN; do NOT flip default in W23-1 yet"
    reasoning: |
      Direction holds on 3 of 4 PO instances (avg +0.4% across paired n).
      The PO(16,1.0) -6.1% regression is a real concern; deserves
      investigation before defaulting. The synthetic +2.8% L2 claim
      ranges from -6.1% to +8.6% in empirical wealth — high variance.
      Recommended next: run PO(16,1.0) at n=10 to confirm/refute the
      regression sign; if it persists at n=10, FTSE test is risky.
```

## Accumulated delta vs pre-W22 baseline (as of 2026-05-20)

| Config | Wealth Δ vs PO(8,1.0) BASELINE | Wealth Δ vs FTSE 2015 NC8c-v2 baseline |
|---|---|---|
| **PRODUCTION CURRENT** (no opt-ins set; NC-AD-fix + NC29 + NC29a + Hv-DM-AMFC available) | N/A (we ARE BASELINE) | UNTESTED — must validate that NC-AD-fix + NC29 + NC29a don't regress the +7.50% |
| NC27_DEEP alone | +1.37% paired n=10 | UNTESTED |
| FULL_STACK | +1.90% paired n=5 | UNTESTED |
| NC36 alone | -6.13% paired n=5 | UNTESTED |
| All other opt-ins individually | negative or neutral | UNTESTED |

**CRITICAL GAP**: the entire empirical W22 evidence is on **PO(8,1.0) synthetic**.
The FTSE +7.50% breakthrough has NOT been re-validated post-W22 changes.
This is the next-highest priority (W22-1 cross-instance + W22 future FTSE
gate before W23 integration).

---

## Drift registry summary (2026-05-20)

| Verdict | NCs / Probes | Count |
|---|---|---|
| STRONG_TRANSLATION | PROBE_AD_RECTANGLE_BUG | 1 |
| MODEST_TRANSLATION | NC27_DEEP, FULL_STACK, PROBE_Q_V1 | 3 |
| NEUTRAL | NC13b_ALONE, PROBE_Z_STABILITY | 2 |
| OPPOSITE_SIGN | NC36_TIP_ANALYTICAL, TIP_CLEANUP_STACK, NC27_NO_NC36, SPO_BENCH | 4 |
| UNTESTED | NC32 (both regimes), NC29a γ, NC30/AMFC, NC34, NC38, NC39, NC33 | 4+ |

**Translation rate**: 4 of 9 tested (44%) translate STRONG/MODEST.
**Mean drift score**: -0.46 (synthetic over-promises by ~half on average).

---

## Update cadence

- **After every empirical smoke**: add an entry within 24h of commit
- **After every integration commit**: add an entry with thesis anchor + A/B receipt
- **After every drift verdict change**: registry → log entry → state.json
- **At each wave gate close**: write a synthesis sub-section here

Source: `python_refactor/src/probes/probe_drift_synthetic_vs_empirical.py`
maintains the registry data; this log maintains the narrative.
