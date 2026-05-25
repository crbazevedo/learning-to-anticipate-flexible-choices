# W22 Probe Y — NC13b smooth TIP clamp impact

**Status:** SHIPPED (standalone analyzer + tests + this doc)
**Linked NC:** NC13b (commit `308de50`) — smooth tanh-based TIP clamp
**Hypothesis:** in TIP-saturation regimes (where raw TIP frequently lies in
[0, 0.05) ∪ (0.95, 1]), the smooth clamp preserves gradient signal that
the hard clamp destroys. Quantify the impact across a TIP distribution.

---

## Why this matters

Inspection 1 found that the TIP clamp at [0.05, 0.95] preserves stability
but destroys tail signal: two raw TIP values of 0.001 and 0.01 both
map to 0.05 under hard clip — indistinguishable. NC13b replaces the
hard clip with a smooth tanh squash that maps R → (c_min, c_max)
monotonically. But whether this matters in PRODUCTION depends on
whether real TIP distributions actually saturate.

Probe Y answers: **given a TIP distribution (synthetic or empirical from
a production ASMS run), how much signal does the smooth clamp recover
that the hard clamp destroys?**

---

## Methodology

Pure-numpy analyzer in `src/probes/probe_y_smooth_clamp_impact.py`:

- `hard_clamp(tip, c_min, c_max)` — legacy `max(c_min, min(c_max, tip))`
- `smooth_clamp(tip, c_min, c_max, k)` — NC13b tanh squash
- `derivative_hard(tip)` / `derivative_smooth(tip)` — local gradients
- `compare_clamps(tips_array)` — produces side-by-side output + gradient
- `saturation_regime_summary(tip_distribution)` — markdown summary with
  HIGH / Moderate / Low verdict

---

## Synthetic high-saturation example

A heavily-saturated TIP distribution (40% tails, 60% inside):

```
## NC13b smooth-clamp impact summary

TIP distribution: 1000 samples
- below c_min (0.05): 200 (20.0%)
- above c_max (0.95): 200 (20.0%)
- inside: 600 (60.0%)
- saturation rate: **40.0%**

Smooth clamp impact:
- signal_recovered_fraction (tail TIPs w/ non-zero gradient): **100.0%**
- mean(smooth - hard) output difference: -3.2854e-03
- mean |smooth - hard| output difference: 4.6997e-02
- mean gradient difference (smooth - hard): 0.0765

**Verdict:** HIGH saturation regime — NC13b smooth clamp likely matters.
```

In this regime, ALL tail TIPs recover non-zero gradient (signal preserved).
The mean absolute output difference is ~0.05 (5% of the clamp range), meaning
downstream λ^H computations would see meaningfully different inputs.

---

## Success criteria

NC13b should be ratified as production default IF:
1. Empirical production TIP distribution shows saturation rate ≥ 20%
2. signal_recovered_fraction ≥ 90% on tail TIPs
3. Downstream λ^H signal becomes noticeably more informative
4. Final wealth (FTSE n≥10) doesn't regress vs hard clamp

**Operator action items** (out of probe scope):
- Instrument the existing TIP calculator to dump per-period TIP values to a
  side-channel during a FTSE n=5 smoke
- Feed the resulting distribution into `saturation_regime_summary`
- If saturation rate ≥ 20%, run FTSE n=10 with `W22_NC13B_SMOOTH_CLAMP=1`
- Compare final wealth + Sharpe to baseline

---

## Honest scars

- **Pure synthetic verdict**: a HIGH-saturation synthetic distribution doesn't
  mean PRODUCTION TIPs are saturated. Need real telemetry.
- **Symmetric synthetic distribution**: my example used uniform tails. Real
  ASMS may have asymmetric saturation (e.g., more frequent TIP≈0.95 than
  TIP≈0.05) depending on how well-separated solutions tend to be.
- **k=4 default**: the smooth steepness k is tunable via `W22_NC13B_K`. A
  more lenient k (e.g., 2) widens the soft transition; stricter k (e.g., 8)
  approaches the hard-clip behavior. Probe Y uses k=4 default; sensitivity
  to k is not yet characterized.
- **Composes with NC31**: if TIP conditional mode (NC31) is enabled, the TIP
  distribution itself changes. Probe Y is silent on the joint effect.

---

## Test coverage

13/13 tests passing in `tests/test_probe_y_smooth_clamp_impact.py`:
- hard / smooth clamp identities and boundary behavior
- monotonicity of smooth clamp
- derivative comparisons (hard zero outside; smooth positive everywhere)
- tail signal recovery (compare_clamps)
- saturation regime classification (HIGH / Moderate / Low verdicts)

---

## Linkage

- **NC13b** (`308de50`): implementation of smooth clamp
- **NC31** (`7940604`): TIP conditional mode — composes; if enabled changes TIP distribution
- **Probe AA**: AMFC telemetry harness — can also capture raw TIP values from telemetry to feed into this probe
