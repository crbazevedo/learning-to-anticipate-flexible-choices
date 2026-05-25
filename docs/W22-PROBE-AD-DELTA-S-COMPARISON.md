# W22 Probe AD — Stochastic vs Deterministic Δ_S Accuracy

**Status:** SHIPPED (commit hash filled in at merge)
**Branch:** `feat/w22-inspection-backlog`
**Module:** `python_refactor/src/probes/probe_ad_delta_s_comparison.py`
**Tests:** `python_refactor/tests/test_probe_ad_delta_s_comparison.py` (10 tests, all PASS)
**Backlog row:** Section D, "Probe AD — Stochastic vs deterministic Δ_S accuracy comparison"

---

## Hypothesis (H-Probe-AD)

When per-solution within-objective covariance ``Cov(ROI, risk) != 0``,
the stochastic Δ_S (Eq 6.41) is a strictly better approximation to a
CRN Monte-Carlo estimate of ``E[Δ_S]`` than the deterministic closed
form is.

When ``Cov(ROI, risk) = 0`` the two formulas are mathematically
equivalent (the ``- Cov(a, d)`` correction in Eq 6.41 vanishes), so
the stochastic version is only **meaningfully different** in the
correlated regime.

---

## Methodology

### Three independent Δ_S estimators (all replicated in this module — NO imports from `sms_emoa.py`)

1. **Deterministic** — closed form per
   `sms_emoa.py::_compute_hypervolume_contributions_class`:

   ```
   sort by ROI ascending
   position 0 (leftmost):  Δ_S = (ROI - R1) * (R2 - risk)
   middle position i:      Δ_S = (ROI_i - ROI_{i+1}) * (risk_{i-1} - risk_i)
   last position:          Δ_S = (ROI_i - ROI_{i-1}) * (R2 - risk_i)
   ```

2. **Stochastic (Eq 6.41 self-cov correction only)** — per
   `sms_emoa.py::_compute_stochastic_hypervolume_contributions_class`:

   ```
   Δ_S^stoch[i] = Δ_S^det[i] - Cov(ROI, risk)_self,i
   ```

   The Eq 6.41 expansion has four Cov terms ``(+,-,-,+)`` on
   ``(Cov(a,c), Cov(a,d), Cov(b,c), Cov(b,d))`` but in the Python
   refactor only ``Cov(a, d)`` (within-solution self) is nonzero —
   per-solution MC sample banks are not stored, so cross-pair Covs
   are 0 by construction.

3. **MC ground truth (CRN)** — for each MC trial we draw
   ``(roi_i, risk_i) ~ N(mus[i], covs[i])`` per solution (using a
   full 2x2 bivariate Gaussian) and evaluate the **deterministic**
   Δ_S on the sampled front. Average over trials. CRN is achieved by
   sharing the ``rng`` across trials so two methods see the same
   noise sequence.

### Synthetic Pareto fronts

We test on a 3-solution front sorted by ROI ascending (risk
descending — the Pareto shape):

```
ROIs  = [0.10, 0.15, 0.20]
risks = [0.30, 0.25, 0.20]
R1, R2 = 0.05, 0.40
```

with per-solution ``Cov(ROI, risk) = 0.008`` and marginal variance
``var = 0.01`` (so the 2x2 is positive semi-definite: ``|cov| <=
sqrt(var * var) = 0.01``).

---

## Expected results

| Test | Expectation | Outcome |
|---|---|---|
| Cov = 0 → stoch == det | exact equality | PASS (algebraic identity) |
| Cov > 0 → stoch < det (element-wise) | per Eq 6.41 sign | PASS |
| Large n_mc + Cov = 0 → MC ≈ det | MC noise band ~5e-3 | PASS (atol=5e-3, n_mc=20000) |
| Cov > 0 → SUM(MC) < SUM(det) | E[Δ_S] correction is negative | PASS |
| **|C|=1 single-solution: |stoch - MC| < |det - MC|** | KEY claim, clean regime | PASS (l1_stoch ≈ 5e-5 vs l1_det ≈ 8e-3) |
| **Multi-solution: |stoch - MC| > |det - MC|** | SCAR (rectangle mismatch) | PASS (l1_stoch ≈ 0.021 vs l1_det ≈ 0.010) |

---

## Findings

### Where H-Probe-AD HOLDS

**Single-solution case (|C|=1):** the stochastic correction is exactly
right. The MC ground truth is essentially indistinguishable from
``(ROI - R1)(R2 - risk) - Cov(ROI, risk)``. At n_mc = 50000:

```
det   = [0.005]
stoch = [-0.003]
mc    = [-0.00295]  (within MC SE of stoch)
L1(stoch - mc) ≈ 5e-5
L1(det   - mc) ≈ 8e-3   (= |Cov|, the full bias)
```

**Boundary positions on multi-solution fronts:** the leftmost/rightmost
position rectangles both involve the same ``(ROI - R1)`` or
``(R2 - risk)`` structure and the ``- Cov`` correction is the natural
one for those rectangles.

### Where H-Probe-AD FAILS (SCAR)

**Middle positions on multi-solution fronts:** the deterministic
middle-branch in `sms_emoa.py` line 723 uses rectangle
``(ROI_i - ROI_{i+1})(risk_{i-1} - risk_i)`` (the legacy Python /
C++ convention) while Eq 6.41 actually specifies
``(ROI_i - ROI_{i-1})(risk_{i+1} - risk_i)``. The ``- Cov``
correction is derived against the **Eq 6.41 rectangle**, so when
subtracted from the legacy rectangle it overshoots — the L1 gap can
end up LARGER for stoch than for det:

```
3-solution front (rois=[0.10,0.15,0.20], risks=[0.30,0.25,0.20],
                  cov_off=0.008, n_mc=20000):
L1(stoch - mc) ≈ 0.021
L1(det   - mc) ≈ 0.010
→ stoch is WORSE on this front.
```

This is exactly the "DETERMINISTIC-VS-STOCHASTIC RECTANGLE NOTE" scar
already documented in `sms_emoa.py` lines 772-781. The probe quantifies
the empirical L1 cost of the mismatch.

---

## Scars

1. **Rectangle mismatch in middle-branch det** — when
   ``Cov(ROI, risk) != 0`` and the front has 3+ solutions, the
   stochastic formula's ``- Cov`` correction does NOT reduce the L1
   gap from MC; in our test setup it nearly doubles it. The fix is
   either to re-align the deterministic middle-branch rectangle with
   Eq 6.41 OR to drop the self-cov subtraction on middle positions.
   This is a future-operator decision.

2. **Zero-correlation regime is trivial** — when ``Cov(ROI, risk) =
   0`` (uncorrelated marginals) stoch == det algebraically, so the
   stochastic estimator is only meaningfully different when the KF
   state P[0,1] is nonzero. In production this depends on whether
   the per-asset Kalman state has off-diagonal mass; this probe does
   not measure that.

3. **No cross-solution market noise modelled** — Eq 6.41's
   ``Cov(a, c), Cov(b, c), Cov(b, d)`` terms (cross-pair) are
   assumed zero throughout — both in the Python refactor and in our
   MC ground truth (which samples each solution independently). The
   C++ legacy DOES use per-solution MC sample banks to estimate
   these cross-pair Covs empirically; that level of fidelity is
   tracked under NC26-deep (Section C, blocked on market-Σ
   plumbing).

4. **Synthetic, not production** — all evaluations are on hand-
   constructed 1- and 3-solution fronts with prescribed Gaussian
   noise. The probe doesn't measure the L1 gap on a real SMS-EMOA
   population on FTSE or NASDAQ. That would require a separate probe
   (e.g. Probe AD-prod) that hooks into the SMS-EMOA selection step.

---

## Related work

- **Inspection 2 (Eq 6.41 truncation, 75% under-estimate, `b67601d`)**
  — diagnosed the missing Eq 6.41 implementation; the FIX shipped in
  W22 was the stochastic version this probe now compares against.
- **NC26-deep (DEFERRED)** — full market-Σ plumbing for cross-pair
  Covs; blocked by Section C plumbing work.
- **`sms_emoa.py::_compute_stochastic_hypervolume_contributions_class`**
  (lines 732-878) — the production implementation this probe
  replicates without importing.

---

## Reproducibility

```bash
python3 -m pytest python_refactor/tests/test_probe_ad_delta_s_comparison.py -v
```

All 10 tests pass with `np.random.default_rng(seed)` fixed seeds so
results are deterministic across runs.
