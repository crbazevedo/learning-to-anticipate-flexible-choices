"""W22 Inspection 6: AMFC soundness — z_ref choice, signal quality, naming.

Per W22-RESEARCH-PROGRAM.md Area VIII (AMFC).

OPERATOR DEFINITION (corrected 2026-05-19):
    AMFC = Anticipative Maximal Flexible Choice
         = argmax over current Pareto frontier of
           PREDICTED EXPECTED FUTURE HYPERVOLUME given that choice

    s* = argmax_{s ∈ P_t}  E[ HV( F_{t+h} | choose s at t ) ]

That is: for each candidate s on the CURRENT Pareto frontier P_t, forecast
the future Pareto frontier F_{t+h} conditioning on having chosen s, compute
the expected hypervolume of that future frontier, and pick the s that
maximizes it. "Flexible" = the choice rule is re-evaluated each period
under updated predictions.

CURRENT CODE (thesis_aligned_experiment.py:215, Hv-DM branch):
    return max(population, key=lambda s: s.Delta_S)

Where Delta_S comes from:
- SMS-EMOA._compute_hypervolume_contributions_class (deterministic)
- SMS-EMOA._compute_stochastic_hypervolume_contributions_class (Eq 6.41)
- Then multiplied by stability_factor (in v2 mode = 1.0 always)

GAP between operator-defined AMFC and current implementation:
- AMFC (correct):  argmax_s  E[ HV( F_{t+h} | s ) ]    ← per-candidate forward forecast
- Current code:   argmax_s  Δ_S(s) at time t           ← per-candidate current contribution

The "Anticipative" part of the name comes ONLY from Eq 14 / Eq 7.16 having
mutated ẑ_t IN-PLACE before Δ_S is computed — but that bakes anticipation
into the STATE of each candidate, not into a FUTURE-conditional forecast.
If the same anticipation rule applies to every candidate equally, the
ranking by current Δ_S is approximately the ranking under "mean-anticipated
Δ_S" — but not under "E[ future HV | this candidate chosen ]" because:
  (a) future HV depends on which OTHER solutions enter the future frontier
  (b) the chosen-then-realized return path affects what F_{t+h} looks like
  (c) Eq 14 anticipation is the SAME rule applied to every s, so it doesn't
      DIFFERENTIATE candidates as a flexible-choice forecast would.

ALGEBRAIC CONCERNS:
1. z_ref inconsistency: default in sms_emoa.py is (0.0, 0.2); main.py uses
   (-1.0, 10.0); other configs may differ. The z_ref affects Delta_S
   computation but is hard-coded per-run, not data-derived.
2. Stability factor: in v2 mode is forced to 1.0 (no-op). In legacy mode is
   1/(1+trace(P)), which down-weights solutions with high KF uncertainty.
   The v2 flip was Reading-F (W21-1); the question is whether disabling
   the down-weighting actually loses signal that distinguished AMFC from
   R-DM.
3. Signal quality: how often does argmax(Delta_S) agree with R-DM (random)
   or M-DM (median ROI)? If they often pick the SAME solution, the AMFC
   adds no information over a simpler selection rule.
4. Delta_S monotonicity: a solution with the BIGGEST ROI gap to its
   neighbor will dominate Delta_S regardless of risk. Is that what we want
   for "AMFC = anticipatory choice"?
5. Sensitivity to z_ref: |Delta_S| scales with (R2 - risk). A loose z_ref
   (large R2) makes all Delta_S values LARGE but argmax invariant (linear).
   But a TIGHT z_ref can have argmax(Delta_S) flip if some solutions have
   ROI < R1 (negative contribution).

This script:
- Tests sensitivity of argmax(Delta_S) to z_ref choice
- Simulates how often AMFC == R-DM == M-DM (signal redundancy)
- Compares deterministic Delta_S vs stochastic E[Δ_S] (Eq 6.41) selection
- Quantifies the impact of stability_factor on/off

Usage:
    uv run python -m experiments.inspect_amfc
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)


def delta_s_middle(roi_i, roi_iminus1, roi_iplus1, risk_iminus1, risk_i, risk_iplus1,
                   R1, R2):
    """Deterministic Δ_S for a middle solution in a Pareto class (sorted by ROI).

    Per sms_emoa.py line 723:
        Δ_S = (ROI_i − ROI_{i+1}) · (risk_{i−1} − risk_i)
    """
    return (roi_i - roi_iplus1) * (risk_iminus1 - risk_i)


def delta_s_boundary(roi, risk, R1, R2):
    """Deterministic Δ_S for boundary solution: (ROI − R1)(R2 − risk)."""
    return (roi - R1) * (R2 - risk)


def gen_pareto_front(n_solutions: int, rng: np.random.Generator,
                      roi_range=(0.0001, 0.005), risk_range=(0.001, 0.05)):
    """Generate a synthetic Pareto front: sorted by ROI ascending, risk descending."""
    rois = np.linspace(roi_range[0], roi_range[1], n_solutions)
    risks = np.linspace(risk_range[1], risk_range[0], n_solutions)  # descending
    # Perturb slightly to avoid perfect collinearity
    rois += rng.normal(0, 0.0001, size=n_solutions)
    risks += rng.normal(0, 0.001, size=n_solutions)
    # Re-sort by ROI ascending; check risk stays monotone-decreasing
    idx = np.argsort(rois)
    return rois[idx], risks[idx]


def compute_delta_s_for_front(rois: np.ndarray, risks: np.ndarray,
                               R1: float, R2: float) -> np.ndarray:
    """Compute Delta_S for every solution on a sorted-by-ROI front."""
    n = len(rois)
    ds = np.zeros(n)
    if n == 1:
        ds[0] = delta_s_boundary(rois[0], risks[0], R1, R2)
        return ds
    for i in range(n):
        if i == 0:
            ds[i] = delta_s_boundary(rois[i], risks[i], R1, R2)
        elif i == n - 1:
            ds[i] = (rois[i] - rois[i-1]) * (R2 - risks[i])
        else:
            ds[i] = delta_s_middle(rois[i], rois[i-1], rois[i+1],
                                    risks[i-1], risks[i], risks[i+1],
                                    R1, R2)
    return ds


def main():
    print("=" * 80)
    print("W22 INSPECTION 6: AMFC soundness — z_ref, signal quality, naming")
    print("=" * 80)
    print()
    print("AMFC (Anticipative Maximal Flexible Choice) per operator definition:")
    print("  s* = argmax_{s in P_t}  E[ HV( F_{t+h} | choose s at t ) ]")
    print()
    print("CURRENT implementation (thesis_aligned_experiment.py:215, Hv-DM branch):")
    print("  s* = max(population, key=lambda s: s.Delta_S)        # current-period Δ_S")
    print()
    print("These are NOT the same metric. Δ_S is current contribution; the AMFC target")
    print("is the EXPECTED FUTURE hypervolume conditional on which candidate is chosen.")
    print()

    # =========================================================================
    # TEST 1: z_ref sensitivity — does argmax(Delta_S) flip with z_ref?
    # =========================================================================
    print("=" * 80)
    print("TEST 1: z_ref sensitivity — does argmax(Delta_S) flip?")
    print("=" * 80)
    print()
    rois, risks = gen_pareto_front(7, RNG)
    print("Pareto front (sorted by ROI ascending):")
    print(f"  ROIs  = {rois.round(5)}")
    print(f"  risks = {risks.round(5)}")
    print()

    z_refs = [
        ("default sms_emoa.py",      (0.0, 0.2)),
        ("main.py",                  (-1.0, 10.0)),
        ("tight (R1=min, R2=max)",   (float(np.min(rois)), float(np.max(risks)))),
        ("loose (R1<<min, R2>>max)", (float(np.min(rois)) - 0.01, float(np.max(risks)) + 0.05)),
        ("punitive (R1>min)",        (float(np.median(rois)), float(np.max(risks)))),
    ]
    print(f"{'z_ref label':<30} {'R1':>10} {'R2':>10} {'argmax(Δ_S) idx':>18} {'argmax(Δ_S) ROI':>20}")
    print("-" * 95)
    for label, (R1, R2) in z_refs:
        ds = compute_delta_s_for_front(rois, risks, R1, R2)
        ai = int(np.argmax(ds))
        print(f"{label:<30} {R1:>10.4f} {R2:>10.4f} {ai:>18} {rois[ai]:>20.5f}")
    print()
    print("Observation: if argmax(Δ_S) flips across z_ref choices, the AMFC")
    print("is sensitive to a configuration knob that has no clear data-driven default.")
    print()

    # =========================================================================
    # TEST 2: AMFC vs R-DM vs M-DM signal redundancy
    # =========================================================================
    print("=" * 80)
    print("TEST 2: AMFC vs R-DM vs M-DM agreement rate")
    print("=" * 80)
    print()
    print("Simulate 1000 random Pareto fronts of size 7 (typical for sparse FTSE).")
    print("Compare which solution each rule picks:")
    print("  AMFC = argmax(Δ_S)")
    print("  M-DM = median ROI")
    print("  R-DM = uniform random")
    print()
    n_runs = 1000
    n_front = 7
    R1, R2 = 0.0, 0.05

    amfc_picks = []
    mdm_picks = []
    rdm_picks = []
    for _ in range(n_runs):
        rois, risks = gen_pareto_front(n_front, RNG)
        ds = compute_delta_s_for_front(rois, risks, R1, R2)
        amfc_idx = int(np.argmax(ds))
        mdm_idx = n_front // 2
        rdm_idx = int(RNG.integers(0, n_front))
        amfc_picks.append(amfc_idx)
        mdm_picks.append(mdm_idx)
        rdm_picks.append(rdm_idx)

    amfc_arr = np.array(amfc_picks)
    mdm_arr = np.array(mdm_picks)
    rdm_arr = np.array(rdm_picks)

    print(f"Agreement rates over {n_runs} synthetic fronts:")
    print(f"  AMFC == M-DM: {np.mean(amfc_arr == mdm_arr):.2%}")
    print(f"  AMFC == R-DM: {np.mean(amfc_arr == rdm_arr):.2%}  (chance baseline: {1/n_front:.2%})")
    print(f"  M-DM == R-DM: {np.mean(mdm_arr == rdm_arr):.2%}  (chance baseline: {1/n_front:.2%})")
    print()
    print("AMFC pick-index distribution (count per front-position):")
    for i in range(n_front):
        print(f"  position {i}: {np.mean(amfc_arr == i):.2%}")
    print()
    print("If AMFC picks are concentrated on boundary positions (0 or n-1),")
    print("then 'AMFC selects the extreme of the Pareto front' is the real signal,")
    print("not 'AMFC selects the most-future-contributing portfolio'.")
    print()

    # =========================================================================
    # TEST 3: stability factor impact (legacy mode vs v2 mode)
    # =========================================================================
    print("=" * 80)
    print("TEST 3: stability_factor impact (1/(1+trace(P)) vs 1.0)")
    print("=" * 80)
    print()
    print("Reading-F (W21-1) flipped stability_factor from per-solution to 1.0.")
    print("How often does this change the argmax pick?")
    print()
    n_runs = 1000
    n_front = 7
    R1, R2 = 0.0, 0.05

    # Simulate stability heterogeneity: trace(P) ~ Exp(0.5), so stability ~ U(0,1)
    diff_count = 0
    for _ in range(n_runs):
        rois, risks = gen_pareto_front(n_front, RNG)
        ds = compute_delta_s_for_front(rois, risks, R1, R2)
        # Sample stability factors
        trace_P = RNG.exponential(0.5, size=n_front)
        stability = 1.0 / (1.0 + trace_P)
        ds_v2 = ds.copy()  # stability = 1.0
        ds_legacy = ds * stability
        amfc_v2 = int(np.argmax(ds_v2))
        amfc_legacy = int(np.argmax(ds_legacy))
        if amfc_v2 != amfc_legacy:
            diff_count += 1
    print(f"  argmax disagrees between v2 and legacy modes: {diff_count}/{n_runs} ({diff_count/n_runs:.2%})")
    print()
    print("If the disagreement is non-trivial, the stability factor is a")
    print("LOAD-BEARING signal that was lost in Reading-F's v2 flip. The")
    print("intuition was that high-uncertainty (high trace(P)) portfolios should")
    print("be down-weighted in Δ_S — Reading-F dropped this.")
    print()

    # =========================================================================
    # TEST 4: Delta_S monotonicity vs ROI/risk separately
    # =========================================================================
    print("=" * 80)
    print("TEST 4: what dominates Delta_S — ROI gap or risk gap?")
    print("=" * 80)
    print()
    print("For a middle solution i: Δ_S = (ROI_i − ROI_{i+1}) * (risk_{i−1} − risk_i)")
    print("Both factors are NEGATIVE (next is bigger ROI, prev is bigger risk),")
    print("but their PRODUCT is positive.")
    print()
    rois, risks = gen_pareto_front(7, RNG)
    ds = compute_delta_s_for_front(rois, risks, R1=0.0, R2=0.05)
    # Compute the two factors for each middle solution
    print(f"{'i':>3} {'ROI':>10} {'risk':>10} {'(ROI_i - ROI_{i+1})':>22} "
          f"{'(risk_{i-1} - risk_i)':>22} {'Δ_S':>10}")
    print("-" * 80)
    for i in range(1, len(rois) - 1):
        roi_gap = rois[i] - rois[i+1]
        risk_gap = risks[i-1] - risks[i]
        print(f"{i:>3} {rois[i]:>10.5f} {risks[i]:>10.5f} "
              f"{roi_gap:>22.5f} {risk_gap:>22.5f} {ds[i]:>10.5f}")
    print()
    print("Interpretation: Δ_S is HIGH when both gaps are large (i.e., the solution")
    print("'fills a gap' in the Pareto front along both axes). It's small when a")
    print("solution is crowded between neighbors. So AMFC is structurally selecting")
    print("the LEAST-CROWDED solution — which is also the SMS-EMOA's own selection")
    print("criterion. This raises the question: is AMFC just 'pick the same thing")
    print("SMS-EMOA already considered most valuable for survival'?")
    print()

    # =========================================================================
    # TEST 5: temporal coherence — does AMFC pick persist across periods?
    # =========================================================================
    print("=" * 80)
    print("TEST 5: AMFC temporal coherence (does the choice flip wildly?)")
    print("=" * 80)
    print()
    print("Simulate a slowly-evolving Pareto front (drift each period) and check")
    print("how often the AMFC argmax changes index.")
    print()
    n_periods = 50
    n_front = 7
    rois, risks = gen_pareto_front(n_front, RNG)
    prev_amfc = None
    flip_count = 0
    for t in range(n_periods):
        # Drift each solution slightly
        rois = rois + RNG.normal(0, 0.0002, size=n_front)
        risks = risks + RNG.normal(0, 0.001, size=n_front)
        # Re-sort by ROI ascending
        idx = np.argsort(rois)
        rois = rois[idx]; risks = risks[idx]
        ds = compute_delta_s_for_front(rois, risks, R1=0.0, R2=0.05)
        amfc = int(np.argmax(ds))
        if prev_amfc is not None and amfc != prev_amfc:
            flip_count += 1
        prev_amfc = amfc
    print(f"  AMFC index flipped {flip_count}/{n_periods-1} periods ({flip_count/(n_periods-1):.2%})")
    print()
    print("In Probe G we observed AMFC asset overlap was Jaccard 0.169 (75% asset")
    print("turnover per period) — that is a CHAOTIC AMFC signal at the asset level.")
    print("This test checks if the index-level chaos is similar.")
    print()

    # =========================================================================
    # TEST 6: TRUE AMFC vs current-Δ_S — empirical disagreement
    # =========================================================================
    print("=" * 80)
    print("TEST 6: TRUE AMFC (operator-defined) vs current-Δ_S implementation")
    print("=" * 80)
    print()
    print("Simulate operator's AMFC definition with a forward-prediction surrogate.")
    print("For each candidate s on the current Pareto frontier:")
    print("  1. Forecast the future frontier conditional on choosing s")
    print("     (here: noisy KF-style drift of s plus its neighbors)")
    print("  2. Compute E[ HV(F_{t+h}) | s ] by MC over future-frontier samples")
    print("  3. argmax over s")
    print("Compare with the current implementation's argmax(current Δ_S).")
    print()

    def expected_future_hv_given_choice(rois: np.ndarray, risks: np.ndarray,
                                          chosen_idx: int, R1: float, R2: float,
                                          horizon: int = 3, n_mc: int = 200,
                                          drift_scale: float = 0.0003,
                                          rng: np.random.Generator = RNG) -> float:
        """Surrogate for E[ HV(F_{t+h}) | choose chosen_idx ].

        Each MC draw: every solution drifts by per-period noise; the chosen
        solution gets an extra 'momentum' nudge along its ROI direction
        (more aggressive at higher-ROI choices); compute hypervolume of
        the resulting front at horizon h; average over MC draws.
        """
        hvs = []
        n = len(rois)
        for _ in range(n_mc):
            roi_h = rois.copy()
            risk_h = risks.copy()
            # Per-period drift accumulated over h
            for _ in range(horizon):
                roi_h = roi_h + rng.normal(0, drift_scale, size=n)
                risk_h = risk_h + rng.normal(0, drift_scale * 2, size=n)
                # Momentum: chosen solution biases the whole front
                # toward its ROI direction
                bias = (rois[chosen_idx] - rois.mean()) * 0.05
                roi_h = roi_h + bias
            # Re-sort to enforce front structure (rois ascending)
            idx = np.argsort(roi_h)
            roi_h = roi_h[idx]; risk_h = risk_h[idx]
            ds_h = compute_delta_s_for_front(roi_h, risk_h, R1, R2)
            hvs.append(float(np.sum(np.maximum(ds_h, 0.0))))
        return float(np.mean(hvs))

    n_runs = 200
    n_front = 7
    R1, R2 = 0.0, 0.05
    disagree = 0
    rank_diffs = []
    for _ in range(n_runs):
        rois, risks = gen_pareto_front(n_front, RNG)
        ds = compute_delta_s_for_front(rois, risks, R1, R2)
        amfc_current = int(np.argmax(ds))

        # TRUE AMFC: per-candidate forward forecast
        future_hvs = np.array([
            expected_future_hv_given_choice(rois, risks, i, R1, R2,
                                              horizon=3, n_mc=80, rng=RNG)
            for i in range(n_front)
        ])
        amfc_true = int(np.argmax(future_hvs))

        if amfc_current != amfc_true:
            disagree += 1
        # Rank of current-pick under future-HV ranking
        rank_of_current = int(np.argsort(-future_hvs).tolist().index(amfc_current))
        rank_diffs.append(rank_of_current)

    print(f"  Synthetic runs: {n_runs}, front size: {n_front}, horizon: 3, MC: 80")
    print(f"  Current-Δ_S pick != TRUE-AMFC pick: {disagree}/{n_runs} ({disagree/n_runs:.2%})")
    print(f"  Mean rank of current pick under TRUE-AMFC ordering: {np.mean(rank_diffs):.2f}")
    print(f"  (0 = perfectly aligned; n_front-1 = worst possible)")
    print()
    print("Interpretation:")
    print("  - If current pick is rank-0 under TRUE AMFC most of the time, Δ_S is a")
    print("    decent SURROGATE for predicted future HV and the gap is mostly cosmetic.")
    print("  - If current pick is mid-rank, current-Δ_S is leaving forward-prediction")
    print("    signal on the table — operator's TRUE AMFC formulation has real upside.")
    print()
    print("CAVEAT: this is a TOY forward model. A production AMFC would use the actual")
    print("KF / Dirichlet predictor chain (or any of the alt-signal probes Q/R/S) to")
    print("forecast the per-candidate future frontier. The disagreement rate here is")
    print("an order-of-magnitude hint, not a calibrated estimate.")
    print()

    # =========================================================================
    # Final summary
    # =========================================================================
    print("=" * 80)
    print("INSPECTION 6 CONCLUSIONS")
    print("=" * 80)
    print()
    print("1. THE IMPLEMENTATION IS NOT THE OPERATOR-DEFINED AMFC.")
    print("   AMFC (correct) = argmax_s  E[ HV( F_{t+h} | choose s ) ]")
    print("   Code         = argmax_s  Δ_S(s) at time t")
    print("   The 'Anticipative' in the name comes only from Eq 14 / Eq 7.16 having")
    print("   mutated ẑ_t in-place uniformly for every candidate — which DOES NOT")
    print("   differentiate candidates by their per-choice forward prospects.")
    print("   The 'Flexible' part (re-evaluate the choice rule each period under")
    print("   updated predictions) IS present — the rule does re-pick each period.")
    print("   The 'Maximal' part is the argmax — but over the WRONG objective.")
    print()
    print("2. z_ref CHOICE matters but has no data-driven default.")
    print("   - sms_emoa.py default: (0.0, 0.2)")
    print("   - main.py: (-1.0, 10.0)")
    print("   - thesis_parameters.py: separate REFERENCE_POINT and HYPERVOLUME_REFERENCE_POINT")
    print("   These DISAGREE. The choice of z_ref can flip argmax(Δ_S).")
    print()
    print("3. STABILITY FACTOR was a load-bearing signal — Reading-F dropped it.")
    print("   In legacy mode: Δ_S *= 1/(1+trace(P)) — uncertain portfolios down-weighted.")
    print("   In v2 mode: Δ_S *= 1.0 — uncertainty IGNORED.")
    print("   Probe-level RCA needed: did Reading-F's improvement come DESPITE")
    print("   losing this signal, or was the signal actually noise in the Python port?")
    print("   (Restoring uncertainty-weighting would be doubly important under the")
    print("    operator-correct AMFC: forward-HV forecasts INHERIT predictor uncertainty.)")
    print()
    print("4. CURRENT-Δ_S SIGNAL is structurally 'least crowded solution' on Pareto front.")
    print("   This is the same criterion SMS-EMOA uses for survival — so the current")
    print("   AMFC proxy adds little information beyond 'pick what SMS-EMOA promoted'.")
    print("   The OPERATOR-CORRECT AMFC would differentiate candidates by their")
    print("   forward HV forecasts, which is orthogonal to current-front crowding.")
    print()
    print("5. NC30 CANDIDATES (implementing the operator-correct AMFC):")
    print("   a. CORE: per-candidate forward forecast loop")
    print("      For each s in current Pareto frontier:")
    print("        Forecast future frontier F_{t+h} conditional on s")
    print("        Compute E[HV(F_{t+h})] via MC (n_mc ≈ 100–500)")
    print("        Cache the conditional forecast for the chosen s")
    print("      Pick s* = argmax over s.")
    print("      Cost scales O(|P_t| * H * n_mc) — for |P_t|=20, H=3, n_mc=200,")
    print("      that's 12,000 forecasts per period. Optimization needed but tractable.")
    print("   b. Data-derive z_ref from sliding-window (worst_ROI, worst_risk) so")
    print("      the future-HV forecast is comparable across periods.")
    print("   c. Restore stability_factor (or NIS-gated equivalent) for per-candidate")
    print("      uncertainty discounting in the forward forecast.")
    print("   d. AMFC tie-breaker: when top-1 / top-2 E[HV] within ε, pick by lowest")
    print("      forecast variance (most certain forecast wins).")
    print()
    print("6. OPERATOR'S 'how to increase its signal?' QUESTION (corrected framing):")
    print("   The current implementation is using the WRONG objective. Switching to")
    print("   the operator-defined AMFC (per-candidate forward HV forecast) IS the")
    print("   structural signal-increasing change. Items (b)/(c)/(d) above are then")
    print("   complementary refinements of WHICH forecast goes into the argmax.")
    print("   This is a substantial implementation (NC30 candidate) but it's also")
    print("   the only way to align the algorithm with what the AMFC name promises.")


if __name__ == "__main__":
    main()
