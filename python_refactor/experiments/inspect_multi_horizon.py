"""W22 Inspection 5: Multi-period discount model — are h-horizon weights principled?

Per W22-RESEARCH-PROGRAM.md Area VII (Multi-Period Discount).

OPERATOR FLAGGED: examine the discount multi-period model.

CURRENT CODE (MultiHorizonAnticipatoryLearning):

  calculate_multi_horizon_lambda_rates (Eq 7.16):
    for h in 1..H-1:
        tip = TIP(current, predicted_h)
        entropy = binary_entropy(tip)
        lambda_h = (1.0 / (H-1)) * (1.0 - entropy)        # Eq 6.6 — λ^H
        lambda_h = clamp(lambda_h, 0.0, 0.5)
        lambda_k = _compute_lambda_k_with_branch(...)      # Eq 6.9 — λ^K (constant per period)
        lambda_combined = 0.5 * (lambda_h + lambda_k)      # Eq 7.16

  apply_anticipatory_learning_rule (Eq 6.10 / Eq 14):
    lambda_sum = sum(lambda_rates)
    if lambda_sum > 1.0:
        lambda_rates = [r / lambda_sum for r in lambda_rates]  # NORMALIZE
    anticipatory_state = (1 - lambda_sum) * z_t
    for h, predicted in enumerate(predicted_states):
        anticipatory_state += lambda_rates[h] * predicted_states[h]

ALGEBRAIC CONCERNS:
1. The "discount" is FLAT, not geometric. Each horizon h gets the SAME (1/(H-1))
   prefactor in λ^H — only the TIP entropy term differs per horizon.
   In particular, λ^H does NOT decay with h in any structural way.
2. λ^K is solution-invariant per period — the SAME value gets added to every
   horizon's λ_combined. So λ^K contributes a CONSTANT offset across horizons.
3. The clamp (0, 0.5) for λ^H, plus the 0.5 * (λ^H + λ^K) combination, means
   λ_combined is in [0, 0.5] per horizon. With H=3 (2 horizons), Σλ ≤ 1.0
   already — but with H=4 (3 horizons) Σλ can hit 1.5, triggering the
   safety normalization which then DESTROYS the discount structure.
4. INCONSISTENCY: n_step_prediction.py:205 uses horizon_factor = 1/(1 + 0.1*h)
   — a totally DIFFERENT discount formula — in compute_n_step_expected_hypervolume.
   This is the kind of inconsistency that breeds bugs.
5. The "convex combination" interpretation requires Σλ ≤ 1. Beyond that
   point the normalize-by-Σλ branch produces (1 - 1) = 0 weight on z_t —
   the current observation gets ZERO weight in the anticipatory state.
   That's a degenerate case the literature flags as runaway anticipation.

This script:
- Computes the effective per-horizon weight profile w_0, w_1, ..., w_{H-1}
  as a function of H, TIP, λ^K
- Visualizes when the "safety normalization" fires and what it destroys
- Compares with: geometric discount (γ^h), exponential discount (exp(-βh)),
  uniform 1/H discount
- Highlights the n_step_prediction.py:205 inconsistency

Usage:
    uv run python -m experiments.inspect_multi_horizon
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(42)


def binary_entropy(p: float) -> float:
    """Binary entropy h(p) = -p log p - (1-p) log(1-p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def lambda_h_eq66(H: int, tip: float) -> float:
    """Eq 6.6 — λ^H = (1/(H-1)) * (1 - entropy(TIP)), clamped to [0, 0.5]."""
    if H < 2:
        return 0.0
    raw = (1.0 / (H - 1)) * (1.0 - binary_entropy(tip))
    return max(0.0, min(0.5, raw))


def lambda_combined_eq716(lambda_h: float, lambda_k: float) -> float:
    """Eq 7.16 — λ_combined = 0.5 * (λ^H + λ^K)."""
    return 0.5 * (lambda_h + lambda_k)


def effective_weights(lambda_rates: list[float]) -> tuple[float, list[float]]:
    """Mirror of apply_anticipatory_learning_rule.
    Returns (w_0, [w_1, ..., w_{H-1}]) and whether normalization fired.
    """
    lambda_sum = sum(lambda_rates)
    normalized = False
    if lambda_sum > 1.0:
        lambda_rates = [r / lambda_sum for r in lambda_rates]
        lambda_sum = 1.0
        normalized = True
    w_0 = 1.0 - lambda_sum
    return w_0, list(lambda_rates), normalized


def main():
    print("=" * 80)
    print("W22 INSPECTION 5: Multi-period discount — are h-horizon weights principled?")
    print("=" * 80)
    print()
    print("Eq 6.6:   λ^H_{t+h} = (1/(H-1)) * (1 - entropy(TIP_h)),  clamp(0, 0.5)")
    print("Eq 6.9:   λ^K = solution-invariant per-period, derived from KF residuals")
    print("Eq 7.16:  λ_combined = 0.5 * (λ^H + λ^K)")
    print("Eq 14:    z_anticip = (1 - Σλ) * z_t + Σλ_h * z_{t+h}")
    print()
    print("Note: the prefactor (1/(H-1)) is the SAME for every horizon h.")
    print("Only TIP differs per h. So the 'discount' is purely entropic — there")
    print("is NO structural h-decay in λ^H.")
    print()

    # =========================================================================
    # TEST 1: λ^H profile by horizon h, fixing TIP per horizon
    # =========================================================================
    print("=" * 80)
    print("TEST 1: λ^H profile by horizon h, for fixed TIP_h schedules")
    print("=" * 80)
    print()
    H = 4  # 3 forward horizons
    print(f"H = {H} (so horizons h = 1, 2, 3)")
    print()
    schedules = [
        ("All TIP_h = 0.5 (max ambiguity)",         [0.5, 0.5, 0.5]),
        ("TIP_h linearly decays 0.5→0.3→0.1",        [0.5, 0.3, 0.1]),
        ("TIP_h linearly grows 0.1→0.3→0.5",         [0.1, 0.3, 0.5]),
        ("All TIP_h = 0.1 (decisive predictions)",   [0.1, 0.1, 0.1]),
        ("All TIP_h = 0.95 (NC13a clamp boundary)",  [0.95, 0.95, 0.95]),
    ]
    print(f"{'schedule':<45} {'λ^H_1':>8} {'λ^H_2':>8} {'λ^H_3':>8} {'Σλ^H':>8}")
    print("-" * 80)
    for name, tips in schedules:
        lambdas = [lambda_h_eq66(H, t) for t in tips]
        print(f"{name:<45} {lambdas[0]:>8.4f} {lambdas[1]:>8.4f} "
              f"{lambdas[2]:>8.4f} {sum(lambdas):>8.4f}")
    print()
    print("Observation: when TIP is constant across horizons, λ^H is constant too.")
    print("The model has no intrinsic discount — only TIP variation gives discount.")
    print("But TIP itself is bounded [0.05, 0.95] (NC13a clamp) so the dynamic")
    print("range of the 'discount' is small in saturated regimes.")
    print()

    # =========================================================================
    # TEST 2: λ_combined profile with λ^K offset
    # =========================================================================
    print("=" * 80)
    print("TEST 2: λ_combined (Eq 7.16) with varying λ^K")
    print("=" * 80)
    print()
    print(f"H = {H}, all TIP_h = 0.5 (so λ^H_h is constant per horizon)")
    print()
    print(f"{'λ^K':>8} {'λ^H':>8} {'λ_combined':>12} {'Σλ_combined':>14} {'w_0 (=1-Σλ)':>14} {'normalized?':>14}")
    print("-" * 80)
    tips_flat = [0.5] * (H - 1)
    for lambda_k in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        lambdas_h = [lambda_h_eq66(H, t) for t in tips_flat]
        lambdas_combined = [lambda_combined_eq716(lh, lambda_k) for lh in lambdas_h]
        w_0, w_rest, normalized = effective_weights(lambdas_combined)
        print(f"{lambda_k:>8.2f} {lambdas_h[0]:>8.4f} {lambdas_combined[0]:>12.4f} "
              f"{sum(lambdas_combined):>14.4f} {w_0:>14.4f} {str(normalized):>14}")
    print()
    print("Observation: at λ^K ≈ 0.7, Σλ_combined hits 1.0 and the normalization")
    print("kicks in. Beyond that, w_0 = 0 — the current observation z_t is given")
    print("ZERO weight. This is the 'runaway anticipation' degenerate case.")
    print("The current code doesn't WARN about this — it logs a warning at >1.0,")
    print("but at exactly 1.0 (or 0.99), we're already in a regime where z_t")
    print("contributes negligibly.")
    print()

    # =========================================================================
    # TEST 3: compare with principled discount schemes
    # =========================================================================
    print("=" * 80)
    print("TEST 3: comparison with geometric / exponential / uniform discounts")
    print("=" * 80)
    print()
    print("Principled multi-horizon weighting schemes from RL / DP literature:")
    print()
    H_test = 5  # 4 forward horizons
    print(f"H_test = {H_test} (horizons h = 1..{H_test-1})")
    print()
    horizons = list(range(1, H_test))

    # Current code: assume TIP=0.5 everywhere, λ^K=0
    current_weights = [lambda_h_eq66(H_test, 0.5) for _ in horizons]
    print(f"  current Eq 7.16 (TIP=0.5 flat, λ^K=0): {[f'{w:.4f}' for w in current_weights]}")
    print(f"     sum = {sum(current_weights):.4f}")

    # Uniform 1/H
    uniform = [1.0 / H_test for _ in horizons]
    print(f"  uniform 1/H={1/H_test:.4f}:              {[f'{w:.4f}' for w in uniform]}")
    print(f"     sum = {sum(uniform):.4f}")

    # Geometric γ^h, γ=0.9
    gamma = 0.9
    raw_geo = [gamma ** h for h in horizons]
    geo = [w / sum(raw_geo) for w in raw_geo]
    print(f"  geometric γ^h normalized (γ=0.9):       {[f'{w:.4f}' for w in geo]}")
    print(f"     sum = {sum(geo):.4f}")

    # Exponential e^{-β h}, β=0.3
    beta = 0.3
    raw_exp = [np.exp(-beta * h) for h in horizons]
    exp_d = [w / sum(raw_exp) for w in raw_exp]
    print(f"  exponential exp(-β h) normalized (β=0.3): {[f'{w:.4f}' for w in exp_d]}")
    print(f"     sum = {sum(exp_d):.4f}")

    # Hyperbolic 1/(1 + κ h), κ=0.5 (behavioral econ canonical)
    kappa = 0.5
    raw_hyp = [1.0 / (1.0 + kappa * h) for h in horizons]
    hyp = [w / sum(raw_hyp) for w in raw_hyp]
    print(f"  hyperbolic 1/(1+κh) normalized (κ=0.5):  {[f'{w:.4f}' for w in hyp]}")
    print(f"     sum = {sum(hyp):.4f}")
    print()
    print("Observations:")
    print("  - Current Eq 7.16 produces FLAT weights when TIP is constant — no")
    print("    decay with h. Far horizons get the same weight as near.")
    print("  - Principled schemes (geometric, exp, hyperbolic) all decay with h,")
    print("    placing more weight on near-term predictions where the KF is more")
    print("    reliable (covariance lower, model linear-quadratic approximation tighter).")
    print("  - The flat-weight regime makes the algorithm sensitive to long-horizon")
    print("    KF extrapolation, which is the LEAST reliable part of the predictor.")
    print()

    # =========================================================================
    # TEST 4: inconsistency with n_step_prediction.py:205
    # =========================================================================
    print("=" * 80)
    print("TEST 4: INCONSISTENCY — two different horizon-discount formulas in code")
    print("=" * 80)
    print()
    print("In n_step_prediction.py:205, _compute_solution_expected_hypervolume uses:")
    print("    horizon_factor = 1.0 / (1.0 + 0.1 * h)    # hyperbolic, κ=0.1")
    print("    expected_hv = base_hv * state_factor * weight_factor * horizon_factor")
    print()
    print("In multi_horizon_anticipatory.py:220, λ^H uses:")
    print("    lambda_h = (1.0 / (H-1)) * (1.0 - entropy(TIP_h))  # flat-by-h")
    print()
    print("These are DIFFERENT DISCOUNT POLICIES applied at DIFFERENT POINTS in the")
    print("pipeline (HV scoring vs anticipation state-blending). There is no design")
    print("rationale documented for why two different formulas are used, and no")
    print("guarantee that they compose to a coherent multi-period objective.")
    print()
    print(f"  hyperbolic κ=0.1 for h=1..5: {[1.0/(1.0+0.1*h) for h in range(1,6)]}")
    print(f"  flat (1/(H-1))=1/4 for h=1..4 (H=5): {[1/4]*4}")
    print()

    # =========================================================================
    # TEST 5: degenerate case — Σλ_combined boundary
    # =========================================================================
    print("=" * 80)
    print("TEST 5: degenerate boundary — Σλ → 1, w_0 → 0")
    print("=" * 80)
    print()
    print("With H=4, λ^H per horizon ∈ [0, 0.5], so Σλ^H ∈ [0, 1.5].")
    print("Σλ_combined = 0.5 * (Σλ^H + 3*λ^K) ∈ [0, 0.5*(1.5 + 3)] = [0, 2.25].")
    print("Code normalizes when Σλ > 1.0, killing the discount structure entirely.")
    print()
    print(f"{'H':>4} {'max(λ^H_h)':>12} {'Σλ^H':>10} {'λ^K':>6} {'Σλ_combined':>14} {'normalized?':>14}")
    print("-" * 70)
    for H_ in [2, 3, 4, 5, 6]:
        max_l_h = 0.5  # clamp ceiling
        n_h = H_ - 1
        sum_l_h = max_l_h * n_h  # all horizons at clamp ceiling
        lambda_k = 0.5
        sum_l_combined = 0.5 * (sum_l_h + n_h * lambda_k)
        normalized = sum_l_combined > 1.0
        print(f"{H_:>4} {max_l_h:>12.4f} {sum_l_h:>10.4f} {lambda_k:>6.2f} "
              f"{sum_l_combined:>14.4f} {str(normalized):>14}")
    print()
    print("Observation: for H ≥ 4 with λ^K = 0.5, the normalization ALWAYS fires.")
    print("Anything > H=3 in this regime is operating in the runaway-anticipation")
    print("zone. The original paper (Eq 14) doesn't specify H — implementation chose H=3.")
    print()

    # =========================================================================
    # Final summary
    # =========================================================================
    print("=" * 80)
    print("INSPECTION 5 CONCLUSIONS")
    print("=" * 80)
    print()
    print("1. THE DISCOUNT IS NOT REALLY A DISCOUNT.")
    print("   Eq 6.6 uses a FLAT (1/(H-1)) prefactor for every horizon h. The only")
    print("   horizon-variation comes from TIP_h, which is itself clamped to [0.05,")
    print("   0.95] and often saturates. Result: near-flat weights across horizons.")
    print()
    print("2. λ^K IS CONSTANT ACROSS HORIZONS in the same period.")
    print("   It adds a uniform offset to every λ_combined_h, which doesn't help")
    print("   the per-horizon discount discrimination either.")
    print()
    print("3. THE NORMALIZATION SAFETY KILLS DISCOUNT STRUCTURE.")
    print("   When Σλ > 1, the code normalizes all λ_h to sum to 1, setting w_0 = 0.")
    print("   At this point the anticipatory state is PURE prediction with zero")
    print("   weight on the current observation — runaway anticipation.")
    print()
    print("4. INCONSISTENCY between two discount formulas in the codebase:")
    print("   - n_step_prediction.py:205 uses hyperbolic 1/(1 + 0.1*h)")
    print("   - multi_horizon_anticipatory.py:220 uses flat 1/(H-1)")
    print("   No design doc reconciles them; no test guards their composition.")
    print()
    print("5. NC29 CANDIDATE — Principled geometric discount:")
    print("   a. Replace (1/(H-1)) prefactor with γ^h (γ ≈ 0.9 default)")
    print("      → λ^H_h = γ^h * (1 - entropy(TIP_h))")
    print("      → Σ_h γ^h = γ(1-γ^(H-1))/(1-γ) bounded for any H")
    print("   b. Add HARD floor on w_0 (e.g., w_0 ≥ 0.2) — keep some weight on z_t")
    print("      always; never go to pure-prediction.")
    print("   c. Reconcile n_step_prediction.py:205 horizon_factor with the chosen")
    print("      anticipation discount, or document why they differ.")
    print()
    print("6. OPERATOR'S 'discount multi-period model' QUESTION:")
    print("   - The model is structurally NOT a discount model in the standard")
    print("     sense (no γ^h, no exp(-βh) decay). It's a TIP-modulated equal-")
    print("     weight blend with a safety normalization that can kill z_t entirely.")
    print("   - Whether this is 'principled' depends on whether you read paper Eq 14")
    print("     as 'sum over horizons with equal eligibility' (current) vs 'discounted")
    print("     sum favoring near-term' (textbook).")
    print("   - The paper text says λ_{t+h} should reflect anticipation rate at")
    print("     horizon h. In practice, the current code makes λ_{t+h} nearly")
    print("     constant per period — closer to 'select multi-horizon average' than")
    print("     'discounted anticipation'.")


if __name__ == "__main__":
    main()
