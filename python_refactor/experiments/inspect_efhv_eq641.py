"""W22 Inspection 2: Expected Hypervolume Eq 6.41 soundness.

Per W22-RESEARCH-PROGRAM.md Area V (Expected Hypervolume Theory).

OPERATOR FLAGGED: "conditional over the mean — is that sound?"

Eq 6.41 (thesis p.131): E[Δ_S(ẑ_t^(i))] = E[(a-b)(c-d)]
where (Eq 6.36):
  a = ẑ_{1,t}^(i) | m̂_{2,t}^(i)      ← self ROI conditioned on self risk-MEAN
  b = ẑ_{1,t}^(i-1) | m̂_{2,t}^(i-1)
  c = ẑ_{2,t}^(i+1) | m̂_{1,t}^(i+1)
  d = ẑ_{2,t}^(i) | m̂_{1,t}^(i)      ← self risk conditioned on self ROI-MEAN

ALGEBRAIC INSPECTION:

1. For bivariate Gaussian (X, Y) with means (μ_X, μ_Y), Cov [[σ_X², σ_XY], [σ_XY, σ_Y²]]:
   E[X | Y = y] = μ_X + (σ_XY / σ_Y²) (y - μ_Y)
   Var[X | Y = y] = σ_X² (1 - ρ²) where ρ = σ_XY / (σ_X σ_Y)

2. When y = μ_Y (i.e., conditioning on the MEAN):
   E[X | Y = μ_Y] = μ_X + (σ_XY / σ_Y²)(μ_Y - μ_Y) = μ_X
   ← THE REGRESSION TERM VANISHES; conditional mean = unconditional mean!

3. The conditional VARIANCE is reduced: Var[X | Y = μ_Y] = σ_X² (1 - ρ²)

So "conditional on the mean" preserves the unconditional mean but reduces
variance by factor (1 - ρ²). This is mathematically correct (Aitken's
theorem on conditional Gaussian) but the FRAMING is misleading: the
formula recovers unconditional means.

4. Per Eq 6.39: E[XY] = E[X]E[Y] + Cov(X, Y). Expanding (a-b)(c-d):
   E[(a-b)(c-d)] = E[ac] - E[ad] - E[bc] + E[bd]
                 = E[a]E[c] + Cov(a,c) - E[a]E[d] - Cov(a,d) - E[b]E[c] - Cov(b,c) + E[b]E[d] + Cov(b,d)
                 = (E[a] - E[b])(E[c] - E[d]) + Cov(a,c) - Cov(a,d) - Cov(b,c) + Cov(b,d)
                 = (m_1^(i) - m_1^(i-1))(m_2^(i+1) - m_2^(i)) + [4 Cov terms]

5. For DIFFERENT solutions (b is from i-1, c from i+1, etc.) → assumed
   independent → cross-pair Cov = 0.
   Only within-solution Cov(a, d) is nonzero = Cov(ẑ_1^(i), ẑ_2^(i)) = P[0,1].

6. So implementation: E[Δ_S] = (m_1^(i) - m_1^(i-1))(m_2^(i+1) - m_2^(i)) - P_i[0,1]
   (the negative sign on Cov(a,d) per Eq 6.41).

VERIFY VIA MONTE CARLO:

Compare:
- Closed-form (Eq 6.41 truncated to within-solution Cov(a,d)): what code does
- Monte Carlo from joint bivariate Gaussian per solution: ground truth
- Full Eq 6.41 with all 4 Cov terms (when per-solution samples are available)

ALSO CHECK: is the conditional even WELL-DEFINED?
If conditioning on Y = exactly its mean, the conditional distribution X|Y=μ_Y
exists but Y=μ_Y is a measure-zero event. The formula uses E[X|Y=μ_Y] which
is the conditional mean function evaluated at y=μ_Y, which equals μ_X.
Standard but care needed in interpretation.

Usage:
    uv run python -m experiments.inspect_efhv_eq641
"""
from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(0)


def mc_e_delta_s(solutions_means, solutions_covs, n_mc=100000):
    """Monte Carlo estimate of E[Δ_S] = E[(a-b)(c-d)] for each middle solution.

    For each MC iteration:
      Sample (ROI_i, risk_i) ~ N(mean_i, cov_i) for each solution i
      For each middle solution i:
        a = sampled ROI_i
        b = sampled ROI_{i-1}
        c = sampled risk_{i+1}
        d = sampled risk_i
        delta_s_i = (a - b) * (c - d)
      Track mean delta_s_i per i across MC iterations
    """
    n_sol = len(solutions_means)
    if n_sol < 3:
        return None

    # For each MC iteration, sample all solutions; collect (ROI, risk) per i
    delta_s_acc = {i: 0.0 for i in range(1, n_sol - 1)}
    for _ in range(n_mc):
        samples = [
            RNG.multivariate_normal(solutions_means[i], solutions_covs[i])
            for i in range(n_sol)
        ]
        for i in range(1, n_sol - 1):
            roi_i = samples[i][0]
            roi_i_minus_1 = samples[i - 1][0]
            risk_i_plus_1 = samples[i + 1][1]
            risk_i = samples[i][1]
            delta_s_acc[i] += (roi_i - roi_i_minus_1) * (risk_i_plus_1 - risk_i)

    return {i: delta_s_acc[i] / n_mc for i in delta_s_acc}


def closed_form_truncated(solutions_means, solutions_covs):
    """Code's current implementation: Eq 6.41 with only within-solution
    Cov(a, d) = Cov(ẑ_1^(i), ẑ_2^(i)) = P_i[0, 1].

    Returns {middle_i: E[Δ_S]_truncated}
    """
    n_sol = len(solutions_means)
    result = {}
    for i in range(1, n_sol - 1):
        m_roi_i = solutions_means[i][0]
        m_risk_i = solutions_means[i][1]
        m_roi_prev = solutions_means[i - 1][0]
        m_risk_next = solutions_means[i + 1][1]
        # First term: mean product of differences
        mean_term = (m_roi_i - m_roi_prev) * (m_risk_next - m_risk_i)
        # Cov(a, d) = Cov(ẑ_1^(i), ẑ_2^(i)) = P_i[0, 1]
        # Sign: -Cov(a, d) per Eq 6.41
        within_cov = solutions_covs[i][0, 1]
        result[i] = mean_term - within_cov
    return result


def closed_form_full(solutions_means, solutions_covs, sample_buffer):
    """Full Eq 6.41 with all 4 Cov terms from per-solution sample buffer."""
    n_sol = len(solutions_means)
    result = {}
    for i in range(1, n_sol - 1):
        # Means
        m_roi_i = solutions_means[i][0]
        m_risk_i = solutions_means[i][1]
        m_roi_prev = solutions_means[i - 1][0]
        m_risk_next = solutions_means[i + 1][1]
        mean_term = (m_roi_i - m_roi_prev) * (m_risk_next - m_risk_i)

        # Empirical Cov from sample buffer
        # a = ROI_i, b = ROI_{i-1}, c = risk_{i+1}, d = risk_i
        a_samples = sample_buffer[i]["roi"]
        b_samples = sample_buffer[i - 1]["roi"]
        c_samples = sample_buffer[i + 1]["risk"]
        d_samples = sample_buffer[i]["risk"]
        cov_ac = float(np.cov(a_samples, c_samples)[0, 1])
        cov_ad = float(np.cov(a_samples, d_samples)[0, 1])
        cov_bc = float(np.cov(b_samples, c_samples)[0, 1])
        cov_bd = float(np.cov(b_samples, d_samples)[0, 1])

        # Eq 6.41
        result[i] = mean_term + cov_ac - cov_ad - cov_bc + cov_bd
    return result


def main():
    print("=" * 80)
    print("W22 INSPECTION 2: Eq 6.41 'conditional over the mean' soundness")
    print("=" * 80)
    print()
    print("KEY ALGEBRAIC FACT (Aitken, conditional Gaussian):")
    print("  For bivariate (X, Y) with mean (μ_X, μ_Y), conditioning on Y=μ_Y:")
    print("  E[X | Y=μ_Y] = μ_X + (σ_XY/σ_Y²)(μ_Y - μ_Y) = μ_X  ← NO REGRESSION SHIFT")
    print("  Var[X | Y=μ_Y] = σ_X² (1 - ρ²)  ← variance reduced by ρ²")
    print()
    print("So 'conditional over the mean' RECOVERS THE UNCONDITIONAL MEAN.")
    print("The framing is misleading but the formula is correct.")
    print()
    print("The Cov terms in Eq 6.41 carry the conditional information:")
    print("  Cov(a, d) = Cov(ẑ_1^(i), ẑ_2^(i)) = P_i[0, 1] (within-solution)")
    print("  Cov(a, c), Cov(b, c), Cov(b, d) = 0 (independent solutions)")
    print()

    # Scenario: 5-solution Pareto front
    print("=" * 80)
    print("SCENARIO: 5-solution Pareto front (FTSE-realistic)")
    print("=" * 80)
    print()
    # Sort by ROI ascending; risk descending (typical Pareto front shape)
    solutions_means = [
        np.array([0.001, 0.014]),  # i=0: low ROI, high risk
        np.array([0.002, 0.012]),  # i=1
        np.array([0.003, 0.010]),  # i=2 (middle)
        np.array([0.004, 0.008]),  # i=3
        np.array([0.005, 0.006]),  # i=4: high ROI, low risk
    ]
    # Within-solution Cov(ROI, risk) — typical positive for portfolios
    # (high-return assets tend to have high vol → positive Cov in portfolio)
    solutions_covs = [
        np.array([[1e-5, 5e-6], [5e-6, 1e-5]]),
        np.array([[1e-5, 5e-6], [5e-6, 1e-5]]),
        np.array([[1e-5, 5e-6], [5e-6, 1e-5]]),
        np.array([[1e-5, 5e-6], [5e-6, 1e-5]]),
        np.array([[1e-5, 5e-6], [5e-6, 1e-5]]),
    ]

    # MC ground truth
    print("Running MC (n=100000) for ground truth E[Δ_S]...")
    mc_result = mc_e_delta_s(solutions_means, solutions_covs, n_mc=100000)
    # Closed-form truncated (current code)
    truncated = closed_form_truncated(solutions_means, solutions_covs)
    # Full (with per-solution samples)
    sample_buffer = {}
    for i in range(len(solutions_means)):
        samples = RNG.multivariate_normal(solutions_means[i], solutions_covs[i], size=100000)
        sample_buffer[i] = {"roi": samples[:, 0], "risk": samples[:, 1]}
    full = closed_form_full(solutions_means, solutions_covs, sample_buffer)

    print()
    print("Per-middle-solution comparison:")
    print(f"{'i':>3} | {'MC ground truth':>20} | {'Code (truncated)':>20} | "
          f"{'Full Eq 6.41':>20} | {'truncated err':>15} | {'full err':>15}")
    print("-" * 110)
    for i in mc_result:
        err_t = truncated[i] - mc_result[i]
        err_f = full[i] - mc_result[i]
        print(f"{i:>3} | {mc_result[i]:>+.6e} | {truncated[i]:>+.6e} | "
              f"{full[i]:>+.6e} | {err_t:>+.6e} | {err_f:>+.6e}")

    print()
    print("FINDINGS:")
    print("- MC ground truth recovers the same E[Δ_S] as both closed-form variants")
    print("- 'Truncated' (current code) and 'Full' agree because for INDEPENDENT")
    print("  solutions the cross-pair Cov terms are statistically zero")
    print("- The within-solution Cov(a, d) correction (Cov(ROI_i, risk_i) = P_i[0, 1])")
    print("  is essential — without it, the mean-product term over-estimates E[Δ_S]")
    print()

    # Scenario 2: solutions are CORRELATED via shared market noise
    print("=" * 80)
    print("SCENARIO 2: Solutions with CORRELATED noise (shared market factor)")
    print("=" * 80)
    print()
    print("If portfolios share asset exposure, their (ROI, risk) shocks are correlated.")
    print("Pure independence assumption (Cov(a, c) = 0 etc.) is VIOLATED.")
    print()
    n_mc = 100000
    # Shared market shock
    market_shocks = RNG.multivariate_normal([0, 0], np.array([[2e-5, 1e-5], [1e-5, 2e-5]]), size=n_mc)
    corr_samples = []
    for i in range(5):
        own_noise = RNG.multivariate_normal([0, 0], np.array([[5e-6, 2e-6], [2e-6, 5e-6]]), size=n_mc)
        own_corr = solutions_means[i] + market_shocks + own_noise  # shared + idiosyncratic
        corr_samples.append({"roi": own_corr[:, 0], "risk": own_corr[:, 1]})

    # MC ground truth on these correlated samples
    delta_s_acc = {i: 0.0 for i in range(1, 4)}
    for k in range(n_mc):
        for i in range(1, 4):
            a = corr_samples[i]["roi"][k]
            b = corr_samples[i - 1]["roi"][k]
            c = corr_samples[i + 1]["risk"][k]
            d = corr_samples[i]["risk"][k]
            delta_s_acc[i] += (a - b) * (c - d)
    mc_corr = {i: delta_s_acc[i] / n_mc for i in delta_s_acc}

    # Closed-form truncated on solutions_means/covs (treats as independent)
    print(f"{'i':>3} | {'MC (correlated)':>20} | {'Truncated':>20} | {'Δ':>15}")
    print("-" * 75)
    for i in mc_corr:
        err = truncated[i] - mc_corr[i]
        print(f"{i:>3} | {mc_corr[i]:>+.6e} | {truncated[i]:>+.6e} | {err:>+.6e}")

    print()
    print("FINDING: When solutions share noise (realistic case), the truncated")
    print("closed-form DEVIATES from MC ground truth. The 3 missing Cov terms")
    print("(Cov(a,c), Cov(b,c), Cov(b,d)) become non-zero and contribute.")
    print()
    print("CODE SCAR (already documented): 'Python lacks per-solution MC samples;")
    print("cross-pair Cov(a,c), Cov(b,c), Cov(b,d) are 0; only within-solution")
    print("Cov(a,d) = KF P[0,1] retained.'")
    print()
    print("THIS IS THE 'MISSING SIGNAL' from NC EQ1 scar. Implementing per-solution")
    print("MC sample buffer (size 100-1000) would restore the full formula. The")
    print("incremental signal on FTSE may or may not move the breakthrough.")


if __name__ == "__main__":
    main()
