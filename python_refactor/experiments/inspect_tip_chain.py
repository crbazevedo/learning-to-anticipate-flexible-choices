"""W22 Inspection 1: TIP chain of computations — soundness, numeric stability,
assumption violations (operator focus).

Per W22-RESEARCH-PROGRAM.md Area IV (TIP Soundness).

ALGEBRAIC CONCERN (operator-flagged):
Paper Definition 6.1: TIP(t, t+h) = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]

The conditional `| ẑ_t` means: GIVEN we observe ẑ_t (treated as KNOWN),
what is the probability ẑ_{t+h} is mutually non-dominated with ẑ_t?

Code does:
    current_sample ~ N(current_mean, current_cov)   # ← SAMPLES current
    predicted_sample ~ N(predicted_mean, predicted_cov)
    check_mutual_non_dominance(current_sample, predicted_sample)

This violates the conditional: per Defn 6.1, current should be FIXED at
its observed value, only predicted should be sampled.

This script DEMONSTRATES the impact with both methods on the same
inputs and reports TIP_joint vs TIP_conditional + a numerical
analytical reference for simple cases.

Usage:
    uv run python -m experiments.inspect_tip_chain
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# Reproducibility
RNG = np.random.default_rng(42)


def tip_joint_mc(current_mean, current_cov, predicted_mean, predicted_cov,
                 n_mc=10000):
    """JOINT-sampling TIP (current code's behavior):
    samples BOTH current and predicted as Gaussians; counts mutual non-dominance.
    """
    n_nd = 0
    for _ in range(n_mc):
        c = RNG.multivariate_normal(current_mean, current_cov)
        p = RNG.multivariate_normal(predicted_mean, predicted_cov)
        # ROI maximized (index 0), risk minimized (index 1)
        c_dom = (c[0] > p[0]) and (c[1] < p[1])
        p_dom = (p[0] > c[0]) and (p[1] < c[1])
        if not c_dom and not p_dom:
            n_nd += 1
    return n_nd / n_mc


def tip_conditional_mc(current_observed, predicted_mean, predicted_cov,
                       n_mc=10000):
    """CONDITIONAL TIP (Defn 6.1 correct):
    current is FIXED at observed value; only predicted is sampled.
    """
    n_nd = 0
    for _ in range(n_mc):
        c = current_observed  # fixed
        p = RNG.multivariate_normal(predicted_mean, predicted_cov)
        c_dom = (c[0] > p[0]) and (c[1] < p[1])
        p_dom = (p[0] > c[0]) and (p[1] < c[1])
        if not c_dom and not p_dom:
            n_nd += 1
    return n_nd / n_mc


def tip_analytical_conditional(current_observed, predicted_mean, predicted_cov):
    """Analytical TIP_conditional under bivariate Gaussian.

    With current = (c_roi, c_risk) FIXED, predicted = (P_roi, P_risk) ~ N(μ, Σ):
        P[current dominates predicted] = P[P_roi < c_roi AND P_risk > c_risk]
        P[predicted dominates current] = P[P_roi > c_roi AND P_risk < c_risk]
        TIP_cond = 1 - P[current dominates] - P[predicted dominates]

    For BIVARIATE Gaussian with correlation ρ:
        P[X < a AND Y > b] is computed via Owen's T function or scipy mvn_cdf.
    """
    c_roi, c_risk = current_observed
    mu_roi, mu_risk = predicted_mean
    sigma_roi = np.sqrt(predicted_cov[0, 0])
    sigma_risk = np.sqrt(predicted_cov[1, 1])
    if sigma_roi == 0 or sigma_risk == 0:
        return float("nan")
    rho = predicted_cov[0, 1] / (sigma_roi * sigma_risk)

    # Standardize
    # P[P_roi < c_roi] = Φ((c_roi - μ_roi) / σ_roi) = Φ(z1)
    # P[P_risk > c_risk] = Φ(-(c_risk - μ_risk) / σ_risk) = Φ(-z2)
    z1 = (c_roi - mu_roi) / sigma_roi
    z2 = (c_risk - mu_risk) / sigma_risk

    # For bivariate normal, joint probability P[Z1 < z1 AND Z2 < z2] = Φ_2(z1, z2; ρ)
    # We want P[P_roi < c_roi AND P_risk > c_risk] = P[Z1 < z1 AND Z2 > z2]
    #       = P[Z1 < z1] - P[Z1 < z1 AND Z2 < z2]
    #       = Φ(z1) - Φ_2(z1, z2; ρ)
    # Where Z1, Z2 are standard bivariate normal with correlation ρ.

    Phi_z1 = scipy_stats.norm.cdf(z1)
    Phi_z2 = scipy_stats.norm.cdf(z2)
    # Bivariate normal CDF using scipy.stats.multivariate_normal
    bvn = scipy_stats.multivariate_normal(
        mean=[0, 0], cov=[[1.0, rho], [rho, 1.0]]
    )
    # scipy mvn.cdf accepts the upper-right corner directly
    Phi2_z1_z2 = bvn.cdf([z1, z2])

    # P[current dominates predicted] = P[P_roi < c_roi AND P_risk > c_risk]
    P_c_dom = Phi_z1 - Phi2_z1_z2
    # P[predicted dominates current] = P[P_roi > c_roi AND P_risk < c_risk]
    P_p_dom = Phi_z2 - Phi2_z1_z2

    return float(1.0 - P_c_dom - P_p_dom)


def main():
    print("=" * 80)
    print("W22 INSPECTION 1: TIP chain — joint vs conditional sampling")
    print("=" * 80)
    print()
    print("Definition 6.1: TIP = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]")
    print()
    print("The conditional `| ẑ_t` requires ẑ_t to be FIXED (observed),")
    print("not sampled. The code's joint sampling is mathematically incorrect.")
    print()

    # Test scenarios
    scenarios = [
        {
            "name": "Close objectives, small cov (FTSE realistic, post-NC8c-v2)",
            "current": [0.001, 0.005],
            "current_cov": np.diag([0.01, 0.01]),
            "predicted": [0.0012, 0.0048],
            "predicted_cov": np.diag([0.01, 0.01]),
        },
        {
            "name": "Same point, small cov (degenerate case)",
            "current": [0.001, 0.005],
            "current_cov": np.diag([0.01, 0.01]),
            "predicted": [0.001, 0.005],
            "predicted_cov": np.diag([0.01, 0.01]),
        },
        {
            "name": "Predicted DOMINATES current (deterministic)",
            "current": [0.001, 0.010],   # lower ROI, higher risk
            "current_cov": np.diag([1e-12, 1e-12]),  # near-deterministic
            "predicted": [0.005, 0.005],  # higher ROI, lower risk
            "predicted_cov": np.diag([1e-12, 1e-12]),
        },
        {
            "name": "Large cov (Probe E post-NC7 pathology: P[ROI,ROI]=486)",
            "current": [0.001, 0.005],
            "current_cov": np.array([[486.0, 0], [0, 486.0]]),
            "predicted": [0.001, 0.005],
            "predicted_cov": np.array([[1000.0, 0], [0, 1000.0]]),
        },
        {
            "name": "Post-NC13a clamped (P_pred ≤ 1.0)",
            "current": [0.001, 0.005],
            "current_cov": np.diag([0.05, 0.05]),
            "predicted": [0.0015, 0.0048],
            "predicted_cov": np.diag([1.0, 1.0]),
        },
    ]

    for sc in scenarios:
        print(f"### Scenario: {sc['name']}")
        print(f"  current = {sc['current']}, cov_diag = {np.diag(sc['current_cov'])}")
        print(f"  predicted = {sc['predicted']}, cov_diag = {np.diag(sc['predicted_cov'])}")

        tip_j = tip_joint_mc(
            np.array(sc["current"]), np.array(sc["current_cov"]),
            np.array(sc["predicted"]), np.array(sc["predicted_cov"]),
            n_mc=10000,
        )
        tip_c = tip_conditional_mc(
            np.array(sc["current"]),  # FIXED
            np.array(sc["predicted"]), np.array(sc["predicted_cov"]),
            n_mc=10000,
        )
        try:
            tip_a = tip_analytical_conditional(
                np.array(sc["current"]),
                np.array(sc["predicted"]), np.array(sc["predicted_cov"]),
            )
        except Exception as e:
            tip_a = float("nan")

        print(f"  TIP_joint (current code, MC=10000):              {tip_j:.4f}")
        print(f"  TIP_conditional (Defn 6.1 correct, MC=10000):    {tip_c:.4f}")
        print(f"  TIP_conditional (analytical, bivariate Gaussian): {tip_a:.4f}")
        print(f"  Δ (joint - conditional MC):                       {tip_j - tip_c:+.4f}")
        print()


if __name__ == "__main__":
    main()
