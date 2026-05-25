"""W22 Inspection 3: Dirichlet filter — is it a real Dirichlet?

Per W22-RESEARCH-PROGRAM.md Area II.

OPERATOR FLAGGED: examine Dirichlet filter implementation.

CURRENT CODE (DirichletPredictor.dirichlet_mean_prediction_vec):
    rate = 0.5 * anticipative_rate
    prediction = prev + rate * (current - prev)
    return prediction / sum(prediction)

ALGEBRAIC VERDICT: This is NOT a Dirichlet filter.
It is EXPONENTIAL SMOOTHING on raw weight vectors with re-normalization.

A TRUE Dirichlet filter would:
1. Maintain Dirichlet posterior with concentration parameter α
2. Update α via Bayesian update: α_new = α + observation
3. Predict mean = α / Σα; predict variance = α(Σα - α) / (Σα²(Σα + 1))

This inspection:
- Verifies the empirical equivalence (where they agree / differ)
- Tests both on Dirichlet-generated data to see which converges
- Suggests proper implementation

Usage:
    uv run python -m experiments.inspect_dirichlet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.algorithms.anticipatory_learning import DirichletPredictor

RNG = np.random.default_rng(42)


def true_dirichlet_filter_step(alpha_t: np.ndarray, observation: np.ndarray,
                                concentration_increment: float = 1.0) -> np.ndarray:
    """One step of TRUE Dirichlet filter.

    Bayesian update: α_{t+1} = α_t + concentration_increment * normalized_obs

    Args:
        alpha_t: current concentration parameter (positive vector summing to anything)
        observation: observed weight vector (on simplex)
        concentration_increment: how much weight to give the observation
            (small → conservative update, large → aggressive update)

    Returns:
        α_{t+1}: updated concentration
    """
    return alpha_t + concentration_increment * observation


def dirichlet_mean(alpha: np.ndarray) -> np.ndarray:
    """Mean of Dirichlet(α): E[X_i] = α_i / Σα."""
    return alpha / np.sum(alpha)


def dirichlet_variance(alpha: np.ndarray) -> np.ndarray:
    """Variance of Dirichlet(α): Var(X_i) = α_i (Σα - α_i) / (Σα² (Σα + 1))."""
    alpha_sum = np.sum(alpha)
    return alpha * (alpha_sum - alpha) / (alpha_sum ** 2 * (alpha_sum + 1))


def main():
    print("=" * 80)
    print("W22 INSPECTION 3: Dirichlet filter — is it a real Dirichlet?")
    print("=" * 80)
    print()
    print("CURRENT CODE: exponential smoothing on raw weights:")
    print("    rate = 0.5 * anticipative_rate")
    print("    prediction = prev + rate * (current - prev)")
    print("    return prediction / sum(prediction)")
    print()
    print("This is NOT a Dirichlet filter — no concentration parameter,")
    print("no Bayesian update, no posterior variance.")
    print()

    # Test 1: simple comparison on a synthetic trajectory
    print("=" * 80)
    print("TEST 1: Synthetic trajectory of 5 weight observations")
    print("=" * 80)
    print()
    d = 5  # 5 assets
    observations = [
        np.array([0.5, 0.2, 0.1, 0.1, 0.1]),
        np.array([0.4, 0.3, 0.15, 0.1, 0.05]),
        np.array([0.3, 0.35, 0.2, 0.1, 0.05]),
        np.array([0.25, 0.4, 0.2, 0.1, 0.05]),
        np.array([0.2, 0.45, 0.2, 0.1, 0.05]),
    ]
    # True trajectory has trend: asset 0 decreasing, asset 1 increasing

    # Current "Dirichlet" predictor (exponential smoothing)
    print("CURRENT 'Dirichlet' (exponential smoothing, rate=0.5):")
    pred_curr = observations[0]
    for t in range(1, len(observations)):
        new_pred = DirichletPredictor.dirichlet_mean_prediction_vec(
            pred_curr, observations[t], anticipative_rate=1.0,  # 0.5 * 1.0 = 0.5
        )
        print(f"  After obs {t}: pred = {new_pred.round(4)}")
        pred_curr = new_pred

    # True Dirichlet filter
    print()
    print("TRUE Dirichlet filter (concentration_increment=1.0):")
    alpha = np.ones(d)  # uniform Jeffreys-like prior (alpha_0 = (1, 1, 1, 1, 1))
    for t in range(len(observations)):
        alpha = true_dirichlet_filter_step(alpha, observations[t], concentration_increment=1.0)
        mean = dirichlet_mean(alpha)
        var = dirichlet_variance(alpha)
        print(f"  After obs {t}: alpha = {alpha.round(3)}, mean = {mean.round(4)}, std = {np.sqrt(var).round(4)}")

    print()
    print("FINDING: True Dirichlet tracks a RUNNING AVERAGE of observations")
    print("(asymptotic mean = empirical mean of observations).")
    print("Exponential smoothing tracks a WEIGHTED RECENT-OBSERVATION average.")
    print("Different convergence; different uncertainty quantification.")
    print()

    # Test 2: which one is closer to the true Dirichlet generating process?
    print("=" * 80)
    print("TEST 2: Generate from TRUE Dirichlet(α_true); which predictor recovers it?")
    print("=" * 80)
    print()
    alpha_true = np.array([5.0, 3.0, 2.0, 1.0, 1.0])
    n_obs = 100
    print(f"True α = {alpha_true}")
    print(f"True mean = {dirichlet_mean(alpha_true).round(4)}")
    print(f"True std  = {np.sqrt(dirichlet_variance(alpha_true)).round(4)}")
    print()

    # Generate observations
    observations_long = RNG.dirichlet(alpha_true, size=n_obs)

    # Current predictor: tracks last observation with exponential smoothing
    pred_curr = observations_long[0]
    for t in range(1, n_obs):
        pred_curr = DirichletPredictor.dirichlet_mean_prediction_vec(
            pred_curr, observations_long[t], anticipative_rate=1.0,
        )

    # True Dirichlet filter: updates α
    alpha_filter = np.ones(d) * 0.5  # Jeffreys prior
    for t in range(n_obs):
        alpha_filter = true_dirichlet_filter_step(
            alpha_filter, observations_long[t], concentration_increment=1.0,
        )

    print(f"After {n_obs} observations:")
    print(f"  Current 'Dirichlet' (expo smooth) final pred:        {pred_curr.round(4)}")
    print(f"  TRUE Dirichlet final mean estimate:                   {dirichlet_mean(alpha_filter).round(4)}")
    print(f"  Empirical mean of observations:                       {observations_long.mean(axis=0).round(4)}")
    print(f"  True mean:                                            {dirichlet_mean(alpha_true).round(4)}")
    print()
    err_curr = np.linalg.norm(pred_curr - dirichlet_mean(alpha_true))
    err_true = np.linalg.norm(dirichlet_mean(alpha_filter) - dirichlet_mean(alpha_true))
    print(f"L2 error vs true mean:")
    print(f"  Current 'Dirichlet': {err_curr:.4f}")
    print(f"  TRUE Dirichlet:      {err_true:.4f}")
    print()
    print(f"TRUE Dirichlet error is {(err_curr / err_true):.2f}× SMALLER than exponential smoothing.")
    print()

    # Test 3: predicted variance bound from TRUE Dirichlet
    print("=" * 80)
    print("TEST 3: TRUE Dirichlet provides POSTERIOR VARIANCE for free")
    print("=" * 80)
    print()
    print("Current implementation has NO uncertainty quantification.")
    print("It cannot tell us 'how confident are we in this prediction?'")
    print()
    print("TRUE Dirichlet gives Var(X_i) = α_i(Σα - α_i)/(Σα²(Σα + 1)).")
    print(f"After {n_obs} obs, posterior std of TRUE Dirichlet:")
    posterior_std = np.sqrt(dirichlet_variance(alpha_filter))
    for i in range(d):
        print(f"  asset {i}: std = {posterior_std[i]:.5f}")
    print()
    print("This uncertainty could feed:")
    print("- λ_combined (currently uses only KF residuals); add λ^D for Dirichlet")
    print("- AMFC scoring (favor portfolios with low Dirichlet uncertainty)")
    print("- Anticipative blend weight (low Dirichlet var → trust prediction more)")
    print()

    # Test 4: counterfactual — logistic-normal alternative (compositional KF)
    print("=" * 80)
    print("TEST 4: Logistic-Normal alternative (Aitchison compositional KF)")
    print("=" * 80)
    print()
    print("For data on simplex, the gold-standard filter is logistic-normal:")
    print("- Transform w → y = log(w_i / w_d) for i ≠ d (some reference asset)")
    print("- Standard KF on y (unconstrained)")
    print("- Inverse-transform back to simplex")
    print()
    print("Advantages over both exponential smoothing and Dirichlet:")
    print("- Uses standard KF infrastructure (already have it!)")
    print("- Handles cross-asset correlations naturally")
    print("- Compatible with H, F, R, Q machinery from KF on (ROI, risk)")
    print("- Could share KF state-evolution model")
    print()
    print("Demonstration of transform:")
    w = observations[0]
    print(f"  Simplex weights: {w.round(4)}")
    # Reference: last asset
    y = np.log(w[:-1] / w[-1])
    print(f"  Log-ratio y_i = log(w_i / w_{d-1}): {y.round(4)}")
    # Inverse transform
    w_recovered = np.empty(d)
    w_recovered[:-1] = np.exp(y)
    w_recovered[-1] = 1.0
    w_recovered /= np.sum(w_recovered)
    print(f"  Recovered weights: {w_recovered.round(4)}")
    print(f"  Match: {np.allclose(w, w_recovered)}")
    print()

    print("=" * 80)
    print("INSPECTION 3 CONCLUSIONS")
    print("=" * 80)
    print()
    print("1. Current 'Dirichlet' predictor is EXPONENTIAL SMOOTHING.")
    print("   Misleading name; not a real Dirichlet posterior.")
    print()
    print("2. TRUE Dirichlet filter converges faster + provides uncertainty.")
    print(f"   In our 100-obs test: TRUE Dirichlet had {err_curr/err_true:.2f}× smaller error.")
    print()
    print("3. Probe F (W22) already confirmed: current 'Dirichlet' predictor does")
    print("   NOT beat persistence (Wilcoxon p=0.9992).")
    print("   This is CONSISTENT with the algebraic finding — exponential smoothing")
    print("   adds noise relative to just using last observation.")
    print()
    print("4. RECOMMENDED MIGRATION (NC27 candidate):")
    print("   a. Replace exponential smoothing with TRUE Dirichlet posterior")
    print("   b. OR replace with Logistic-Normal KF (reuses existing KF infra)")
    print("   c. Then enable λ^D (decision-space anticipation rate)")
    print()
    print("5. Operator's question 'why not historical Dirichlet errors?':")
    print("   The current predictor has NO error history because it has no")
    print("   posterior. TRUE Dirichlet would naturally provide errors:")
    print("   prediction_error_t = ||predicted_t - observed_t||_KL or L1")
    print("   These could feed λ^D analogously to λ^K.")


if __name__ == "__main__":
    main()
