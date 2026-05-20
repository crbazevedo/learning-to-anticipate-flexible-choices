"""W22 Probe AG — process noise (Q) velocity sensitivity for KF.

Question: Probe AC+AF found that the production KF has |lag-1
autocorrelation| ≈ 0.5 on risk innovations (MODERATE/borderline
MIS-TUNED). Root cause traced to `P_next = F P F^T` (no Q term)
matching legacy C++ exactly.

Probe AG quantifies — on SYNTHETIC time series with KNOWN dynamics —
how much adding `Q = diag([0, 0, σ²_v, σ²_v])` (velocity-only process
noise) improves:
  - prediction MSE
  - innovation autocorrelation (Mehra whiteness)

This is STANDALONE — does NOT touch shared code paths. The KF here
is a self-contained re-implementation of `kalman_filter.py` with an
optional Q parameter. The production KF remains untouched.

Per docs/W22-PROBE-AC-AF-KF-DIAGNOSTICS.md "Future Probe AG" section.
"""
from __future__ import annotations

import numpy as np


def kalman_predict_with_q(x: np.ndarray, P: np.ndarray, F: np.ndarray,
                            Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prediction with optional Q. Q can be zeros to match production."""
    x_next = F @ x
    P_next = F @ P @ F.T + Q
    return x_next, P_next


def kalman_update(x_next: np.ndarray, P_next: np.ndarray, z: np.ndarray,
                   H: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard KF update. Returns (x_new, P_new, innovation, S)."""
    y = z - H @ x_next
    S = H @ P_next @ H.T + R
    K = P_next @ H.T @ np.linalg.inv(S)
    x = x_next + K @ y
    P = (np.eye(P_next.shape[0]) - K @ H) @ P_next
    return x, P, y, S


def simulate_kf_run(true_states: np.ndarray, true_obs: np.ndarray,
                     F: np.ndarray, H: np.ndarray, R: np.ndarray,
                     Q: np.ndarray) -> dict:
    """Run KF on a known time series with given Q.

    Args:
        true_states: shape (T, 4) ground-truth state trajectory
        true_obs: shape (T, 2) noisy observations
        F, H, R: KF matrices (constant)
        Q: process noise (4x4); use np.zeros((4,4)) for production-equivalent

    Returns dict with:
      - predicted_obs: shape (T, 2) one-step-ahead predictions
      - innovations: shape (T, 2)
      - P_traces: shape (T,) trace of P at each step
      - mse_obs: float overall MSE between predicted_obs and true_obs
      - lag1_autocorr: per-objective lag-1 autocorrelation of innovations
    """
    T = true_obs.shape[0]
    x = np.array([true_obs[0, 0], true_obs[0, 1], 0.0, 0.0])
    P = np.diag([0.1, 0.1, 1000, 1000])  # production init
    predicted_obs = np.zeros((T, 2))
    innovations = np.zeros((T, 2))
    P_traces = np.zeros(T)

    for t in range(T):
        x_next, P_next = kalman_predict_with_q(x, P, F, Q)
        predicted_obs[t] = H @ x_next
        if t > 0:  # innovation requires comparison with observation
            x, P, y, S = kalman_update(x_next, P_next, true_obs[t], H, R)
            innovations[t] = y
        P_traces[t] = float(np.trace(P))

    # MSE over t >= 1
    mse = float(np.mean(np.sum((predicted_obs[1:] - true_obs[1:]) ** 2, axis=1)))

    # Per-objective lag-1 autocorrelation of innovations (skip t=0)
    def lag1(x):
        x = x - np.mean(x)
        num = float(np.sum(x[:-1] * x[1:]))
        den = float(np.sum(x ** 2))
        return num / den if den > 1e-12 else 0.0
    lag1_roi = lag1(innovations[1:, 0])
    lag1_risk = lag1(innovations[1:, 1])

    return {
        "predicted_obs": predicted_obs,
        "innovations": innovations,
        "P_traces": P_traces,
        "mse_obs": mse,
        "lag1_autocorr_ROI": lag1_roi,
        "lag1_autocorr_risk": lag1_risk,
    }


def simulate_synthetic_dynamics(T: int = 23, sigma_v: float = 0.01,
                                   sigma_obs: float = 0.05,
                                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate true state + noisy obs from a velocity-random-walk model.

    The TRUE dynamics ARE noisy in velocity (Q != 0 in truth). The
    production KF assumes Q=0 (no velocity drift), so its predictions
    should be SYSTEMATICALLY worse on this generator than a KF with the
    correct Q.

    Args:
        T: number of time steps
        sigma_v: std-dev of velocity random walk
        sigma_obs: std-dev of measurement noise
        seed: RNG seed

    Returns:
        (true_states[T, 4], obs[T, 2])
    """
    rng = np.random.default_rng(seed)
    F = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    x = np.array([0.0001, 0.05, 0.0, 0.0])  # plausible initial portfolio state
    true_states = np.zeros((T, 4))
    obs = np.zeros((T, 2))
    for t in range(T):
        # Add velocity drift to the velocity components only (random walk).
        x = F @ x
        x[2] += rng.normal(0, sigma_v) * 1e-3  # ROI velocity drift
        x[3] += rng.normal(0, sigma_v)         # risk velocity drift
        true_states[t] = x
        obs[t] = x[:2] + rng.normal(0, sigma_obs, size=2) * np.array([1e-3, 1.0])
    return true_states, obs


def compare_q_settings(T: int = 23, n_seeds: int = 50,
                          sigma_v: float = 0.01) -> dict:
    """Run KF with Q=0 vs Q=σv²I on synthetic data; report aggregate stats."""
    F = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    R = np.diag([0.01, 0.01])  # match production

    Q_zero = np.zeros((4, 4))
    Q_velocity = np.zeros((4, 4))
    Q_velocity[2, 2] = (sigma_v * 1e-3) ** 2  # ROI velocity process noise
    Q_velocity[3, 3] = sigma_v ** 2           # risk velocity process noise

    mse_zero_q, mse_vel_q = [], []
    lag1_roi_zero, lag1_risk_zero = [], []
    lag1_roi_vel, lag1_risk_vel = [], []
    for seed in range(n_seeds):
        true_states, obs = simulate_synthetic_dynamics(T=T, sigma_v=sigma_v,
                                                          seed=seed)
        res_zero = simulate_kf_run(true_states, obs, F, H, R, Q_zero)
        res_vel = simulate_kf_run(true_states, obs, F, H, R, Q_velocity)
        mse_zero_q.append(res_zero["mse_obs"])
        mse_vel_q.append(res_vel["mse_obs"])
        lag1_roi_zero.append(res_zero["lag1_autocorr_ROI"])
        lag1_risk_zero.append(res_zero["lag1_autocorr_risk"])
        lag1_roi_vel.append(res_vel["lag1_autocorr_ROI"])
        lag1_risk_vel.append(res_vel["lag1_autocorr_risk"])
    return {
        "n_seeds": n_seeds,
        "T": T,
        "sigma_v": sigma_v,
        "Q_zero": {
            "mse_obs_mean": float(np.mean(mse_zero_q)),
            "mse_obs_std": float(np.std(mse_zero_q)),
            "lag1_autocorr_ROI_mean": float(np.mean(lag1_roi_zero)),
            "lag1_autocorr_ROI_std": float(np.std(lag1_roi_zero)),
            "lag1_autocorr_risk_mean": float(np.mean(lag1_risk_zero)),
            "lag1_autocorr_risk_std": float(np.std(lag1_risk_zero)),
        },
        "Q_velocity": {
            "mse_obs_mean": float(np.mean(mse_vel_q)),
            "mse_obs_std": float(np.std(mse_vel_q)),
            "lag1_autocorr_ROI_mean": float(np.mean(lag1_roi_vel)),
            "lag1_autocorr_ROI_std": float(np.std(lag1_roi_vel)),
            "lag1_autocorr_risk_mean": float(np.mean(lag1_risk_vel)),
            "lag1_autocorr_risk_std": float(np.std(lag1_risk_vel)),
        },
    }
