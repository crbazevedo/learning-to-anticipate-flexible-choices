"""W22-NC32: Logistic-Normal Kalman Filter (LNKF) for portfolio weight tracking.

Hypothesis (per docs/W22-NEXT-STEPS-NC32-36.md Section B2): portfolio weights
w in Delta^{d-1} (the (d-1)-simplex) can be tracked by a STANDARD Kalman
filter operating in unconstrained Aitchison log-ratio coordinates

    y_i = log(w_i / w_d)  for i in {0, ..., d-2}

Inverse transform:

    w_d = 1 / (1 + sum_{i=0}^{d-2} exp(y_i))
    w_i = exp(y_i) * w_d

This couples the (ROI, risk) KF infrastructure with the weight tracker into
a single mathematical framework, eliminating the artificial separation
between the "Dirichlet predictor" (decision space) and the "Kalman filter"
(objective space). Closed-form posterior variance per weight is now
available via standard KF covariance propagation.

Phase 1 scope (this module): INDEPENDENT y-space KF with random-walk default
F = I_{d-1}, isotropic process noise Q = process_noise * I, and isotropic
measurement noise R = measurement_noise * I.

Phase 2 (deferred to NC32+): joint state x_joint = [ROI, risk, y_1, ..., y_{d-1}]
with block-diagonal F and off-diagonal Q for coupling.

Scars (honest):
- Reference-asset choice (here: the LAST component w_d) is mathematically
  equivariant under permutation, but numerically affects log-stability if
  the chosen reference is near zero.
- The log-ratio transform fails on exact-zero weights; handled by EPS
  clipping at 1e-10 before forward transform.
- Q and R hyperparameters need tuning per regime (calibration scar).

Pure numpy; no scipy dependency for the KF math.
"""

from __future__ import annotations

import numpy as np


EPS = 1e-10


class LogisticNormalKF:
    """Standalone Kalman filter for simplex-valued portfolio weights.

    State y in R^{d-1} = Aitchison log-ratio coordinates of w in Delta^{d-1}.
    F = I_{d-1} (random walk), Q = process_noise * I, R = measurement_noise * I.

    Parameters
    ----------
    d : int
        Dimension of the simplex (number of portfolio assets). Must be >= 2.
        State dimension is d - 1.
    process_noise : float
        Diagonal magnitude of Q (isotropic process noise). Default 0.01.
    measurement_noise : float
        Diagonal magnitude of R (isotropic measurement noise). Default 0.001.
    initial_y : np.ndarray | None
        Optional initial state vector in R^{d-1}. Default = zeros, which
        inverse-transforms to the uniform simplex point w = (1/d, ..., 1/d).
    initial_P : np.ndarray | None
        Optional initial covariance in R^{(d-1)x(d-1)}. Default = I.
    """

    def __init__(
        self,
        d: int,
        process_noise: float = 0.01,
        measurement_noise: float = 0.001,
        initial_y: np.ndarray | None = None,
        initial_P: np.ndarray | None = None,
    ) -> None:
        if d < 2:
            raise ValueError(f"d must be >= 2 (got {d})")
        self.d = int(d)
        self.state_dim = self.d - 1

        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        # F = I (random walk), H = I (we observe y directly via forward(w))
        self.F = np.eye(self.state_dim)
        self.H = np.eye(self.state_dim)
        self.Q = self.process_noise * np.eye(self.state_dim)
        self.R = self.measurement_noise * np.eye(self.state_dim)

        # Stash initial state for reset()
        self._initial_y_default = (
            np.zeros(self.state_dim) if initial_y is None else np.asarray(initial_y, dtype=float).copy()
        )
        self._initial_P_default = (
            np.eye(self.state_dim) if initial_P is None else np.asarray(initial_P, dtype=float).copy()
        )

        if self._initial_y_default.shape != (self.state_dim,):
            raise ValueError(
                f"initial_y must have shape ({self.state_dim},) got {self._initial_y_default.shape}"
            )
        if self._initial_P_default.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"initial_P must have shape ({self.state_dim}, {self.state_dim}) got "
                f"{self._initial_P_default.shape}"
            )

        # Live state
        self.y = self._initial_y_default.copy()
        self.P = self._initial_P_default.copy()
        self.n_observations = 0

    # ------------------------------------------------------------------ #
    # Aitchison log-ratio forward / inverse transforms                    #
    # ------------------------------------------------------------------ #

    def _forward(self, w: np.ndarray) -> np.ndarray:
        """Simplex -> y. Reference asset is the last component w_d.

        y_i = log(w_i / w_d) for i in {0, ..., d-2}.
        EPS-clipped to avoid log(0) on sparse weights.
        """
        w = np.asarray(w, dtype=float).reshape(-1)
        if w.shape[0] != self.d:
            raise ValueError(f"w must have length d={self.d} (got {w.shape[0]})")
        w_clipped = np.clip(w, EPS, None)
        # Renormalize so the clipped vector still sums to 1 to within EPS scale
        w_clipped = w_clipped / np.sum(w_clipped)
        ref = w_clipped[-1]
        return np.log(w_clipped[:-1] / ref)

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        """y -> simplex. Numerically stable via log-sum-exp.

        w_d = 1 / (1 + sum exp(y_i)), w_i = exp(y_i) * w_d.
        """
        y = np.asarray(y, dtype=float).reshape(-1)
        if y.shape[0] != self.state_dim:
            raise ValueError(
                f"y must have length state_dim={self.state_dim} (got {y.shape[0]})"
            )
        # Stable softmax-style: append 0 (== log 1 for the reference) and softmax
        logits = np.concatenate([y, [0.0]])
        max_l = np.max(logits)
        exps = np.exp(logits - max_l)
        w = exps / np.sum(exps)
        return w

    # ------------------------------------------------------------------ #
    # Standard KF primitives                                              #
    # ------------------------------------------------------------------ #

    def predict(self, h: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """h-step ahead prediction (does NOT mutate state).

        For random-walk F = I:
            mu_y(t+h | t) = y(t)
            Sigma_y(t+h | t) = P(t) + h * Q

        For general F, the result is F^h y and F^h P (F^h)^T + sum_{k=0}^{h-1} F^k Q (F^k)^T.

        Returns
        -------
        (mu_y, Sigma_y) : tuple of np.ndarray
            Predictive mean (shape state_dim,) and covariance (state_dim x state_dim).
        """
        if h < 1:
            raise ValueError(f"h must be >= 1 (got {h})")
        # General multi-step: roll forward h applications of F and accumulate Q
        F_pow = np.eye(self.state_dim)
        mu = self.y.copy()
        Sigma = self.P.copy()
        for _ in range(h):
            mu = self.F @ mu
            Sigma = self.F @ Sigma @ self.F.T + self.Q
            F_pow = self.F @ F_pow  # not directly used but kept for clarity
        return mu, Sigma

    def update(self, y_obs: np.ndarray) -> None:
        """KF update step with an observation in y-space (mutates state).

        Standard equations (with H = I):
            mu_pred, Sigma_pred = predict(h=1)
            S = Sigma_pred + R         (innovation covariance)
            K = Sigma_pred @ S^{-1}    (Kalman gain)
            y_new = mu_pred + K @ (y_obs - mu_pred)
            P_new = (I - K) @ Sigma_pred
        """
        y_obs = np.asarray(y_obs, dtype=float).reshape(-1)
        if y_obs.shape[0] != self.state_dim:
            raise ValueError(
                f"y_obs must have length state_dim={self.state_dim} (got {y_obs.shape[0]})"
            )

        # 1-step predict (without mutating self)
        mu_pred = self.F @ self.y
        Sigma_pred = self.F @ self.P @ self.F.T + self.Q

        # Innovation
        innovation = y_obs - self.H @ mu_pred
        S = self.H @ Sigma_pred @ self.H.T + self.R

        # Gain (small dim; direct solve is fine)
        # K = Sigma_pred H^T S^{-1}
        K = np.linalg.solve(S.T, (Sigma_pred @ self.H.T).T).T

        # Posterior
        self.y = mu_pred + K @ innovation
        I = np.eye(self.state_dim)
        # Joseph-form-equivalent simplification (H = I): P = (I - K) Sigma_pred
        self.P = (I - K @ self.H) @ Sigma_pred
        # Symmetrize defensively
        self.P = 0.5 * (self.P + self.P.T)
        self.n_observations += 1

    # ------------------------------------------------------------------ #
    # Convenience layer                                                   #
    # ------------------------------------------------------------------ #

    def observe(self, w_obs: np.ndarray) -> None:
        """Forward-transform a simplex observation and run a KF update."""
        y_obs = self._forward(w_obs)
        self.update(y_obs)

    def predict_simplex_mean(self, h: int = 1) -> np.ndarray:
        """h-step ahead predictive mean inverse-transformed to the simplex.

        Note: inverse(E[y]) != E[inverse(y)] (Jensen). For an unbiased
        simplex estimate use predict_simplex_samples().
        """
        mu_y, _ = self.predict(h=h)
        return self._inverse(mu_y)

    def predict_simplex_samples(
        self,
        h: int = 1,
        n_mc: int = 100,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample from N(mu_y, Sigma_y) at horizon h and inverse-transform.

        Returns
        -------
        np.ndarray of shape (n_mc, d) — each row a simplex point.
        """
        if rng is None:
            rng = np.random.default_rng()
        mu_y, Sigma_y = self.predict(h=h)
        # Symmetrize and add tiny jitter for numerical PSD safety
        Sigma_y = 0.5 * (Sigma_y + Sigma_y.T) + 1e-12 * np.eye(self.state_dim)
        samples_y = rng.multivariate_normal(mean=mu_y, cov=Sigma_y, size=n_mc)
        samples_w = np.empty((n_mc, self.d), dtype=float)
        for i in range(n_mc):
            samples_w[i] = self._inverse(samples_y[i])
        return samples_w

    def reset(
        self,
        initial_y: np.ndarray | None = None,
        initial_P: np.ndarray | None = None,
    ) -> None:
        """Restore the filter to its initial state (or override defaults)."""
        if initial_y is not None:
            initial_y = np.asarray(initial_y, dtype=float).copy()
            if initial_y.shape != (self.state_dim,):
                raise ValueError(
                    f"initial_y must have shape ({self.state_dim},) got {initial_y.shape}"
                )
            self._initial_y_default = initial_y
        if initial_P is not None:
            initial_P = np.asarray(initial_P, dtype=float).copy()
            if initial_P.shape != (self.state_dim, self.state_dim):
                raise ValueError(
                    f"initial_P must have shape ({self.state_dim}, {self.state_dim}) got "
                    f"{initial_P.shape}"
                )
            self._initial_P_default = initial_P
        self.y = self._initial_y_default.copy()
        self.P = self._initial_P_default.copy()
        self.n_observations = 0
