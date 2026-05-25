"""W22 Probe Q-v1: per-asset AR(1) return predictor.

Standalone module — does NOT touch shared code (``sms_emoa.py``,
``anticipatory_learning.py``, ``kalman_filter.py``). The portfolio
optimizer integration is a *future* operator decision; this module only
ships the predictor + a no-change baseline so we can falsify the
hypothesis that AR(1) beats persistence at the per-asset level
(Q-H1 in ``docs/W22-ALT-SIGNAL-PROBES.md``).

Hypotheses (from the probe spec):
    Q-H1  Per-asset AR(1) predictions of r_{k,t+1} beat the implicit
          "no change" baseline (r_{k,t+1} = r_{k,t}).
    Q-H2  Using these predictions as inputs to a mean-variance portfolio
          optimizer changes the chosen assets (downstream HV gain).

This module addresses Q-H1 ONLY. Q-H2 requires wiring into the
optimizer, which is intentionally deferred.

AR(1) model
-----------
For each asset k independently::

    r_{k,t+1} = mu_k + rho_k * (r_{k,t} - mu_k) + eps

with ``mu_k`` = window mean and ``rho_k`` = lag-1 autocorrelation
estimated by ordinary least squares on the centred series. The
one-step-ahead prediction reduces to::

    r_hat_{k,t+1} = mu_k + rho_k * (r_{k,t} - mu_k)

Numerical-safety contract
-------------------------
* ``window_size < 2``        -> fall back to no-change (rho = 0, mu = r_t)
* ``len(buffer) < 2``        -> fall back to no-change
* near-zero variance         -> rho = 0 (predict the mean)
* NaN / inf in observations  -> dropped from the per-asset window before
                                fitting (the per-asset window length may
                                therefore be smaller than ``window_size``)
* fewer than 2 finite obs    -> per-asset no-change fallback

The class is stateless except for the rolling per-asset buffers; it does
not mutate any module-level globals and is safe to instantiate multiple
times in the same process.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


_DEFAULT_WINDOW = 60
_MIN_WINDOW = 2
_VAR_FLOOR = 1e-12  # below this we treat the centred series as constant


class AR1AssetPredictor:
    """Per-asset AR(1) predictor with a rolling window.

    Parameters
    ----------
    d
        Number of assets (dimension of each return vector). Must be > 0.
    window_size
        Length of the rolling per-asset window used to fit AR(1).
        Values < 2 trigger the no-change fallback for ``predict_next``.

    Notes
    -----
    The predictor stores one ``collections.deque`` per asset so each
    asset's window evolves independently. This keeps the memory cost
    O(d * window_size) and avoids re-allocating arrays on every
    ``observe`` call.
    """

    def __init__(self, d: int, window_size: int = _DEFAULT_WINDOW) -> None:
        if d <= 0:
            raise ValueError(f"d must be > 0 (got {d})")
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0 (got {window_size})")
        self.d = int(d)
        self.window_size = int(window_size)
        # maxlen=window_size gives O(1) rolling eviction.
        # window_size==0 is technically valid (always no-change) but a
        # deque(maxlen=0) silently drops everything; use 1 so we can
        # still return the latest observation in the no-change baseline.
        buf_len = max(1, self.window_size)
        self._buffers: list[Deque[float]] = [
            deque(maxlen=buf_len) for _ in range(self.d)
        ]
        self._n_observed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, returns: np.ndarray) -> None:
        """Append one per-asset return vector to each rolling buffer.

        Parameters
        ----------
        returns
            1-D array of length ``d``. Non-finite entries are stored as
            NaN and filtered out at fit time (the per-asset window
            length may be shorter than ``window_size`` as a result).
        """
        arr = np.asarray(returns, dtype=float).reshape(-1)
        if arr.shape[0] != self.d:
            raise ValueError(
                f"returns must have length d={self.d} (got {arr.shape[0]})"
            )
        for k in range(self.d):
            v = arr[k]
            # Store NaN for non-finite so the fit code can drop them.
            self._buffers[k].append(v if np.isfinite(v) else np.nan)
        self._n_observed += 1

    def predict_no_change(self) -> np.ndarray:
        """No-change baseline: predict r_{t+1} = r_t.

        Returns
        -------
        np.ndarray
            Length-``d`` vector. For assets with no observations yet,
            returns 0.0 (a deliberately neutral fallback so the caller
            can still feed the vector to a downstream optimizer).
        """
        out = np.zeros(self.d, dtype=float)
        for k in range(self.d):
            buf = self._buffers[k]
            if len(buf) == 0:
                continue
            # Walk backwards for the most recent finite value.
            for v in reversed(buf):
                if np.isfinite(v):
                    out[k] = v
                    break
        return out

    def predict_next(self) -> np.ndarray:
        """One-step-ahead AR(1) prediction per asset.

        Returns
        -------
        np.ndarray
            Length-``d`` vector of predicted next-period returns.

        Notes
        -----
        Falls back to ``predict_no_change`` for any asset whose window
        has fewer than 2 finite observations or whose centred variance
        is below ``_VAR_FLOOR``. Falls back globally if
        ``window_size < _MIN_WINDOW`` (i.e. AR is not estimable by
        contract).
        """
        if self.window_size < _MIN_WINDOW:
            return self.predict_no_change()

        no_change = self.predict_no_change()
        out = no_change.copy()
        for k in range(self.d):
            buf = self._buffers[k]
            if len(buf) < _MIN_WINDOW:
                continue  # no_change already
            arr = np.fromiter(buf, dtype=float, count=len(buf))
            finite = arr[np.isfinite(arr)]
            if finite.size < _MIN_WINDOW:
                continue  # no_change already

            mu = float(finite.mean())
            centred = finite - mu
            # Lag-1 autocorrelation via OLS on (x_{t-1}, x_t).
            x_prev = centred[:-1]
            x_curr = centred[1:]
            denom = float(np.dot(x_prev, x_prev))
            if denom < _VAR_FLOOR:
                # Degenerate: window is constant -> rho undefined.
                # Predict the mean (which equals the constant).
                out[k] = mu
                continue
            rho = float(np.dot(x_prev, x_curr) / denom)
            # Clip to (-1, 1) for stationarity sanity. AR(1) with
            # |rho| >= 1 is non-stationary; we don't extrapolate.
            if not np.isfinite(rho):
                out[k] = mu
                continue
            rho = float(np.clip(rho, -0.999, 0.999))
            r_t = float(finite[-1])
            out[k] = mu + rho * (r_t - mu)
        return out

    # ------------------------------------------------------------------
    # Introspection helpers (no side effects; safe in tests)
    # ------------------------------------------------------------------

    @property
    def n_observed(self) -> int:
        """Total number of ``observe`` calls (NOT per-asset finite count)."""
        return self._n_observed

    def window_length(self, asset_idx: int) -> int:
        """Current rolling-window length for one asset (incl. NaNs)."""
        if not 0 <= asset_idx < self.d:
            raise IndexError(asset_idx)
        return len(self._buffers[asset_idx])
