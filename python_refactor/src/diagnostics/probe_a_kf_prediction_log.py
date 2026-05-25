"""W22 Probe A: per-period KF prediction logging.

Per-period (NOT per-generation) capture of:
- Each Pareto-front portfolio's KF prediction for period t+1
- Persistence baseline (current period t observation)
- Actual t+1 observation (computed by evaluating weights under
  period-t+1's MLE μ̂/Σ̂)

Wired into walk_forward.py:run_walk_forward. Singleton sink writes
JSONL to W22_PROBE_A_LOG_PATH env var; unset → no-op.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path


_LOG_PATH_ENV = "W22_PROBE_A_LOG_PATH"
_LOCK = threading.Lock()
_PATH_CACHE: Path | None = None


def _get_log_path() -> Path | None:
    """Resolve log path from env var. None if disabled."""
    global _PATH_CACHE
    raw = os.environ.get(_LOG_PATH_ENV, "").strip()
    if not raw:
        return None
    if _PATH_CACHE is None or str(_PATH_CACHE) != raw:
        _PATH_CACHE = Path(raw)
        _PATH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    return _PATH_CACHE


def is_enabled() -> bool:
    """Check if probe-A logging is active."""
    return _get_log_path() is not None


def log_period_kf_prediction(
    *,
    scenario: str,
    seed: int,
    period_t: int,
    portfolio_idx: int,
    weights: list[float],
    # KF prediction for t+1 (the anticipative arm's output)
    kf_predicted_ROI_t_plus_1: float,
    kf_predicted_risk_t_plus_1: float,
    # Persistence baseline (current t observation)
    persistence_ROI_t: float,
    persistence_risk_t: float,
    # Actual observed t+1 (weights evaluated under period-t+1 MLE)
    actual_ROI_t_plus_1: float,
    actual_risk_t_plus_1: float,
    # Bonus context: KF covariance diagonal (uncertainty proxy)
    kf_P_diag: list[float],
) -> None:
    """Append a per-period per-portfolio KF prediction record."""
    path = _get_log_path()
    if path is None:
        return

    record = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "scenario": str(scenario),
        "seed": int(seed),
        "period_t": int(period_t),
        "portfolio_idx": int(portfolio_idx),
        "weights": [float(w) for w in weights],
        "kf_predicted_ROI_t_plus_1": float(kf_predicted_ROI_t_plus_1),
        "kf_predicted_risk_t_plus_1": float(kf_predicted_risk_t_plus_1),
        "persistence_ROI_t": float(persistence_ROI_t),
        "persistence_risk_t": float(persistence_risk_t),
        "actual_ROI_t_plus_1": float(actual_ROI_t_plus_1),
        "actual_risk_t_plus_1": float(actual_risk_t_plus_1),
        "kf_P_diag": [float(x) for x in kf_P_diag],
    }
    line = json.dumps(record)
    with _LOCK:
        with path.open("a") as f:
            f.write(line + "\n")
