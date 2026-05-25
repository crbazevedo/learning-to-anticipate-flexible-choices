"""W22 Probe AB — λ^K per-portfolio differentiation analyzer.

Standalone module that quantifies what λ^K *would* look like if it were
computed per-portfolio (versus the current solution-invariant version
shipped in :mod:`src.algorithms.anticipatory_learning`).

The analyzer makes NO modifications to shared code paths
(``anticipatory_learning.py``, ``sms_emoa.py``, ``kalman_filter.py``).
Any future wiring that records per-portfolio KF residual windows would
be a *separate* operator decision; this probe ships only the contained
analyzer + a unit-test battery + a hypothesis document so we can
falsify the claim::

    AB-H1  A per-portfolio λ^K (computed from per-portfolio KF residual
           windows) provides meaningful discrimination across portfolios
           in heterogeneous-tracking regimes, where the current
           shared-window λ^K provides exactly zero.

Inspection 6 reference
----------------------
``docs/W22-INSPECTIONS-SYNTHESIS.md`` ("Anticipation rate is poorly
differentiated per period") records the finding:

    λ^K is solution-invariant per period (constant across portfolios)

Because the live implementation accumulates a single shared residual
window via ``record_kf_residual`` and exposes it through
``_compute_lambda_k(...)`` as a function of *the window only*, every
solution in the population is assigned the identical λ^K at a given
period. The "differentiation" claimed by AMFC degenerates to current-
state argmax modulo a uniform anticipation offset.

Canonical formula (mirrored from ``_compute_lambda_k``)
-------------------------------------------------------
For a non-empty residual window ``w = [r_1, ..., r_K]``::

    residual_sum = sum(w)
    scale        = max(1.0, mean(w))
    normalized   = 1 - exp(-residual_sum / (len(w) * scale))
    lambda_k     = 0.5 * normalized           # mapped to [0, 0.5]

For an empty window, λ^K = 0 (we report the warm-up branch as 0 here;
in production the live code falls back to a per-solution traditional
rate during warm-up — that fallback is NOT replicated here because
this probe deliberately isolates the residual-window contribution).

Discrimination thresholds (closed-enum)
---------------------------------------
``DISCRIMINATION_THRESHOLDS`` maps the per-portfolio range
(``max - min``) to a label::

    NEGLIGIBLE   range < 0.05
    MODEST       0.05 <= range < 0.15
    STRONG       range >= 0.15

The thresholds match the rough scale of λ^K's [0, 0.5] codomain: a
range of 0.05 is 10% of the codomain — easily lost in noise; 0.15 is
30% — visibly discriminating across portfolios.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Closed-enum discrimination thresholds (paired-diff invariant — changes
# require updating the comparator below + the markdown doc)
# ---------------------------------------------------------------------------
DISCRIMINATION_THRESHOLDS: dict[str, tuple[float, float]] = {
    "NEGLIGIBLE": (0.0, 0.05),
    "MODEST": (0.05, 0.15),
    "STRONG": (0.15, math.inf),
}

# Canonical keys returned by compare_lambda_k_modes — closed-enum.
COMPARE_KEYS: tuple[str, ...] = (
    "shared_lambda_k",
    "per_portfolio_lambda_k",
    "per_portfolio_std",
    "per_portfolio_range",
    "discrimination_significance",
)


# ---------------------------------------------------------------------------
# Canonical λ^K formula (single-source-of-truth for shared + per-portfolio
# computations — both call into this helper so the formula matches bit-for-bit
# with the production ``_compute_lambda_k`` implementation up to numerical
# associativity in summation)
# ---------------------------------------------------------------------------
def _lambda_k_from_window(window: Sequence[float]) -> float:
    """Compute λ^K from a single residual window (sigmoid → [0, 0.5]).

    Mirrors ``anticipatory_learning._compute_lambda_k`` line-for-line on
    the non-empty branch::

        residual_sum = float(np.sum(window))
        scale        = max(1.0, float(np.mean(window)))
        normalized   = 1.0 - float(np.exp(-residual_sum / (len(window) * scale)))
        return 0.5 * normalized

    Empty windows return 0.0 (this probe does NOT replicate the
    per-solution traditional-rate fallback used during production
    warm-up — see module docstring).
    """
    w = list(window)
    if not w:
        return 0.0
    arr = np.asarray(w, dtype=float)
    residual_sum = float(np.sum(arr))
    scale = max(1.0, float(np.mean(arr)))
    normalized = 1.0 - float(np.exp(-residual_sum / (len(arr) * scale)))
    return 0.5 * normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_shared_lambda_k(residual_window: Sequence[float]) -> float:
    """Replicate the current solution-invariant λ^K implementation.

    Parameters
    ----------
    residual_window
        Sequence of squared-residual scalars accumulated across ALL
        portfolios over the K-period sliding window (this is the same
        buffer that ``record_kf_residual`` populates in production).

    Returns
    -------
    float
        λ^K ∈ [0, 0.5]. Empty input returns 0.0.
    """
    return _lambda_k_from_window(residual_window)


def compute_per_portfolio_lambda_k(
    per_portfolio_residual_windows: Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute one λ^K per portfolio from per-portfolio residual windows.

    Parameters
    ----------
    per_portfolio_residual_windows
        Outer sequence indexes the portfolio. Inner sequence is that
        portfolio's own K-period residual window. Windows may have
        different lengths (e.g. fresh / pruned portfolios still in
        warm-up).

    Returns
    -------
    np.ndarray
        Shape ``(n_portfolios,)``. Each element is the λ^K that would
        be assigned to that portfolio if the residual window were
        recorded per-solution rather than as a shared aggregate.
    """
    if not per_portfolio_residual_windows:
        return np.zeros(0, dtype=float)
    out = np.array(
        [_lambda_k_from_window(w) for w in per_portfolio_residual_windows],
        dtype=float,
    )
    return out


def _classify_discrimination(range_value: float) -> str:
    """Map a per-portfolio range to a closed-enum significance label."""
    for label, (lo, hi) in DISCRIMINATION_THRESHOLDS.items():
        if lo <= range_value < hi:
            return label
    # Defensive — DISCRIMINATION_THRESHOLDS covers [0, inf) by construction.
    return "STRONG"


def compare_lambda_k_modes(
    per_portfolio_residual_windows: Sequence[Sequence[float]],
) -> dict:
    """Compare the shared-window λ^K vs the per-portfolio λ^K.

    The "shared" baseline concatenates every portfolio's residual window
    into one flat list and applies the production formula — this is
    the closest analytic stand-in for what the live
    ``_kf_residual_window`` would hold if it had recorded each
    portfolio's residual contribution sequentially.

    Returns
    -------
    dict
        Closed-enum keyed by :data:`COMPARE_KEYS`::

            shared_lambda_k              float in [0, 0.5]
            per_portfolio_lambda_k       np.ndarray of shape (n_portfolios,)
            per_portfolio_std            float (std across portfolios)
            per_portfolio_range          float (max - min)
            discrimination_significance  one of {'NEGLIGIBLE','MODEST','STRONG'}
    """
    per_portfolio = compute_per_portfolio_lambda_k(per_portfolio_residual_windows)

    # Shared baseline = concatenated residuals as one window.
    concatenated: list[float] = []
    for w in per_portfolio_residual_windows:
        concatenated.extend(float(x) for x in w)
    shared = compute_shared_lambda_k(concatenated)

    if per_portfolio.size == 0:
        std_v = 0.0
        range_v = 0.0
    else:
        std_v = float(np.std(per_portfolio))
        range_v = float(np.max(per_portfolio) - np.min(per_portfolio))

    return {
        "shared_lambda_k": float(shared),
        "per_portfolio_lambda_k": per_portfolio,
        "per_portfolio_std": std_v,
        "per_portfolio_range": range_v,
        "discrimination_significance": _classify_discrimination(range_v),
    }


# ---------------------------------------------------------------------------
# Synthetic residual-window generator (for unit tests + future driver
# scripts that want to probe λ^K discrimination at varying heterogeneity)
# ---------------------------------------------------------------------------
def synthesize_residual_windows(
    n_portfolios: int,
    window_size: int,
    mean_residual: float,
    std_residual: float,
    heterogeneity: float,
    rng: np.random.Generator,
) -> list[list[float]]:
    """Generate synthetic per-portfolio residual windows.

    Parameters
    ----------
    n_portfolios
        Number of portfolios (outer length of return value).
    window_size
        Size K of each portfolio's residual window (inner length).
    mean_residual
        Population-level mean residual (positive — squared residuals).
        Per-portfolio means are sampled from
        ``N(mean_residual, heterogeneity * mean_residual)`` and then
        clipped to be positive.
    std_residual
        Within-portfolio noise standard deviation around its portfolio
        mean. Per-residual draws are clipped at 0 to remain
        non-negative (squared-residual semantics).
    heterogeneity
        Controls how much per-portfolio means differ.
        ``heterogeneity = 0`` → all portfolios share ``mean_residual``
        (homogeneous tracking regime; shared λ^K dominates).
        ``heterogeneity = 1.0`` → portfolio means spread with σ equal
        to ``mean_residual`` (heterogeneous tracking regime; per-
        portfolio λ^K is hypothesized to discriminate).
    rng
        Numpy ``Generator`` for reproducibility.

    Returns
    -------
    list of list of float
        Outer length ``n_portfolios``, inner length ``window_size``.
        All elements are ≥ 0 (squared-residual semantics).
    """
    if n_portfolios <= 0 or window_size <= 0:
        return []
    mean_residual = float(mean_residual)
    std_residual = float(std_residual)
    heterogeneity = float(heterogeneity)
    if mean_residual < 0:
        raise ValueError("mean_residual must be >= 0 (squared residuals)")
    if std_residual < 0:
        raise ValueError("std_residual must be >= 0")
    if heterogeneity < 0:
        raise ValueError("heterogeneity must be >= 0")

    # Per-portfolio means: N(mean_residual, heterogeneity * mean_residual)
    # clipped to be strictly positive (squared residuals can't be < 0).
    portfolio_means = rng.normal(
        loc=mean_residual,
        scale=heterogeneity * mean_residual,
        size=n_portfolios,
    )
    portfolio_means = np.clip(portfolio_means, a_min=1e-12, a_max=None)

    out: list[list[float]] = []
    for mu in portfolio_means:
        draws = rng.normal(loc=mu, scale=std_residual, size=window_size)
        draws = np.clip(draws, a_min=0.0, a_max=None)
        out.append([float(x) for x in draws])
    return out
