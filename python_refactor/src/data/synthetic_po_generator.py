"""
Synthetic Portfolio Optimization (PO) Benchmark Generator.

Implements the random dynamic portfolio-selection instance generator from
Azevedo & Von Zuben's IEEE TCYB 2015 paper (Eqs 31-32) and the canonical
thesis specification (Azevedo, PhD thesis, Unicamp 2014, Eqs 7.6-7.9,
pages 142-143).

Equations
---------
The generator produces a stream of joint-asset-return distribution
parameters ``X_t = (mu_t, Sigma_t)`` that disrupt every ``tau`` periods
with severity ``eta in [0, 1]``.

Let ``X_1 = (mu_1, Sigma_1)`` be the initial parameter set and ``P_{t_k}``
a random parameter set sampled at the k-th disruption boundary ``t_k``
(``t_k = k * tau``, with ``t_0 = tau`` being the first disruption).

* **Eq 7.6 (first disruption)**::

      X_{t_0} = (1 - eta) * X_1 + eta * P_{t_0}

* **Eq 7.7 (smooth update for t in (1, t_0))**::

      X_t = (1 - t/t_0) * X_1 + (t/t_0) * X_{t_0}

* **Eq 7.8 (k-th disruption boundary update)**::

      X_{t_k} = (1 - eta) * X_{t_{k-1}} + eta * P_{t_k}

  (paper Eq 32 phrasing; thesis Eq 7.8 written with slightly different
  but equivalent indexing.)

* **Eq 7.9 (smooth update for t in (t_{k-1}, t_k))**::

      X_t = (1 - t/t_{k-1}) * X_{t_{k-1}} + (t/t_{k-1}) * P_{t_k}

  In this implementation we use the equivalent intra-segment linear
  interpolation::

      X_t = (1 - alpha) * X_{t_{k-1}} + alpha * X_{t_k},
      alpha = (t - t_{k-1}) / (t_k - t_{k-1})

  which matches the paper's intra-segment smooth dynamics
  (paper page 8: "the sequence of parameters in h ∈ (1, t_0)
  ... undergo smooth linear dynamics").

Random parameter set ``P``
--------------------------
* ``mu_P``: uniform sample from the hyperbox ``[mu_lb, mu_ub]^d``
  (thesis page 143; paper Eq 32 prose).
* ``Sigma_P``: Gram-matrix construction (Holmes 1991, ref [33] in
  paper / [117] in thesis). For each row independently sample
  ``x ~ N(0, I_d)`` and project to the unit hypersphere
  ``G^{d-1}: x / ||x||``. The correlation matrix is ``rho = X X^T``
  where ``X`` is the matrix of unit-sphere rows. The covariance is
  then ``Sigma(i, j) = rho(i, j) * sigma_i * sigma_j`` with
  ``sigma_i ~ Uniform([sigma_lb, sigma_ub])``.

Daily returns from per-period parameters
----------------------------------------
At each period ``t`` (``t = 1, ..., T_periods``) we sample
``days_per_period`` IID daily return vectors from
``N(mu_t, Sigma_t)`` per the paper's Step 1 (page 8).

Defaults reproduce the PO(8, 1.0) instance described in the paper
(section V-C / table II) and the thesis (section 7.2.1 / 7.3):

* ``d = 30`` simulated assets
* ``T_periods = 25`` investment periods
* ``days_per_period = 50`` (50 lagged daily returns per period — the
  paper's "stage" granularity, page 9)
* ``tau = 8`` periods between disruptions ("low frequency")
* ``eta = 1.0`` severity ("maximum")

References
----------
* C. R. B. Azevedo and F. J. Von Zuben, "Anticipatory stochastic
  multi-objective optimization for uncertainty handling in portfolio
  selection," IEEE Trans. Cybern., 2015 — Eqs 31-32, section V-C.
* C. R. B. Azevedo, "Anticipation in multiple criteria decision-making
  under uncertainty," PhD thesis, Unicamp, 2014 — Eqs 7.6-7.9,
  section 7.2.1, pages 142-143.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class POGeneratorConfig:
    """Configuration for the PO(tau, eta) synthetic benchmark generator.

    All defaults reproduce PO(8, 1.0) as documented in the paper /
    thesis. Lower/upper bounds for mu and sigma follow the thesis
    convention (low-magnitude daily returns and volatilities consistent
    with the "50 lagged daily returns" granularity, paper page 9).
    """

    d: int = 30
    """Number of risky assets (paper page 9, thesis page 145: d=30)."""

    T_periods: int = 25
    """Total number of investment periods (paper page 9, thesis
    page 145: T=25)."""

    days_per_period: int = 50
    """Number of daily-return observations per period (paper page 9,
    thesis page 145-146: 50 business days per period)."""

    tau: int = 8
    """Periodicity of disruptive changes — disruption every ``tau``
    periods (thesis section 7.2.3, page 146: tau in {2, 4, 8})."""

    eta: float = 1.0
    """Severity of disruptive changes in [0, 1] (thesis section 7.2.3,
    page 146: eta in {0.5, 1.0}). eta=1.0 means the new random P
    completely replaces the previous X at each disruption boundary."""

    mu_lb: float = -0.005
    """Lower bound for the uniform daily-return hyperbox (thesis
    page 143: "mu_lb and mu_ub are the lower and upper bounds of
    return for all assets")."""

    mu_ub: float = 0.005
    """Upper bound for the uniform daily-return hyperbox."""

    sigma_lb: float = 0.005
    """Lower bound for per-asset standard deviation (thesis page 143:
    "sigma_i, uniformly sampled from [sigma_lb, sigma_ub]")."""

    sigma_ub: float = 0.04
    """Upper bound for per-asset standard deviation."""

    seed: int = 42
    """RNG seed for reproducibility — same seed must produce
    bit-identical output."""


def _sample_random_P(
    rng: np.random.Generator, cfg: POGeneratorConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a random parameter set ``(mu_P, Sigma_P)``.

    ``mu_P`` is uniform on ``[mu_lb, mu_ub]^d``. ``Sigma_P`` is built
    via the Gram-matrix method (thesis page 143; paper page 8): sample
    a random matrix ``X`` whose rows are unit vectors uniformly
    distributed on ``G^{d-1}``, form the correlation matrix
    ``rho = X X^T``, then scale by uniformly-sampled per-asset
    standard deviations.

    Returns
    -------
    mu_P : ndarray, shape (d,)
    Sigma_P : ndarray, shape (d, d)
        Symmetric positive-semidefinite covariance matrix.
    """
    d = cfg.d

    # mu_P ~ Uniform([mu_lb, mu_ub]^d)
    mu_P = rng.uniform(cfg.mu_lb, cfg.mu_ub, size=d)

    # Gram-matrix construction of the correlation matrix.
    # Sample X ~ N(0, I); each row is then projected to the unit
    # hypersphere by row-normalisation. This is the canonical
    # "random correlation matrix" method (Holmes 1991, ref [33]
    # in paper / [117] in thesis).
    X_mat = rng.standard_normal(size=(d, d))
    row_norms = np.linalg.norm(X_mat, axis=1, keepdims=True)
    # Guard against the (measure-zero) all-zero row.
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    X_unit = X_mat / row_norms  # rows on G^{d-1}
    rho = X_unit @ X_unit.T  # rho(i, j) = X_unit[i] . X_unit[j]

    # Per-asset standard deviations.
    sigmas = rng.uniform(cfg.sigma_lb, cfg.sigma_ub, size=d)

    # Sigma(i, j) = rho(i, j) * sigma_i * sigma_j
    Sigma_P = rho * np.outer(sigmas, sigmas)

    # Symmetrise to remove any floating-point asymmetry from the
    # outer-product scaling.
    Sigma_P = 0.5 * (Sigma_P + Sigma_P.T)

    return mu_P, Sigma_P


def generate_po_parameter_stream(
    cfg: Optional[POGeneratorConfig] = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate the stream of per-period ``(mu_t, Sigma_t)`` parameters.

    Implements thesis Eqs 7.6-7.9 (paper Eqs 31-32):

    * At ``t = 1`` we start at ``X_1 = (mu_1, Sigma_1)``.
    * At each disruption boundary ``t_k = k * tau`` for ``k >= 1``,
      we form ``X_{t_k} = (1 - eta) * X_{t_{k-1}} + eta * P_{t_k}``
      where ``P_{t_k}`` is a fresh random parameter set.
      (``X_{t_0} = X_1`` for the first segment, so the first
      disruption at ``t_1 = tau`` reduces to Eq 7.6.)
    * Between two disruption boundaries ``t_{k-1}`` and ``t_k`` we
      linearly interpolate (Eq 7.9, paper page 8 smooth dynamics)
      from ``X_{t_{k-1}}`` to ``X_{t_k}``.

    Returns
    -------
    mus : list of ndarray, length ``T_periods``
        ``mus[t-1]`` is ``mu_t``.
    sigmas : list of ndarray, length ``T_periods``
        ``sigmas[t-1]`` is ``Sigma_t``.
    """
    if cfg is None:
        cfg = POGeneratorConfig()

    rng = np.random.default_rng(cfg.seed)

    # Initial parameter set X_1.
    mu_1, Sigma_1 = _sample_random_P(rng, cfg)

    # Identify disruption boundaries within the T_periods window.
    # Disruption at period t_k = k * tau (1-indexed) for k = 1, 2, ...
    # In 0-indexed array terms, disruption-period index is k*tau - 1.
    # We include the initial period (index 0) and any subsequent k*tau
    # that falls within [1, T_periods].
    T = cfg.T_periods
    tau = cfg.tau
    eta = cfg.eta

    # boundaries (1-indexed): [1, tau, 2*tau, 3*tau, ...] capped at T.
    # We always pin t=1 as the initial parameter; the first true
    # disruption per Eq 7.6 is at t_0 = tau.
    boundaries: list[int] = [1]
    k = 1
    while k * tau <= T:
        boundaries.append(k * tau)
        k += 1
    # Append a sentinel beyond T so we have a target for the final
    # segment (linear interpolation needs both endpoints).
    if boundaries[-1] < T:
        boundaries.append(boundaries[-1] + tau)

    # Build boundary parameters X_{t_k} per Eq 7.6 / 7.8.
    # boundary_params[k] = (mu, Sigma) at boundaries[k].
    boundary_params: list[tuple[np.ndarray, np.ndarray]] = [(mu_1, Sigma_1)]
    for _ in range(1, len(boundaries)):
        mu_prev, Sigma_prev = boundary_params[-1]
        mu_P, Sigma_P = _sample_random_P(rng, cfg)
        mu_new = (1.0 - eta) * mu_prev + eta * mu_P
        Sigma_new = (1.0 - eta) * Sigma_prev + eta * Sigma_P
        boundary_params.append((mu_new, Sigma_new))

    # Build the per-period stream via intra-segment linear
    # interpolation (Eq 7.7 for the first segment, Eq 7.9 for
    # subsequent segments).
    mus: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    for t in range(1, T + 1):
        # Find the segment containing t: boundaries[seg_idx] <= t
        # < boundaries[seg_idx + 1].
        seg_idx = 0
        for j in range(len(boundaries) - 1):
            if boundaries[j] <= t < boundaries[j + 1]:
                seg_idx = j
                break
        else:
            # t == boundaries[-1]; clamp to the last interior
            # segment so we land exactly on the right endpoint.
            if t >= boundaries[-1]:
                seg_idx = len(boundaries) - 2

        t_lo = boundaries[seg_idx]
        t_hi = boundaries[seg_idx + 1]
        mu_lo, Sigma_lo = boundary_params[seg_idx]
        mu_hi, Sigma_hi = boundary_params[seg_idx + 1]

        # alpha = (t - t_lo) / (t_hi - t_lo). At t == t_lo we're
        # exactly at the segment start (alpha = 0); at t == t_hi
        # we're exactly at the end (alpha = 1).
        denom = float(t_hi - t_lo)
        alpha = (t - t_lo) / denom if denom > 0 else 0.0

        mu_t = (1.0 - alpha) * mu_lo + alpha * mu_hi
        Sigma_t = (1.0 - alpha) * Sigma_lo + alpha * Sigma_hi
        mus.append(mu_t)
        sigmas.append(Sigma_t)

    return mus, sigmas


def generate_synthetic_po_returns(
    cfg: Optional[POGeneratorConfig] = None,
    start_date: str = "2007-01-01",
) -> pd.DataFrame:
    """Generate a synthetic daily-returns DataFrame matching PO(tau, eta).

    Per the paper's experimental methodology (page 8, Step 1):
    at each period ``t`` we sample ``days_per_period`` IID daily
    return vectors from ``N(mu_t, Sigma_t)``.

    Parameters
    ----------
    cfg : POGeneratorConfig, optional
        Configuration. Defaults to PO(8, 1.0) with d=30, T=25,
        50 BD/period (1250 days total).
    start_date : str, default "2007-01-01"
        Calendar start date for the synthetic business-day index.
        Roughly aligns with the paper's real-world window
        (2006/11/20 - 2012/12/31).

    Returns
    -------
    DataFrame
        Shape ``(T_periods * days_per_period, d)``. Index is a
        business-day ``DatetimeIndex``. Columns are
        ``asset_00, asset_01, ..., asset_{d-1:02d}``. Values are
        simulated daily returns suitable for direct consumption by
        ``walk_forward_report.py``-style loaders.
    """
    if cfg is None:
        cfg = POGeneratorConfig()

    mus, sigmas = generate_po_parameter_stream(cfg)

    # Sample daily returns per period.
    rng_daily = np.random.default_rng(cfg.seed + 1)  # decoupled stream
    daily_blocks: list[np.ndarray] = []
    for t in range(cfg.T_periods):
        mu_t = mus[t]
        Sigma_t = sigmas[t]
        # multivariate_normal handles non-PSD by issuing a warning;
        # our Gram-matrix construction yields PSD, but we still
        # tolerate tiny floating-point drift.
        block = rng_daily.multivariate_normal(
            mean=mu_t, cov=Sigma_t, size=cfg.days_per_period
        )
        daily_blocks.append(block)

    returns = np.vstack(daily_blocks)  # shape (T*days_per_period, d)

    total_days = cfg.T_periods * cfg.days_per_period
    dates = pd.bdate_range(start=start_date, periods=total_days)
    columns = [f"asset_{i:02d}" for i in range(cfg.d)]
    df = pd.DataFrame(returns, index=dates, columns=columns)
    df.index.name = "Date"
    return df


def returns_to_close_prices(
    returns: pd.DataFrame, initial_price: float = 100.0
) -> pd.DataFrame:
    """Convert a daily-returns frame to a synthetic close-price frame.

    The DataLoader in ``python_refactor.src.experiments.data_loader``
    expects per-asset CSVs with ``Date,Close`` columns and computes
    returns internally via ``pct_change()``. This helper inverts that:
    ``close[t] = close[t-1] * (1 + return[t])`` starting from
    ``initial_price``.

    Parameters
    ----------
    returns : DataFrame
        Daily returns (shape ``(T, d)``).
    initial_price : float, default 100.0
        Anchor price applied to every asset at ``t = 0``.

    Returns
    -------
    DataFrame
        Synthetic close prices with the same index/columns as
        ``returns``. The first row is ``initial_price`` (i.e. the
        return at day 0 is implicitly 0 after the loader's
        ``pct_change()`` discards the leading NaN).
    """
    # cumprod(1 + r) gives the price-relative path.
    growth = (1.0 + returns).cumprod(axis=0)
    # Anchor so that the first row equals initial_price; that way
    # pct_change() applied downstream by the DataLoader recovers
    # the input ``returns`` exactly from row 1 onward.
    # We prepend a row of initial_price values via reindexing.
    prices_path = initial_price * growth
    # Shift so that the first observation acts as the anchor and
    # subsequent rows reflect the day's return.
    anchor = pd.DataFrame(
        np.full((1, returns.shape[1]), initial_price),
        index=[returns.index[0] - pd.Timedelta(days=1)],
        columns=returns.columns,
    )
    full = pd.concat([anchor, prices_path], axis=0)
    full.index.name = "Date"
    return full
