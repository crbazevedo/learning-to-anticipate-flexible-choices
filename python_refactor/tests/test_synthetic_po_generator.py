"""
Tests for the synthetic PO(tau, eta) benchmark generator.

Validates the implementation of thesis Eqs 7.6-7.9 / paper Eqs 31-32
in :mod:`python_refactor.src.data.synthetic_po_generator`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src.data.* importable regardless of how pytest is invoked.
_REPO_PY = Path(__file__).resolve().parents[1]
if str(_REPO_PY) not in sys.path:
    sys.path.insert(0, str(_REPO_PY))

from src.data.synthetic_po_generator import (  # noqa: E402
    POGeneratorConfig,
    _sample_random_P,
    generate_po_parameter_stream,
    generate_synthetic_po_returns,
    returns_to_close_prices,
)

# ---------------------------------------------------------------------------
# Shape and basic invariants
# ---------------------------------------------------------------------------

def test_po_8_1_0_default_shape_is_1250_by_30() -> None:
    """PO(tau=8, eta=1.0, d=30, T=25, 50 BD/period) -> (1250, 30).

    Receipt for the task spec: "PO(tau=8, eta=1.0, d=30) produces
    correct shape (1250, 30)."
    """
    cfg = POGeneratorConfig()  # defaults reproduce PO(8, 1.0)
    assert cfg.tau == 8
    assert cfg.eta == 1.0
    assert cfg.d == 30
    assert cfg.T_periods == 25
    assert cfg.days_per_period == 50

    df = generate_synthetic_po_returns(cfg)
    assert df.shape == (1250, 30)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == [f"asset_{i:02d}" for i in range(30)]


def test_param_stream_length_matches_T_periods() -> None:
    """One (mu, Sigma) pair per investment period."""
    cfg = POGeneratorConfig(T_periods=25)
    mus, sigmas = generate_po_parameter_stream(cfg)
    assert len(mus) == 25
    assert len(sigmas) == 25
    for mu_t in mus:
        assert mu_t.shape == (cfg.d,)
    for Sigma_t in sigmas:
        assert Sigma_t.shape == (cfg.d, cfg.d)


def test_sigma_is_symmetric_and_psd() -> None:
    """Gram-matrix construction must yield symmetric PSD covariance.

    Pins thesis page 143: "we use a Gram Matrix-based method ...
    Sigma(i, j) = rho(i, j) * sigma_i * sigma_j".
    """
    cfg = POGeneratorConfig()
    rng = np.random.default_rng(0)
    _, Sigma = _sample_random_P(rng, cfg)
    # Symmetric.
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)
    # PSD: smallest eigenvalue >= 0 (allow tiny floating-point noise).
    min_eig = np.linalg.eigvalsh(Sigma).min()
    assert min_eig >= -1e-10, (
        f"Sigma should be PSD; got min eigenvalue {min_eig}"
    )


# ---------------------------------------------------------------------------
# Disruption dynamics
# ---------------------------------------------------------------------------

def test_regime_shifts_happen_every_tau_periods() -> None:
    """Boundary jumps occur exactly at periods k*tau (1-indexed).

    For PO(8, 1.0): boundaries at periods {1, 8, 16, 24} (k*tau for
    k = 0..3, capped at T=25). The interpolation segment between
    boundaries is linear; the period-to-period delta is approximately
    constant inside a segment, then changes direction/magnitude when
    crossing a boundary.

    This pins paper Eq 31 (smooth dynamics) + Eq 32 (boundary jump).
    """
    cfg = POGeneratorConfig(tau=8, eta=1.0, T_periods=25, seed=123)
    mus, _ = generate_po_parameter_stream(cfg)

    # Per-period deltas in the mean vector.
    deltas = [mus[t] - mus[t - 1] for t in range(1, len(mus))]

    # Inside a segment, the delta is constant (linear interp).
    # boundaries (1-indexed): 1, 8, 16, 24, with a synthetic "next"
    # at 32 for the final tail segment.
    # Segment 1 covers periods 1..8 -> delta indices 0..6 (between
    # periods 1->2, 2->3, ..., 7->8). All must be (approximately)
    # equal.
    for seg_start, seg_end in [(1, 8), (8, 16), (16, 24)]:
        # delta between period t and t-1 has index (t-2) in `deltas`
        # since deltas[0] = mus[1] - mus[0] = period 2 - period 1.
        idx_start = seg_start - 1  # delta from seg_start -> seg_start+1
        idx_end = seg_end - 1  # delta from seg_end-1 -> seg_end
        seg_deltas = deltas[idx_start:idx_end]
        if len(seg_deltas) < 2:
            continue
        first = seg_deltas[0]
        for d in seg_deltas[1:]:
            np.testing.assert_allclose(
                d, first, atol=1e-10,
                err_msg=(
                    f"Within-segment delta drift detected for segment "
                    f"({seg_start}, {seg_end}); linear interpolation "
                    f"should give constant per-period deltas."
                ),
            )

    # Cross-boundary deltas should differ from the in-segment delta
    # (the random P at the new boundary changes direction). Compare
    # the in-segment delta of segment 1 (between p1 and p2) against
    # the cross-boundary delta from segment 1 to segment 2 (p8 -> p9).
    in_seg1 = deltas[0]  # p1 -> p2
    cross_boundary = deltas[7]  # p8 -> p9
    # Direction or magnitude should change at the boundary; assert
    # they are not numerically equal.
    assert not np.allclose(in_seg1, cross_boundary, atol=1e-6), (
        "Cross-boundary delta should differ from in-segment delta; "
        "otherwise Eq 32 disruption is not firing."
    )


def test_eta_1_0_replaces_boundary_with_new_P() -> None:
    """Severity eta=1.0 means new boundary X_{t_k} == P_{t_k}.

    Per Eq 32 (paper) / Eq 7.8 (thesis):
        X_{t_k} = (1 - eta) * X_{t_{k-1}} + eta * P_{t_k}
    With eta = 1.0, X_{t_k} = P_{t_k} (the previous state is
    fully discarded at each boundary). This pins the severity
    semantics.
    """
    cfg = POGeneratorConfig(tau=4, eta=1.0, T_periods=12, seed=7)
    mus, _ = generate_po_parameter_stream(cfg)

    # Reconstruct what _sample_random_P would yield with the same
    # seed: 1 call for X_1, then 1 call per additional boundary
    # (k = 1, 2, ...). For T=12 and tau=4, boundaries (1-indexed)
    # are [1, 4, 8, 12] + sentinel [16] -> 5 boundary slots ->
    # 5 calls to _sample_random_P.
    rng = np.random.default_rng(cfg.seed)
    expected_boundary_mus: list[np.ndarray] = []
    for _ in range(5):  # 1 initial + 4 subsequent boundaries
        mu_P, _ = _sample_random_P(rng, cfg)
        expected_boundary_mus.append(mu_P)

    # At eta=1.0, X_{t_k} equals the freshly-sampled P_{t_k}
    # for every k >= 1. Verify boundaries at periods 4, 8, 12.
    # boundaries in code: [1, 4, 8, 12, 16].
    for boundary_period, expected_mu in zip(
        [1, 4, 8, 12], expected_boundary_mus[:4]
    ):
        # Period boundary_period is index (boundary_period - 1).
        actual_mu = mus[boundary_period - 1]
        np.testing.assert_allclose(
            actual_mu, expected_mu, atol=1e-12,
            err_msg=(
                f"At eta=1.0, X at boundary period {boundary_period} "
                f"should equal freshly sampled P (Eq 32 with eta=1)."
            ),
        )


def test_eta_0_5_blends_old_and_new() -> None:
    """Severity eta=0.5 produces midpoint blend (1-eta)*X_old + eta*P.

    Cross-checks the linear blend semantics with a non-extremal eta.
    """
    cfg = POGeneratorConfig(tau=4, eta=0.5, T_periods=8, seed=99)
    mus, sigmas = generate_po_parameter_stream(cfg)

    # Replay the RNG: 1 call for X_1, 1 for boundary at t=4, 1 for
    # boundary at t=8, 1 for sentinel beyond T (4 calls total).
    rng = np.random.default_rng(cfg.seed)
    mu_1, Sigma_1 = _sample_random_P(rng, cfg)
    mu_P_4, Sigma_P_4 = _sample_random_P(rng, cfg)

    # Expected blend at boundary t=4: 0.5 * X_1 + 0.5 * P_4.
    expected_mu_4 = 0.5 * mu_1 + 0.5 * mu_P_4
    expected_Sigma_4 = 0.5 * Sigma_1 + 0.5 * Sigma_P_4

    np.testing.assert_allclose(mus[3], expected_mu_4, atol=1e-12)
    np.testing.assert_allclose(sigmas[3], expected_Sigma_4, atol=1e-12)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_yields_bit_identical_output() -> None:
    """Same seed -> bit-identical DataFrame.

    Required by the task spec: "Reproducibility is REQUIRED — same
    seed must produce bit-identical CSVs."
    """
    cfg = POGeneratorConfig(seed=42)
    df1 = generate_synthetic_po_returns(cfg)
    df2 = generate_synthetic_po_returns(cfg)
    pd.testing.assert_frame_equal(df1, df2)
    # And every value bit-identical.
    np.testing.assert_array_equal(df1.values, df2.values)


def test_different_seed_yields_different_output() -> None:
    """Sanity check that the seed actually drives randomness."""
    df1 = generate_synthetic_po_returns(POGeneratorConfig(seed=42))
    df2 = generate_synthetic_po_returns(POGeneratorConfig(seed=43))
    assert not np.array_equal(df1.values, df2.values), (
        "Different seeds must produce different sample paths."
    )


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------

def test_returns_to_close_prices_roundtrip() -> None:
    """Returns -> prices -> pct_change should recover the original returns.

    Pins the loader-compatibility contract: DataLoader applies
    ``pct_change()`` to ``Close``; our CSVs must therefore encode
    prices whose pct_change equals the simulated returns.
    """
    cfg = POGeneratorConfig(d=5, T_periods=4, days_per_period=10, seed=11)
    returns = generate_synthetic_po_returns(cfg)
    prices = returns_to_close_prices(returns, initial_price=100.0)

    # The first row of `prices` is the anchor (one bday before
    # returns.index[0]); the next rows correspond to returns.index.
    recovered = prices["asset_00"].pct_change().dropna()
    # Drop the first NaN; the remaining should match returns row-by-row.
    np.testing.assert_allclose(
        recovered.to_numpy(),
        returns["asset_00"].to_numpy(),
        atol=1e-12,
    )
