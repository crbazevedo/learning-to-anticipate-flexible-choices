"""W22-NC39 — anticipatory cardinality constraint handler.

Operator directive 2026-05-20: "anticipatory constraint handling, anticipatory
operators."

Background: portfolio optimization typically enforces a CARDINALITY constraint
(K = max number of assets with non-zero weight). Standard implementations
project after mutation by keeping the top-K weights and zeroing the rest —
this is a NAIVE projection: it picks based on CURRENT weight, ignoring future
expected contribution.

NC39 implements an ANTICIPATORY projection: at the moment of cardinality
enforcement, score each asset by its predicted contribution to future ROI
(per-asset KF or simple statistical estimator) and pick the top-K by that
ANTICIPATED metric, not the current weight.

Math:
- For each non-zero asset i in the post-mutation weight vector w (length d):
  - Score s_i = w_i · (1 + α · predicted_growth_i)
    where predicted_growth_i is the forecast (e.g., from a simple AR(1) per
    asset, or from a market model)
  - α is the anticipation strength (0.0 = pure top-K by weight; >0 = tilt
    toward anticipated growth)
- Keep top-K assets by s_i; zero the rest; renormalize to simplex

This couples cardinality enforcement with anticipation, addressing the
operator's "anticipatory constraint handling" directive.

Standalone analyzer + handler module. NO modifications to shared code.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def naive_top_k_projection(weights: np.ndarray, cardinality: int) -> np.ndarray:
    """Standard naive top-K projection: keep K largest weights, zero rest,
    renormalize to simplex. Reference baseline for comparison."""
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    cardinality = max(1, min(n, cardinality))
    if cardinality >= n:
        s = np.sum(w)
        return w / s if s > 0 else np.ones(n) / n
    idx = np.argsort(w)[::-1][:cardinality]  # top-K by current weight
    out = np.zeros(n)
    out[idx] = w[idx]
    s = np.sum(out)
    return out / s if s > 0 else _uniform_over_idx(n, idx)


def _uniform_over_idx(n: int, idx: np.ndarray) -> np.ndarray:
    """Fallback: uniform over the kept indices."""
    out = np.zeros(n)
    out[idx] = 1.0 / len(idx)
    return out


def anticipatory_top_k_projection(
    weights: np.ndarray,
    cardinality: int,
    predicted_growth: Optional[np.ndarray] = None,
    anticipation_strength: float = 0.5,
) -> np.ndarray:
    """Anticipatory cardinality projection.

    Args:
        weights: post-mutation weight vector (length d), need not be on simplex
        cardinality: max number of non-zero assets to keep (1 ≤ K ≤ d)
        predicted_growth: per-asset predicted next-period return (length d).
            If None, falls back to naive top-K.
        anticipation_strength: α ∈ [0, ∞). 0 = naive top-K by weight; larger
            = stronger tilt toward predicted-growth assets.

    Returns:
        Projected weights (simplex, exactly `cardinality` non-zero entries
        unless input weights have fewer non-zeros).

    Behavior:
        - α=0: equivalent to naive_top_k_projection (only `weights` matters)
        - α > 0: scoring becomes s_i = w_i · (1 + α · predicted_growth_i),
          biasing the top-K choice toward high-growth assets
        - Negative α: tilt AWAY from high-growth (defensive selection)
    """
    w = np.asarray(weights, dtype=float).copy()
    n = len(w)
    cardinality = max(1, min(n, cardinality))
    if predicted_growth is None or anticipation_strength == 0.0:
        return naive_top_k_projection(w, cardinality)
    g = np.asarray(predicted_growth, dtype=float)
    if len(g) != n:
        raise ValueError(
            f"predicted_growth length {len(g)} != weights length {n}"
        )
    # Anticipatory score
    scores = w * (1.0 + anticipation_strength * g)
    # Edge: if all scores negative (rare), use naive fallback
    if np.all(scores <= 0):
        return naive_top_k_projection(w, cardinality)
    # Top-K by score
    if cardinality >= n:
        s = np.sum(np.maximum(w, 0.0))
        return np.maximum(w, 0.0) / s if s > 0 else np.ones(n) / n
    idx = np.argsort(scores)[::-1][:cardinality]
    out = np.zeros(n)
    out[idx] = np.maximum(w[idx], 0.0)  # keep non-negative
    s = np.sum(out)
    return out / s if s > 0 else _uniform_over_idx(n, idx)


def compare_projections(
    weights: np.ndarray,
    cardinality: int,
    predicted_growth: np.ndarray,
    anticipation_strength: float = 0.5,
) -> dict:
    """Compare naive vs anticipatory projections.

    Returns dict with:
      - naive_weights, anticipatory_weights
      - naive_kept_indices, anticipatory_kept_indices
      - jaccard_kept: overlap of kept-asset sets
      - anticipatory_growth_weighted_score: sum(out · predicted_growth)
      - naive_growth_weighted_score
      - signed_growth_uplift: anticipatory_growth_score - naive_growth_score
    """
    naive = naive_top_k_projection(weights, cardinality)
    antic = anticipatory_top_k_projection(
        weights, cardinality, predicted_growth, anticipation_strength,
    )
    naive_idx = set(np.where(naive > 1e-12)[0])
    antic_idx = set(np.where(antic > 1e-12)[0])
    jaccard = (len(naive_idx & antic_idx) / len(naive_idx | antic_idx)) if (naive_idx | antic_idx) else 1.0
    naive_growth_score = float(np.dot(naive, predicted_growth))
    antic_growth_score = float(np.dot(antic, predicted_growth))
    return {
        "naive_weights": naive,
        "anticipatory_weights": antic,
        "naive_kept_indices": naive_idx,
        "anticipatory_kept_indices": antic_idx,
        "jaccard_kept": jaccard,
        "naive_growth_weighted_score": naive_growth_score,
        "anticipatory_growth_weighted_score": antic_growth_score,
        "signed_growth_uplift": antic_growth_score - naive_growth_score,
    }


def sweep_anticipation_strength(
    weights: np.ndarray,
    cardinality: int,
    predicted_growth: np.ndarray,
    strengths: Optional[list[float]] = None,
) -> dict:
    """For each α in strengths, compute the kept-asset set + growth score.

    Returns dict {α: comparison_dict}.
    """
    if strengths is None:
        strengths = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    out = {}
    for a in strengths:
        out[a] = compare_projections(weights, cardinality, predicted_growth, a)
    return out
