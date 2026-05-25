"""Probe S-v1 — asset network eigenvector centrality.

Implements the S-v1 (hub-tilt) variant from
``docs/W22-ALT-SIGNAL-PROBES.md``.

Scope (contained):
    - Build a correlation network from a T x K returns matrix.
    - Compute eigenvector and (approximate) betweenness centralities,
      pure NumPy / SciPy.csgraph only — NO ``networkx``.
    - Quantify hub concentration via the Gini coefficient.
    - Provide a portfolio weight tilt ``w_k <- w_k * (1 + gamma * cent_k)``
      followed by a simplex renormalization.

The module exposes pure functions only. Wiring into ASMOO / the portfolio
optimizer is left to a separate experiment script (see
``docs/W22-PROBE-S-V1-CENTRALITY.md`` § integration sketch). No imports
from ``sms_emoa.py`` or any other shared code path.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Correlation network construction
# ---------------------------------------------------------------------------


def build_correlation_network(
    returns_matrix: np.ndarray,
    top_k_edges_per_node: int | None = None,
) -> np.ndarray:
    """Build a K x K weighted adjacency matrix from a T x K returns matrix.

    Parameters
    ----------
    returns_matrix
        Array of shape ``(T, K)`` with one column per asset and one row
        per period. Must have at least 2 rows (correlation undefined for
        single-period data).
    top_k_edges_per_node
        Optional sparsification — keep only the ``top_k_edges_per_node``
        strongest absolute-correlation edges per node, in addition to the
        diagonal. ``None`` keeps the dense network. Sparsification is
        symmetrized: an edge survives if it is in the top-K of EITHER
        endpoint (logical OR).

    Returns
    -------
    np.ndarray
        Symmetric ``(K, K)`` adjacency matrix. Diagonal is always 1.
        Off-diagonals are Pearson correlations of the corresponding
        columns. NaN correlations (zero-variance columns) are coerced
        to 0.
    """
    if returns_matrix.ndim != 2:
        raise ValueError(
            f"returns_matrix must be 2-D (T x K), got shape {returns_matrix.shape}"
        )
    n_periods, n_assets = returns_matrix.shape
    if n_periods < 2:
        raise ValueError(
            f"returns_matrix needs at least 2 rows, got {n_periods}"
        )

    # np.corrcoef expects variables-as-rows; transpose so rows = assets.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(returns_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Force exact symmetry (numpy guarantees this, but be defensive).
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)

    if top_k_edges_per_node is None:
        return corr

    if top_k_edges_per_node < 0:
        raise ValueError(
            f"top_k_edges_per_node must be >= 0, got {top_k_edges_per_node}"
        )
    if top_k_edges_per_node >= n_assets - 1:
        # K-1 or larger -> retains every off-diagonal edge anyway.
        return corr

    # Per-node mask of top-K strongest off-diagonal edges by |corr|.
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, -np.inf)  # exclude self
    # argpartition is O(n) per row vs O(n log n) full sort.
    keep_idx = np.argpartition(-abs_corr, kth=top_k_edges_per_node - 1, axis=1)[
        :, :top_k_edges_per_node
    ]
    keep_mask = np.zeros_like(corr, dtype=bool)
    rows = np.repeat(np.arange(n_assets), top_k_edges_per_node)
    keep_mask[rows, keep_idx.ravel()] = True
    # Symmetrize: edge survives if either endpoint keeps it.
    keep_mask = keep_mask | keep_mask.T
    np.fill_diagonal(keep_mask, True)

    sparsified = np.where(keep_mask, corr, 0.0)
    return sparsified


# ---------------------------------------------------------------------------
# Eigenvector centrality (power iteration)
# ---------------------------------------------------------------------------


def eigenvector_centrality(
    adj: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Power-iteration eigenvector centrality, pure NumPy.

    For unsigned interpretation, we take the absolute value of the
    adjacency matrix (matching the conventional "magnitude of co-
    movement" reading for correlation networks). The dominant
    eigenvector is non-negative by Perron-Frobenius for non-negative
    irreducible matrices.

    Parameters
    ----------
    adj
        Square ``(K, K)`` adjacency matrix. May be signed; magnitude is
        used internally.
    max_iter
        Maximum power-iteration steps. Convergence is typically reached
        in 20-50 iterations for well-conditioned correlation networks.
    tol
        L1 convergence tolerance on successive iterates.

    Returns
    -------
    np.ndarray
        Length-K vector summing to 1 (L1-normalized) with entries in
        ``[0, 1]``. If the graph is degenerate (all-zero magnitudes),
        returns the uniform distribution.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"adj must be square 2-D, got shape {adj.shape}"
        )
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0)

    mag = np.abs(adj.astype(float))

    if not np.any(mag > 0):
        return np.full(n, 1.0 / n)

    # Initialize with uniform vector. Any positive vector works under
    # Perron-Frobenius; uniform is reproducible and avoids RNG state.
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        x_new = mag @ x
        norm = np.linalg.norm(x_new, ord=2)
        if norm == 0:
            return np.full(n, 1.0 / n)
        x_new = x_new / norm
        if np.linalg.norm(x_new - x, ord=1) < tol:
            x = x_new
            break
        x = x_new

    # L1 normalize so the centralities sum to 1 (probability-of-hubness
    # interpretation; makes the Gini computation directly meaningful).
    total = x.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return x / total


# ---------------------------------------------------------------------------
# Betweenness centrality (BFS-based, scipy.csgraph fast path)
# ---------------------------------------------------------------------------


def betweenness_centrality(adj: np.ndarray) -> np.ndarray:
    """Approximate betweenness centrality via shortest-path counting.

    Uses ``scipy.sparse.csgraph.shortest_path`` when available (fast
    Floyd-Warshall / Dijkstra C path). Edge weights are converted to
    "distances" via ``d_ij = 1 - |corr_ij|`` so that strong correlations
    are short paths. Self-loops are removed.

    A pure-NumPy fallback (BFS on the unweighted thresholded graph at
    ``|corr| > 0``) is used if SciPy is not importable.

    Parameters
    ----------
    adj
        Symmetric ``(K, K)`` adjacency matrix.

    Returns
    -------
    np.ndarray
        Length-K vector of betweenness scores in ``[0, 1]``,
        normalized by ``(K-1)(K-2)/2`` (the maximum possible number of
        unordered source-target pairs).
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"adj must be square 2-D, got shape {adj.shape}"
        )
    n = adj.shape[0]
    if n < 3:
        # Need at least 3 nodes for any node to lie on a path between
        # two others.
        return np.zeros(n)

    mag = np.abs(adj.astype(float))
    np.fill_diagonal(mag, 0.0)

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
    except ImportError:  # pragma: no cover - scipy is a project dep
        return _betweenness_unweighted_bfs(mag > 0)

    # Convert correlation magnitudes to distances. Zero magnitude -> no
    # edge (sparse zero -> infinity distance).
    with np.errstate(invalid="ignore"):
        dist_dense = np.where(mag > 0, 1.0 - mag, 0.0)
    # Clip tiny negative distances from floating-point error.
    dist_dense = np.clip(dist_dense, 0.0, None)
    # csr_matrix treats explicit zeros as absent edges, which is what
    # we want for nodes with zero correlation. Use a small epsilon for
    # zero-distance edges (perfect correlation, |c|=1) so they remain
    # represented.
    eps = 1e-12
    dist_for_graph = np.where(mag > 0, np.maximum(dist_dense, eps), 0.0)
    graph = csr_matrix(dist_for_graph)
    dist_matrix, predecessors = shortest_path(
        graph, directed=False, return_predecessors=True
    )

    betweenness = np.zeros(n)
    for s in range(n):
        for t in range(s + 1, n):
            if not np.isfinite(dist_matrix[s, t]):
                continue
            # Walk predecessor chain t -> ... -> s, counting interior
            # nodes (excluding the two endpoints).
            node = predecessors[s, t]
            while node != s and node >= 0:
                betweenness[node] += 1.0
                node = predecessors[s, node]

    max_pairs = (n - 1) * (n - 2) / 2.0
    if max_pairs > 0:
        betweenness = betweenness / max_pairs
    return betweenness


def _betweenness_unweighted_bfs(adjacency_bool: np.ndarray) -> np.ndarray:
    """Pure-NumPy BFS fallback for the unweighted graph."""
    from collections import deque

    n = adjacency_bool.shape[0]
    betweenness = np.zeros(n)
    neighbors = [np.flatnonzero(adjacency_bool[i]) for i in range(n)]

    for s in range(n):
        # Brandes (2001) single-source shortest-path counting.
        stack: list[int] = []
        pred: list[list[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n)
        sigma[s] = 1.0
        dist = np.full(n, -1)
        dist[s] = 0
        queue: deque[int] = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in neighbors[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(n)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Brandes counts each pair twice in undirected graph; divide by 2.
    betweenness = betweenness / 2.0
    max_pairs = (n - 1) * (n - 2) / 2.0
    if max_pairs > 0:
        betweenness = betweenness / max_pairs
    return betweenness


# ---------------------------------------------------------------------------
# Gini coefficient (concentration of centrality mass)
# ---------------------------------------------------------------------------


def centrality_gini(centrality: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution.

    Returns 0 for a perfectly uniform distribution and approaches 1 as
    the distribution becomes maximally concentrated on a single
    element.

    Parameters
    ----------
    centrality
        Length-K non-negative array.

    Returns
    -------
    float
        Gini coefficient in ``[0, 1)``.
    """
    x = np.asarray(centrality, dtype=float).ravel()
    if x.size == 0:
        return 0.0
    if np.any(x < 0):
        raise ValueError("Gini is undefined for negative values")
    total = x.sum()
    if total <= 0:
        return 0.0
    sorted_x = np.sort(x)
    n = sorted_x.size
    # Sen / Brown formula:
    # G = (2 * sum_i i * x_i) / (n * sum x) - (n + 1) / n
    indices = np.arange(1, n + 1)
    gini = (2.0 * np.sum(indices * sorted_x)) / (n * total) - (n + 1.0) / n
    # Clip tiny floating-point excursions outside [0, 1).
    return float(np.clip(gini, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Hub-tilt rule (Probe S-v1 weight perturbation)
# ---------------------------------------------------------------------------


def hub_tilt_weights(
    base_weights: np.ndarray,
    centrality: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Apply the S-v1 hub-tilt and renormalize back to the simplex.

    For each asset k:
        w_k <- max(0, w_k * (1 + gamma * cent_k))
    then divide by the sum so that the output lies on the K-1 simplex.

    Parameters
    ----------
    base_weights
        Length-K non-negative weights (need not be exactly normalized;
        we renormalize the result regardless).
    centrality
        Length-K centrality scores. Should be non-negative for
        gamma > 0; signed centralities are tolerated but will be passed
        through the clip below.
    gamma
        Tilt strength. ``gamma = 0`` is identity (after renormalization).
        Positive gamma favors hubs (S-H2). Negative gamma favors anti-
        hubs (S-H3). For very negative gamma, some entries can hit the
        clip floor.

    Returns
    -------
    np.ndarray
        Length-K vector summing to 1 with non-negative entries. If the
        tilt zeros every weight (pathological combination of inputs),
        returns the uniform distribution.
    """
    w = np.asarray(base_weights, dtype=float).ravel()
    c = np.asarray(centrality, dtype=float).ravel()
    if w.shape != c.shape:
        raise ValueError(
            f"base_weights {w.shape} and centrality {c.shape} must match"
        )
    if w.size == 0:
        return np.zeros(0)
    if np.any(w < 0):
        raise ValueError("base_weights must be non-negative")

    tilted = w * (1.0 + gamma * c)
    # Clip strictly to non-negative — under negative gamma, the
    # multiplier can drive a weight below zero, which is meaningless on
    # the simplex. Set the floor at 0 rather than a small epsilon so
    # that the math identity at gamma = 0 is bit-exact.
    tilted = np.clip(tilted, 0.0, None)

    total = tilted.sum()
    if total <= 0:
        return np.full(w.size, 1.0 / w.size)
    return tilted / total
