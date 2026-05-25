"""W22 Probe AD: stochastic vs deterministic Δ_S accuracy comparison.

Standalone module — does NOT touch shared code (``sms_emoa.py``,
``anticipatory_learning.py``, ``kalman_filter.py``). The four functions
in this module replicate (independently) the SMS-EMOA per-solution
hypervolume contribution formulas so we can quantify — on synthetic
Pareto fronts with KNOWN per-solution covariances — how much the
stochastic (Eq 6.41) version differs from the deterministic closed
form, and which is closer to a Monte-Carlo (MC) ground truth.

Hypothesis (H-Probe-AD)
-----------------------
When per-solution within-objective covariance ``Cov(ROI, risk) != 0``,
the stochastic Δ_S (Eq 6.41) is closer to a CRN Monte-Carlo estimate
of E[Δ_S] than the deterministic closed form is. When ``Cov = 0`` the
two are mathematically equivalent (the - Cov correction vanishes) so
the stochastic version is only meaningfully different in the
correlated regime.

Formulas implemented
--------------------
DETERMINISTIC (replicating ``sms_emoa.py::_compute_hypervolume_contributions_class``
WITHOUT importing it — pure numpy):

    - sort by ROI ascending
    - position 0 (leftmost):  Δ_S = (ROI - R1) * (R2 - risk)
    - middle position i:      Δ_S = (ROI_i - ROI_{i+1}) * (risk_{i-1} - risk_i)
    - last position:          Δ_S = (ROI_i - ROI_{i-1}) * (R2 - risk_i)

STOCHASTIC (Eq 6.41 self-cov correction only — the only nonzero Cov
term in the Python refactor per the thesis Eq 6.41 expansion):

    Δ_S^stoch = Δ_S^det - Cov(ROI, risk)_self

per the (+,-,-,+) sign pattern on (Cov(a,c), Cov(a,d), Cov(b,c),
Cov(b,d)), where in the Python refactor only Cov(a,d) (the
within-solution self term) is nonzero because per-solution MC sample
banks are not stored.

MC GROUND TRUTH (CRN — common random numbers shared across solutions
to reduce variance of the comparison):

    for trial j in 1..n_mc:
        sample (roi_i, risk_i) ~ N(mus[i], covs[i]) for each i
        compute deterministic Δ_S on the SAMPLED front
        average over trials

Citations
---------
- Thesis Theorem 6.3.1, Eq 6.41, p.131
- Thesis Eq 6.36 (rectangle definition), p.130
- Thesis Eq 6.39 (E[xy] = E[x]E[y] + Cov(x,y)), p.130
- ``python_refactor/src/algorithms/sms_emoa.py`` lines 698-878
  (the implementations this module replicates without importing)
"""

from __future__ import annotations

import numpy as np


def deterministic_delta_s(
    rois: np.ndarray,
    risks: np.ndarray,
    R1: float,
    R2: float,
) -> np.ndarray:
    """Per-solution deterministic Δ_S (closed form).

    Replicates ``sms_emoa.py::_compute_hypervolume_contributions_class``
    middle/edge branches.  ``rois`` and ``risks`` are NOT assumed
    pre-sorted; this function sorts internally and returns Δ_S in the
    SAME ORDER as the input arrays.

    Parameters
    ----------
    rois
        1-D length-N array of per-solution mean ROIs.
    risks
        1-D length-N array of per-solution mean risks.
    R1
        ROI lower reference (return floor).
    R2
        Risk upper reference (risk ceiling).

    Returns
    -------
    np.ndarray
        1-D length-N array of per-solution Δ_S, in the input order.
    """
    rois = np.asarray(rois, dtype=float).reshape(-1)
    risks = np.asarray(risks, dtype=float).reshape(-1)
    if rois.shape != risks.shape:
        raise ValueError(
            f"rois and risks shape mismatch: {rois.shape} vs {risks.shape}"
        )
    n = rois.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)

    if n == 1:
        return np.array([(rois[0] - R1) * (R2 - risks[0])], dtype=float)

    order = np.argsort(rois, kind="stable")
    rois_s = rois[order]
    risks_s = risks[order]

    delta_s_sorted = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            delta_s_sorted[i] = (rois_s[i] - R1) * (R2 - risks_s[i])
        elif i == n - 1:
            delta_s_sorted[i] = (rois_s[i] - rois_s[i - 1]) * (R2 - risks_s[i])
        else:
            # Middle: matches sms_emoa.py line 723.
            delta_s_sorted[i] = (rois_s[i] - rois_s[i + 1]) * (
                risks_s[i - 1] - risks_s[i]
            )

    # Restore input order.
    out = np.zeros(n, dtype=float)
    out[order] = delta_s_sorted
    return out


def stochastic_delta_s(
    rois: np.ndarray,
    risks: np.ndarray,
    covs: np.ndarray,
    R1: float,
    R2: float,
) -> np.ndarray:
    """Per-solution stochastic Δ_S (Eq 6.41) — det form minus self-Cov.

    In the Python refactor only the within-solution ``Cov(a, d)`` term
    is nonzero (cross-pair Covs require MC sample banks that aren't
    stored). So the stochastic formula reduces to::

        Δ_S^stoch[i] = Δ_S^det[i] - covs[i]

    Parameters
    ----------
    rois
        1-D length-N array of per-solution mean ROIs.
    risks
        1-D length-N array of per-solution mean risks.
    covs
        1-D length-N array of within-solution ``Cov(ROI, risk)``
        values (one scalar per solution — the KF state P[0,1]).
    R1
        ROI lower reference.
    R2
        Risk upper reference.

    Returns
    -------
    np.ndarray
        1-D length-N array of per-solution stochastic Δ_S, input order.
    """
    rois = np.asarray(rois, dtype=float).reshape(-1)
    risks = np.asarray(risks, dtype=float).reshape(-1)
    covs = np.asarray(covs, dtype=float).reshape(-1)
    if rois.shape != risks.shape or rois.shape != covs.shape:
        raise ValueError(
            "rois, risks, covs must share shape; got "
            f"{rois.shape}, {risks.shape}, {covs.shape}"
        )
    det = deterministic_delta_s(rois, risks, R1, R2)
    return det - covs


def mc_ground_truth_delta_s(
    mus: np.ndarray,
    covs: np.ndarray,
    R1: float,
    R2: float,
    n_mc: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Monte-Carlo ground truth for E[Δ_S] under bivariate Gaussian.

    For each MC trial we draw (roi_i, risk_i) from
    ``N(mus[i], covs_full_i)`` for every solution i and evaluate the
    deterministic Δ_S on the SAMPLED front. We average per-solution
    Δ_S across trials.

    To match what the Python refactor actually has access to (only the
    within-solution ``Cov(ROI, risk)`` per solution; cross-solution
    Cov is assumed zero — Eq 6.41 expansion in sms_emoa.py:760-770),
    we sample each solution INDEPENDENTLY of the others. Within each
    solution we use the FULL 2x2 covariance ``[[var_ROI, cov], [cov,
    var_risk]]``. The variance of the marginals is parameterised
    through the ``mus`` and ``covs`` arrays' shape:

    Parameters
    ----------
    mus
        Shape ``(N, 2)`` — per-solution mean vectors ``[mean_ROI,
        mean_risk]``.
    covs
        Shape ``(N, 2, 2)`` — per-solution 2x2 bivariate Gaussian
        covariance matrices. ``covs[i, 0, 1]`` and ``covs[i, 1, 0]``
        are the within-solution ``Cov(ROI, risk)`` (the only nonzero
        Cov term in Eq 6.41 expansion under the Python refactor).
    R1
        ROI lower reference (deterministic constant — same every trial).
    R2
        Risk upper reference (deterministic constant — same every trial).
    n_mc
        Number of MC trials. Default 5000.
    rng
        Optional ``np.random.Generator``. If ``None``, a fresh
        ``default_rng()`` is created.

    Returns
    -------
    np.ndarray
        1-D length-N array of per-solution E[Δ_S] estimates.

    Notes
    -----
    CRN (common random numbers) is applied at the trial level: every
    solution sees the same RNG draw sequence, so when we compare two
    methods (e.g. stoch vs det) the comparison is on the SAME MC
    estimate — the noise cancels.
    """
    mus = np.asarray(mus, dtype=float)
    covs = np.asarray(covs, dtype=float)
    if mus.ndim != 2 or mus.shape[1] != 2:
        raise ValueError(f"mus must be (N, 2); got {mus.shape}")
    n = mus.shape[0]
    if covs.shape != (n, 2, 2):
        raise ValueError(f"covs must be (N, 2, 2); got {covs.shape}")
    if n_mc <= 0:
        raise ValueError(f"n_mc must be > 0; got {n_mc}")

    if rng is None:
        rng = np.random.default_rng()

    delta_s_accum = np.zeros(n, dtype=float)
    # CRN: pre-draw all trial samples in one go for reproducibility +
    # so every method that takes this rng sees the same sequence.
    # multivariate_normal handles (N, 2) means + (N, 2, 2) covs in a
    # loop; we draw per-solution then per-trial inside the loop.
    samples = np.empty((n_mc, n, 2), dtype=float)
    for i in range(n):
        samples[:, i, :] = rng.multivariate_normal(
            mus[i], covs[i], size=n_mc
        )

    for j in range(n_mc):
        rois_j = samples[j, :, 0]
        risks_j = samples[j, :, 1]
        delta_s_accum += deterministic_delta_s(rois_j, risks_j, R1, R2)

    return delta_s_accum / n_mc


def compare_methods(
    rois: np.ndarray,
    risks: np.ndarray,
    covs: np.ndarray,
    R1: float,
    R2: float,
    n_mc: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compare deterministic + stochastic Δ_S against MC ground truth.

    Builds the (N, 2) ``mus`` and (N, 2, 2) ``covs_full`` arrays from
    the scalar ``rois``, ``risks``, ``covs`` inputs, assuming the
    marginal variances come from ``covs`` (Cov(ROI, risk)) plus
    user-implied unit-variance for the marginals (Var(ROI) = Var(risk)
    = max(|Cov(ROI, risk)|, 1e-6) so the resulting 2x2 is positive
    semi-definite). Tests that need a specific marginal variance
    pattern should call ``mc_ground_truth_delta_s`` directly with a
    hand-built ``covs_full``.

    Parameters
    ----------
    rois
        Length-N per-solution mean ROIs.
    risks
        Length-N per-solution mean risks.
    covs
        Length-N per-solution Cov(ROI, risk) (the off-diagonal).
    R1, R2
        ROI floor and risk ceiling.
    n_mc
        Number of MC trials. Default 5000.
    rng
        Optional RNG for CRN reproducibility.

    Returns
    -------
    dict
        Keys:
          - ``deterministic`` : np.ndarray, det Δ_S per solution
          - ``stochastic``    : np.ndarray, stoch Δ_S per solution
          - ``mc``            : np.ndarray, MC E[Δ_S] per solution
          - ``l1_det_vs_mc``  : float, L1 norm |det - mc| summed over solutions
          - ``l1_stoch_vs_mc``: float, L1 norm |stoch - mc| summed over solutions
    """
    rois = np.asarray(rois, dtype=float).reshape(-1)
    risks = np.asarray(risks, dtype=float).reshape(-1)
    covs = np.asarray(covs, dtype=float).reshape(-1)
    n = rois.shape[0]
    if risks.shape[0] != n or covs.shape[0] != n:
        raise ValueError(
            "rois, risks, covs must share length; got "
            f"{rois.shape}, {risks.shape}, {covs.shape}"
        )

    det = deterministic_delta_s(rois, risks, R1, R2)
    stoch = stochastic_delta_s(rois, risks, covs, R1, R2)

    # Build (N, 2) mus + (N, 2, 2) bivariate covariance for MC.
    # Use marginal variance = max(|cov|, 1e-6) to keep the 2x2
    # positive semi-definite (since |cov| <= sqrt(var_ROI * var_risk)).
    # When the covariance is zero we still need a strictly positive
    # marginal variance for multivariate_normal to be well-defined;
    # we use a tiny floor so the MC noise is bounded but not zero.
    mus = np.column_stack([rois, risks])
    var_floor = 1e-6
    marg_vars = np.maximum(np.abs(covs), var_floor)
    covs_full = np.zeros((n, 2, 2), dtype=float)
    covs_full[:, 0, 0] = marg_vars
    covs_full[:, 1, 1] = marg_vars
    covs_full[:, 0, 1] = covs
    covs_full[:, 1, 0] = covs

    mc = mc_ground_truth_delta_s(mus, covs_full, R1, R2, n_mc=n_mc, rng=rng)

    return {
        "deterministic": det,
        "stochastic": stoch,
        "mc": mc,
        "l1_det_vs_mc": float(np.sum(np.abs(det - mc))),
        "l1_stoch_vs_mc": float(np.sum(np.abs(stoch - mc))),
    }


# ─── W22 Probe AD empirical extension (2026-05-20) ───────────────────
# The original `deterministic_delta_s` above replicates the CURRENT (and
# potentially-buggy) Python `sms_emoa.py:_compute_hypervolume_contributions_class`
# middle-branch rectangle `(ROI_i − ROI_{i+1})(risk_{i−1} − risk_i)`.
# The functions below add two extra estimators so we can isolate the
# rectangle-mismatch effect empirically:
#
# 1. `deterministic_eq636_delta_s` — uses Eq 6.36 rectangle
#    `(ROI_i − ROI_{i−1})(risk_{i+1} − risk_i)`, which IS the true
#    unique 2D hypervolume contribution. This is what `stochastic_delta_s`
#    (Eq 6.41) reduces to when Cov = 0, and matches the C++ stochastic
#    `mean_delta_product` arithmetic when its DESC indices are mapped
#    to ASC.
#
# 2. `true_hv_contribution_2d` — non-MC closed-form 2D HV contribution
#    computed directly from sorted means (no covariance, no sampling).
#    This serves as the ARGMAX ground truth: in 2D the unique
#    hypervolume contribution of each Pareto-front point is exactly
#    the Eq 6.36 rectangle.

def deterministic_eq636_delta_s(
    rois: np.ndarray,
    risks: np.ndarray,
    R1: float,
    R2: float,
) -> np.ndarray:
    """Per-solution deterministic Δ_S using the Eq 6.36 rectangle.

    Sorted by ROI ASC, the unique 2D HV contribution of solution i is:

        Δ_S^eq636[i] = (ROI_i − ROI_{i−1}) × (risk_{i+1} − risk_i)

    with edge conventions:
        - i=0:   width = ROI_0 − R_1,         height = risk_1 − risk_0
        - i=n−1: width = ROI_{n−1} − ROI_{n−2}, height = R_2 − risk_{n−1}
        - n=1:   width = ROI_0 − R_1,         height = R_2 − risk_0
    """
    rois = np.asarray(rois, dtype=float).reshape(-1)
    risks = np.asarray(risks, dtype=float).reshape(-1)
    n = rois.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    if n == 1:
        return np.array([(rois[0] - R1) * (R2 - risks[0])], dtype=float)
    order = np.argsort(rois, kind="stable")
    rois_s = rois[order]
    risks_s = risks[order]
    out_sorted = np.zeros(n, dtype=float)
    for i in range(n):
        if i == 0:
            width = rois_s[0] - R1
            height = risks_s[1] - risks_s[0]
        elif i == n - 1:
            width = rois_s[i] - rois_s[i - 1]
            height = R2 - risks_s[i]
        else:
            width = rois_s[i] - rois_s[i - 1]
            height = risks_s[i + 1] - risks_s[i]
        out_sorted[i] = width * height
    out = np.zeros(n, dtype=float)
    out[order] = out_sorted
    return out


def compare_methods_with_eq636(
    rois: np.ndarray,
    risks: np.ndarray,
    covs: np.ndarray,
    R1: float,
    R2: float,
    n_mc: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Extended compare: adds `deterministic_eq636` to original comparison.

    Returns same keys as `compare_methods` plus:
      - `deterministic_eq636`: np.ndarray
      - `l1_eq636_vs_mc`: float

    Uses `compare_methods` MC ground truth, which itself uses the
    original (possibly-buggy) `deterministic_delta_s`. To get a clean
    "Eq 6.36 vs everything else" comparison, also computes an
    `eq636_mc_ground_truth_delta_s` (re-using `deterministic_eq636_delta_s`
    inside the MC loop). Both MCs are reported.
    """
    base = compare_methods(rois, risks, covs, R1, R2, n_mc=n_mc, rng=rng)
    eq636 = deterministic_eq636_delta_s(rois, risks, R1, R2)
    base["deterministic_eq636"] = eq636
    base["l1_eq636_vs_mc"] = float(np.sum(np.abs(eq636 - base["mc"])))

    # Also compute Eq 6.36-based MC (the "right" MC ground truth in 2D).
    # mc_eq636[i] should equal eq636[i] EXACTLY in the n_mc → ∞ limit
    # since the rectangle is deterministic given sampled means.
    n = np.asarray(rois).reshape(-1).shape[0]
    mus = np.column_stack([np.asarray(rois, dtype=float).reshape(-1),
                            np.asarray(risks, dtype=float).reshape(-1)])
    var_floor = 1e-6
    marg_vars = np.maximum(np.abs(np.asarray(covs, dtype=float).reshape(-1)),
                            var_floor)
    covs_full = np.zeros((n, 2, 2), dtype=float)
    covs_full[:, 0, 0] = marg_vars
    covs_full[:, 1, 1] = marg_vars
    covs_full[:, 0, 1] = np.asarray(covs, dtype=float).reshape(-1)
    covs_full[:, 1, 0] = np.asarray(covs, dtype=float).reshape(-1)

    if rng is None:
        rng = np.random.default_rng()
    samples = np.empty((n_mc, n, 2), dtype=float)
    for i in range(n):
        samples[:, i, :] = rng.multivariate_normal(mus[i], covs_full[i],
                                                    size=n_mc)
    mc_eq636_accum = np.zeros(n, dtype=float)
    for j in range(n_mc):
        rois_j = samples[j, :, 0]
        risks_j = samples[j, :, 1]
        mc_eq636_accum += deterministic_eq636_delta_s(rois_j, risks_j, R1, R2)
    mc_eq636 = mc_eq636_accum / n_mc

    base["mc_eq636"] = mc_eq636
    base["l1_det_vs_mc_eq636"] = float(np.sum(np.abs(base["deterministic"]
                                                      - mc_eq636)))
    base["l1_eq636_vs_mc_eq636"] = float(np.sum(np.abs(eq636 - mc_eq636)))
    base["l1_stoch_vs_mc_eq636"] = float(np.sum(np.abs(base["stochastic"]
                                                        - mc_eq636)))
    return base
