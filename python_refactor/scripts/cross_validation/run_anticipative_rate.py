"""W19-4 cross-check G: anticipative rate λ — THREE formulas compared.

Each formula consumes a TIP (= nd_probability) value and produces α / λ.
We sweep TIP over a grid + compare:

  v1  C++ (legacy-cpp/source/nsga2.cpp:565)
      alpha = 1.0 - linear_entropy(nd_probability)
        where linear_entropy = triangular tent: 2p if p<=0.5; 2(1-p) else

  v2  C++ (legacy-cpp-v2/source/asms_emoa.cpp:44)
      alpha = 1.0 - nd_probability

  Python current production (W16-1, anticipatory_learning.py:compute_anticipatory_learning_rate)
      lambda = 0.5 * (lambda_H + lambda_K) per thesis Eq 7.16
      where lambda_H per Eq 6.6 = (1/(H-1)) * (1 - binary_entropy(TIP))

  Thesis Eq 6.6 (λ^H component for H=2)
      lambda_H = (1 - binary_entropy(TIP)) / (H-1)

Re-implements v1 and v2 formulas in pure Python (no C++ link needed —
arithmetic faithful). For Python, uses the ACTUAL production
DispatchPredictor.dirichlet_mean_prediction_vec NO — wrong class!

Actually for the rate, the production class is AnticipatoryLearning.
This driver exercises the rate formula in isolation by direct math.
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def v1_linear_entropy(p: float) -> float:
    """legacy-cpp/source/statistics.cpp:228 — triangular tent."""
    if p <= 0.5:
        return 2.0 * p
    return 2.0 * (1.0 - p)


def v1_alpha(nd_probability: float) -> float:
    """v1 C++ (nsga2.cpp:565): alpha = 1 - linear_entropy(nd_probability)."""
    return 1.0 - v1_linear_entropy(nd_probability)


def v2_alpha(nd_probability: float) -> float:
    """v2 C++ (asms_emoa.cpp:44): alpha = 1 - nd_probability."""
    return 1.0 - nd_probability


def binary_entropy(p: float) -> float:
    """Shannon binary entropy (in bits, base 2)."""
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def python_lambda_h(tip: float, H: int = 2) -> float:
    """Python λ^H per thesis Eq 6.6 (multi-horizon form)."""
    return (1.0 - binary_entropy(tip)) / max(1, H - 1)


def python_lambda_eq716(tip: float, lambda_k: float = 0.0, H: int = 2) -> float:
    """Python λ per thesis Eq 7.16 (the production formula in
    compute_anticipatory_learning_rate after W16-1).

    Default lambda_k=0 = no K-period residual contribution; representative
    of warm-up / production saturation regime per W17-5 trace."""
    return 0.5 * (python_lambda_h(tip, H) + lambda_k)


def main():
    """Compare all three formulas on a TIP grid."""
    print("# W19-4 cross-check G: anticipative rate λ — three formulas")
    print()
    print("All formulas consume TIP (= nd_probability) ∈ [0, 1] and produce α/λ.")
    print()
    print("| TIP     | v1 (1 - linear_entropy) | v2 (1 - TIP) | Python λ^H | Python λ (Eq 7.16) |")
    print("|---------|-------------------------|--------------|------------|--------------------|")
    for tip in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        v1 = v1_alpha(tip)
        v2 = v2_alpha(tip)
        py_h = python_lambda_h(tip)
        py_full = python_lambda_eq716(tip)
        print(f"| {tip:.2f}    | {v1:.4f}                  | {v2:.4f}       | {py_h:.4f}     | {py_full:.4f}            |")

    print()
    print("## Key observations")
    print()
    print("- v1 is SYMMETRIC around TIP=0.5 (alpha=0 at TIP=0.5; alpha=1 at TIP=0 or 1)")
    print("- v2 is MONOTONIC DECREASING in TIP (alpha=1 at TIP=0; alpha=0 at TIP=1)")
    print("- Python λ^H is binary-Shannon (alpha→0 as TIP→0.5; alpha→1 as TIP→0 or 1)")
    print("- Python λ (Eq 7.16) = λ^H/2 when λ^K=0 (the W17-5 production regime)")
    print()
    print("At W17-5 production TIP ≈ 0.50:")
    print(f"  v1 alpha       = {v1_alpha(0.50):.4f}  (zero — no anticipation)")
    print(f"  v2 alpha       = {v2_alpha(0.50):.4f}  (half — moderate anticipation)")
    print(f"  Python λ^H     = {python_lambda_h(0.50):.4f}  (zero — Shannon collapse)")
    print(f"  Python λ Eq7.16= {python_lambda_eq716(0.50):.4f}  (zero — W17-5 'saturation' reproduced)")
    print()
    print("So Python and v1 BOTH go to 0 at TIP=0.5 (the saturation regime),")
    print("while v2 maintains alpha = 0.5 — meaningful anticipation strength.")
    print()
    print("**Hypothesis**: v2's monotonic-decreasing formula keeps anticipation")
    print("active even when nd_probability is near 0.5 (max uncertainty), whereas")
    print("v1 and Python both COLLAPSE to zero in this regime. If the headline")
    print("paper-replication result was generated with v2, then ASMS_mHDM_K3")
    print("would still be doing meaningful anticipation in saturation regimes,")
    print("explaining its outperformance vs SMS baseline. Python's Eq 7.16")
    print("faithful implementation collapses anticipation in exactly this regime")
    print("→ ASMS reduces to SMS+noise → loses.")


if __name__ == "__main__":
    main()
