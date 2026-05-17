"""
W18-1 cross-validation: comparison framework.

Per operator directive: "There is the possibility of the C++ reference
implementation also being wrong, so do not merely treat it as the
oracle." → comparison flags ANY discrepancy without privileging either
side.

Usage:
    report = compare_csvs(cpp_path, py_path, atol=1e-9, rtol=1e-9)
    # report is dict with:
    #   verdict: AGREE | DISAGREE
    #   n_compared: int
    #   per_column: {col: {abs_max, rel_max, n_violations, scale_ratio}}
    #   markdown: str (suitable for docs/CROSS-VALIDATION-<CHECK>.md)
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compare_csvs(
    cpp_path: Path | str,
    py_path: Path | str,
    *,
    atol: float = 1e-9,
    rtol: float = 1e-9,
    columns: list[str] | None = None,
    label_cpp: str = "C++ legacy",
    label_py: str = "Python port",
) -> dict[str, Any]:
    """Compare two cross-validation CSVs.

    Args:
        cpp_path: C++ driver output CSV
        py_path: Python driver output CSV
        atol: absolute tolerance for `np.isclose`
        rtol: relative tolerance
        columns: optional explicit column list (else compare all
            numeric columns that exist in BOTH files)
        label_cpp: label for the C++ side in the report
        label_py: label for the Python side in the report

    Returns:
        Structured report dict + markdown receipt
    """
    cpp_df = pd.read_csv(cpp_path)
    py_df = pd.read_csv(py_path)

    if columns is None:
        cpp_numeric = set(cpp_df.select_dtypes(include=[np.number]).columns)
        py_numeric = set(py_df.select_dtypes(include=[np.number]).columns)
        columns = sorted(cpp_numeric & py_numeric)

    if len(cpp_df) != len(py_df):
        return _emit_report(
            verdict="DISAGREE",
            mismatch_reason=(
                f"row count differs: {label_cpp} has {len(cpp_df)} rows, "
                f"{label_py} has {len(py_df)} rows"
            ),
            atol=atol, rtol=rtol, columns=columns,
            label_cpp=label_cpp, label_py=label_py,
            cpp_df=cpp_df, py_df=py_df,
        )

    per_column: dict[str, dict[str, float]] = {}
    any_violation = False
    for col in columns:
        cpp_v = cpp_df[col].to_numpy(dtype=float)
        py_v = py_df[col].to_numpy(dtype=float)
        abs_diff = np.abs(cpp_v - py_v)
        # Avoid divide-by-zero in rel-diff
        denom = np.maximum(np.abs(cpp_v), np.abs(py_v))
        denom = np.where(denom == 0, 1.0, denom)
        rel_diff = abs_diff / denom

        violations = ~np.isclose(cpp_v, py_v, atol=atol, rtol=rtol, equal_nan=True)
        n_viol = int(violations.sum())
        if n_viol > 0:
            any_violation = True

        cpp_nonzero_mean = np.mean(np.abs(cpp_v[np.abs(cpp_v) > 1e-30])) if np.any(np.abs(cpp_v) > 1e-30) else 0.0
        py_nonzero_mean = np.mean(np.abs(py_v[np.abs(py_v) > 1e-30])) if np.any(np.abs(py_v) > 1e-30) else 0.0
        scale_ratio = (cpp_nonzero_mean / py_nonzero_mean) if py_nonzero_mean > 0 else float("nan")

        per_column[col] = {
            "abs_max": float(np.max(abs_diff)),
            "rel_max": float(np.max(rel_diff)),
            "n_violations": n_viol,
            "scale_ratio_cpp_over_py": float(scale_ratio),
            "cpp_mean_abs": float(cpp_nonzero_mean),
            "py_mean_abs": float(py_nonzero_mean),
        }

    verdict = "DISAGREE" if any_violation else "AGREE"
    return _emit_report(
        verdict=verdict,
        mismatch_reason=None,
        atol=atol, rtol=rtol, columns=columns,
        label_cpp=label_cpp, label_py=label_py,
        cpp_df=cpp_df, py_df=py_df,
        per_column=per_column,
    )


def _emit_report(
    verdict: str,
    mismatch_reason: str | None,
    atol: float, rtol: float, columns: list[str],
    label_cpp: str, label_py: str,
    cpp_df: pd.DataFrame, py_df: pd.DataFrame,
    per_column: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Emit a structured report + markdown."""
    md_lines = []
    md_lines.append(f"# Cross-validation comparison — {label_cpp} vs {label_py}")
    md_lines.append("")
    md_lines.append(f"**Verdict**: `{verdict}`")
    md_lines.append("")
    md_lines.append(f"- Tolerance: `atol={atol:g}`, `rtol={rtol:g}`")
    md_lines.append(f"- Rows: `{label_cpp}` = {len(cpp_df)}, `{label_py}` = {len(py_df)}")
    md_lines.append(f"- Columns compared: {len(columns)}")
    md_lines.append("")

    if mismatch_reason is not None:
        md_lines.append(f"## Structural mismatch")
        md_lines.append("")
        md_lines.append(f"{mismatch_reason}")
        md_lines.append("")
        return {
            "verdict": verdict,
            "n_compared": 0,
            "per_column": {},
            "markdown": "\n".join(md_lines),
        }

    if per_column is None:
        per_column = {}
    md_lines.append("## Per-column results")
    md_lines.append("")
    md_lines.append(
        "| Column | abs_max | rel_max | n_violations | "
        f"scale_ratio ({label_cpp}/{label_py}) | "
        f"{label_cpp} mean(|x|) | {label_py} mean(|x|) |"
    )
    md_lines.append("|---|---|---|---|---|---|---|")
    for col, stats in per_column.items():
        md_lines.append(
            f"| `{col}` | {stats['abs_max']:.3g} | {stats['rel_max']:.3g} | "
            f"{stats['n_violations']} | "
            f"{stats['scale_ratio_cpp_over_py']:.4f} | "
            f"{stats['cpp_mean_abs']:.3g} | {stats['py_mean_abs']:.3g} |"
        )
    md_lines.append("")

    md_lines.append("## Mutual-skepticism reporting")
    md_lines.append("")
    md_lines.append(
        "Per W18 directive, neither side is treated as oracle. "
        "Discrepancies are flagged; resolution (which side is correct) "
        "must reference the thesis equation cross-cited in the unit "
        "contract's BACKLOG §6 grounding section."
    )

    return {
        "verdict": verdict,
        "n_compared": len(cpp_df),
        "per_column": per_column,
        "markdown": "\n".join(md_lines),
    }


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="W18 cross-validation comparison")
    parser.add_argument("cpp_path", help="C++ driver output CSV")
    parser.add_argument("py_path", help="Python driver output CSV")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--output", help="Markdown report output path (default stdout)")
    args = parser.parse_args()

    report = compare_csvs(args.cpp_path, args.py_path,
                            atol=args.atol, rtol=args.rtol)
    if args.output:
        Path(args.output).write_text(report["markdown"])
    else:
        sys.stdout.write(report["markdown"] + "\n")
    sys.exit(0 if report["verdict"] == "AGREE" else 1)
