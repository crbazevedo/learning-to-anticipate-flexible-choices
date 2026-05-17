"""W19-3 cross-check E: Dirichlet predictor (v2 reference).

Calls the ACTUAL production class DirichletPredictor (the methods
invoked from anticipatory_learning.py:822/831/865 in production).
Same fixture format as legacy-cpp-v2/build/drivers/dirichlet_driver.cpp.
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.anticipatory_learning import DirichletPredictor


def _read_vec(it, n):
    line = next(it)
    return np.array([float(x) for x in line.strip().split(",")])


def main(stream_in=sys.stdin, stream_out=sys.stdout):
    lines = iter(stream_in.read().splitlines())
    next(lines)  # ###DIRICHLET_META
    n_assets, n_cases = (int(x) for x in next(lines).split(","))

    next(lines)  # ###CASES
    cases = []
    for _ in range(n_cases):
        parts = next(lines).split(",")
        cases.append({
            "id": int(parts[0]),
            "rate": float(parts[1]),
            "concentration": float(parts[2]),
        })

    header = ["test_case_id", "kind"] + [f"c{a}" for a in range(n_assets)]
    stream_out.write(",".join(header) + "\n")

    for k in cases:
        next(lines)  # ###prev_<id>
        prev = _read_vec(lines, n_assets)
        next(lines)  # ###curr_<id>
        curr = _read_vec(lines, n_assets)
        next(lines)  # ###obs_<id>
        obs = _read_vec(lines, n_assets)

        pred = DirichletPredictor.dirichlet_mean_prediction_vec(prev, curr, k["rate"])
        updated = DirichletPredictor.dirichlet_mean_map_update(pred, obs, k["concentration"])

        stream_out.write(f"{k['id']},mean_pred,"
                          + ",".join(f"{x:.17g}" for x in pred) + "\n")
        stream_out.write(f"{k['id']},map_update,"
                          + ",".join(f"{x:.17g}" for x in updated) + "\n")


def build_dirichlet_fixture(seed: int = 42, n_assets: int = 5, n_cases: int = 5) -> str:
    """Build a deterministic Dirichlet fixture (prev, curr, obs triples + params)."""
    rng = np.random.default_rng(seed)
    out = StringIO()
    out.write("###DIRICHLET_META\n")
    out.write(f"{n_assets},{n_cases}\n")
    out.write("###CASES\n")
    for i in range(n_cases):
        rate = float(rng.uniform(0.0, 2.0))
        conc = float(rng.uniform(5.0, 50.0))
        out.write(f"{i},{rate},{conc}\n")
    for i in range(n_cases):
        prev = rng.dirichlet(np.ones(n_assets))
        curr = rng.dirichlet(np.ones(n_assets))
        obs = rng.dirichlet(np.ones(n_assets))
        out.write(f"###prev_{i}\n")
        out.write(",".join(f"{x:.17g}" for x in prev) + "\n")
        out.write(f"###curr_{i}\n")
        out.write(",".join(f"{x:.17g}" for x in curr) + "\n")
        out.write(f"###obs_{i}\n")
        out.write(",".join(f"{x:.17g}" for x in obs) + "\n")
    return out.getvalue()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-fixture", action="store_true")
    args = parser.parse_args()
    if args.build_fixture:
        sys.stdout.write(build_dirichlet_fixture())
    else:
        main()
