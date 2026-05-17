"""W19-2 Python driver: Kalman filter cross-check.

Reads same KF fixture format as the C++ driver; emits identical
step-by-step (x, P) CSV.
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.kalman_filter import KalmanParams, kalman_filter


def _read_matrix(it, rows, cols):
    M = np.zeros((rows, cols))
    for r in range(rows):
        line = next(it)
        vals = [float(x) for x in line.strip().split(",")]
        for c in range(cols):
            M[r, c] = vals[c]
    return M


def main(stream_in=sys.stdin, stream_out=sys.stdout):
    lines = iter(stream_in.read().splitlines())
    next(lines)  # ###KF_META
    state_dim, obs_dim, n_steps = (int(x) for x in next(lines).split(","))

    next(lines)  # ###F
    F = _read_matrix(lines, state_dim, state_dim)
    next(lines)  # ###H
    H = _read_matrix(lines, obs_dim, state_dim)
    next(lines)  # ###R
    R = _read_matrix(lines, obs_dim, obs_dim)

    params = KalmanParams()
    # Per-instance matrix assignment (dataclass default is None; class-level
    # assignment doesn't always propagate through dataclass descriptor)
    params.F = F
    params.H = H
    params.R = R
    next(lines)  # ###x0
    params.x = _read_matrix(lines, state_dim, 1)[:, 0]
    params.u = np.zeros(state_dim)
    next(lines)  # ###P0
    params.P = _read_matrix(lines, state_dim, state_dim)
    params.x_next = params.x.copy()
    params.P_next = params.P.copy()

    # CSV emit
    header = ["step", "kind", "row_idx"] + [f"c{j}" for j in range(state_dim)]
    stream_out.write(",".join(header) + "\n")

    def _emit(step):
        # x
        row = [str(step), "x", "0"] + [f"{params.x[i]:.17g}" for i in range(state_dim)]
        stream_out.write(",".join(row) + "\n")
        # P
        for i in range(state_dim):
            row = [str(step), "P", str(i)] + [f"{params.P[i, j]:.17g}" for j in range(state_dim)]
            stream_out.write(",".join(row) + "\n")

    _emit(0)
    for s in range(1, n_steps + 1):
        next(lines)  # ###MEAS_<i>
        z = _read_matrix(lines, obs_dim, 1)[:, 0]
        kalman_filter(params, z)
        _emit(s)


def build_kf_fixture(seed: int = 42, state_dim: int = 4, obs_dim: int = 2,
                       n_steps: int = 5) -> str:
    """Build a deterministic KF fixture (synthetic F, H, R, x0, P0, measurements)."""
    rng = np.random.default_rng(seed)
    # Paper Eq (11): F is the constant-velocity model
    if state_dim == 4 and obs_dim == 2:
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
    else:
        F = np.eye(state_dim)
        H = np.zeros((obs_dim, state_dim))
        H[:obs_dim, :obs_dim] = np.eye(obs_dim)
    R = np.eye(obs_dim) * 0.01
    x0 = rng.standard_normal(state_dim) * 0.01
    P0 = np.eye(state_dim) * 0.1
    measurements = rng.standard_normal((n_steps, obs_dim)) * 0.01

    out = StringIO()
    out.write("###KF_META\n")
    out.write(f"{state_dim},{obs_dim},{n_steps}\n")
    out.write("###F\n")
    np.savetxt(out, F, delimiter=",", fmt="%.17g")
    out.write("###H\n")
    np.savetxt(out, H, delimiter=",", fmt="%.17g")
    out.write("###R\n")
    np.savetxt(out, R, delimiter=",", fmt="%.17g")
    out.write("###x0\n")
    np.savetxt(out, x0.reshape(-1, 1), delimiter=",", fmt="%.17g")
    out.write("###P0\n")
    np.savetxt(out, P0, delimiter=",", fmt="%.17g")
    for i, z in enumerate(measurements):
        out.write(f"###MEAS_{i}\n")
        np.savetxt(out, z.reshape(-1, 1), delimiter=",", fmt="%.17g")
    return out.getvalue()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-fixture", action="store_true",
                          help="If set, emit a default fixture to stdout instead of running KF.")
    args = parser.parse_args()
    if args.build_fixture:
        sys.stdout.write(build_kf_fixture())
    else:
        main()
