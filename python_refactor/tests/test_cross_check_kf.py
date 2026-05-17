"""W19-2 KF cross-check tests. Reading-C critical test."""
from __future__ import annotations

import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.run_kf import build_kf_fixture


CPP_DRIVER = REPO_ROOT / "legacy-cpp" / "build" / "drivers" / "kf_driver"


@pytest.mark.skipif(not CPP_DRIVER.exists(), reason="kf_driver not built")
class TestKFParityMachinePrecision:
    """Reading C: C++ and Python KFs produce identical posteriors."""

    def test_5_step_predict_update_identical(self):
        fixture = build_kf_fixture(seed=42, state_dim=4, obs_dim=2, n_steps=5)
        cpp = subprocess.run([str(CPP_DRIVER)], input=fixture,
                              capture_output=True, text=True, timeout=10)
        assert cpp.returncode == 0, f"C++ KF failed: {cpp.stderr}"

        from scripts.cross_validation.run_kf import main as run_py
        py_out = StringIO()
        run_py(stream_in=StringIO(fixture), stream_out=py_out)

        cpp_df = pd.read_csv(StringIO(cpp.stdout))
        py_df = pd.read_csv(StringIO(py_out.getvalue()))
        assert cpp_df.shape == py_df.shape

        for col in ["c0", "c1", "c2", "c3"]:
            np.testing.assert_allclose(cpp_df[col].to_numpy(),
                                          py_df[col].to_numpy(),
                                          atol=1e-12,
                                          err_msg=f"KF parity failure on {col}")

    def test_different_seed_still_identical(self):
        fixture = build_kf_fixture(seed=99, state_dim=4, obs_dim=2, n_steps=3)
        cpp = subprocess.run([str(CPP_DRIVER)], input=fixture,
                              capture_output=True, text=True, timeout=10)
        assert cpp.returncode == 0
        from scripts.cross_validation.run_kf import main as run_py
        py_out = StringIO()
        run_py(stream_in=StringIO(fixture), stream_out=py_out)
        cpp_df = pd.read_csv(StringIO(cpp.stdout))
        py_df = pd.read_csv(StringIO(py_out.getvalue()))
        np.testing.assert_allclose(cpp_df["c0"].to_numpy(),
                                      py_df["c0"].to_numpy(), atol=1e-12)
