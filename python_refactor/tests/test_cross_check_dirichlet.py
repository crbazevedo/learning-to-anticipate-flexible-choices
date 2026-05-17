"""W19-3 Dirichlet cross-check tests (vs legacy-cpp-v2)."""
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

from scripts.cross_validation.run_dirichlet import build_dirichlet_fixture
from src.algorithms.anticipatory_learning import DirichletPredictor


CPP_DRIVER = REPO_ROOT / "legacy-cpp-v2" / "build" / "drivers" / "dirichlet_driver"


class TestPythonDirichletFormulas:
    """Python DirichletPredictor formulas match v2 dirichlet.cpp line-for-line."""

    def test_mean_prediction_vec_convex_combination(self):
        prev = np.array([0.1, 0.2, 0.3, 0.4])
        curr = np.array([0.4, 0.3, 0.2, 0.1])
        # rate = 0 → returns prev (after normalization)
        pred = DirichletPredictor.dirichlet_mean_prediction_vec(prev, curr, 0.0)
        np.testing.assert_allclose(pred, prev, atol=1e-15)

    def test_map_update_zero_variance_returns_zero(self):
        """If a position has p_predicted = 0, p_updated = 0."""
        pred = np.array([0.0, 0.3, 0.7])
        obs = np.array([0.5, 0.3, 0.2])
        out = DirichletPredictor.dirichlet_mean_map_update(pred, obs, 10.0)
        assert out[0] == 0.0


@pytest.mark.skipif(not CPP_DRIVER.exists(),
                      reason="dirichlet_driver not built; cd legacy-cpp-v2/build && make ...")
class TestCPPv2VsPythonDirichletParity:
    def test_agree_machine_precision(self, tmp_path: Path):
        fixture = build_dirichlet_fixture(seed=42, n_assets=5, n_cases=5)
        cpp = subprocess.run([str(CPP_DRIVER)], input=fixture,
                              capture_output=True, text=True, timeout=10)
        assert cpp.returncode == 0, f"C++ v2 dirichlet_driver failed: {cpp.stderr}"

        from scripts.cross_validation.run_dirichlet import main as run_py
        py_out = StringIO()
        run_py(stream_in=StringIO(fixture), stream_out=py_out)

        cpp_df = pd.read_csv(StringIO(cpp.stdout))
        py_df = pd.read_csv(StringIO(py_out.getvalue()))
        assert cpp_df.shape == py_df.shape

        for col in [c for c in cpp_df.columns if c.startswith("c")]:
            np.testing.assert_allclose(cpp_df[col].to_numpy(),
                                          py_df[col].to_numpy(),
                                          atol=1e-12,
                                          err_msg=f"Dirichlet parity failure on {col}")
