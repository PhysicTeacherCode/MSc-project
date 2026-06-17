import io
import os
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

import src.ising_coniii as ising


class IsingSharedSamplesTests(unittest.TestCase):
    def setUp(self):
        self.n_users = 21
        self.n_keywords = 8
        base = np.arange(self.n_users * self.n_keywords).reshape(self.n_users, self.n_keywords)
        self.spin_matrix = np.where(base % 2 == 0, 1, -1).astype(np.int64)
        self.model_samples = np.where(
            np.arange(1000 * self.n_users).reshape(1000, self.n_users) % 3 == 0,
            1.0,
            -1.0,
        )
        n_pairs = self.n_users * (self.n_users - 1) // 2
        self.multipliers = np.zeros(self.n_users + n_pairs, dtype=np.float64)
        self.resultados = {
            "MCH-Custom": {
                "multipliers": self.multipliers,
                "J": np.zeros((self.n_users, self.n_users), dtype=np.float64),
            }
        }

    def test_figures_reuse_provided_model_samples_without_resampling(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ising, "_amostrar_modelo_metropolis", side_effect=AssertionError("resampled")
        ), patch.object(
            ising, "_distribuicao_Q_pairwise_mc", side_effect=AssertionError("resampled Q")
        ), patch.object(
            ising, "_bootstrap_tripletos", return_value=np.zeros(self.n_users * (self.n_users - 1) * (self.n_users - 2) // 6)
        ):
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    ising.gerar_figura2(
                        self.spin_matrix,
                        self.resultados,
                        gexf_path="missing.gexf",
                        node_names=[f"user_{i}" for i in range(self.n_users)],
                        session_id="test",
                        model_samples=self.model_samples,
                    )
                    ising.gerar_figura3_multimodo(
                        self.spin_matrix,
                        self.resultados,
                        session_id="test",
                        triplet_modes=["all"],
                        n_amostras_mc=100_000,
                        model_samples=self.model_samples,
                    )
                    ising.gerar_figura4(
                        self.spin_matrix,
                        self.resultados,
                        session_id="test",
                        model_samples=self.model_samples,
                    )
            finally:
                os.chdir(old_cwd)

    def test_figura2_uses_two_covariance_panels_and_masks_diagonal(self):
        captured = []

        def capture_imshow(self, mat, *args, **kwargs):
            captured.append((np.asarray(mat, dtype=np.float64).copy(), kwargs))
            return original_imshow(self, mat, *args, **kwargs)

        original_imshow = ising.plt.Axes.imshow
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ising.plt.Axes, "imshow", new=capture_imshow
        ):
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    ising.gerar_figura2(
                        self.spin_matrix,
                        self.resultados,
                        gexf_path="missing.gexf",
                        node_names=[f"user_{i}" for i in range(self.n_users)],
                        session_id="test",
                        model_samples=self.model_samples,
                    )
            finally:
                os.chdir(old_cwd)

        self.assertEqual(len(captured), 2)
        for mat, kwargs in captured:
            self.assertTrue(np.isnan(np.diag(mat)).all())
            self.assertEqual(kwargs["vmin"], 0.0)
            self.assertGreater(kwargs["vmax"], 0.0)


if __name__ == "__main__":
    unittest.main()
