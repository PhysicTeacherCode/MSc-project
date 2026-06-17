import unittest
import os
import tempfile
import warnings

import numpy as np
from pyparsing import PyparsingDeprecationWarning

warnings.filterwarnings("ignore", message=r".*deprecated.*")
warnings.filterwarnings("ignore", category=PyparsingDeprecationWarning)

from src.susceptibility import (
    calcular_suscetibilidade_individuos_forcados,
    calcular_curva_suscetibilidade,
    executar_analise_suscetibilidade,
    estimar_entropia_energia,
)


class SusceptibilityAnalysisTests(unittest.TestCase):
    def test_susceptibility_matches_independent_exact_case(self):
        spin_matrix = np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int64,
        )
        multipliers = np.zeros(3, dtype=np.float64)

        df = calcular_curva_suscetibilidade(
            spin_matrix,
            multipliers,
            field_values=np.array([0.0]),
            n_samples=1000,
        )

        self.assertAlmostEqual(float(df.loc[0, "chi_pairwise"]), 0.25, places=6)
        self.assertAlmostEqual(float(df.loc[0, "chi_independente"]), 0.25, places=6)

    def test_entropy_energy_exact_has_normalized_cumulative_entropy(self):
        multipliers = np.zeros(3, dtype=np.float64)

        df, info = estimar_entropia_energia(
            multipliers,
            n_users=2,
            n_bins=8,
            wl_steps=100,
            exact_threshold=20,
        )

        self.assertEqual(info["method"], "exact")
        self.assertGreater(len(df), 0)
        self.assertAlmostEqual(float(df["S"].max()), 2 * np.log(2), places=6)
        self.assertTrue(np.all(np.diff(df["E"].to_numpy()) >= 0))

    def test_forced_individuals_matches_independent_exact_case(self):
        multipliers = np.zeros(6, dtype=np.float64)

        df = calcular_suscetibilidade_individuos_forcados(
            multipliers,
            n_users=3,
            max_forced=2,
            n_configurations=5,
            n_samples=1000,
        )

        self.assertEqual(df["n_forced"].tolist(), [0, 1, 2])
        self.assertTrue(np.allclose(df["chi_mean"].to_numpy(), 0.25))
        self.assertTrue(np.allclose(df["chi_std"].to_numpy(), 0.0))

    def test_susceptibility_analysis_writes_hall_figures_5_and_6(self):
        spin_matrix = np.array(
            [
                [1, -1, 1, -1],
                [-1, 1, -1, 1],
            ],
            dtype=np.int64,
        )
        multipliers = np.zeros(3, dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = executar_analise_suscetibilidade(
                spin_matrix,
                multipliers,
                output_dir=tmpdir,
                artifact_stem="unit",
                field_min=-0.1,
                field_max=0.1,
                field_points=3,
                samples_per_field=1000,
                max_forced=1,
                forced_configurations=3,
                samples_per_forced_configuration=1000,
            )

            self.assertIn("figura6_individuos_forcados", outputs["figura6_png"])
            self.assertTrue(os.path.exists(outputs["figura5_png"]))
            self.assertTrue(os.path.exists(outputs["figura6_png"]))
            self.assertEqual(outputs["figura6_definition"], "forced individuals")


if __name__ == "__main__":
    unittest.main()
