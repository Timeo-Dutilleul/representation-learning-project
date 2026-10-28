import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.src.d_ref import load_d_ref_indices
from redundancy_project.src.loader import (
    load_checkpoint,
    load_run_config,
)
from redundancy_project.src.measures_R import (
    R_C_layer,
    R_L_layer,
    covariance_spectrum,
    get_activations,
)
from redundancy_project.src.participation import (
    spectral_energy_participation_ratio,
)
from redundancy_project.run_pipeline import _summarize_off_diagonal
from redundancy_project.src.model_defs import build_model_from_checkpoint
from redundancy_project.tests.test_single_run import RUN_DIR, ZOO_ROOT


CHECKPOINT_EPOCH = 50
TRIAL_SEED = 550
ACTIVATION_LAYER = 10
MAX_SAMPLES = 500


class MeasuresRTests(unittest.TestCase):
    def test_constant_units_have_undefined_correlations_and_are_excluded(self):
        values = torch.tensor(
            [
                [0.0, 0.0, 5.0],
                [1.0, 2.0, 5.0],
                [2.0, 4.0, 5.0],
                [3.0, 6.0, 5.0],
            ]
        )
        correlations = R_L_layer(values)
        self.assertTrue(torch.isnan(correlations[2]).all())
        self.assertTrue(torch.isnan(correlations[:, 2]).all())
        self.assertAlmostEqual(correlations[0, 1].item(), 1.0)
        self.assertEqual(
            _summarize_off_diagonal(correlations),
            (1.0, 1.0),
        )
        self.assertAlmostEqual(R_C_layer(values), 1.0)

    def test_correlations_are_stable_for_small_activation_variance(self):
        generator = torch.Generator().manual_seed(7)
        values = 1e-4 + 1e-8 * torch.randn(
            500, 4, generator=generator, dtype=torch.float32
        )
        correlations = R_L_layer(values)
        self.assertTrue(torch.all(correlations <= 1.0))
        self.assertTrue(torch.all(correlations >= -1.0))
        self.assertTrue(
            torch.allclose(
                torch.diag(correlations), torch.ones(4), atol=1e-6
            )
        )

    def test_R_C_matches_participation_ratio_of_covariance_spectrum(self):
        torch.manual_seed(9)
        activations = torch.randn(12, 5, dtype=torch.float64)
        spectrum = covariance_spectrum(activations)
        self.assertEqual(spectrum.shape, (5,))
        value = R_C_layer(activations)
        self.assertAlmostEqual(
            value, spectral_energy_participation_ratio(spectrum), places=10
        )
        self.assertGreaterEqual(value, 1.0)
        self.assertLessEqual(value, activations.shape[1])

    @classmethod
    def setUpClass(cls):
        config = load_run_config(ZOO_ROOT)
        checkpoint = load_checkpoint(RUN_DIR, epoch=CHECKPOINT_EPOCH)
        cls.model = build_model_from_checkpoint(
            config, checkpoint, seed=TRIAL_SEED
        )
        with torch.serialization.safe_globals([TensorDataset]):
            dataset = torch.load(
                ZOO_ROOT / "dataset.pt",
                map_location="cpu",
                weights_only=True,
            )
        testset = dataset["testset"]
        indices = load_d_ref_indices(dataset_size=len(testset))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        cls.testloader = DataLoader(
            TensorDataset(
                testset.tensors[0].index_select(0, index_tensor),
                testset.tensors[1].index_select(0, index_tensor),
            ),
            batch_size=128,
            shuffle=False,
        )

    def test_R_measures_on_real_run(self):
        activations = get_activations(
            self.model,
            self.testloader,
            layer_indices=[ACTIVATION_LAYER],
            max_samples=MAX_SAMPLES,
        )
        layer_activations = activations[ACTIVATION_LAYER]
        self.assertEqual(tuple(layer_activations.shape), (500, 20))

        correlation = R_L_layer(layer_activations)
        self.assertEqual(tuple(correlation.shape), (20, 20))
        self.assertTrue(torch.all(correlation <= 1.0 + 1e-4))
        self.assertTrue(torch.all(correlation >= -1.0 - 1e-4))
        self.assertTrue(
            torch.allclose(
                torch.diag(correlation), torch.ones(20), atol=1e-3
            )
        )

        effective_rank = R_C_layer(layer_activations)
        self.assertGreaterEqual(effective_rank, 1.0)
        self.assertLessEqual(effective_rank, 20.0 + 1e-3)

        print(f"R_C_layer={effective_rank:.6f}")
        print("R_L_layer correlation matrix:")
        print(correlation)


if __name__ == "__main__":
    unittest.main()
