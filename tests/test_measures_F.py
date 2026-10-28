import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.src.d_ref import load_d_ref_indices
from redundancy_project.src.loader import load_checkpoint, load_run_config
from redundancy_project.src.measures_F import (
    F_C_subspace,
    F_L_layer,
    F_L_layer_diagnostics,
    get_logprobs,
    low_variance_directions,
    random_orthonormal_directions,
)
from redundancy_project.src.measures_R import get_activations
from redundancy_project.src.model_defs import build_model_from_checkpoint
from redundancy_project.tests.test_single_run import RUN_DIR, ZOO_ROOT


CHECKPOINT_EPOCH = 50
TRIAL_SEED = 550
NONLINEARITY_LAYER = 10
N_UNITS = 20
MAX_SAMPLES = 500


class MeasuresFTests(unittest.TestCase):
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
        cls.testset = dataset["testset"]
        index_tensor = torch.tensor(
            load_d_ref_indices(dataset_size=len(cls.testset)),
            dtype=torch.long,
        )
        cls.x = cls.testset.tensors[0].index_select(0, index_tensor)
        cls.targets = cls.testset.tensors[1].index_select(0, index_tensor)
        cls.testloader = DataLoader(
            TensorDataset(cls.x, cls.targets),
            batch_size=128,
            shuffle=False,
        )

    def test_F_L_runs_and_is_nonnegative(self):
        scores = F_L_layer(
            self.model, self.x, NONLINEARITY_LAYER, N_UNITS
        )
        self.assertEqual(tuple(scores.shape), (N_UNITS,))
        self.assertTrue(torch.all(scores >= -1e-6))
        self.assertTrue(torch.isfinite(scores).all())
        print("F_L scores sorted (unit, KL):")
        for unit_idx in torch.argsort(scores).tolist():
            print(f"  {unit_idx:2d}: {scores[unit_idx].item():.8f}")
        positive_scores = scores[scores > 1e-12]
        self.assertGreaterEqual(positive_scores.numel(), 2)
        ratio = positive_scores.max() / positive_scores.min()
        print(f"F_L max/min positive ratio: {ratio.item():.3f}")
        print(
            "F_L variation reaches one order of magnitude:",
            ratio.item() >= 10.0,
        )

    def test_dead_unit_activation_has_low_ablation_score_if_present(self):
        activations = get_activations(
            self.model,
            self.testloader,
            layer_indices=[NONLINEARITY_LAYER],
            max_samples=MAX_SAMPLES,
        )[NONLINEARITY_LAYER]
        variances = activations.var(dim=0, unbiased=False)
        print(
            "hidden activation variance min/max:",
            variances.min().item(),
            variances.max().item(),
        )
        dead_units = torch.nonzero(variances <= 1e-12).flatten()
        diagnostics = F_L_layer_diagnostics(
            self.model,
            self.x,
            self.targets,
            NONLINEARITY_LAYER,
            N_UNITS,
        )
        if dead_units.numel():
            scores = diagnostics["kl"]
            dead_scores = scores[dead_units]
            self.assertLessEqual(
                dead_scores.max().item(),
                torch.quantile(scores, 0.25).item() + 1e-6,
            )
            print("dead units and F_L:", list(zip(
                dead_units.tolist(), dead_scores.tolist()
            )))
        else:
            print("No exactly constant-zero hidden unit found on D_ref.")

        print(
            "F_L secondary delta NLL min/max:",
            diagnostics["nll_delta"].min().item(),
            diagnostics["nll_delta"].max().item(),
        )
        print(
            "F_L secondary delta accuracy min/max:",
            diagnostics["accuracy_delta"].min().item(),
            diagnostics["accuracy_delta"].max().item(),
        )

    def test_F_C_low_variance_vs_random_subspaces(self):
        activations = get_activations(
            self.model,
            self.testloader,
            layer_indices=[NONLINEARITY_LAYER],
            max_samples=MAX_SAMPLES,
        )[NONLINEARITY_LAYER]
        logprobs_orig = get_logprobs(self.model, self.x)
        mean_activation = activations.mean(dim=0)
        table = []

        for k in (1, 5, 10):
            low_variance = low_variance_directions(activations, k)
            low_variance_score = F_C_subspace(
                self.model,
                self.x,
                logprobs_orig,
                NONLINEARITY_LAYER,
                low_variance,
                mean_activation,
            )
            random_scores = []
            for seed in range(5):
                directions = random_orthonormal_directions(
                    N_UNITS, k, seed=seed
                )
                random_scores.append(
                    F_C_subspace(
                        self.model,
                        self.x,
                        logprobs_orig,
                        NONLINEARITY_LAYER,
                        directions,
                        mean_activation,
                    )
                )
            random_values = torch.tensor(random_scores)
            table.append(
                (
                    k,
                    low_variance_score,
                    random_values.mean().item(),
                    random_values.std(unbiased=True).item(),
                )
            )

        print("k | F_C low variance | F_C random mean | random std")
        for k, low_score, random_mean, random_std in table:
            print(
                f"{k:2d} | {low_score:.8f} | "
                f"{random_mean:.8f} | {random_std:.8f}"
            )
            self.assertTrue(
                torch.isfinite(
                    torch.tensor([low_score, random_mean, random_std])
                ).all()
            )
            self.assertGreaterEqual(low_score, -1e-6)
            self.assertGreaterEqual(random_mean, -1e-6)

        for k in (1, 5, 10):
            directions = random_orthonormal_directions(N_UNITS, k, seed=0)
            gram = directions @ directions.T
            self.assertTrue(
                torch.allclose(gram, torch.eye(k), atol=1e-5)
            )


if __name__ == "__main__":
    unittest.main()
