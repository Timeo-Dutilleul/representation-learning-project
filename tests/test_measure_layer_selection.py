import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.run_pipeline import (
    _find_measurement_layers,
    _validate_layer_activations,
)
from redundancy_project.src.measures_F import F_L_layer
from redundancy_project.src.measures_R import get_activations
from redundancy_project.src.measures_R import activation_spectral_diagnostics
from redundancy_project.src.model_defs import NNmodule


ACTIVATIONS = {
    "leakyrelu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def make_model(architecture, activation):
    config = {
        "model::type": architecture,
        "model::channels_in": 1,
        "model::o_dim": 10,
        "model::nlin": activation,
        "model::dropout": 0.0,
        "model::init_type": "kaiming_uniform",
        "optim::optimizer": "sgd",
        "optim::lr": 0.001,
        "optim::wd": 0.0,
        "optim::momentum": 0.9,
        "optim::nesterov": False,
        "optim::scheduler": None,
        "training::task": "classification",
        "training::loss": "nll",
        "training::epochs_train": 1,
    }
    return NNmodule(config, cuda=False, seed=17, verbosity=0)


class MeasurementLayerSelectionTests(unittest.TestCase):
    def test_d_ref_constant_unit_has_zero_ablation_discrepancy(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.module_list = nn.ModuleList(
                    [
                        nn.Linear(2, 2),
                        nn.ReLU(),
                        nn.Linear(2, 2, bias=False),
                    ]
                )
                with torch.no_grad():
                    self.module_list[0].weight.copy_(
                        torch.tensor([[0.0, 0.0], [1.0, 1.0]])
                    )
                    self.module_list[0].bias.copy_(torch.tensor([1.0, 0.0]))
                    self.module_list[0].weight[1].zero_()
                    self.module_list[2].weight.copy_(
                        torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
                    )

            def forward(self, inputs):
                for module in self.module_list:
                    inputs = module(inputs)
                return F.log_softmax(inputs, dim=1)

        class ModelAdapter:
            def __init__(self):
                self.model = TinyModel()

            def forward(self, inputs):
                return self.model(inputs)

        model = ModelAdapter()
        inputs = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]])
        scores = F_L_layer(model, inputs, nonlin_layer_idx=1, n_units=2)
        self.assertGreater(scores[0].item(), 0.0)
        self.assertEqual(scores[1].item(), 0.0)

    def test_constant_hidden_layer_is_classified_without_raising(self):
        activations = torch.zeros(5, 20)
        self.assertEqual(
            _validate_layer_activations(activations),
            (list(range(20)), list(range(20))),
        )
        structure = activation_spectral_diagnostics(activations)
        self.assertEqual(structure["representation_status"], "all_constant")
        self.assertEqual(structure["constant_unit_count"], 20)
        self.assertEqual(structure["nonconstant_unit_count"], 0)
        self.assertEqual(structure["numerical_rank"], 0)
        self.assertEqual(structure["nullity"], 20)
        self.assertEqual(structure["total_activation_variance"], 0.0)

    def test_constant_units_are_counted(self):
        activations = torch.randn(5, 3)
        activations[:, 1] = 0
        self.assertEqual(_validate_layer_activations(activations), ([1], [1]))

    def test_single_active_unit_is_classified_for_pairwise_measurement(self):
        activations = torch.zeros(5, 3)
        activations[:, 0] = torch.arange(5, dtype=torch.float32)
        self.assertEqual(
            _validate_layer_activations(activations),
            ([1, 2], [1, 2]),
        )
        structure = activation_spectral_diagnostics(activations)
        self.assertEqual(
            structure["representation_status"], "single_nonconstant_unit"
        )
        self.assertEqual(structure["numerical_rank"], 1)

    def test_proportional_nonconstant_units_have_rank_one_covariance(self):
        values = torch.arange(6, dtype=torch.float32)
        activations = torch.stack((values, 2 * values, 3 * values), dim=1)
        structure = activation_spectral_diagnostics(activations)
        self.assertEqual(structure["nonconstant_unit_count"], 3)
        self.assertEqual(structure["numerical_rank"], 1)
        self.assertEqual(structure["nullity"], 2)
        self.assertEqual(structure["representation_status"], "rank_deficient")
        self.assertGreater(structure["total_activation_variance"], 0.0)

    def test_cnn2_and_cnn3_select_twenty_unit_hidden_layer_for_all_activations(self):
        labels = torch.zeros(4, dtype=torch.long)

        for architecture in ("CNN2", "CNN3"):
            input_size = 32 if architecture == "CNN3" else 28
            images = torch.randn(4, 1, input_size, input_size)
            loader = DataLoader(TensorDataset(images, labels), batch_size=4)
            for activation, activation_type in ACTIVATIONS.items():
                with self.subTest(
                    architecture=architecture, activation=activation
                ):
                    model = make_model(architecture, activation)
                    layer_idx, output_idx, activation_idx, width = (
                        _find_measurement_layers(model)
                    )
                    modules = list(model.model.module_list)
                    self.assertEqual(width, 20)
                    self.assertEqual(modules[layer_idx].out_features, 20)
                    self.assertEqual(modules[output_idx].out_features, 10)
                    self.assertIsInstance(
                        modules[activation_idx], activation_type
                    )

                    captured = get_activations(
                        model,
                        loader,
                        layer_indices=[activation_idx],
                        max_samples=4,
                    )[activation_idx]
                    self.assertEqual(tuple(captured.shape), (4, 20))
                    self.assertTrue(torch.isfinite(captured).all())
                    _validate_layer_activations(captured)


if __name__ == "__main__":
    unittest.main()
