import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from redundancy_project.src.metric_definitions import (
    F_C_INTERVENTION,
    METRIC_DEFINITION_VERSION,
    P_C_DEFINITION,
)
from redundancy_project.src.temporal_measurements import measure_checkpoint
from redundancy_project.src import temporal_measurements


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.ModuleList(
            [
                nn.Linear(4, 20),
                nn.Tanh(),
                nn.Linear(20, 10),
                nn.LogSoftmax(dim=1),
            ]
        )

    def forward(self, inputs):
        for layer in self.module_list:
            inputs = layer(inputs)
        return inputs


class _ModelWrapper:
    def __init__(self):
        self.model = _TinyClassifier()

    def forward(self, inputs):
        return self.model(inputs)


class TemporalMeasurementSmokeTests(unittest.TestCase):
    def test_zero_variance_reference_sample_keeps_independent_measures(self):
        model = _ModelWrapper()
        fixed_inputs = torch.zeros(6, 4)
        targets = torch.zeros(6, dtype=torch.long)
        samples = (fixed_inputs, targets)
        with patch.object(
            temporal_measurements,
            "random_orthonormal_directions",
            side_effect=AssertionError("random controls must not be generated"),
        ):
            result = measure_checkpoint(
                model,
                samples,
                samples,
                fc_subspace_dims=(1, 20),
            )

        for split_name in ("train", "validation"):
            split = result[split_name]
            self.assertEqual(split["representation_status"], "all_constant")
            self.assertEqual(split["constant_unit_count"], 20)
            self.assertEqual(split["nonconstant_unit_count"], 0)
            self.assertEqual(split["total_activation_variance"], 0.0)
            self.assertEqual(split["numerical_rank"], 0)
            self.assertEqual(split["nullity"], 20)
            self.assertIsNone(split["R_L"])
            self.assertIsNone(split["R_C"])
            self.assertIsNone(split["F_C"])
            self.assertEqual(len(split["F_L"]), 20)
        json.dumps(result, allow_nan=False)

    def test_rank_one_with_multiple_nonconstant_units_measures_nullspace_fc(self):
        torch.manual_seed(19)
        model = _ModelWrapper()
        with torch.no_grad():
            hidden = model.model.module_list[0]
            hidden.weight.zero_()
            hidden.bias.zero_()
            hidden.weight[0].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
            hidden.weight[1].copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        inputs = torch.randn(12, 4)
        labels = torch.randint(0, 10, (12,))
        result = measure_checkpoint(
            model,
            (inputs, labels),
            (inputs, labels),
            fc_subspace_dims=(1, 19, 20),
        )

        for split_name in ("train", "validation"):
            split = result[split_name]
            self.assertEqual(split["nonconstant_unit_count"], 2)
            self.assertEqual(split["representation_status"], "rank_deficient")
            self.assertEqual(split["numerical_rank"], 1)
            self.assertEqual(split["nullity"], 19)
            self.assertAlmostEqual(split["R_C"], 1.0, places=6)
            self.assertEqual(split["R_L"][0][1], 1.0)
            nullspace_score = split["F_C"]["1"]["low_variance"]
            self.assertIsInstance(nullspace_score, float)
            self.assertAlmostEqual(nullspace_score, 0.0, delta=1e-7)
            self.assertTrue(
                split["F_C"]["1"]["contains_zero_variance_directions"]
            )
            self.assertFalse(
                split["F_C"]["1"][
                    "zero_variance_directions_individually_identifiable"
                ]
            )
            self.assertEqual(
                split["F_C"]["1"][
                    "nullspace_ablation_effect_on_reference_sample"
                ],
                "zero_to_numerical_precision",
            )
            self.assertFalse(
                split["zero_variance_directions_individually_identifiable"]
            )
            self.assertEqual(
                split["centered_nullspace_ablation_effect"],
                "zero_to_numerical_precision",
            )
            self.assertTrue(
                split["F_C"]["1"]["low_variance"] is not None
            )
            self.assertEqual(len(split["F_C"]["1"]["random"]), 50)
            self.assertIsInstance(split["F_C"]["19"]["low_variance"], float)
            self.assertIsInstance(split["F_C"]["20"]["low_variance"], float)
        json.dumps(result, allow_nan=False)

    def test_emits_versioned_centered_fc_schema_for_both_fixed_splits(self):
        torch.manual_seed(11)
        model = _ModelWrapper()
        train = (torch.randn(8, 4), torch.randint(0, 10, (8,)))
        validation = (torch.randn(7, 4), torch.randint(0, 10, (7,)))
        result = measure_checkpoint(
            model,
            train,
            validation,
            fc_subspace_dims=(1, 20),
            train_sample_identity={"split": "train", "size": 8},
            validation_sample_identity={
                "split": "validation",
                "size": 7,
            },
        )

        self.assertEqual(
            result["metric_definition_version"], METRIC_DEFINITION_VERSION
        )
        self.assertEqual(result["p_c_definition"], P_C_DEFINITION)
        self.assertEqual(result["fc_intervention"], F_C_INTERVENTION)
        self.assertGreaterEqual(result["P_C"], 1.0)
        self.assertLessEqual(
            result["P_C"], result["activation_width"] + 1e-6
        )
        self.assertEqual(len(result["P_L_in"]), 20)
        self.assertEqual(len(result["P_L_out"]), 20)
        for combined_row, incoming_row, outgoing_row in zip(
            result["P_L"], result["P_L_in"], result["P_L_out"]
        ):
            for combined, incoming, outgoing in zip(
                combined_row, incoming_row, outgoing_row
            ):
                self.assertAlmostEqual(
                    combined, (incoming + outgoing) / 2, places=6
                )
        for split_name, sample_count in (("train", 8), ("validation", 7)):
            split = result[split_name]
            self.assertEqual(split["sample_count"], sample_count)
            self.assertEqual(split["reference_sample"]["size"], sample_count)
            self.assertAlmostEqual(
                split["F_C"]["0"]["R_C_remaining_lowvar"],
                split["R_C"],
                delta=1e-5,
            )
            self.assertEqual(len(split["reference_activation_mean"]), 20)
            self.assertEqual(len(split["activation_covariance_eigenvalues"]), 20)
            self.assertEqual(split["F_C"]["0"]["low_variance"], 0.0)
            self.assertEqual(len(split["F_C"]["1"]["random"]), 50)
            self.assertEqual(split["F_C"]["20"]["R_C_remaining_lowvar"], 0.0)
            self.assertEqual(split["F_C"]["20"]["V_remaining_lowvar"], 0.0)
            self.assertGreaterEqual(split["R_C"], 1.0)
            self.assertLessEqual(split["R_C"], 20.0 + 1e-6)

            full_dimension_low_variance = split["F_C"]["20"]["low_variance"]
            full_dimension_random = split["F_C"]["20"]["random"]
            self.assertTrue(
                all(
                    abs(score - full_dimension_low_variance) <= 1e-5
                    for score in full_dimension_random
                )
            )

        raw_scores = result["validation"]["F_C"]["1"]["random"]
        with tempfile.TemporaryDirectory() as temp_dir:
            serialized_scores = Path(temp_dir) / "random_scores.json"
            serialized_scores.write_text(json.dumps(raw_scores), encoding="utf-8")
            restored_scores = json.loads(
                serialized_scores.read_text(encoding="utf-8")
            )
        self.assertEqual(len(restored_scores), 50)
        self.assertEqual(restored_scores, raw_scores)


if __name__ == "__main__":
    unittest.main()
