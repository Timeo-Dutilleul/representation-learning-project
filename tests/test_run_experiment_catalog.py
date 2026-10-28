import io
import json
import re
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.utils.data import TensorDataset

from redundancy_project import run_experiment_catalog


class RunExperimentCatalogTests(unittest.TestCase):
    def test_catalog_scopes_raw_measurements_to_output_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "experiment-unique"
            output_root = root / "runs"
            output_root.mkdir(parents=True)
            run_record = {
                "run_id": "run-stable-id",
                "config": {
                    "training": {
                        "epochs": 3,
                        "checkpoint_epochs": [1],
                    }
                },
            }
            catalog_path = root / "catalog.json"
            catalog_path.write_text(
                json.dumps({"runs": [run_record]}), encoding="utf-8"
            )
            run_dir = output_root / run_record["run_id"]
            run_dir.mkdir()

            output = io.StringIO()
            with redirect_stdout(output):
                with ExitStack() as stack:
                    train_one_run = stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "train_one_run",
                            return_value=run_dir,
                        )
                    )
                    compute_measures = stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "compute_all_measures",
                            return_value={"summaries": {}},
                        )
                    )
                    stack.enter_context(
                        patch.object(run_experiment_catalog, "write_dashboard")
                    )
                    run_experiment_catalog.train_catalog(
                        catalog_path=catalog_path,
                        output_root=output_root,
                        temporal_dashboard=True,
                        measurement_every=2,
                        fc_subspace_dims=(2, 7),
                    )

            self.assertEqual(
                train_one_run.call_args.kwargs["fc_subspace_dims"], (2, 7)
            )
            self.assertEqual(
                compute_measures.call_args.kwargs["fc_subspace_dims"], (2, 7)
            )
            self.assertEqual(
                compute_measures.call_args.kwargs["raw_results_root"],
                root / "raw",
            )
            self.assertEqual(compute_measures.call_args.kwargs["epoch"], 3)
            self.assertTrue((run_dir / "measurements.json").is_file())
            self.assertRegex(
                output.getvalue(),
                r"Total run 1/1 — run-stable-id: \d+\.\d+s",
            )
            self.assertRegex(
                output.getvalue(),
                r"Total execution time for all runs \(1 run\(s\)\): "
                r"\d+\.\d+s",
            )

    def test_temporal_run_measures_only_selected_epochs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "cached.pt").touch()
            samples = TensorDataset(
                torch.arange(8, dtype=torch.float32).reshape(4, 2),
                torch.zeros(4, dtype=torch.long),
            )
            dataset = {
                "trainset": samples,
                "valset": samples,
                "testset": samples,
            }
            config = {
                "architecture": "CNN",
                "activation": "relu",
                "dropout": 0.0,
                "initializer": "uniform",
                "use_bias": True,
                "optimizer": "SGD",
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "dataset": {
                    "path": "cached.pt",
                    "train_fraction": 1.0,
                    "subset_seed": 11,
                },
                "training": {
                    "epochs": 3,
                    "checkpoint_epochs": [3],
                    "batch_size": 2,
                    "device": "cpu",
                },
            }
            run_record = {
                "run_id": "run-temporal-test",
                "condition_id": "condition-test",
                "seed": 5,
                "config": config,
            }
            model = Mock()
            model.model.state_dict.return_value = {"weight": torch.tensor(1)}
            model.train_epoch.return_value = (0.5, 0.8)
            model.test_epoch.return_value = (0.6, 0.7)
            measure_result = {
                "P_L": [[1.0]],
                "P_C": 1.0,
                "train": {"R_L": [[1.0]], "R_C": 1.0, "F_L": [0.0], "F_C": {}},
                "validation": {
                    "R_L": [[1.0]],
                    "R_C": 1.0,
                    "F_L": [0.0],
                    "F_C": {},
                },
            }

            progress_output = io.StringIO()
            with redirect_stdout(progress_output):
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "PROJECT_ROOT",
                            project_root,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "_load_cached_dataset",
                            return_value=dataset,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "prepare_dataset_for_architecture",
                            side_effect=lambda value, architecture: value,
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "NNmodule",
                            return_value=model,
                        )
                    )
                    measure_checkpoint = stack.enter_context(
                        patch.object(
                            run_experiment_catalog,
                            "measure_checkpoint",
                            return_value=measure_result,
                        )
                    )
                    run_dir = run_experiment_catalog.train_one_run(
                        run_record,
                        output_root=project_root / "runs",
                        temporal_dashboard=True,
                        measurement_samples=2,
                        measurement_every=2,
                        fc_subspace_dims=(2, 7),
                    )

            with open(run_dir / "temporal_measurements.jsonl") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(
                [record["training_iteration"] for record in records], [2, 3]
            )
            self.assertEqual(measure_checkpoint.call_count, 2)
            self.assertEqual(
                [call.args[3] for call in measure_checkpoint.call_args_list],
                [(2, 7), (2, 7)],
            )
            first_call, second_call = measure_checkpoint.call_args_list
            for first_split, second_split in zip(
                first_call.args[1:3], second_call.args[1:3]
            ):
                for first_tensor, second_tensor in zip(
                    first_split, second_split
                ):
                    self.assertTrue(torch.equal(first_tensor, second_tensor))
            self.assertEqual(
                first_call.kwargs["train_sample_identity"],
                second_call.kwargs["train_sample_identity"],
            )
            self.assertEqual(
                first_call.kwargs["validation_sample_identity"],
                second_call.kwargs["validation_sample_identity"],
            )
            with open(run_dir / "run_metadata.json") as stream:
                run_metadata = json.load(stream)
            sample_metadata = run_metadata["temporal_measurement_samples"]
            self.assertEqual(
                sample_metadata["train_dataset_indices"],
                [
                    run_metadata["train_subset_indices"][position]
                    for position in sample_metadata[
                        "train_positions_in_training_subset"
                    ]
                ],
            )
            self.assertEqual(
                sample_metadata["validation_split_indices"],
                sample_metadata["validation_positions"],
            )
            self.assertEqual(
                [
                    record["training_iteration"]
                    for record in records
                ],
                sample_metadata["measurement_epochs"],
            )
            with open(run_dir / "result.json") as stream:
                metrics = [json.loads(line) for line in stream]
            self.assertIsNone(metrics[0]["train_eval_loss"])
            self.assertIsNotNone(metrics[1]["train_eval_loss"])
            self.assertIsNotNone(metrics[2]["train_eval_loss"])
            self.assertIn(
                "Training run-temporal-test: epochs 1-2/3... done in",
                progress_output.getvalue(),
            )
            self.assertIn(
                "Temporal measurements for run-temporal-test at epoch 2... done in",
                progress_output.getvalue(),
            )
            self.assertIn(
                "Training run-temporal-test: epochs 3-3/3... done in",
                progress_output.getvalue(),
            )


if __name__ == "__main__":
    unittest.main()
