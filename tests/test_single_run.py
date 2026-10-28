import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.src.loader import (
    load_checkpoint,
    load_run_config,
    load_run_result,
)
from redundancy_project.src.model_defs import build_model_from_checkpoint, evaluate
from redundancy_project.src.weights_utils import (
    vector_to_checkpoint,
    vectorize_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZOO_ROOT = (
    PROJECT_ROOT
    / "redundancy_project"
    / "data"
    / "mnist_zoo_raw"
    / "tune_zoo_mnist_uniform"
)
RUN_DIR = (
    ZOO_ROOT
    / "NN_tune_trainable_c0371_00549_549_seed=550_2021-07-02_19-38-33"
)
LAYER_LIST = ((0, "conv"), (3, "conv"), (6, "conv"), (9, "linear"), (11, "linear"))
CHECKPOINT_EPOCH = 50
TRIAL_SEED = 550


class SingleRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_run_config(ZOO_ROOT)
        cls.checkpoint = load_checkpoint(RUN_DIR, epoch=CHECKPOINT_EPOCH)

    def test_vectorize_roundtrip(self):
        vector = vectorize_checkpoint(
            self.checkpoint, LAYER_LIST, use_bias=True
        )
        reconstructed = vector_to_checkpoint(
            self.checkpoint, vector, LAYER_LIST, use_bias=True
        )
        for key, original in self.checkpoint.items():
            torch.testing.assert_close(reconstructed[key], original)

    def test_reconstructed_model_matches_stored_accuracy(self):
        model = build_model_from_checkpoint(
            self.config, self.checkpoint, seed=TRIAL_SEED
        )
        with torch.serialization.safe_globals([TensorDataset]):
            dataset = torch.load(
                ZOO_ROOT / "dataset.pt",
                map_location="cpu",
                weights_only=True,
            )
        testset = dataset["testset"]
        testloader = DataLoader(
            testset, batch_size=len(testset), shuffle=False
        )

        _, accuracy = evaluate(model, testloader)
        results = load_run_result(RUN_DIR)
        epoch_result = next(
            result
            for result in results
            if result["training_iteration"] == CHECKPOINT_EPOCH
        )
        self.assertAlmostEqual(
            accuracy, epoch_result["test_acc"], delta=1e-3
        )


if __name__ == "__main__":
    unittest.main()
