import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.src.d_ref import load_d_ref_indices
from redundancy_project.src.loader import load_checkpoint
from redundancy_project.src.measures_P import P_C_layer, P_L_layer
from redundancy_project.src.measures_R import (
    R_C_layer,
    R_L_layer,
    get_activations,
)
from redundancy_project.src.model_defs import build_model_from_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
CHECKPOINT_EPOCH = 50
TRIAL_SEED = 550
ACTIVATION_LAYER = 10
MAX_SAMPLES = 500


def summarize_off_diagonal(matrix: torch.Tensor) -> tuple:
    mask = ~torch.eye(
        matrix.shape[0], dtype=torch.bool, device=matrix.device
    )
    off_diagonal = matrix[mask]
    return off_diagonal.abs().max().item(), off_diagonal.abs().mean().item()


def main():
    with open(ZOO_ROOT / "config.json") as config_file:
        config = json.load(config_file)
    checkpoint = load_checkpoint(RUN_DIR, epoch=CHECKPOINT_EPOCH)
    model = build_model_from_checkpoint(
        config, checkpoint, seed=TRIAL_SEED
    )

    with torch.serialization.safe_globals([TensorDataset]):
        dataset = torch.load(
            ZOO_ROOT / "dataset.pt",
            map_location="cpu",
            weights_only=True,
        )
    testset = dataset["testset"]
    indices = torch.tensor(
        load_d_ref_indices(dataset_size=len(testset)), dtype=torch.long
    )
    d_ref = TensorDataset(
        testset.tensors[0].index_select(0, indices),
        testset.tensors[1].index_select(0, indices),
    )
    testloader = DataLoader(
        d_ref, batch_size=128, shuffle=False
    )

    # The output layer has no following nonlinearity; R is measured on hidden layer 9 only.
    hidden_weights = checkpoint["module_list.9.weight"]
    next_layer_weights = checkpoint["module_list.11.weight"]
    pl = P_L_layer(hidden_weights, next_layer_weights.T)
    pc = P_C_layer(hidden_weights)

    activations = get_activations(
        model,
        testloader,
        layer_indices=[ACTIVATION_LAYER],
        max_samples=MAX_SAMPLES,
    )[ACTIVATION_LAYER]
    rl = R_L_layer(activations)
    rc = R_C_layer(activations)

    pl_max, pl_mean = summarize_off_diagonal(pl)
    rl_max, rl_mean = summarize_off_diagonal(rl)
    print(f"P_L (max |off-diagonal|): {pl_max:.6f}")
    print(f"P_L (mean |off-diagonal|): {pl_mean:.6f}")
    print(f"P_C: {pc:.6f}")
    print(f"R_L (max |off-diagonal|): {rl_max:.6f}")
    print(f"R_L (mean |off-diagonal|): {rl_mean:.6f}")
    print(f"R_C: {rc:.6f}")


if __name__ == "__main__":
    main()
