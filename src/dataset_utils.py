import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset


def prepare_dataset_for_architecture(dataset: dict, architecture: str) -> dict:
    """Apply the fixed input geometry required by a supported MNIST CNN."""
    expected_size = 32 if architecture == "CNN3" else 28
    prepared = dict(dataset)
    for split_name in ("trainset", "valset", "testset"):
        split = prepared.get(split_name)
        if split is None:
            continue
        if not isinstance(split, TensorDataset) or len(split.tensors) != 2:
            raise TypeError(
                f"{split_name} must be a two-tensor classification dataset"
            )
        images, labels = split.tensors
        if (
            images.ndim != 4
            or images.shape[1] != 1
            or images.shape[2] != images.shape[3]
        ):
            raise ValueError(
                f"{split_name} images must have shape (N, 1, H, H)"
            )
        size = images.shape[-1]
        if size == expected_size:
            continue
        if architecture == "CNN3" and size == 28:
            images = F.pad(images, (2, 2, 2, 2))
        else:
            raise ValueError(
                f"{architecture} expects {expected_size}x{expected_size} "
                f"inputs; {split_name} contains {size}x{size} images"
            )
        prepared[split_name] = TensorDataset(images, labels)
    return prepared
