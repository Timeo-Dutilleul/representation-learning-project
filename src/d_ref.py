import json
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZOO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "mnist_zoo_raw"
    / "tune_zoo_mnist_uniform"
)
D_REF_PATH = PROJECT_ROOT / "data" / "d_ref_indices.json"
D_REF_SIZE = 500


def load_d_ref_indices(
    path: Path = D_REF_PATH, dataset_size: int = 10000
) -> List[int]:
    """Load the frozen first 500 test-set indices, creating them only once."""
    if dataset_size < D_REF_SIZE:
        raise ValueError(
            f"test set has {dataset_size} items, fewer than D_ref size {D_REF_SIZE}"
        )

    if path.exists():
        with open(path) as index_file:
            indices = json.load(index_file)
    else:
        indices = list(range(D_REF_SIZE))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "x") as index_file:
            json.dump(indices, index_file, indent=2)
            index_file.write("\n")

    if (
        not isinstance(indices, list)
        or len(indices) != D_REF_SIZE
        or any(not isinstance(index, int) for index in indices)
        or len(set(indices)) != D_REF_SIZE
        or any(index < 0 or index >= dataset_size for index in indices)
    ):
        raise ValueError(f"{path} must contain {D_REF_SIZE} unique valid indices")
    return indices
