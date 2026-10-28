import json
from pathlib import Path

import torch


def load_checkpoint(run_dir: Path, epoch: int) -> dict:
    """Load a single checkpoint from a Model Zoos trial directory."""
    ckpt_path = run_dir / f"checkpoint_{epoch:06d}" / "checkpoints"
    return torch.load(str(ckpt_path), map_location="cpu")


def load_run_config(zoo_root: Path) -> dict:
    """Load the config.json at the root of a zoo."""
    with open(zoo_root / "config.json") as f:
        return json.load(f)


def load_run_result(run_dir: Path) -> list:
    """Load the per-epoch JSON records from a trial's result.json."""
    results = []
    with open(run_dir / "result.json") as f:
        for line in f:
            results.append(json.loads(line))
    return results
