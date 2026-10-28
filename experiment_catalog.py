import hashlib
import itertools
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN = PROJECT_ROOT / "experiment_design.yaml"
DEFAULT_CATALOG = PROJECT_ROOT / "results" / "experiment_catalog.json"

ARCHITECTURES = {"CNN", "CNN2", "CNN3", "Resnet18"}
OPTIMIZERS = {"adam", "sgd", "rms_prop"}
ACTIVATIONS = {"leakyrelu", "relu", "tanh", "sigmoid", "silu", "gelu"}
INITIALIZERS = {
    "uniform",
    "normal",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
}
CONDITION_FIELDS = {
    "architecture",
    "activation",
    "dropout",
    "initializer",
    "use_bias",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "momentum",
    "nesterov",
}
REQUIRED_CONDITION_FIELDS = {
    "architecture",
    "activation",
    "dropout",
    "initializer",
    "use_bias",
    "optimizer",
    "learning_rate",
    "weight_decay",
}


def _as_values(value):
    return value if isinstance(value, list) else [value]


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()


def load_plan(plan_path: Path = DEFAULT_PLAN) -> Dict:
    with open(plan_path) as plan_file:
        plan = yaml.safe_load(plan_file)
    _validate_plan(plan)
    return plan


def _validate_plan(plan: Dict) -> None:
    if not isinstance(plan, dict) or set(plan) != {"experiment"}:
        raise ValueError("plan root must contain only the 'experiment' object")
    experiment = plan["experiment"]
    required = {"name", "dataset", "training", "conditions"}
    if not isinstance(experiment, dict) or not required.issubset(experiment):
        raise ValueError(f"experiment must define {sorted(required)}")
    dataset = experiment["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "source",
        "path",
        "train_fraction",
        "subset_seed",
    }:
        raise ValueError(
            "dataset must define only source, path, train_fraction, subset_seed"
        )
    if dataset.get("source") != "cached_mnist_tensor":
        raise ValueError("only dataset source 'cached_mnist_tensor' is supported")
    if not isinstance(dataset["path"], str) or not dataset["path"]:
        raise ValueError("dataset.path must be a non-empty string")
    fraction = dataset.get("train_fraction")
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("dataset.train_fraction must be in (0, 1]")
    if not isinstance(dataset["subset_seed"], int):
        raise ValueError("dataset.subset_seed must be an integer")

    training = experiment["training"]
    if not isinstance(training, dict) or set(training) != {
        "batch_size",
        "epochs",
        "checkpoint_epochs",
        "device",
    }:
        raise ValueError(
            "training must define only batch_size, epochs, "
            "checkpoint_epochs, device; schedulers are not supported yet"
        )
    epochs = training.get("epochs")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("training.epochs must be a positive integer")
    checkpoint_epochs = training.get("checkpoint_epochs")
    if (
        not isinstance(checkpoint_epochs, list)
        or not checkpoint_epochs
        or any(
            not isinstance(epoch, int) or epoch < 1 or epoch > epochs
            for epoch in checkpoint_epochs
        )
        or len(set(checkpoint_epochs)) != len(checkpoint_epochs)
    ):
        raise ValueError(
            "checkpoint_epochs must be unique positive epochs no greater than training.epochs"
        )
    if training.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("training.device must be 'cpu' or 'cuda'")
    if (
        not isinstance(training.get("batch_size"), int)
        or isinstance(training["batch_size"], bool)
        or training["batch_size"] < 1
    ):
        raise ValueError("training.batch_size must be a positive integer")

    conditions = experiment["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("experiment.conditions must be a non-empty list")
    condition_ids = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("each condition must be an object")
        missing = REQUIRED_CONDITION_FIELDS - set(condition)
        unknown = set(condition) - CONDITION_FIELDS - {"id", "seeds"}
        if missing or unknown:
            raise ValueError(
                f"condition has missing fields {sorted(missing)} "
                f"and unsupported fields {sorted(unknown)}"
            )
        condition_id = condition.get("id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            raise ValueError("each condition needs a non-empty string id")
        condition_ids.append(condition_id)
        if not isinstance(condition.get("seeds"), list) or not condition["seeds"]:
            raise ValueError(f"{condition_id}: seeds must be a non-empty list")
        if any(not isinstance(seed, int) for seed in condition["seeds"]):
            raise ValueError(f"{condition_id}: every seed must be an integer")
        if len(set(condition["seeds"])) != len(condition["seeds"]):
            raise ValueError(f"{condition_id}: seeds must be unique")

        for field, allowed in (
            ("architecture", ARCHITECTURES),
            ("activation", ACTIVATIONS),
            ("optimizer", OPTIMIZERS),
            ("initializer", INITIALIZERS),
        ):
            unsupported = set(_as_values(condition[field])) - allowed
            if unsupported:
                raise ValueError(
                    f"{condition_id}: unsupported {field} value(s): "
                    f"{sorted(unsupported)}"
                )
        if "Resnet18" in _as_values(condition["architecture"]):
            raise ValueError(
                "Resnet18 is present in the source but not yet supported by "
                "this MNIST experiment runner"
            )
        if any(
            value is False for value in _as_values(condition["use_bias"])
        ):
            raise ValueError(
                "use_bias=false is unsupported: existing model constructors "
                "do not reliably disable biases"
            )
        if (
            "rms_prop" in _as_values(condition["optimizer"])
            and "momentum" not in condition
        ):
            raise ValueError("rms_prop requires an explicit momentum value")

        for field in ("dropout", "learning_rate", "weight_decay", "momentum"):
            if field in condition and any(
                not isinstance(value, (int, float))
                for value in _as_values(condition[field])
            ):
                raise ValueError(f"{condition_id}: {field} values must be numeric")
        for dropout in _as_values(condition["dropout"]):
            if not 0 <= dropout < 1:
                raise ValueError(f"{condition_id}: dropout must be in [0, 1)")
        for learning_rate in _as_values(condition["learning_rate"]):
            if learning_rate <= 0:
                raise ValueError(f"{condition_id}: learning_rate must be positive")
        for weight_decay in _as_values(condition["weight_decay"]):
            if weight_decay < 0:
                raise ValueError(f"{condition_id}: weight_decay cannot be negative")
        if any(
            not isinstance(value, bool)
            for value in _as_values(condition["use_bias"])
        ):
            raise ValueError(f"{condition_id}: use_bias must be boolean")
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("condition ids must be unique")


def expand_plan(plan: Dict) -> List[Dict]:
    experiment = plan["experiment"]
    expanded_runs = []
    for condition in experiment["conditions"]:
        condition_fields = sorted(CONDITION_FIELDS & condition.keys())
        values = [_as_values(condition[field]) for field in condition_fields]
        for combination in itertools.product(*values):
            condition_config = dict(zip(condition_fields, combination))
            full_condition = {
                **condition_config,
                "dataset": experiment["dataset"],
                "training": experiment["training"],
            }
            canonical = json.dumps(
                full_condition, sort_keys=True, separators=(",", ":")
            )
            condition_hash = hashlib.sha256(canonical.encode()).hexdigest()[:10]
            resolved_condition_id = (
                f"{_slug(condition['id'])}-{condition_hash}"
            )
            for seed in condition["seeds"]:
                run_hash = hashlib.sha256(
                    f"{resolved_condition_id}:{seed}".encode()
                ).hexdigest()[:10]
                run_config = {
                    **condition_config,
                    "seed": seed,
                    "dataset": dict(experiment["dataset"]),
                    "training": dict(experiment["training"]),
                }
                expanded_runs.append(
                    {
                        "run_id": f"run-{run_hash}",
                        "condition_id": resolved_condition_id,
                        "seed": seed,
                        "config": run_config,
                    }
                )
    ids = [run["run_id"] for run in expanded_runs]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded plan contains duplicate run IDs")
    return expanded_runs


def build_catalog(plan_path: Path, catalog_path: Path) -> List[Dict]:
    if catalog_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing experiment catalog: {catalog_path}"
        )
    plan = load_plan(plan_path)
    runs = expand_plan(plan)
    catalog = {
        "experiment_name": plan["experiment"]["name"],
        "run_count": len(runs),
        "runs": runs,
    }
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(catalog_path, "x") as catalog_file:
        json.dump(catalog, catalog_file, indent=2)
        catalog_file.write("\n")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand an experiment YAML plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    runs = build_catalog(args.plan, args.output)
    print(f"Expanded {len(runs)} runs into {args.output}")
