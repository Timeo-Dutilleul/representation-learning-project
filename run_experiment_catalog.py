import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from redundancy_project.experiment_catalog import DEFAULT_CATALOG
from redundancy_project.temporal_dashboard import write_dashboard
from redundancy_project.src.model_defs import NNmodule
from redundancy_project.src.temporal_measurements import (
    get_temporal_measurement_epochs,
    measure_checkpoint,
)
from redundancy_project.run_pipeline import compute_all_measures
from redundancy_project.src.dataset_utils import (
    prepare_dataset_for_architecture,
)
from redundancy_project.src.measures_F import (
    DEFAULT_FC_SUBSPACE_DIMS,
    validate_subspace_dimensions,
)
from redundancy_project.src.metric_definitions import (
    F_C_INTERVENTION,
    METRIC_DEFINITION_VERSION,
    P_C_DEFINITION,
    REFERENCE_SAMPLE_POLICY,
    ordered_sample_identity,
)
from redundancy_project.src.progress import format_duration, progress_step


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "generated_runs"


def _load_cached_dataset(dataset_path: Path) -> Dict:
    with torch.serialization.safe_globals([TensorDataset]):
        return torch.load(
            dataset_path,
            map_location="cpu",
            weights_only=True,
        )


def _resolve_nn_config(run_config: Dict, epochs: int, device: str) -> Dict:
    optimizer = run_config["optimizer"]
    nn_config = {
        "model::type": run_config["architecture"],
        "model::channels_in": 1,
        "model::o_dim": 10,
        "model::nlin": run_config["activation"],
        "model::dropout": run_config["dropout"],
        "model::init_type": run_config["initializer"],
        "model::use_bias": run_config["use_bias"],
        "optim::optimizer": optimizer,
        "optim::lr": run_config["learning_rate"],
        "optim::wd": run_config["weight_decay"],
        "optim::momentum": run_config.get("momentum", 0.0),
        "optim::nesterov": run_config.get("nesterov", False),
        "optim::scheduler": None,
        "training::task": "classification",
        "training::epochs_train": epochs,
        "training::start_epoch": 1,
        "training::batchsize": run_config["training"]["batch_size"],
        "training::loss": "nll",
        "cuda": device == "cuda",
    }
    return nn_config


def _selected_checkpoint_epochs(
    config: Dict, temporal_dashboard: bool, measurement_every: int
):
    checkpoint_epochs = set(config["training"]["checkpoint_epochs"])
    if temporal_dashboard:
        checkpoint_epochs.update(
            get_temporal_measurement_epochs(
                config["training"]["epochs"], measurement_every
            )
        )
    return checkpoint_epochs


def _select_training_subset(trainset, fraction: float, seed: int):
    count = max(1, int(len(trainset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(trainset), generator=generator)[:count].tolist()
    return Subset(trainset, indices), indices


def _fixed_sample(dataset, sample_count: int, seed: int):
    count = min(sample_count, len(dataset))
    if count < 2:
        raise ValueError("temporal measurements need at least two examples")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    examples = [dataset[index] for index in indices]
    inputs = torch.stack([example[0] for example in examples])
    labels = torch.stack([example[1] for example in examples])
    return (inputs, labels), indices


def train_one_run(
    run_record: Dict,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    temporal_dashboard: bool = False,
    measurement_samples: int = 500,
    measurement_every: int = 1,
    fc_subspace_dims=DEFAULT_FC_SUBSPACE_DIMS,
) -> Path:
    run_id = run_record["run_id"]
    config = run_record["config"]
    output_dir = output_root / run_id
    if output_dir.exists():
        raise FileExistsError(
            f"run output already exists; refusing to overwrite: {output_dir}"
        )

    dataset_path = PROJECT_ROOT / config["dataset"]["path"]
    with progress_step(f"Loading data for {run_id}"):
        if not dataset_path.is_file():
            raise FileNotFoundError(f"cached dataset not found: {dataset_path}")
        dataset = prepare_dataset_for_architecture(
            _load_cached_dataset(dataset_path), config["architecture"]
        )
        trainset, subset_indices = _select_training_subset(
            dataset["trainset"],
            config["dataset"]["train_fraction"],
            config["dataset"]["subset_seed"],
        )
        valset = dataset.get("valset")
        if temporal_dashboard and valset is None:
            raise ValueError(
                "temporal dashboard requires a validation split in the dataset"
            )
        testset = dataset["testset"]
        if temporal_dashboard:
            measurement_epochs = get_temporal_measurement_epochs(
                config["training"]["epochs"], measurement_every
            )
            train_samples, train_sample_indices = _fixed_sample(
                trainset, measurement_samples, config["dataset"]["subset_seed"]
            )
            validation_samples, validation_sample_indices = _fixed_sample(
                valset, measurement_samples, config["dataset"]["subset_seed"] + 1
            )
        else:
            measurement_epochs = []
            train_samples = validation_samples = None
            train_sample_indices = validation_sample_indices = []

    requested_device = config["training"].get("device", "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("plan requested CUDA, but CUDA is unavailable")
    device = requested_device

    nn_config = _resolve_nn_config(
        config, config["training"]["epochs"], device
    )
    output_dir.mkdir(parents=True)
    effective_record = {
        "run_id": run_id,
        "condition_id": run_record["condition_id"],
        "seed": run_record["seed"],
        "requested_config": config,
        "effective_model_training_config": nn_config,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_preprocessing": (
            "zero_pad_2px_to_32x32"
            if config["architecture"] == "CNN3"
            else "identity_28x28"
        ),
        "dataset_seed": dataset.get("dataset_seed"),
        "train_subset_size": len(trainset),
        "train_subset_indices": subset_indices,
        "temporal_measurement_samples": (
            {
                "train_count": len(train_sample_indices),
                "train_positions_in_training_subset": train_sample_indices,
                "train_dataset_indices": [
                    subset_indices[position]
                    for position in train_sample_indices
                ],
                "train_identity": ordered_sample_identity(
                    "train",
                    train_sample_indices,
                    "positions_in_fixed_training_subset",
                ),
                "validation_count": len(validation_sample_indices),
                "validation_positions": validation_sample_indices,
                "validation_split_indices": validation_sample_indices,
                "validation_identity": ordered_sample_identity(
                    "validation",
                    validation_sample_indices,
                    "positions_in_fixed_validation_split",
                ),
                "measurement_epochs": measurement_epochs,
                "measurement_every": measurement_every,
            }
            if temporal_dashboard
            else None
        ),
        "fc_subspace_dims": list(fc_subspace_dims),
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "p_c_definition": P_C_DEFINITION,
        "fc_intervention": F_C_INTERVENTION,
        "reference_sample_policy": REFERENCE_SAMPLE_POLICY,
        "device": device,
    }
    with open(output_dir / "run_metadata.json", "x") as metadata_file:
        json.dump(effective_record, metadata_file, indent=2)
        metadata_file.write("\n")
    with open(output_dir / "config.json", "x") as config_file:
        json.dump(nn_config, config_file, indent=2)
        config_file.write("\n")

    with progress_step(f"Initializing model for {run_id}"):
        model = NNmodule(
            config=nn_config,
            cuda=device == "cuda",
            seed=run_record["seed"],
            verbosity=0,
        )
    generator = torch.Generator().manual_seed(run_record["seed"])
    trainloader = DataLoader(
        trainset,
        batch_size=nn_config["training::batchsize"],
        shuffle=True,
        generator=generator,
    )
    validation_loader = (
        DataLoader(
            valset,
            batch_size=256,
            shuffle=False,
        )
        if valset is not None
        else None
    )
    testloader = DataLoader(testset, batch_size=256, shuffle=False)
    checkpoint_epochs = _selected_checkpoint_epochs(
        config, temporal_dashboard, measurement_every
    )
    final_measurement_epoch = max(checkpoint_epochs)
    metrics_path = output_dir / "result.json"
    temporal_path = output_dir / "temporal_measurements.jsonl"
    if temporal_dashboard:
        temporal_file = open(temporal_path, "x")
    else:
        temporal_file = None
    with open(metrics_path, "x") as metrics_file:
        try:
            total_epochs = config["training"]["epochs"]
            progress_interval = 5
            checkpoint_epochs_in_range = {
                epoch for epoch in checkpoint_epochs if 1 <= epoch <= total_epochs
            }
            progress_boundaries = (
                set(range(progress_interval, total_epochs + 1, progress_interval))
                | set(measurement_epochs)
                | checkpoint_epochs_in_range
                | {total_epochs}
            )
            block_start = 1
            for block_end in sorted(progress_boundaries):
                with progress_step(
                    f"Training {run_id}: epochs {block_start}-{block_end}/{total_epochs}"
                ):
                    for epoch in range(block_start, block_end + 1):
                        train_loss, train_accuracy = model.train_epoch(
                            trainloader,
                            epoch,
                            idx_out=max(1, len(trainloader) + 1),
                        )
                        validation_loss, validation_accuracy = (
                            model.test_epoch(validation_loader, epoch)
                            if validation_loader is not None
                            else (None, None)
                        )
                        if temporal_dashboard and epoch in measurement_epochs:
                            diagnostic_start = time.perf_counter()
                            train_eval_loss, train_eval_accuracy = model.test_epoch(
                                DataLoader(
                                    TensorDataset(*train_samples),
                                    batch_size=128,
                                    shuffle=False,
                                ),
                                epoch,
                            )
                            validation_eval_loss, validation_eval_accuracy = (
                                model.test_epoch(
                                    DataLoader(
                                        TensorDataset(*validation_samples),
                                        batch_size=128,
                                        shuffle=False,
                                    ),
                                    epoch,
                                )
                            )
                            diagnostic_evaluation_seconds = (
                                time.perf_counter() - diagnostic_start
                            )
                        else:
                            train_eval_loss = train_eval_accuracy = None
                            validation_eval_loss = validation_eval_accuracy = None
                            diagnostic_evaluation_seconds = None

                        if epoch == final_measurement_epoch:
                            test_loss, test_accuracy = model.test_epoch(
                                testloader, epoch
                            )
                        else:
                            test_loss, test_accuracy = None, None
                        record = {
                            "training_iteration": epoch,
                            "train_loss": train_loss,
                            "train_acc": train_accuracy,
                            "validation_loss": validation_loss,
                            "validation_acc": validation_accuracy,
                            "train_eval_loss": train_eval_loss,
                            "train_eval_acc": train_eval_accuracy,
                            "validation_eval_loss": validation_eval_loss,
                            "validation_eval_acc": validation_eval_accuracy,
                            "diagnostic_evaluation_seconds": diagnostic_evaluation_seconds,
                            "test_loss": test_loss,
                            "test_acc": test_accuracy,
                        }
                        metrics_file.write(json.dumps(record) + "\n")
                        metrics_file.flush()

                        if epoch in checkpoint_epochs:
                            checkpoint_dir = output_dir / f"checkpoint_{epoch:06d}"
                            checkpoint_dir.mkdir()
                            torch.save(
                                model.model.state_dict(),
                                checkpoint_dir / "checkpoints",
                            )

                if temporal_dashboard and block_end in measurement_epochs:
                    with progress_step(
                        f"Temporal measurements for {run_id} at epoch {block_end}"
                    ):
                        measurement_start = time.perf_counter()
                        epoch_measures = measure_checkpoint(
                            model,
                            train_samples,
                            validation_samples,
                            fc_subspace_dims,
                            train_sample_identity=ordered_sample_identity(
                                "train",
                                train_sample_indices,
                                "positions_in_fixed_training_subset",
                            ),
                            validation_sample_identity=ordered_sample_identity(
                                "validation",
                                validation_sample_indices,
                                "positions_in_fixed_validation_split",
                            ),
                        )
                        epoch_measures["training_iteration"] = block_end
                        epoch_measures["measurement_seconds"] = (
                            time.perf_counter() - measurement_start
                        )
                        temporal_file.write(
                            json.dumps(epoch_measures, allow_nan=False) + "\n"
                        )
                        temporal_file.flush()
                block_start = block_end + 1
        finally:
            if temporal_file is not None:
                temporal_file.close()

    with open(output_dir / "status.json", "x") as status_file:
        json.dump({"status": "trained"}, status_file)
        status_file.write("\n")

    return output_dir


def train_catalog(
    catalog_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    limit: int = 0,
    temporal_dashboard: bool = False,
    measurement_samples: int = 500,
    measurement_every: int = 1,
    fc_subspace_dims=DEFAULT_FC_SUBSPACE_DIMS,
    run_ids=None,
) -> None:
    with open(catalog_path) as catalog_file:
        catalog = json.load(catalog_file)
    runs: Iterable[Dict] = catalog["runs"]
    runs = list(runs)
    if run_ids is not None:
        requested_ids = list(run_ids)
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("run_ids must not contain duplicates")
        catalog_ids = {record["run_id"] for record in runs}
        unknown_ids = set(requested_ids) - catalog_ids
        if unknown_ids:
            raise ValueError(
                f"run_ids are absent from the catalog: {sorted(unknown_ids)}"
            )
        requested = set(requested_ids)
        runs = [record for record in runs if record["run_id"] in requested]
        if not runs:
            raise ValueError("run_ids selected no catalog runs")
    if limit:
        runs = runs[:limit]
    fc_subspace_dims = validate_subspace_dimensions(
        fc_subspace_dims, 20
    )
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_root = output_root.parent / "raw"
    failures = []

    print(f"Starting {len(runs)} run(s).", flush=True)
    all_runs_started_at = time.perf_counter()
    for run_index, run_record in enumerate(runs, start=1):
        run_id = run_record["run_id"]
        run_dir = output_root / run_id
        run_started_at = time.perf_counter()
        print(f"\nRun {run_index}/{len(runs)} — {run_id}", flush=True)
        try:
            status_path = run_dir / "status.json"
            if status_path.is_file():
                with open(status_path) as status_file:
                    status = json.load(status_file)
                if (
                    status.get("status") == "completed"
                    and (run_dir / "measurements.json").is_file()
                ):
                    if temporal_dashboard and not (
                        run_dir / "dashboard.html"
                    ).is_file():
                        failures.append(
                            (
                                run_id,
                                "run is already complete without temporal "
                                "measurements; use a fresh run ID/output "
                                "directory to collect epoch-by-epoch data",
                            )
                        )
                        LOGGER.error(
                            "Run %s is complete but has no temporal "
                            "measurements; a dashboard cannot be reconstructed "
                            "without measurements from each epoch",
                            run_id,
                        )
                        continue
                    print(f"  Already completed — skipping {run_id}", flush=True)
                    continue
            run_dir = train_one_run(
                run_record,
                output_root=output_root,
                temporal_dashboard=temporal_dashboard,
                measurement_samples=measurement_samples,
                measurement_every=measurement_every,
                fc_subspace_dims=fc_subspace_dims,
            )
            with progress_step(f"Final P/R/F measurements for {run_id}"):
                result = compute_all_measures(
                    run_dir,
                    epoch=max(
                        _selected_checkpoint_epochs(
                            run_record["config"],
                            temporal_dashboard,
                            measurement_every,
                        )
                    ),
                    layer_idx=None,
                    run_id=run_id,
                    raw_results_root=raw_results_root,
                    fc_subspace_dims=fc_subspace_dims,
                )
            if result is None:
                raise RuntimeError("measurement pipeline rejected the trained run")
            with open(run_dir / "measurements.json", "x") as measurements_file:
                json.dump(result, measurements_file, indent=2)
                measurements_file.write("\n")
            if temporal_dashboard:
                with progress_step(f"Run dashboard for {run_id}"):
                    write_dashboard(run_dir)
            temporal_status = {}
            temporal_path = run_dir / "temporal_measurements.jsonl"
            if temporal_path.is_file():
                with open(temporal_path) as temporal_file:
                    for line in temporal_file:
                        if line.strip():
                            record = json.loads(line)
                            temporal_status = {
                                split: record[split]["representation_status"]
                                for split in ("train", "validation")
                            }
            final_status = result.get("representation_status", "unknown")
            with open(run_dir / "status.json", "w") as status_file:
                json.dump(
                    {
                        "status": "completed",
                        "observed_reference_sample_status": {
                            "D_ref_test": final_status,
                            **temporal_status,
                        },
                        "observed_reference_sample_collapse": (
                            final_status
                            in {"all_constant", "single_nonconstant_unit"}
                            or any(
                                value
                                in {"all_constant", "single_nonconstant_unit"}
                                for value in temporal_status.values()
                            )
                        ),
                    },
                    status_file,
                    indent=2,
                )
                status_file.write("\n")
            LOGGER.info("Completed training and measurements for %s", run_id)
        except Exception as error:
            failures.append((run_id, str(error)))
            LOGGER.exception("Run %s failed: %s", run_id, error)
            if run_dir.is_dir():
                with open(run_dir / "status.json", "w") as status_file:
                    json.dump(
                        {"status": "failed", "error": str(error)},
                        status_file,
                        indent=2,
                    )
                    status_file.write("\n")
        finally:
            run_elapsed = time.perf_counter() - run_started_at
            print(
                f"Total run {run_index}/{len(runs)} — {run_id}: "
                f"{format_duration(run_elapsed)}",
                flush=True,
            )

    all_runs_elapsed = time.perf_counter() - all_runs_started_at
    print(
        f"\nTotal execution time for all runs "
        f"({len(runs)} run(s)): {format_duration(all_runs_elapsed)}",
        flush=True,
    )
    LOGGER.info(
        "Finished catalog: %d succeeded, %d failed",
        len(runs) - len(failures),
        len(failures),
    )
    print(
        f"\nFinished catalog: {len(runs) - len(failures)} succeeded, "
        f"{len(failures)} failed.",
        flush=True,
    )
    representation_counts = {}
    for run_record in runs:
        status_path = output_root / run_record["run_id"] / "status.json"
        if not status_path.is_file():
            continue
        with open(status_path) as status_file:
            status = json.load(status_file)
        if status.get("status") == "completed":
            values = status.get("observed_reference_sample_status", {})
            key = ", ".join(
                f"{split}={value}" for split, value in sorted(values.items())
            ) or "status unavailable"
            representation_counts[key] = representation_counts.get(key, 0) + 1
    if representation_counts:
        print("Reference-sample representation status summary:", flush=True)
        for status, count in sorted(representation_counts.items()):
            print(f"  {status}: {count}", flush=True)
    if failures:
        raise RuntimeError(
            "one or more catalog runs failed: "
            + "; ".join(f"{run_id}: {reason}" for run_id, reason in failures)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train the runs in a previously generated experiment catalog."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="train only the first N runs (0 means all)",
    )
    parser.add_argument(
        "--temporal-dashboard",
        action="store_true",
        help="measure P/R/F on fixed train and validation samples after every epoch",
    )
    parser.add_argument(
        "--measurement-samples",
        type=int,
        default=500,
        help="maximum fixed train and validation examples used for temporal R/F",
    )
    parser.add_argument(
        "--measurement-every",
        type=int,
        default=1,
        help=(
            "measure P/R/F every N epochs in temporal-dashboard mode; "
            "the final epoch is always measured"
        ),
    )
    parser.add_argument(
        "--fc-subspace-dims",
        type=int,
        nargs="+",
        default=list(DEFAULT_FC_SUBSPACE_DIMS),
        help="F_C subspace dimensions (each between 1 and 20)",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help="train only these exact run IDs from the catalog",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.measurement_samples < 2:
        parser.error("--measurement-samples must be at least 2")
    if args.measurement_every < 1:
        parser.error("--measurement-every must be a positive integer")
    train_catalog(
        args.catalog,
        args.output_root,
        args.limit,
        args.temporal_dashboard,
        args.measurement_samples,
        args.measurement_every,
        args.fc_subspace_dims,
        args.run_ids,
    )


if __name__ == "__main__":
    main()
