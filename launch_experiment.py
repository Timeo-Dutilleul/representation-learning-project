import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from redundancy_project.experiment_catalog import (
    DEFAULT_PLAN,
    build_catalog,
)
from redundancy_project.run_experiment_catalog import train_catalog
from redundancy_project.src.measures_F import (
    DEFAULT_FC_SUBSPACE_DIMS,
    validate_subspace_dimensions,
)
from redundancy_project.src.progress import progress_step
from redundancy_project.temporal_dashboard import write_experiment_dashboard


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SETTINGS = PROJECT_ROOT / "experiment_launch.yaml"


def _archive_launch_provenance(
    output_root: Path,
    settings_path: Path,
    plan_path: Path,
    effective_settings: dict,
    catalog_run_count: int,
    run_limit_override,
    selected_run_ids=None,
) -> None:
    provenance_dir = output_root / "provenance"
    provenance_dir.mkdir()
    shutil.copy2(settings_path, provenance_dir / "experiment_launch.yaml")
    shutil.copy2(plan_path, provenance_dir / "experiment_design.yaml")
    shutil.copy2(
        PROJECT_ROOT / "analysis_plan.md", provenance_dir / "analysis_plan.md"
    )

    code_files = sorted(
        path
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and path.suffix == ".py"
        and not {"data", "results", "__pycache__", ".venv"}.intersection(
            path.relative_to(PROJECT_ROOT).parts
        )
    )
    source_hashes = {
        str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in code_files
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    protocol_files = {
        "experiment_launch.yaml": settings_path,
        "experiment_design.yaml": plan_path,
        "analysis_plan.md": PROJECT_ROOT / "analysis_plan.md",
    }

    with open(provenance_dir / "code_version.json", "x") as version_file:
        json.dump(
            {
                "git_revision": revision,
                "git_revision_note": (
                    None
                    if revision is not None
                    else "Unavailable; use the archived SHA-256 manifests."
                ),
                "python_version": sys.version,
                "sha256_python_sources": source_hashes,
                "sha256_protocol_files": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in protocol_files.items()
                },
            },
            version_file,
            indent=2,
        )
        version_file.write("\n")

    run_limit = effective_settings["run_limit"]
    selected_run_count = (
        len(selected_run_ids)
        if selected_run_ids is not None
        else catalog_run_count
        if run_limit == 0
        else min(run_limit, catalog_run_count)
    )
    with open(output_root / "launch_metadata.json", "x") as metadata_file:
        json.dump(
            {
                "effective_settings": effective_settings,
                "run_limit_override": run_limit_override,
                "catalog_run_count": catalog_run_count,
                "selected_run_count": selected_run_count,
                "selected_run_ids": selected_run_ids,
                "archived_protocol": "provenance/",
            },
            metadata_file,
            indent=2,
        )
        metadata_file.write("\n")


def launch(
    settings_path: Path = DEFAULT_SETTINGS,
    run_limit_override=None,
    run_ids=None,
) -> Path:
    with open(settings_path, encoding="utf-8") as settings_file:
        settings = yaml.safe_load(settings_file)
    if not isinstance(settings, dict):
        raise ValueError("launcher settings must be a YAML mapping")

    allowed_keys = {
        "plan",
        "run_limit",
        "temporal_dashboard",
        "measurement_every",
        "measurement_samples",
        "fc_subspace_dims",
        "output_prefix",
        "save_each_launch",
    }
    unknown_keys = set(settings) - allowed_keys
    if unknown_keys:
        raise ValueError(f"unsupported launcher settings: {sorted(unknown_keys)}")

    plan_path = Path(settings.get("plan", DEFAULT_PLAN))
    if not plan_path.is_absolute():
        plan_path = PROJECT_ROOT / plan_path
    configured_run_limit = settings.get("run_limit", 1)
    run_limit = (
        configured_run_limit
        if run_limit_override is None
        else run_limit_override
    )
    measurement_every = settings.get("measurement_every", 1)
    measurement_samples = settings.get("measurement_samples", 500)
    fc_subspace_dims = settings.get(
        "fc_subspace_dims", DEFAULT_FC_SUBSPACE_DIMS
    )
    temporal_dashboard = settings.get("temporal_dashboard", True)
    output_prefix = settings.get("output_prefix", "experiment")
    save_each_launch = settings.get("save_each_launch", False)

    if not isinstance(run_limit, int) or run_limit < 0:
        raise ValueError("run_limit must be a non-negative integer (0 means all)")
    if run_limit_override is not None and (
        not isinstance(run_limit_override, int) or run_limit_override < 1
    ):
        raise ValueError("run_limit_override must be a positive integer")
    if run_ids is not None and run_limit != 0:
        raise ValueError(
            "run_ids cannot be combined with a non-zero run_limit; "
            "set run_limit: 0 to select exact catalog entries"
        )
    if (
        not isinstance(measurement_every, int)
        or measurement_every < 1
    ):
        raise ValueError("measurement_every must be a positive integer")
    if (
        not isinstance(measurement_samples, int)
        or measurement_samples < 2
    ):
        raise ValueError("measurement_samples must be an integer of at least 2")
    fc_subspace_dims = validate_subspace_dimensions(
        fc_subspace_dims, 20
    )
    if not isinstance(temporal_dashboard, bool):
        raise ValueError("temporal_dashboard must be true or false")
    if not isinstance(output_prefix, str) or not output_prefix.strip():
        raise ValueError("output_prefix must be a non-empty string")
    if (
        Path(output_prefix).name != output_prefix
        or output_prefix in {".", ".."}
    ):
        raise ValueError("output_prefix must be a single directory name")
    if not isinstance(save_each_launch, bool):
        raise ValueError("save_each_launch must be true or false")

    results_root = PROJECT_ROOT / "results"
    if save_each_launch:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output_root = results_root / f"{output_prefix}-{timestamp}"
        with progress_step(f"Preparing output directory {output_root.name}"):
            output_root.mkdir(parents=True, exist_ok=False)
    else:
        output_root = results_root / f"{output_prefix}-exploratory"
        with progress_step(
            f"Replacing exploratory output directory {output_root.name}"
        ):
            if output_root.exists():
                if not output_root.is_dir() or output_root.is_symlink():
                    raise ValueError(
                        "exploratory output path must be a real directory: "
                        f"{output_root}"
                    )
                shutil.rmtree(output_root)
            output_root.mkdir(parents=True, exist_ok=False)
    catalog_path = output_root / "experiment_catalog.json"
    with progress_step("Building run catalog"):
        catalog_runs = build_catalog(plan_path, catalog_path)
    selected_run_ids = None
    if run_ids is not None:
        selected_run_ids = list(run_ids)
        if len(set(selected_run_ids)) != len(selected_run_ids):
            raise ValueError("run_ids must not contain duplicates")
        catalog_ids = {record["run_id"] for record in catalog_runs}
        unknown_ids = set(selected_run_ids) - catalog_ids
        if unknown_ids:
            raise ValueError(
                f"run_ids are absent from the catalog: {sorted(unknown_ids)}"
            )
        if not selected_run_ids:
            raise ValueError("at least one run ID must be selected")
    effective_settings = dict(settings)
    effective_settings["run_limit"] = run_limit
    _archive_launch_provenance(
        output_root,
        Path(settings_path),
        plan_path,
        effective_settings,
        len(catalog_runs),
        run_limit_override,
        selected_run_ids,
    )
    try:
        train_catalog(
            catalog_path=catalog_path,
            output_root=output_root / "runs",
            limit=run_limit,
            temporal_dashboard=temporal_dashboard,
            measurement_samples=measurement_samples,
            measurement_every=measurement_every,
            fc_subspace_dims=fc_subspace_dims,
            run_ids=selected_run_ids,
        )
    except RuntimeError as training_error:
        if temporal_dashboard:
            try:
                with progress_step("Building partial experiment overview"):
                    dashboard_path = write_experiment_dashboard(output_root)
            except (OSError, ValueError) as dashboard_error:
                raise RuntimeError(
                    f"{training_error}\n"
                    "Could not generate a partial experiment dashboard: "
                    f"{dashboard_error}"
                ) from training_error
            raise RuntimeError(
                f"{training_error}\n"
                f"Partial experiment dashboard: {dashboard_path}"
            ) from training_error
        raise
    if temporal_dashboard:
        with progress_step("Building experiment overview dashboard"):
            write_experiment_dashboard(output_root)
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a fresh run catalog and launch it using "
            "experiment_launch.yaml."
        )
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=DEFAULT_SETTINGS,
        help="launcher settings YAML",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run only the first N catalog entries (for a benchmark/smoke run)",
    )
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=None,
        help="run only these exact run IDs from the generated catalogue",
    )
    args = parser.parse_args()
    output_root = launch(
        args.settings, run_limit_override=args.limit, run_ids=args.run_ids
    )
    print(f"Experiment outputs: {output_root}")


if __name__ == "__main__":
    main()
