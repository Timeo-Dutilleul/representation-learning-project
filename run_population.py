import json
import logging
from pathlib import Path

import pandas as pd

from redundancy_project.run_pipeline import (
    RESULTS_ROOT,
    SUMMARY_COLUMNS,
    compute_all_measures,
)
from redundancy_project.src.d_ref import ZOO_ROOT
LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = RESULTS_ROOT / "selected_runs_manifest.json"
SPLIT_PATH = RESULTS_ROOT / "discovery_confirmation_split.json"
TABLE_PATH = RESULTS_ROOT / "population_table_fc9_epoch50.csv"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    if not MANIFEST_PATH.is_file() or not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            "run build_population.py first to freeze the sample and split"
        )

    with open(MANIFEST_PATH) as manifest_file:
        manifest = json.load(manifest_file)
    with open(SPLIT_PATH) as split_file:
        split = json.load(split_file)

    manifest_runs = manifest["runs"]
    run_ids = [item["run_id"] for item in manifest_runs]
    discovery_ids = split["discovery"]
    confirmation_ids = split["confirmation"]
    if (
        len(run_ids) != 40
        or len(set(run_ids)) != 40
        or len(discovery_ids) != 28
        or len(confirmation_ids) != 12
        or set(discovery_ids) & set(confirmation_ids)
        or set(discovery_ids) | set(confirmation_ids) != set(run_ids)
    ):
        raise ValueError("frozen split/manifest failed the 28+12 integrity check")

    split_by_id = {
        run_id: split_name
        for split_name, identifiers in (
            ("discovery", discovery_ids),
            ("confirmation", confirmation_ids),
        )
        for run_id in identifiers
    }
    records = []
    failures = []
    for item in manifest_runs:
        run_id = item["run_id"]
        trial_dir = ZOO_ROOT / item["trial_dir"]
        result = compute_all_measures(
            trial_dir,
            epoch=manifest["epoch"],
            layer_idx=9,
            run_id=run_id,
        )
        if result is None:
            failures.append(run_id)
            continue
        summaries = result["summaries"]
        if set(summaries) != set(SUMMARY_COLUMNS):
            raise RuntimeError(
                f"{run_id} returned unexpected summary columns"
            )
        records.append(
            {
                "run_id": run_id,
                "split": split_by_id[run_id],
                "metric_definition_version": result[
                    "metric_definition_version"
                ],
                "p_c_definition": result["p_c_definition"],
                "fc_intervention": result["fc_intervention"],
                "reference_sample_id": result["reference_sample"]["identity"][
                    "ordered_indices_sha256"
                ],
                **summaries,
            }
        )

    columns = [
        "run_id",
        "split",
        "metric_definition_version",
        "p_c_definition",
        "fc_intervention",
        "reference_sample_id",
        *SUMMARY_COLUMNS,
    ]
    table = pd.DataFrame.from_records(records, columns=columns)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_PATH, index=False)

    LOGGER.info(
        "Population computation finished: %d succeeded, %d failed",
        len(records),
        len(failures),
    )
    if failures:
        LOGGER.warning("Failed opaque run IDs: %s", ", ".join(failures))
    LOGGER.info("Saved summaries to %s", TABLE_PATH)
    print(table.head().to_string(index=False))
    print(f"Successful runs: {len(records)}; failed runs: {len(failures)}")


if __name__ == "__main__":
    main()
