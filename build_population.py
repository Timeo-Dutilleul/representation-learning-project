import json
import random
from pathlib import Path

from redundancy_project.src.d_ref import ZOO_ROOT


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SPLIT_PATH = RESULTS_ROOT / "discovery_confirmation_split.json"
MANIFEST_PATH = RESULTS_ROOT / "selected_runs_manifest.json"

N_RUNS = 40
DISCOVERY_COUNT = 28
# Sampling and split use independent fixed seeds; preserve outputs once written.
RUN_SAMPLE_SEED = 41027
SPLIT_ASSIGNMENT_SEED = 93811
EPOCH = 50


def main():
    if SPLIT_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            "the population split/manifest already exists; refusing to "
            "resample or overwrite the frozen assignment"
        )

    trial_dirs = sorted(
        path
        for path in ZOO_ROOT.iterdir()
        if path.is_dir()
        and (path / "result.json").is_file()
        and (path / f"checkpoint_{EPOCH:06d}" / "checkpoints").is_file()
    )
    if len(trial_dirs) < N_RUNS:
        raise RuntimeError(
            f"found only {len(trial_dirs)} eligible trial directories; "
            f"need {N_RUNS}"
        )

    selected_dirs = random.Random(RUN_SAMPLE_SEED).sample(
        trial_dirs, N_RUNS
    )
    run_ids = [
        f"run_{index:03d}" for index in range(1, N_RUNS + 1)
    ]
    manifest_runs = [
        {"run_id": run_id, "trial_dir": trial_dir.name}
        for run_id, trial_dir in zip(run_ids, selected_dirs)
    ]

    discovery_ids = set(
        random.Random(SPLIT_ASSIGNMENT_SEED).sample(
            run_ids, DISCOVERY_COUNT
        )
    )
    split = {
        "discovery": [
            run_id for run_id in run_ids if run_id in discovery_ids
        ],
        "confirmation": [
            run_id for run_id in run_ids if run_id not in discovery_ids
        ],
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "x") as manifest_file:
        json.dump(
            {"epoch": EPOCH, "runs": manifest_runs},
            manifest_file,
            indent=2,
        )
        manifest_file.write("\n")
    with open(SPLIT_PATH, "x") as split_file:
        json.dump(split, split_file, indent=2)
        split_file.write("\n")

    print(f"Eligible trial directories: {len(trial_dirs)}")
    print(f"Sampled runs: {N_RUNS} (epoch {EPOCH})")
    print(
        f"Frozen split: {len(split['discovery'])} discovery, "
        f"{len(split['confirmation'])} confirmation"
    )
    print(f"Saved split: {SPLIT_PATH}")
    print(f"Saved opaque-ID manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
