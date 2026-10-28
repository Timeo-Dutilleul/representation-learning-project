# Repository guide

## Main entry points

- `launch_experiment.py` validates launch settings, creates a fresh
  timestamped results directory, archives provenance, and launches the
  selected catalogue.
- `experiment_catalog.py` expands `experiment_design.yaml` into stable
  run records.
- `run_experiment_catalog.py` trains and measures each selected run.
- `run_pipeline.py` provides the final-checkpoint measurement pipeline.
- `run_single_layer_demo.py` is a separate demonstration using a saved zoo
  checkpoint; it is not the 32-run screening entry point.

## Measures and configuration

The measure implementations are in `src/measures_P.py`, `src/measures_R.py`,
`src/measures_F.py`, and shared utilities in `src/`. Temporal measurement
orchestration is in `src/temporal_measurements.py`; dashboard generation is in
`temporal_dashboard.py`.

`experiment_design.yaml` defines the model conditions, seeds, dataset settings,
and training duration. `experiment_launch.yaml` defines the run limit,
measurement cadence and sample count, `F_C` dimensions, dashboard setting,
and timestamped-output policy. `analysis_plan.md` records the pre-specified
screening analysis.

## Outputs and separation

Generated outputs are local under `results/`. A timestamped launch contains
the catalogue, launch metadata, provenance and version hashes, one directory
per selected run, temporal JSONL measurements, final JSON/NPZ measurements,
checkpoints, and a unified dashboard. The result tree and large data/model
files are intentionally Git-ignored. Keep private working material, local
paths, secrets, and ad hoc exports outside public documentation. Use the
ignored `local/` directory for private material.

Raw runs, checkpoints, generated dashboards, and machine-readable measurement
files remain local under the ignored `results/` directory. Do not publish
historical dashboards or raw results. After the screening and analysis, only
carefully reviewed, lightweight figures or tables should be considered for a
dedicated public location such as `docs/results/`; no such exports are
currently included.

The cached MNIST tensors and the separate `ModelZooDataset` source tree are
not bundled. See the README for the expected sibling layout and launch
environment.

The configured cached tensor file belongs at
`data/mnist_zoo_raw/tune_zoo_mnist_uniform/dataset.pt`. Dataset and reference-
sample files are local inputs and are Git-ignored. Obtain and use the data
under its original terms.

The report source and bibliography are in `report/`. Compile from that
directory with `pdflatex`, `bibtex`, and two further `pdflatex` passes; the
expected PDF name is `redundancy_in_cnns_report.pdf`. LaTeX auxiliary files
and the `report/build/` directory are ignored by Git.
