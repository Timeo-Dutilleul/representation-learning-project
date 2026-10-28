# Neuronal Redundancy in Convolutional Networks

An independent, ongoing exploratory research project on how parameter
similarity, representational geometry, and functional sensitivity relate in
trained convolutional neural networks (CNNs). The current controlled setting
uses CNNs trained on MNIST under a factorial exploratory design.

The project separates three families of measurements:

- **P — parameter space:** structure and similarity in the weights associated
  with hidden units;
- **R — representation space:** activation correlations and the effective
  dimensionality of hidden representations on fixed samples;
- **F — functional sensitivity:** changes in model output distributions when
  an individual unit or an activation subspace is intervened upon.

Geometric redundancy and functional dispensability are not assumed to be
equivalent. Similar weights or activations do not, by themselves, show that a
unit or direction can be removed without behavioural consequences. The
pipeline and screening protocol are implemented; experiment curation and
analysis are ongoing. No general relationship or established scientific
finding is claimed.

The research report is available as a [PDF](report/redundancy_in_cnns_report.pdf).
Its canonical source is `report/redundancy_in_cnns_report.tex`.

## Getting started

The experiment runner imports model definitions from the separate
`ModelZooDataset` project and reads a locally prepared cached MNIST tensor
dataset. Neither the dataset nor model checkpoints are included here. Arrange
the working folders as siblings, with `ModelZooDataset/code` present and the
cached dataset at the path configured in `experiment_design.yaml`.

From the repository root, create an environment and install the declared
runtime dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the tests without launching an experiment:

```sh
PYTHONPATH=.. python -m unittest discover -s tests -v
```

Review `experiment_design.yaml` for the model conditions and
`experiment_launch.yaml` for launch-wide settings. The current launch selects
all 32 catalogue entries, trains for 20 epochs, measures every epoch on fixed
500-example train and validation samples, evaluates all configured
`F_C` dimensions `k = 1, ..., 20`, and stores 50 random controls per dimension.
From the repository root, launch with:

```sh
PYTHONPATH=.. python -m redundancy_project.launch_experiment
```

Each launch is written to a new timestamped directory under `results/` and
produces a unified dashboard. Results, checkpoints, raw arrays, dashboards,
datasets and generated audit exports are local research artefacts and are
excluded from Git by default. Preserve and review them locally; do not commit
them as part of the source release.

The 32-run catalogue has two seeds per condition and is explicitly a screening
design, not a confirmatory study. See the [research protocol](docs/research_protocol.md)
before running it.

## Project roadmap

The planned stages are described in [docs/project_roadmap.md](docs/project_roadmap.md).
The project aims to develop independent research practice through explicit
hypothesis formulation, reproducible experimentation, engagement with the
literature, and critical interpretation of negative as well as positive
results.

## Repository map

- [Research protocol](docs/research_protocol.md)
- [Project roadmap](docs/project_roadmap.md)
- [Repository guide](docs/repository_guide.md)
- [Report PDF](report/redundancy_in_cnns_report.pdf)
- [Experiment design](experiment_design.yaml) and
  [launch settings](experiment_launch.yaml)

## Licence

The project code and documentation are released under the MIT License. This
does not relicense the separately obtained ModelZooDataset project or any
MNIST data; consult their respective terms before use or redistribution.
