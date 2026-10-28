# Pre-specified analysis plan — representation redundancy

## Status and scope

This document specifies a descriptive analysis in advance of the exploratory
configuration screening. The planned catalogue contains 32 runs (two seeds per
condition), which is insufficient to establish robust hyperparameter effects
or support confirmatory conclusions. Runs and epochs will not be treated as
independent replications: the independent unit for between-condition
comparisons is the run, with seeds providing limited replication within a
condition.

This document does not change the training protocol or the measures. It is not
a confirmatory pre-registration: the screening is intended to describe results
and generate hypotheses for a later study.

## Saved measurement format

The analysis uses the per-epoch records, fixed-sample metadata, final test
summary, and raw arrays described below. These are generated locally for each
launch; the public repository does not include run outputs.

### Output tree for a temporal launch

With the launcher, each execution produces the following structure. The output
folder is `experiment-exploratory` when `save_each_launch: false`, or
`experiment-<timestamp>` when that option is enabled (as in the current
configuration):

```text
results/<launch-folder>/
├── experiment_catalog.json
├── dashboard.html                         # overview of completed runs
├── provenance/
│   ├── experiment_catalog.json
│   ├── experiment_design.yaml
│   ├── experiment_launch.yaml
│   ├── analysis_plan.md
│   └── code_version.json
├── raw/
│   └── <run_id>.npz                        # compact final measures (D_ref)
└── runs/
    └── <run_id>/
        ├── run_metadata.json               # run, condition, seed, data, indices
        ├── config.json                      # effective model configuration
        ├── result.json                      # one JSON record per epoch
        ├── temporal_measurements.jsonl      # one record per measured epoch
        ├── measurements.json                # final summary on D_ref (test)
        ├── dashboard.html                   # per-run dashboard
        ├── status.json
        └── checkpoint_<epoch>/checkpoints   # saved weights
```

In temporal mode, a checkpoint is saved at every measured epoch, in addition
to any epochs explicitly requested in the plan.

### Fields recorded per epoch

`result.json` is JSON Lines and contains one record per training epoch with
`training_iteration`, `train_loss`, `train_acc`, `validation_loss`,
`validation_acc`, `test_loss`, and `test_acc`. The test is evaluated only at
the selected final epoch; other records have null values for those fields. At
measured epochs, it also contains `train_eval_loss`, `train_eval_acc`,
`validation_eval_loss`, `validation_eval_acc`, and
`diagnostic_evaluation_seconds`. These diagnostic evaluations use the fixed
measurement samples. Losses are NLL losses from the model: ordinary training
values are aggregated during updates, whereas `*_eval_*` fields are computed
after the epoch on the fixed samples.

`temporal_measurements.jsonl` contains one record per measured epoch. Its
top-level fields include `training_iteration`, `measurement_seconds`,
`metric_definition_version`, measure definitions, layer indices,
`activation_width`, `P_C`, `P_L_in`, `P_L_out`, and the combined `P_L`. Each
`train` and `validation` object contains:

- `sample_count`, `reference_sample` (split, size, and ordered-index
  fingerprint), and `reference_activation_mean`;
- the full `R_L` matrix, `R_C`, and activation covariance eigenvalues;
- the complete `F_L` vector (20 units);
- for every configured dimension, `F_C[k].low_variance`, all 50 raw
  `F_C[k].random` scores, `R_C_remaining_lowvar`, and `V_remaining_lowvar`
  (remaining variance fraction);
- indices of constant or zero-activation units and the `F_C` intervention
  definition.

Training and validation fields are recorded separately at each measured epoch.
`P_C` and the `P_L*` matrices do not depend on the split and are recorded once
per measured epoch. Means and standard deviations of the 50 scores are not
included in temporal records; they can be recomputed without loss from the raw
scores. Activation singular values can be recovered from the saved eigenvalues
(up to the covariance scaling factor), but are not stored as a separate field.

### Metadata and sample auditability

`run_metadata.json` contains `run_id`, `condition_id`, `seed`, requested and
effective configurations, dataset provenance/preprocessing, and
`temporal_measurement_samples`. The latter stores the exact 500 training
example indices within the subset and their indices in the source dataset, as
well as the exact positions of the 500 validation examples in the validation
split. Each temporal record also carries the split, sample size, and ordered
SHA-256 fingerprint. Run/condition/seed identifiers are held in the run
directory and metadata, rather than repeated in every JSONL record; records
are joined by `run_id`/directory.

The final `measurements.json` summary records the exact indices of the
`D_ref_test` split used for final measures. The final `.npz` contains the raw
numeric arrays listed in `measurements.json` (see keys for `P_L*`, `P_C`,
`R_L`, `R_C`, `F_L`, the activation spectrum, and `F_C`). There is no separate
per-epoch file for matrices or spectra: their temporal copies are in the JSONL
file.

### Availability of requested objects

| Object | Saved at every measured epoch? | Notes |
|---|---:|---|
| Training/validation loss (NLL) and accuracy | Yes | `result.json` at every epoch; additional fixed-sample measures at measured epochs. |
| `P_C`, `R_C` | Yes | `P_C` is split-independent; `R_C` is recorded under `train` and `validation`. |
| `P_L_in`, `P_L_out`, combined `P_L`, `R_L` | Yes | Full matrices in JSON; undefined correlations are encoded as null/non-finite according to the writer. |
| Activation covariance spectrum | Yes | 20 eigenvalues per split; corresponding singular values can be derived. |
| Incoming-weight spectrum | No, not as a temporal field | Each measured-epoch checkpoint contains the exact weights, so the spectrum can be recomputed. `P_C` alone is insufficient to recover it. |
| `F_L` vector (20 units) | Yes | The temporal code saves per-unit KL. |
| Per-neuron `NLL delta` / `accuracy delta` | No | A diagnostic function exists, but the temporal runner does not call it. Fixed-sample metrics without ablation are recorded. |
| Low-variance `F_C`, 50 random scores, remaining `R_C`, remaining variance fraction | Yes, for configured k | All 50 raw scores are persisted. Mean/median/standard deviation can be computed afterwards. |
| Identifiers and exact sample indices | Yes, across files | Training/validation identifiers and indices are in metadata; per-epoch fingerprints and test indices are in the final summary. |

Current storage is sufficient for the primary analysis of `F_C` scores at the
last checkpoint, provided all requested dimensions were computed. Missing
dedicated columns or fields do not require duplication: the incoming-weight
spectrum can be recomputed from checkpoints, and random-score summaries can be
recomputed from the 50 saved values. If per-neuron NLL/accuracy deltas become
an analysis question, the minimal change would be to call the existing
diagnostic during temporal measurement and record two 20-value vectors per
split; they are not needed for the primary analysis below.

## Pre-specified primary analysis

### Population, split, and checkpoint

- Population: the 32 runs planned in the catalogue, subject to completion and
  quality checks; two seeds per condition.
- Split: validation only.
- Primary epoch: epoch 20, the last epoch in the current protocol.
- Each point represents a run. The two seeds per condition are shown as
  individual points; they do not by themselves justify robust inference about
  hyperparameter effects.

### Primary criterion

For each run `r` and `k = 1, …, 19`, calculate at epoch 20 on validation:

```text
Q_r(k) = F_C,lowvar,r(k) / median_b(F_C,random,r,b(k)),  b = 1, …, 50.
```

The numerator is the ablation of the low-variance PCA subspace; the denominator
is the empirical median of the 50 random controls recorded for the same run,
split, epoch, and `k`. Retain `k=20` as a consistency check, but exclude it
from the comparison: both ablations remove the entire space, so equality is
expected up to numerical tolerance.

Plot `Q_r(k)` curves by condition and show the individual seed points/curves.
Do not calculate p-values, describe differences as confirmed effects, or hide
between-seed variability.

### Near-zero denominator and complementary contrast

The ratio may become unstable when the median random KL is very small; in that
case it is not a reliable measure of relative magnitude. Do not silently add
an epsilon to the denominator. Retain and report the denominator, as well as
the absolute contrast:

```text
Delta_r(k) = F_C,lowvar,r(k) - median_b(F_C,random,r,b(k)).
```

Reporting rule: if the denominator is less than or equal to `1e-8
nats/example`, mark `Q_r(k)` as numerically uninterpretable (NA) and show
`Delta_r(k)` alongside the raw values. Apply this rule uniformly; additionally
provide a sensitivity analysis of flags using thresholds of `1e-9` and `1e-7
nats/example`. Extremely small KL measures are particularly susceptible to
numerical cancellation, especially in single-precision floating point. The
absolute contrast is more stable but remains descriptive and depends on the
scale of the outputs.

## Secondary and exploratory analyses

## Degenerate reference samples and numerical rank

All representation statuses below describe the *observed fixed reference
sample and split only*. They do not establish that the network has collapsed
on all possible inputs. A run may therefore have different statuses on its
fixed training, validation, and final test reference samples.

For an activation matrix `A` with `n` sampled examples and `d` units, centre
each column to obtain `X = A - mean(A, axis=0)`. Let
`s_1 >= ... >= s_min(n,d) >= 0` be the singular values of `X`. The numerical
rank `r` is the number of singular values strictly greater than

```text
tau = max(n, d) * eps(dtype(A)) * s_1,
```

where `eps(dtype(A))` is the machine precision of the activation tensor's
floating-point dtype (float32 when the input is not floating point). This
scale-relative tolerance is applied to the centred matrix; covariance
eigenvalues at or below `tau^2 / (n - 1)` are set to zero for rank and
variance summaries. The saved diagnostics include the tolerance, full
zero-padded covariance spectrum, total activation variance, `numerical_rank`
and `nullity = d - r`.

The sample-level states are:

| Status | Meaning and reporting policy |
|---|---|
| `all_constant` | Numerical rank and total centred variance are zero. Save `R_C: null`, `F_C: null`, and do not construct PCA or random controls. `P` and unit-wise `F_L` remain separately available. |
| `single_nonconstant_unit` | Positive variance, exactly one non-constant unit. Save `R_C` (normally 1); save `R_L: null` because pairwise summaries require at least two non-constant units. Compute every configured `F_C` value and control, with the null-space caveat below. |
| `rank_deficient` | Positive variance and `0 < r < d`. Retain `R_C`, rank, nullity, spectrum and status; keep `R_L` if at least two units are non-constant. Compute every configured `F_C` value and control, with the null-space caveat below. |
| `ok` | Positive variance and full numerical rank. The standard metrics are available, subject to their ordinary numerical checks. |

For `rank_deficient` and `single_nonconstant_unit`, let `q = d - r` be the
nullity. Compute and save `F_C,lowvar(k)` and all 50 random controls for every
configured `k = 1, ..., d`, including `k < q`. The basis vectors within the
zero-variance eigenspace are not individually identifiable as unique PCA
axes: the numerical SVD may return any orthonormal basis for that space. This
does not make the measurements unavailable. Under the centred ablation used
here, activation deviations on the reference sample have zero projection
onto exact zero-variance directions, so removing those directions has zero
behavioural effect on that sample, up to numerical precision. Record this
caveat in the split metadata and dashboard, alongside the actual measured
scores; do not replace these scores with `null` or suppress them. The complete
nullspace (dimension `q`) is itself invariant even though its individual
basis vectors are not. At `k=d`, both interventions remove the full unit
space; this remains a consistency check. For zero total variance, by contrast,
`R_C` and `F_C` are unavailable and neither PCA directions nor random
controls are generated.

`P` and `F_L` are retained when defined, but every saved split/run carries its
reference-sample representation status so these quantities are not mistaken
for measurements from a healthy/full-rank representation. The runner records
per-split constant/non-constant counts and final reference-sample status in
the run status file, and continues optimisation through the configured final
epoch when representation-dependent measures are unavailable.

### Secondary, descriptive

- Plot `F_C,lowvar(k)`, the median and distribution of the 50 controls, and
  `Delta_r(k)`; report both splits without changing validation's primary
  status.
- Examine `R_C_remaining_lowvar(k)` and `V_remaining_lowvar(k)` alongside `Q`
  and `Delta` to contextualise PCA-spectrum structure, without interpreting
  them as functional measures.
- Describe other measures at checkpoint 20 (`P_C`, `R_C`, `P_L`, `R_L`,
  `F_L`) and performance; flag invalid runs, non-finite values, or constant
  units.
- Extract incoming-weight singular values from epoch-20 checkpoints if a
  spectral description of weights is useful; state that these spectra are not
  saved directly in temporal results.

### Exploratory

- Describe temporal trajectories for all available measures. Epochs are
  repeated measurements within a run, never independent units for computing
  standard errors or between-condition tests.
- Explore associations between the six principal measures and configuration
  parameters, explicitly noting the hypothesis-generating nature of the work
  and the small number of seeds.
- Do not select conclusions or subgroups based on test performance; the test
  split is reserved for the final summary and performance description.

## Execution configuration

The versioned configuration requests `run_limit: 0`,
`fc_subspace_dims: [1, …, 20]`, and `save_each_launch: true`. It retains
`measurement_every: 1`, 500 fixed examples per split, 50 random controls per
dimension, and the 20-epoch plan. The generated catalogue contains 32 runs.

Every launch writes to a separate timestamped directory. The launcher archives
the effective YAML files, catalogue, analysis plan, and code-version
information under `provenance/`; launch metadata records the effective
settings. The planned 32-run screening is exploratory and is not confirmatory
evidence.
