# Research protocol

This document describes the current exploratory pipeline. Mathematical
definitions and implementation details are given in the
[working report](../report/redundancy_in_cnns_report.tex) and
[measure definitions](measure_definitions.md). The experiment is in progress;
the screening is not confirmatory evidence.

## Pipeline and sampling

The active design trains 32 CNN configurations on a deterministic 40% subset
of cached MNIST training data: two activations, two dropout levels, two
optimisers, two learning rates, and two seeds per combination. Training lasts
20 epochs. At every epoch, parameter measures use the current weights, while
representation and functional measures use the same ordered fixed sample of
500 training and 500 validation examples throughout that run. This fixed-
reference-sample principle makes temporal changes comparable within a run;
it does not make the sample representative of every possible input.

The final summary uses a distinct fixed reference sample from the test split.
Train, validation, and test results must remain distinguishable. The screening
is exploratory, has only two seeds per condition, and is intended to support
descriptive analysis and candidate selection—not confirmatory claims.

## Measure families

- **`P_L` — local parameter structure:** a pairwise matrix combining cosine
  similarity of units' incoming and outgoing weight vectors.
- **`P_C` — collective parameter structure:** the spectral-energy participation
  ratio of the row-normalised incoming-weight matrix; it summarises effective
  dimensionality, not exact rank.
- **`R_L` — local representation structure:** pairwise Pearson correlations
  between units' post-activation responses on the fixed reference sample.
  Correlations involving constant units are undefined, not zero.
- **`R_C` — collective representation structure:** the participation ratio of
  the covariance spectrum of centred activations on that sample.
- **`F_L` — local functional sensitivity:** mean KL divergence between original
  and perturbed class-probability distributions when one post-activation unit
  is set to zero, without retraining.
- **`F_C` — collective functional sensitivity:** the same mean KL measure after
  removing a centred activation subspace, without retraining. The intervention
  is centred PCA-space projection and preserves the reference-sample mean.

For `F_C`, low-variance PCA subspaces are compared with 50 random subspaces of
the same dimension. The primary descriptive comparison is on validation at
epoch 20 for `k = 1, ..., 19`, using each run's low-variance score relative to
the median of its raw random-control scores. `k = 20` is retained as a
consistency bound, not as an ordinary low-variance-versus-random comparison.
If random-control KL is near zero, the ratio is marked uninterpretable under
the documented threshold; the absolute difference and raw scores are retained.
See [analysis_plan.md](../analysis_plan.md) for the pre-specified details.

## Representation status

Statuses describe only a split and its fixed measured reference sample; they
do not establish global network behaviour.

- `ok`: positive variance and full numerical rank;
- `rank_deficient`: positive variance but numerical rank below the hidden width;
- `single_nonconstant_unit`: positive variance with exactly one non-constant
  unit; pairwise `R_L` is unavailable;
- `all_constant`: zero total centred activation variance on the measured sample.

For `all_constant`, `R_C` and `F_C` are unavailable and no PCA or random
subspaces are constructed. This does not invalidate the training trajectory:
performance metrics and any separately defined parameter measures remain
recorded, and training continues. For positive-variance rank-deficient
representations, `R_C` and `F_C` remain measured and are accompanied by rank,
nullity, spectrum, and a caveat that individual zero-variance basis vectors
are not uniquely identifiable. `R_L` is unavailable when fewer than two
units are non-constant.

The results describe geometric and functional measurements on specified
samples. Geometric redundancy and functional dispensability are distinct
questions and are not treated as equivalent.
