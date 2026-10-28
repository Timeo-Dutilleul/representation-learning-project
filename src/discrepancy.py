"""Behavioral discrepancy conventions used throughout the project.

For all F measurements, D_ref is fixed to the first 500 examples of the
MNIST zoo test set in its stored order (the same sample used for R).
The primary discrepancy is mean KL(p_original || p_perturbed), computed
from model log-probabilities; mean NLL and accuracy deltas are secondary
descriptors and do not replace this primary metric.
"""

import torch


def behavioral_discrepancy(
    logprobs_orig: torch.Tensor, logprobs_perturbed: torch.Tensor
) -> float:
    """Return mean KL(p_orig || p_perturbed) for batches of log-probabilities."""
    if logprobs_orig.shape != logprobs_perturbed.shape:
        raise ValueError(
            "original and perturbed log-probabilities must have identical shapes"
        )
    if logprobs_orig.ndim != 2:
        raise ValueError("log-probabilities must have shape (N, n_classes)")
    if logprobs_orig.shape[0] == 0:
        raise ValueError("at least one sample is required")

    p_orig = logprobs_orig.exp()
    kl_per_sample = (
        p_orig * (logprobs_orig - logprobs_perturbed)
    ).sum(dim=1)
    return kl_per_sample.mean().item()
