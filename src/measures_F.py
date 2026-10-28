import torch

from redundancy_project.src.discrepancy import behavioral_discrepancy
from redundancy_project.src.participation import (
    spectral_energy_participation_ratio,
)
from redundancy_project.src.measures_R import covariance_spectrum


DEFAULT_FC_SUBSPACE_DIMS = (1, 5, 10)
RANDOM_SUBSPACE_SEEDS = range(50) #range(5)


def validate_subspace_dimensions(dimensions, n_units: int) -> tuple:
    if not isinstance(dimensions, (list, tuple)):
        raise ValueError("F_C subspace dimensions must be a list or tuple")
    if not dimensions:
        raise ValueError("at least one F_C subspace dimension is required")
    if any(type(dimension) is not int for dimension in dimensions):
        raise ValueError("F_C subspace dimensions must be integers")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("F_C subspace dimensions must be unique")
    if any(dimension < 1 or dimension > n_units for dimension in dimensions):
        raise ValueError(
            "F_C subspace dimensions must be between 1 and "
            f"the hidden-layer width ({n_units})"
        )
    return tuple(dimensions)


def get_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    """Return model log-probabilities for the supplied inputs."""
    model.model.eval()
    with torch.no_grad():
        return model.forward(x)


def ablate_unit_hook(unit_idx: int):
    """Zero one unit's activation immediately after its nonlinearity."""
    def hook(_module, _inputs, output):
        if output.ndim != 2:
            raise ValueError("unit ablation expects (batch, units) activations")
        if unit_idx < 0 or unit_idx >= output.shape[1]:
            raise IndexError(f"unit index {unit_idx} is out of range")
        ablated = output.clone()
        ablated[:, unit_idx] = 0.0
        return ablated

    return hook


def _get_ablated_logprobs(
    model, x: torch.Tensor, nonlin_layer_idx: int, unit_idx: int
) -> torch.Tensor:
    layers = list(model.model.module_list)
    if nonlin_layer_idx < 0 or nonlin_layer_idx >= len(layers):
        raise IndexError(f"module index {nonlin_layer_idx} is out of range")
    handle = layers[nonlin_layer_idx].register_forward_hook(
        ablate_unit_hook(unit_idx)
    )
    try:
        return get_logprobs(model, x)
    finally:
        handle.remove()


def F_L_single_unit(
    model,
    x: torch.Tensor,
    logprobs_orig: torch.Tensor,
    nonlin_layer_idx: int,
    unit_idx: int,
) -> float:
    """Measure output KL after zeroing one post-nonlinearity activation."""
    logprobs_ablated = _get_ablated_logprobs(
        model, x, nonlin_layer_idx, unit_idx
    )
    return behavioral_discrepancy(logprobs_orig, logprobs_ablated)


def F_L_layer(model, x: torch.Tensor, nonlin_layer_idx: int, n_units: int):
    """Return the per-unit behavioral discrepancy for one hidden layer."""
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    logprobs_orig = get_logprobs(model, x)
    return torch.tensor(
        [
            F_L_single_unit(
                model, x, logprobs_orig, nonlin_layer_idx, unit_idx
            )
            for unit_idx in range(n_units)
        ],
        dtype=logprobs_orig.dtype,
        device=logprobs_orig.device,
    )


def F_L_layer_diagnostics(
    model,
    x: torch.Tensor,
    targets: torch.Tensor,
    nonlin_layer_idx: int,
    n_units: int,
):
    """Return per-unit KL, NLL delta, and accuracy delta using shared baseline."""
    logprobs_orig = get_logprobs(model, x)
    if targets.ndim != 1 or targets.shape[0] != logprobs_orig.shape[0]:
        raise ValueError("targets must contain one class label per input")
    if targets.device != logprobs_orig.device:
        targets = targets.to(logprobs_orig.device)

    original_nll = -logprobs_orig.gather(1, targets[:, None]).mean()
    original_predictions = logprobs_orig.argmax(dim=1)
    original_accuracy = (original_predictions == targets).float().mean()

    kl_scores = []
    nll_deltas = []
    accuracy_deltas = []
    for unit_idx in range(n_units):
        logprobs_ablated = _get_ablated_logprobs(
            model, x, nonlin_layer_idx, unit_idx
        )
        kl_scores.append(
            behavioral_discrepancy(logprobs_orig, logprobs_ablated)
        )
        ablated_nll = -logprobs_ablated.gather(1, targets[:, None]).mean()
        ablated_accuracy = (
            logprobs_ablated.argmax(dim=1) == targets
        ).float().mean()
        nll_deltas.append((ablated_nll - original_nll).item())
        accuracy_deltas.append((ablated_accuracy - original_accuracy).item())

    return {
        "kl": torch.tensor(kl_scores, dtype=logprobs_orig.dtype),
        "nll_delta": torch.tensor(nll_deltas, dtype=logprobs_orig.dtype),
        "accuracy_delta": torch.tensor(
            accuracy_deltas, dtype=logprobs_orig.dtype
        ),
    }


def ablate_subspace_hook(
    directions: torch.Tensor, mean_activation: torch.Tensor
):
    """Remove centered activations' projection while preserving their mean."""
    if directions.ndim != 2:
        raise ValueError("directions must have shape (k, n_units)")
    if mean_activation.ndim != 1:
        raise ValueError("mean_activation must have shape (n_units,)")
    if mean_activation.shape[0] != directions.shape[1]:
        raise ValueError(
            "mean_activation must match the subspace unit dimension"
        )
    if mean_activation.device != directions.device:
        raise ValueError(
            "mean_activation and directions must have the same device"
        )
    if mean_activation.dtype != directions.dtype:
        raise ValueError(
            "mean_activation and directions must have the same dtype"
        )
    if not torch.isfinite(mean_activation).all():
        raise ValueError("mean_activation must contain only finite values")
    gram = directions @ directions.T
    identity = torch.eye(
        directions.shape[0], dtype=directions.dtype, device=directions.device
    )
    if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
        raise ValueError("directions must have orthonormal rows")

    def hook(_module, _inputs, output):
        if output.ndim != 2 or output.shape[1] != directions.shape[1]:
            raise ValueError(
                "directions must match (batch, units) module activations"
            )
        if directions.device != output.device or directions.dtype != output.dtype:
            raise ValueError(
                "directions must have the same device and dtype as activations"
            )
        if (
            mean_activation.device != output.device
            or mean_activation.dtype != output.dtype
            or mean_activation.shape != (output.shape[1],)
        ):
            raise ValueError(
                "mean_activation must match activation shape, device and dtype"
            )
        if directions.shape[0] == 0:
            return output
        centered = output - mean_activation
        projection = centered @ directions.T @ directions
        return mean_activation + centered - projection

    return hook


def F_C_subspace(
    model,
    x: torch.Tensor,
    logprobs_orig: torch.Tensor,
    nonlin_layer_idx: int,
    directions: torch.Tensor,
    mean_activation: torch.Tensor,
) -> float:
    """Return output KL after centered activation-space subspace ablation."""
    layers = list(model.model.module_list)
    if nonlin_layer_idx < 0 or nonlin_layer_idx >= len(layers):
        raise IndexError(f"module index {nonlin_layer_idx} is out of range")
    handle = layers[nonlin_layer_idx].register_forward_hook(
        ablate_subspace_hook(directions, mean_activation)
    )
    try:
        logprobs_ablated = get_logprobs(model, x)
    finally:
        handle.remove()
    return behavioral_discrepancy(logprobs_orig, logprobs_ablated)


def low_variance_directions(
    activations: torch.Tensor, k: int
) -> torch.Tensor:
    """Return k right singular vectors with the smallest centered variance."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape (n_samples, n_units)")
    if k < 0 or k > activations.shape[1]:
        raise ValueError("k must be between 0 and the number of units")
    if k == 0:
        return activations.new_empty((0, activations.shape[1]))
    centered = activations - activations.mean(dim=0, keepdim=True)
    _, _, right_vectors = torch.linalg.svd(centered, full_matrices=True)
    return right_vectors[-k:]


def random_orthonormal_directions(
    n_units: int, k: int, seed: int = 0
) -> torch.Tensor:
    """Generate k seeded, orthonormal directions in n_units dimensions."""
    if k < 0 or k > n_units:
        raise ValueError("k must be between 0 and n_units")
    if k == 0:
        return torch.empty((0, n_units))
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(n_units, n_units, generator=generator)
    orthogonal, _ = torch.linalg.qr(matrix)
    return orthogonal[:, :k].T.contiguous()


def remaining_effective_dimensions(
    activations: torch.Tensor, dimensions
) -> dict[str, float]:
    """Compute covariance PR after removing the lowest-variance PCA axes."""
    spectrum = covariance_spectrum(activations)
    total_dimensions = spectrum.numel()
    results = {}
    for dimension in dimensions:
        if dimension < 0 or dimension > total_dimensions:
            raise ValueError("dimension must be between 0 and activation width")
        remaining = spectrum[: total_dimensions - dimension]
        results[str(dimension)] = spectral_energy_participation_ratio(
            remaining
        )
    return results


def remaining_variance_fractions(
    activations: torch.Tensor, dimensions
) -> dict[str, float]:
    """Compute covariance variance fractions after removing low-variance axes."""
    spectrum = covariance_spectrum(activations)
    total_variance = spectrum.sum()
    if total_variance == 0:
        raise ValueError("activation covariance has zero total variance")
    total_dimensions = spectrum.numel()
    return {
        str(dimension): (
            spectrum[: total_dimensions - dimension].sum() / total_variance
        ).item()
        for dimension in dimensions
    }
