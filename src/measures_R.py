from typing import Optional

import torch

from redundancy_project.src.participation import (
    spectral_energy_participation_ratio,
)


def get_activations(model, dataloader, layer_indices, max_samples=500):
    """Collect post-module activations from a fixed, ordered data sample."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    activations = {index: [] for index in layer_indices}
    layers = list(model.model.module_list)
    hooks = []

    def make_hook(index):
        def hook(_module, _inputs, output):
            activations[index].append(output.detach().clone())

        return hook

    try:
        for index in layer_indices:
            if index < 0 or index >= len(layers):
                raise IndexError(f"module index {index} is out of range")
            hooks.append(
                layers[index].register_forward_hook(make_hook(index))
            )

        model.model.eval()
        n_collected = 0
        with torch.no_grad():
            for inputs, _ in dataloader:
                remaining = max_samples - n_collected
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                model.model(inputs)
                n_collected += inputs.shape[0]
    finally:
        for hook_handle in hooks:
            hook_handle.remove()

    if n_collected == 0:
        raise ValueError("dataloader did not yield any samples")

    return {
        index: torch.cat(layer_activations, dim=0)[:max_samples]
        for index, layer_activations in activations.items()
    }


def R_L_layer(activations: torch.Tensor) -> torch.Tensor:
    """Return pairwise Pearson correlations; constant-unit pairs are NaN."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape (n_samples, n_neurons)")
    if activations.shape[0] < 2:
        raise ValueError("at least two samples are needed for correlation")

    values = activations.to(torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    norms = centered.norm(dim=0)
    denominator = norms[:, None] * norms[None, :]
    correlations = (centered.T @ centered) / denominator.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    correlations = correlations.clamp(-1.0, 1.0).to(activations.dtype)
    return correlations.masked_fill(denominator == 0, float("nan"))


def R_C_layer(activations: torch.Tensor) -> Optional[float]:
    """Return covariance spectral-energy PR, including zero-variance units."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape (n_samples, n_neurons)")
    spectrum = activation_spectral_diagnostics(activations)[
        "covariance_eigenvalues"
    ]
    if spectrum.sum() == 0:
        return None
    return spectral_energy_participation_ratio(spectrum)


def activation_spectral_diagnostics(
    activations: torch.Tensor,
    constant_tolerance: float = 1e-12,
) -> dict:
    """Summarise reference-sample variance and numerical covariance rank.

    Numerical rank is computed from singular values of the centred activation
    matrix. A singular value is retained iff it is greater than
    ``max(n_samples, n_units) * eps(input_dtype) * largest_singular_value``.
    This is the standard scale-relative floating-point rank tolerance; using
    the input dtype reflects the precision in which the activations were
    produced. The returned covariance spectrum is zeroed below this cutoff.
    """
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError("activations must be a 2D matrix with at least 2 rows")
    if not torch.isfinite(activations).all():
        raise ValueError("hidden-layer activations contain non-finite values")

    n_samples, n_units = activations.shape
    values = activations.to(torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    largest = singular_values[0] if singular_values.numel() else values.new_zeros(())
    input_dtype = activations.dtype
    if not input_dtype.is_floating_point:
        input_dtype = torch.float32
    eps = torch.finfo(input_dtype).eps
    tolerance = max(n_samples, n_units) * eps * largest
    numerical_rank = int((singular_values > tolerance).sum().item())

    covariance_spectrum = singular_values.square() / (n_samples - 1)
    if covariance_spectrum.numel() < n_units:
        covariance_spectrum = torch.cat(
            (
                covariance_spectrum,
                covariance_spectrum.new_zeros(
                    n_units - covariance_spectrum.numel()
                ),
            )
        )
    covariance_cutoff = tolerance.square() / (n_samples - 1)
    covariance_spectrum = covariance_spectrum.masked_fill(
        covariance_spectrum <= covariance_cutoff, 0.0
    )
    total_variance = covariance_spectrum.sum()

    unit_std = activations.std(dim=0, unbiased=False)
    constant_indices = torch.nonzero(
        unit_std <= constant_tolerance, as_tuple=False
    ).flatten().tolist()
    zero_indices = torch.nonzero(
        activations.abs().amax(dim=0) <= constant_tolerance,
        as_tuple=False,
    ).flatten().tolist()
    nonconstant_count = n_units - len(constant_indices)
    nullity = n_units - numerical_rank

    if numerical_rank == 0 or total_variance.item() == 0.0:
        status = "all_constant"
    elif nonconstant_count == 1:
        status = "single_nonconstant_unit"
    elif numerical_rank < n_units:
        status = "rank_deficient"
    else:
        status = "ok"

    return {
        "representation_status": status,
        "constant_unit_indices": constant_indices,
        "zero_activation_unit_indices": zero_indices,
        "constant_unit_count": len(constant_indices),
        "nonconstant_unit_count": nonconstant_count,
        "total_activation_variance": total_variance.item(),
        "numerical_rank": numerical_rank,
        "nullity": nullity,
        "numerical_rank_tolerance": tolerance.item(),
        "covariance_eigenvalues": covariance_spectrum,
    }


def covariance_spectrum(activations: torch.Tensor) -> torch.Tensor:
    """Return descending covariance eigenvalues of centered activations."""
    if activations.ndim != 2:
        raise ValueError("activations must have shape (n_samples, n_neurons)")
    if activations.shape[0] < 2:
        raise ValueError("at least two samples are needed for covariance")
    centered = activations.to(torch.float64) - activations.to(
        torch.float64
    ).mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (activations.shape[0] - 1)
    return torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0)


def remaining_spectral_summaries(spectrum: torch.Tensor, dimensions):
    """Return remaining participation ratios and variance fractions."""
    total_variance = spectrum.sum()
    if total_variance <= 0:
        return None, None
    width = spectrum.numel()
    effective_dimensions = {}
    variance_fractions = {}
    for dimension in dimensions:
        remaining = spectrum[: width - dimension]
        effective_dimensions[str(dimension)] = (
            spectral_energy_participation_ratio(remaining)
        )
        variance_fractions[str(dimension)] = (
            remaining.sum() / total_variance
        ).item()
    return effective_dimensions, variance_fractions
