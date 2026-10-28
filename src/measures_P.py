import torch

from redundancy_project.src.participation import (
    spectral_energy_participation_ratio,
)


def normalize_rows(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each row of W to unit norm."""
    norms = W.norm(dim=1, keepdim=True).clamp_min(eps)
    return W / norms


def pairwise_cosine(W: torch.Tensor) -> torch.Tensor:
    """Return the pairwise cosine similarities between rows of W."""
    normalized = normalize_rows(W)
    return normalized @ normalized.T


def effective_rank(W: torch.Tensor, eps: float = 1e-12) -> float:
    """Return the spectral-energy participation ratio of W."""
    singular_values = torch.linalg.svdvals(W).clamp_min(0)
    return spectral_energy_participation_ratio(singular_values.square())


def P_L_components(
    W_in: torch.Tensor, W_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return incoming and outgoing direction cosines per neuron pair."""
    if W_in.shape[0] != W_out.shape[0]:
        raise ValueError(
            "W_in and W_out must have the same number of neuron rows"
        )
    return pairwise_cosine(W_in), pairwise_cosine(W_out)


def P_L_layer(W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """Average incoming and outgoing direction cosines per neuron pair."""
    cosine_in, cosine_out = P_L_components(W_in, W_out)
    return (cosine_in + cosine_out) / 2


def P_C_layer(W_in: torch.Tensor) -> float:
    """PR of spectral energy in row-normalized incoming weight directions."""
    return effective_rank(normalize_rows(W_in))
