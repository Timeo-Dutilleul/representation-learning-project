import torch


def spectral_energy_participation_ratio(eigenvalues: torch.Tensor) -> float:
    """Return PR of non-negative spectral energy values."""
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a one-dimensional tensor")
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("eigenvalues must be finite")
    if torch.any(eigenvalues < 0):
        raise ValueError("eigenvalues must be non-negative")
    values = eigenvalues.to(torch.float64)
    numerator = values.sum().square()
    denominator = values.square().sum()
    if denominator == 0:
        return 0.0
    return (numerator / denominator).item()
