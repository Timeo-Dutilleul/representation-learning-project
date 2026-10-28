import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.run_pipeline import (
    _find_measurement_layers,
    _validate_layer_activations,
)
from redundancy_project.src.measures_F import (
    DEFAULT_FC_SUBSPACE_DIMS,
    RANDOM_SUBSPACE_SEEDS,
    F_C_subspace,
    F_L_layer,
    get_logprobs,
    low_variance_directions,
    random_orthonormal_directions,
    remaining_effective_dimensions,
    remaining_variance_fractions,
    validate_subspace_dimensions,
)
from redundancy_project.src.measures_P import P_C_layer, P_L_components
from redundancy_project.src.measures_R import (
    activation_spectral_diagnostics,
    R_C_layer,
    R_L_layer,
    get_activations,
    remaining_spectral_summaries,
)
from redundancy_project.src.participation import (
    spectral_energy_participation_ratio,
)
from redundancy_project.src.metric_definitions import (
    F_C_INTERVENTION,
    METRIC_DEFINITION_VERSION,
    P_C_DEFINITION,
    REFERENCE_SAMPLE_POLICY,
)


SUBSPACE_DIMS = DEFAULT_FC_SUBSPACE_DIMS


def get_temporal_measurement_epochs(epochs: int, every: int):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if every < 1:
        raise ValueError("measurement frequency must be positive")
    return sorted(set(range(every, epochs + 1, every)) | {epochs})


def _matrix_values(matrix: torch.Tensor):
    if matrix is None:
        return None
    return [
        [float(value) if math.isfinite(value) else None for value in row]
        for row in matrix.detach().cpu().tolist()
    ]


def _measure_split(
    model,
    x,
    y,
    activation_layer_idx,
    n_units,
    fc_subspace_dims=SUBSPACE_DIMS,
    sample_identity=None,
):
    fc_subspace_dims = validate_subspace_dimensions(
        fc_subspace_dims, n_units
    )
    device = next(model.model.parameters()).device
    x = x.to(device)
    y = y.to(device)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=128,
        shuffle=False,
    )
    activations = get_activations(
        model,
        loader,
        layer_indices=[activation_layer_idx],
        max_samples=len(x),
    )[activation_layer_idx]
    _validate_layer_activations(activations)
    structure = activation_spectral_diagnostics(activations)
    constant_unit_indices = structure["constant_unit_indices"]
    zero_activation_indices = structure["zero_activation_unit_indices"]
    rl_matrix = (
        R_L_layer(activations)
        if structure["nonconstant_unit_count"] >= 2
        else None
    )
    activation_spectrum = structure["covariance_eigenvalues"]
    rc_value = (
        spectral_energy_participation_ratio(activation_spectrum)
        if structure["total_activation_variance"] > 0
        else None
    )
    mean_activation = activations.mean(dim=0)
    logprobs_orig = get_logprobs(model, x)
    fl_vector = F_L_layer(model, x, activation_layer_idx, n_units)
    zero_scores = fl_vector[zero_activation_indices]
    if zero_scores.numel() and not torch.allclose(
        zero_scores,
        torch.zeros_like(zero_scores),
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError(
            "ablation scores for units constantly zero on the measurement "
            "sample must be zero"
        )

    fc_dimensions = tuple(sorted({0, *fc_subspace_dims}))
    fc_scores = None
    if structure["total_activation_variance"] > 0:
        remaining_dimensions, remaining_variance = remaining_spectral_summaries(
            activation_spectrum, fc_dimensions
        )
        fc_scores = {}
        nullity = structure["nullity"]
        for dimension in fc_dimensions:
            if dimension == 0:
                low_variance_score = 0.0
                random_scores = [0.0 for _ in RANDOM_SUBSPACE_SEEDS]
            else:
                low_variance_score = (
                    F_C_subspace(
                        model,
                        x,
                        logprobs_orig,
                        activation_layer_idx,
                        low_variance_directions(activations, dimension),
                        mean_activation,
                    )
                )
                random_scores = []
                for seed in RANDOM_SUBSPACE_SEEDS:
                    directions = random_orthonormal_directions(
                        n_units, dimension, seed=seed
                    ).to(device=device, dtype=activations.dtype)
                    random_scores.append(
                        F_C_subspace(
                            model,
                            x,
                            logprobs_orig,
                            activation_layer_idx,
                            directions,
                            mean_activation,
                        )
                    )
            fc_scores[str(dimension)] = {
                "low_variance": low_variance_score,
                "random": random_scores,
                "R_C_remaining_lowvar": remaining_dimensions[str(dimension)],
                "V_remaining_lowvar": remaining_variance[str(dimension)],
                "contains_zero_variance_directions": (
                    dimension > 0 and nullity > 0
                ),
                "zero_variance_directions_individually_identifiable": (
                    nullity == 0
                ),
                "nullspace_ablation_effect_on_reference_sample": (
                    "zero_to_numerical_precision"
                    if nullity > 0
                    else "not_applicable"
                ),
            }

    return {
        "sample_count": len(x),
        "activation_width": n_units,
        "reference_sample": sample_identity,
        "reference_activation_mean": mean_activation.detach().cpu().tolist(),
        "activation_covariance_eigenvalues": (
            activation_spectrum.detach().cpu().tolist()
        ),
        "representation_status": structure["representation_status"],
        "constant_unit_count": structure["constant_unit_count"],
        "nonconstant_unit_count": structure["nonconstant_unit_count"],
        "total_activation_variance": structure[
            "total_activation_variance"
        ],
        "numerical_rank": structure["numerical_rank"],
        "nullity": structure["nullity"],
        "zero_variance_directions_individually_identifiable": (
            structure["nullity"] == 0
        ),
        "centered_nullspace_ablation_effect": (
            "zero_to_numerical_precision"
            if (
                structure["nullity"] > 0
                and structure["total_activation_variance"] > 0
            )
            else "not_applicable"
        ),
        "numerical_rank_tolerance": structure[
            "numerical_rank_tolerance"
        ],
        "R_L": _matrix_values(rl_matrix),
        "R_C": rc_value,
        "F_L": fl_vector.detach().cpu().tolist(),
        "F_C": fc_scores,
        "fc_intervention": F_C_INTERVENTION,
        "constant_unit_indices": constant_unit_indices,
        "zero_activation_unit_indices": zero_activation_indices,
    }


def measure_checkpoint(
    model,
    train_samples,
    validation_samples,
    fc_subspace_dims=SUBSPACE_DIMS,
    train_sample_identity=None,
    validation_sample_identity=None,
):
    """Measure P once and R/F on fixed train and validation samples."""
    layer_idx, next_layer_idx, activation_layer_idx, n_units = (
        _find_measurement_layers(model)
    )
    checkpoint = model.model.state_dict()
    incoming = checkpoint[f"module_list.{layer_idx}.weight"]
    outgoing = checkpoint[f"module_list.{next_layer_idx}.weight"]
    pl_in, pl_out = P_L_components(incoming, outgoing.T)
    pl_matrix = (pl_in + pl_out) / 2
    return {
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "p_c_definition": P_C_DEFINITION,
        "fc_intervention": F_C_INTERVENTION,
        "reference_sample_policy": REFERENCE_SAMPLE_POLICY,
        "target_layer_idx": layer_idx,
        "activation_layer_idx": activation_layer_idx,
        "activation_width": n_units,
        "P_L_in": _matrix_values(pl_in),
        "P_L_out": _matrix_values(pl_out),
        "P_L": _matrix_values(pl_matrix),
        "P_C": P_C_layer(incoming),
        "train": _measure_split(
            model,
            *train_samples,
            activation_layer_idx,
            n_units,
            fc_subspace_dims,
            train_sample_identity,
        ),
        "validation": _measure_split(
            model,
            *validation_samples,
            activation_layer_idx,
            n_units,
            fc_subspace_dims,
            validation_sample_identity,
        ),
    }
