import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from redundancy_project.src.d_ref import (
    D_REF_SIZE,
    ZOO_ROOT,
    load_d_ref_indices,
)
from redundancy_project.src.dataset_utils import (
    prepare_dataset_for_architecture,
)
from redundancy_project.src.loader import (
    load_checkpoint,
    load_run_result,
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
from redundancy_project.src.measures_P import (
    P_C_layer,
    P_L_components,
)
from redundancy_project.src.measures_R import (
    activation_spectral_diagnostics,
    R_C_layer,
    R_L_layer,
    covariance_spectrum,
    get_activations,
    remaining_spectral_summaries,
)
from redundancy_project.src.model_defs import build_model_from_checkpoint
from redundancy_project.src.participation import (
    spectral_energy_participation_ratio,
)
from redundancy_project.src.metric_definitions import (
    F_C_INTERVENTION,
    METRIC_DEFINITION_VERSION,
    P_C_DEFINITION,
    REFERENCE_SAMPLE_POLICY,
    ordered_sample_identity,
)


LOGGER = logging.getLogger(__name__)
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RAW_RESULTS_ROOT = RESULTS_ROOT / "raw"
ACCURACY_TOLERANCE = 1e-3
FC_LAYER_INDICES = (9,)
FC_SUBSPACE_DIMS = DEFAULT_FC_SUBSPACE_DIMS

BASE_SUMMARY_COLUMNS = (
    "P_L_max",
    "P_L_mean",
    "P_C",
    "R_L_max",
    "R_L_mean",
    "R_C",
    "F_L_min",
    "F_L_max",
    "F_L_mean",
    "F_L_std",
)


def summary_columns_for_subspace_dims(fc_subspace_dims):
    return (
        *BASE_SUMMARY_COLUMNS,
        *(
            column
            for k in fc_subspace_dims
            for column in (
                f"F_C_lowvar_{k}",
                f"F_C_random_mean_{k}",
                f"F_C_random_std_{k}",
            )
        ),
    )

SUMMARY_COLUMNS = summary_columns_for_subspace_dims(FC_SUBSPACE_DIMS)

@lru_cache(maxsize=2)
def _load_zoo_resources(config_path: str, dataset_path: str):
    config_file = Path(config_path)
    dataset_file = Path(dataset_path)
    with open(config_file) as source:
        config = json.load(source)
    with torch.serialization.safe_globals([TensorDataset]):
        dataset = torch.load(
            dataset_file,
            map_location="cpu",
            weights_only=True,
        )
    return config, dataset


def _summarize_off_diagonal(matrix: torch.Tensor):
    if matrix is None:
        return None, None
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("expected a square pairwise similarity matrix")
    if matrix.shape[0] < 2:
        raise ValueError("at least two units are needed for off-diagonal summaries")
    off_diagonal = matrix[~torch.eye(matrix.shape[0], dtype=torch.bool)]
    absolute_values = off_diagonal[torch.isfinite(off_diagonal)].abs()
    if absolute_values.numel() == 0:
        raise ValueError(
            "at least two non-constant units are needed for pairwise summaries"
        )
    return absolute_values.max().item(), absolute_values.mean().item()


def _opaque_run_id(run_dir: Path) -> str:
    token = hashlib.sha256(run_dir.name.encode("utf-8")).hexdigest()[:12]
    return f"run_{token}"


def _verify_accuracy(model, testset, run_dir: Path, epoch: int):
    matching_results = [
        result
        for result in load_run_result(run_dir)
        if result.get("training_iteration") == epoch
    ]
    if len(matching_results) != 1:
        raise ValueError(
            f"expected exactly one result record for epoch {epoch}, "
            f"found {len(matching_results)}"
        )

    testloader = DataLoader(
        testset, batch_size=128, shuffle=False
    )
    _, calculated_accuracy = model.test_epoch(testloader, epoch=0)
    stored_accuracy = matching_results[0].get("test_acc")
    if stored_accuracy is None:
        raise ValueError(f"result.json has no test_acc for epoch {epoch}")
    if abs(calculated_accuracy - stored_accuracy) > ACCURACY_TOLERANCE:
        raise ValueError(
            f"accuracy mismatch at epoch {epoch}: "
            f"calculated={calculated_accuracy:.8f}, "
            f"stored={stored_accuracy:.8f}"
        )
    return calculated_accuracy


def _find_measurement_layers(model):
    """Find the last hidden Linear layer and its supported activation."""
    modules = list(model.model.module_list)
    linear_indices = [
        index
        for index, module in enumerate(modules)
        if isinstance(module, torch.nn.Linear)
    ]
    if len(linear_indices) < 2:
        raise ValueError("model needs at least two Linear modules for FC P_L")

    layer_idx = linear_indices[-2]
    next_layer_idx = linear_indices[-1]
    activation_types = (
        torch.nn.ReLU,
        torch.nn.LeakyReLU,
        torch.nn.Tanh,
        torch.nn.Sigmoid,
        torch.nn.GELU,
        torch.nn.SiLU,
    )
    activation_indices = [
        index
        for index in range(layer_idx + 1, next_layer_idx)
        if isinstance(modules[index], activation_types)
    ]
    if len(activation_indices) != 1:
        raise ValueError(
            "expected exactly one supported activation between the final "
            f"hidden and output Linear layers; found {len(activation_indices)}"
        )

    hidden_layer = modules[layer_idx]
    output_layer = modules[next_layer_idx]
    if hidden_layer.out_features != 20 or output_layer.out_features != 10:
        raise ValueError(
            "current MNIST measures require a 20-unit hidden layer and "
            "10-unit output layer"
        )
    return layer_idx, next_layer_idx, activation_indices[0], hidden_layer.out_features


def _validate_layer_activations(activations):
    if not torch.isfinite(activations).all():
        raise ValueError("hidden-layer activations contain non-finite values")
    unit_std = activations.std(dim=0, unbiased=False)
    constant_unit_indices = torch.nonzero(
        unit_std <= 1e-12, as_tuple=False
    ).flatten().tolist()
    zero_activation_indices = torch.nonzero(
        activations.abs().amax(dim=0) <= 1e-12,
        as_tuple=False,
    ).flatten().tolist()
    return constant_unit_indices, zero_activation_indices


def compute_all_measures(
    run_dir,
    epoch,
    layer_idx=9,
    d_ref_indices=None,
    run_id=None,
    raw_results_root=None,
    fc_subspace_dims=FC_SUBSPACE_DIMS,
) -> Optional[Dict[str, object]]:
    """Compute and persist the FC-layer measures for one validated run."""
    run_dir = Path(run_dir)
    identifier = run_id or _opaque_run_id(run_dir)
    try:
        if layer_idx is not None and layer_idx not in FC_LAYER_INDICES:
            raise ValueError(
                f"this population pass only supports FC layer indices "
                f"{FC_LAYER_INDICES}, or automatic hidden-FC selection"
            )

        config_path = run_dir / "config.json"
        if not config_path.is_file():
            config_path = run_dir.parent / "config.json"
        metadata_path = run_dir / "run_metadata.json"
        if metadata_path.is_file():
            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)
            dataset_path = Path(metadata["dataset_path"])
        else:
            dataset_path = run_dir.parent / "dataset.pt"
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"dataset for run is missing: {dataset_path}"
            )
        config, dataset = _load_zoo_resources(
            str(config_path.resolve()), str(dataset_path.resolve())
        )
        architecture = config.get("model::type")
        if architecture is None:
            requested = metadata.get("requested_config", {}) if metadata_path.is_file() else {}
            architecture = requested.get("architecture", "CNN")
        dataset = prepare_dataset_for_architecture(dataset, architecture)
        testset = dataset["testset"]
        images, labels = testset.tensors
        indices = (
            load_d_ref_indices(dataset_size=len(testset))
            if d_ref_indices is None
            else list(d_ref_indices)
        )
        if (
            len(indices) != D_REF_SIZE
            or len(set(indices)) != D_REF_SIZE
            or any(index < 0 or index >= len(testset) for index in indices)
        ):
            raise ValueError(
                f"D_ref must contain {D_REF_SIZE} unique valid test-set indices"
            )
        index_tensor = torch.tensor(indices, dtype=torch.long)
        x_ref = images.index_select(0, index_tensor)
        y_ref = labels.index_select(0, index_tensor)
        ref_loader = DataLoader(
            TensorDataset(x_ref, y_ref), batch_size=128, shuffle=False
        )

        checkpoint = load_checkpoint(run_dir, epoch=epoch)
        model = build_model_from_checkpoint(config, checkpoint, seed=42)
        verified_accuracy = _verify_accuracy(
            model, testset, run_dir, epoch
        )

        if layer_idx is None:
            layer_idx, next_layer_idx, activation_layer_idx, n_units = (
                _find_measurement_layers(model)
            )
        else:
            modules = list(model.model.module_list)
            next_layer_idx = next(
                (
                    index
                    for index in range(layer_idx + 1, len(modules))
                    if isinstance(modules[index], torch.nn.Linear)
                ),
                None,
            )
            if next_layer_idx is None:
                raise ValueError("no following Linear layer found for FC P_L")
            activation_layer_idx = next(
                (
                    index
                    for index in range(layer_idx + 1, next_layer_idx)
                    if isinstance(
                        modules[index],
                        (
                            torch.nn.ReLU,
                            torch.nn.LeakyReLU,
                            torch.nn.Tanh,
                            torch.nn.Sigmoid,
                            torch.nn.GELU,
                            torch.nn.SiLU,
                        ),
                    )
                ),
                None,
            )
            if activation_layer_idx is None:
                raise ValueError(
                    "no supported hidden activation found after target Linear layer"
                )
            n_units = modules[layer_idx].out_features

        fc_subspace_dims = validate_subspace_dimensions(
            fc_subspace_dims, n_units
        )
        modules = list(model.model.module_list)
        weight_key = f"module_list.{layer_idx}.weight"
        next_weight_key = f"module_list.{next_layer_idx}.weight"
        if weight_key not in checkpoint or next_weight_key not in checkpoint:
            raise ValueError("checkpoint is missing FC layer weights")
        if not isinstance(modules[layer_idx], torch.nn.Linear):
            raise ValueError(f"module {layer_idx} is not a Linear layer")
        if not isinstance(modules[next_layer_idx], torch.nn.Linear):
            raise ValueError(
                f"module {next_layer_idx} is not the following Linear layer"
            )

        incoming = checkpoint[weight_key]
        outgoing = checkpoint[next_weight_key]

        pl_in, pl_out = P_L_components(incoming, outgoing.T)
        pl_matrix = (pl_in + pl_out) / 2
        pc_value = P_C_layer(incoming)
        activations = get_activations(
            model,
            ref_loader,
            layer_indices=[activation_layer_idx],
            max_samples=D_REF_SIZE,
        )[activation_layer_idx]
        _validate_layer_activations(activations)
        structure = activation_spectral_diagnostics(activations)
        constant_unit_indices = structure["constant_unit_indices"]
        zero_activation_indices = structure["zero_activation_unit_indices"]
        constant_unit_count = structure["constant_unit_count"]
        if constant_unit_count:
            LOGGER.warning(
                "Run %s has %d constant hidden units on the D_ref reference "
                "sample (%s)",
                identifier,
                constant_unit_count,
                structure["representation_status"],
            )
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

        logprobs_orig = get_logprobs(model, x_ref)
        fl_vector = F_L_layer(
            model, x_ref, activation_layer_idx, n_units
        )
        zero_activation_scores = fl_vector[zero_activation_indices]
        if zero_activation_scores.numel() and not torch.allclose(
            zero_activation_scores,
            torch.zeros_like(zero_activation_scores),
            atol=1e-7,
            rtol=0,
        ):
            raise ValueError(
                "ablation scores for units constantly zero on D_ref must "
                "be zero"
            )
        fc_dimensions = tuple(sorted({0, *fc_subspace_dims}))
        fc_scores = None
        if structure["total_activation_variance"] > 0:
            remaining_dimensions, remaining_variance = (
                remaining_spectral_summaries(
                    activation_spectrum, fc_dimensions
                )
            )
            fc_scores = {}
            nullity = structure["nullity"]
            for k in fc_dimensions:
                if k == 0:
                    low_variance_score = 0.0
                    random_scores = [0.0 for _ in RANDOM_SUBSPACE_SEEDS]
                else:
                    low_variance_score = (
                        F_C_subspace(
                            model,
                            x_ref,
                            logprobs_orig,
                            activation_layer_idx,
                            low_variance_directions(activations, k),
                            mean_activation,
                        )
                    )
                    random_scores = [
                        F_C_subspace(
                            model,
                            x_ref,
                            logprobs_orig,
                            activation_layer_idx,
                            random_orthonormal_directions(
                                n_units, k, seed=seed
                            ).to(
                                device=activations.device,
                                dtype=activations.dtype,
                            ),
                            mean_activation,
                        )
                        for seed in RANDOM_SUBSPACE_SEEDS
                    ]
                fc_scores[str(k)] = {
                    "low_variance": low_variance_score,
                    "random": random_scores,
                    "R_C_remaining_lowvar": remaining_dimensions[str(k)],
                    "V_remaining_lowvar": remaining_variance[str(k)],
                    "contains_zero_variance_directions": (
                        k > 0 and nullity > 0
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
        fc_summaries = {}
        for k in fc_subspace_dims:
            score_record = None if fc_scores is None else fc_scores[str(k)]
            random_values = (
                torch.tensor(score_record["random"], dtype=torch.float64)
                if score_record is not None
                else None
            )
            fc_summaries.update(
                {
                    f"F_C_lowvar_{k}": (
                        None if score_record is None
                        else score_record["low_variance"]
                    ),
                    f"F_C_random_mean_{k}": (
                        None if random_values is None
                        else random_values.mean().item()
                    ),
                    f"F_C_random_std_{k}": (
                        None if random_values is None
                        else random_values.std(unbiased=True).item()
                    ),
                }
            )

        pl_max, pl_mean = _summarize_off_diagonal(pl_matrix)
        rl_max, rl_mean = _summarize_off_diagonal(rl_matrix)
        fl_values = fl_vector.detach().cpu().to(torch.float64)
        summaries = {
            "P_L_max": pl_max,
            "P_L_mean": pl_mean,
            "P_C": pc_value,
            "R_L_max": rl_max,
            "R_L_mean": rl_mean,
            "R_C": rc_value,
            "F_L_min": fl_values.min().item(),
            "F_L_max": fl_values.max().item(),
            "F_L_mean": fl_values.mean().item(),
            "F_L_std": fl_values.std(unbiased=True).item(),
            **fc_summaries,
        }
        if set(summaries) != set(
            summary_columns_for_subspace_dims(fc_subspace_dims)
        ):
            raise RuntimeError("computed summary columns do not match schema")

        raw_root = (
            RAW_RESULTS_ROOT
            if raw_results_root is None
            else Path(raw_results_root)
        )
        raw_root.mkdir(parents=True, exist_ok=True)
        raw_path = raw_root / f"{identifier}.npz"
        if raw_path.exists():
            raise FileExistsError(
                f"refusing to overwrite raw measurements: {raw_path}"
            )
        raw_arrays = {
            "P_L": pl_matrix.detach().cpu().numpy(),
            "P_L_in": pl_in.detach().cpu().numpy(),
            "P_L_out": pl_out.detach().cpu().numpy(),
            "P_C": np.asarray(pc_value),
            "R_L": (
                np.full((n_units, n_units), np.nan)
                if rl_matrix is None
                else rl_matrix.detach().cpu().numpy()
            ),
            "R_C": np.asarray(np.nan if rc_value is None else rc_value),
            "representation_status": np.asarray(
                structure["representation_status"]
            ),
            "constant_unit_count": np.asarray(constant_unit_count),
            "nonconstant_unit_count": np.asarray(
                structure["nonconstant_unit_count"]
            ),
            "total_activation_variance": np.asarray(
                structure["total_activation_variance"]
            ),
            "numerical_rank": np.asarray(structure["numerical_rank"]),
            "nullity": np.asarray(structure["nullity"]),
            "numerical_rank_tolerance": np.asarray(
                structure["numerical_rank_tolerance"]
            ),
            "F_L": fl_values.numpy(),
            "activation_covariance_eigenvalues": (
                activation_spectrum.detach().cpu().numpy()
            ),
            "reference_activation_mean": mean_activation.detach().cpu().numpy(),
            "metric_definition_version": np.asarray(
                METRIC_DEFINITION_VERSION
            ),
            "p_c_definition": np.asarray(P_C_DEFINITION),
            "fc_intervention": np.asarray(F_C_INTERVENTION),
        }
        for dimension, values in (fc_scores or {}).items():
            raw_arrays[f"F_C_lowvar_{dimension}"] = np.asarray(
                np.nan
                if values["low_variance"] is None
                else values["low_variance"]
            )
            raw_arrays[f"F_C_random_{dimension}"] = np.asarray(
                values["random"]
            )
            raw_arrays[f"R_C_remaining_lowvar_{dimension}"] = np.asarray(
                values["R_C_remaining_lowvar"]
            )
            raw_arrays[f"V_remaining_lowvar_{dimension}"] = np.asarray(
                values["V_remaining_lowvar"]
            )
        np.savez_compressed(raw_path, **raw_arrays)
        return {
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "p_c_definition": P_C_DEFINITION,
            "fc_intervention": F_C_INTERVENTION,
            "reference_sample_policy": REFERENCE_SAMPLE_POLICY,
            "run_id": identifier,
            "target_layer_idx": layer_idx,
            "activation_layer_idx": activation_layer_idx,
            "constant_unit_count": constant_unit_count,
            "constant_unit_indices": constant_unit_indices,
            "active_unit_count": structure["nonconstant_unit_count"],
            "representation_status": structure["representation_status"],
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
            "zero_activation_unit_indices": zero_activation_indices,
            "fc_subspace_dims": list(fc_subspace_dims),
            "F_C": fc_scores,
            "activation_covariance_eigenvalues": (
                activation_spectrum.detach().cpu().tolist()
            ),
            "reference_activation_mean": (
                mean_activation.detach().cpu().tolist()
            ),
            "reference_sample": {
                "split": "D_ref_test",
                "size": len(indices),
                "indices": indices,
                "identity": ordered_sample_identity(
                    "D_ref_test", indices, "indices_in_fixed_test_split"
                ),
            },
            "P_C": pc_value,
            "R_C": rc_value,
            "P_L_in": pl_in.detach().cpu().tolist(),
            "P_L_out": pl_out.detach().cpu().tolist(),
            "P_L": pl_matrix.detach().cpu().tolist(),
            "F_L_constant_unit_scores": fl_vector[constant_unit_indices]
            .detach()
            .cpu()
            .tolist(),
            "F_L_zero_activation_scores": zero_activation_scores.detach()
            .cpu()
            .tolist(),
            "summaries": summaries,
            "verified_accuracy": verified_accuracy,
            "raw_path": str(raw_path),
        }
    except Exception as error:
        LOGGER.warning(
            "Skipping run %s (%s): %s",
            identifier,
            run_dir.name,
            error,
        )
        return None
