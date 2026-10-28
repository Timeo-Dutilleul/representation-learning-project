import hashlib
import json


METRIC_DEFINITION_VERSION = 2
P_C_DEFINITION = "spectral_energy_participation_ratio_of_row_normalized_W_in"
F_C_INTERVENTION = "centered_pca_projection"
REFERENCE_SAMPLE_POLICY = (
    "fixed_ordered_train_and_validation_samples_for_temporal_measures; "
    "fixed_D_ref_test_sample_for_final_summary"
)


def ordered_sample_identity(split, indices, coordinate_system):
    payload = json.dumps(
        {
            "split": split,
            "coordinate_system": coordinate_system,
            "indices": list(indices),
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "split": split,
        "coordinate_system": coordinate_system,
        "size": len(indices),
        "ordered_indices_sha256": hashlib.sha256(payload).hexdigest(),
    }
