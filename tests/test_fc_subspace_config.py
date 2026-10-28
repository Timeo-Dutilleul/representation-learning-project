import unittest

from redundancy_project.run_pipeline import (
    FC_SUBSPACE_DIMS,
    SUMMARY_COLUMNS,
    summary_columns_for_subspace_dims,
)
from redundancy_project.src.measures_F import (
    validate_subspace_dimensions,
)


class FCSubspaceConfigurationTests(unittest.TestCase):
    def test_default_summary_schema_is_unchanged(self):
        self.assertEqual(
            summary_columns_for_subspace_dims(FC_SUBSPACE_DIMS),
            SUMMARY_COLUMNS,
        )

    def test_custom_dimensions_create_matching_summary_columns(self):
        columns = summary_columns_for_subspace_dims((2, 7))
        self.assertEqual(
            columns[-6:],
            (
                "F_C_lowvar_2",
                "F_C_random_mean_2",
                "F_C_random_std_2",
                "F_C_lowvar_7",
                "F_C_random_mean_7",
                "F_C_random_std_7",
            ),
        )

    def test_dimensions_must_be_unique_integers_within_layer_width(self):
        self.assertEqual(validate_subspace_dimensions([2, 7, 20], 20), (2, 7, 20))
        invalid_dimensions = (
            [],
            [1, 1],
            [1, 1.5],
            [0],
            [21],
        )
        for dimensions in invalid_dimensions:
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    validate_subspace_dimensions(dimensions, 20)


if __name__ == "__main__":
    unittest.main()
