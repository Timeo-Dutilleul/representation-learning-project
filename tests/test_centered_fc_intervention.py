import unittest

import torch

from redundancy_project.src.measures_F import (
    ablate_subspace_hook,
    low_variance_directions,
    remaining_effective_dimensions,
    remaining_variance_fractions,
    random_orthonormal_directions,
)
from redundancy_project.src.measures_R import R_C_layer


class CenteredFCInterventionTests(unittest.TestCase):
    def test_zero_dimensional_projection_is_unchanged(self):
        activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mean = activations.mean(dim=0)
        result = ablate_subspace_hook(
            torch.empty((0, 2)), mean
        )(None, (), activations)
        self.assertTrue(torch.equal(result, activations))

    def test_centered_projection_preserves_mean_and_removes_selected_axis(self):
        activations = torch.tensor([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
        mean = activations.mean(dim=0)
        directions = torch.tensor([[1.0, 0.0]])
        result = ablate_subspace_hook(
            directions, mean
        )(None, (), activations)
        self.assertTrue(torch.allclose(result.mean(dim=0), mean, atol=1e-7))
        self.assertTrue(
            torch.allclose(
                (result - mean) @ directions.T,
                torch.zeros((3, 1)),
                atol=1e-7,
            )
        )
        self.assertTrue(
            torch.allclose(result, mean.expand_as(activations), atol=1e-7)
        )

    def test_generated_directions_are_orthonormal_and_pca_is_centered(self):
        torch.manual_seed(4)
        activations = torch.randn(8, 3)
        for directions in (
            random_orthonormal_directions(3, 2, seed=12),
            low_variance_directions(activations, 2),
        ):
            self.assertTrue(
                torch.allclose(
                    directions @ directions.T,
                    torch.eye(2),
                    atol=1e-6,
                )
            )

    def test_remaining_spectrum_includes_origin_and_all_removed_endpoint(self):
        activations = torch.tensor(
            [[-1.0, -2.0], [1.0, 2.0], [-2.0, 1.0], [2.0, -1.0]]
        )
        remaining = remaining_effective_dimensions(activations, (0, 2))
        variance = remaining_variance_fractions(activations, (0, 2))
        self.assertAlmostEqual(remaining["0"], R_C_layer(activations), places=6)
        self.assertAlmostEqual(remaining["0"], 2.0, places=6)
        self.assertAlmostEqual(variance["0"], 1.0, places=6)
        self.assertEqual(remaining["2"], 0.0)
        self.assertEqual(variance["2"], 0.0)


if __name__ == "__main__":
    unittest.main()
