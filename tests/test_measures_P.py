import unittest

import torch

from redundancy_project.src.measures_P import (
    P_C_layer,
    P_L_layer,
    effective_rank,
    normalize_rows,
    pairwise_cosine,
)


class MeasuresPTests(unittest.TestCase):
    def test_cosine_identity_rows(self):
        weights = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 1.0, 0.0]]
        )
        cosine = pairwise_cosine(weights)
        self.assertTrue(
            torch.allclose(cosine[0, 1], torch.tensor(1.0), atol=1e-5)
        )

    def test_effective_rank_bounds(self):
        self.assertAlmostEqual(effective_rank(torch.eye(5)), 5.0, delta=1e-3)
        self.assertAlmostEqual(
            effective_rank(torch.ones(5, 3)), 1.0, delta=1e-3
        )

    def test_effective_rank_uses_spectral_energy_and_is_scale_invariant(self):
        rank_one = torch.outer(torch.tensor([1.0, 2.0]), torch.tensor([2.0, -1.0]))
        equal_rank_three = torch.diag(torch.tensor([4.0, 4.0, 4.0, 0.0]))
        self.assertAlmostEqual(effective_rank(rank_one), 1.0, places=6)
        self.assertAlmostEqual(effective_rank(equal_rank_three), 3.0, places=6)
        self.assertAlmostEqual(
            effective_rank(equal_rank_three * 17.0),
            effective_rank(equal_rank_three),
            places=6,
        )

    def test_P_C_uses_row_normalized_incoming_weights(self):
        weights = torch.tensor([[100.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(P_C_layer(weights), 2.0, places=6)
        self.assertAlmostEqual(
            P_C_layer(weights), effective_rank(normalize_rows(weights)), places=6
        )

    def test_P_C_is_bounded_by_nonzero_matrix_width(self):
        weights = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        )
        value = P_C_layer(weights)
        self.assertGreaterEqual(value, 1.0)
        self.assertLessEqual(value, weights.shape[0])

    def test_permutation_invariance_P_C(self):
        torch.manual_seed(0)
        weights = torch.randn(6, 4)
        permutation = torch.randperm(6)
        self.assertAlmostEqual(
            P_C_layer(weights),
            P_C_layer(weights[permutation]),
            delta=1e-5,
        )

    def test_relu_rescaling_invariance_P_L_and_P_C(self):
        torch.manual_seed(1)
        n_current, n_in, n_out = 5, 4, 3
        weights_in = torch.randn(n_current, n_in)
        weights_out = torch.randn(n_out, n_current)
        scaling = torch.rand(n_current) * 2 + 0.1

        scaled_weights_in = weights_in * scaling.unsqueeze(1)
        scaled_weights_out = weights_out / scaling.unsqueeze(0)

        pl_original = P_L_layer(weights_in, weights_out.T)
        pl_scaled = P_L_layer(scaled_weights_in, scaled_weights_out.T)
        self.assertTrue(torch.allclose(pl_original, pl_scaled, atol=1e-4))
        self.assertAlmostEqual(
            P_C_layer(weights_in),
            P_C_layer(scaled_weights_in),
            delta=1e-4,
        )

    def test_normalize_rows_handles_zero_rows(self):
        normalized = normalize_rows(torch.tensor([[0.0, 0.0], [3.0, 4.0]]))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized[0], torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
