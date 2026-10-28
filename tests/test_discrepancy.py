import unittest

import torch

from redundancy_project.src.discrepancy import behavioral_discrepancy


class BehavioralDiscrepancyTests(unittest.TestCase):
    def test_discrepancy_zero_when_identical(self):
        logprobs = torch.log_softmax(torch.randn(50, 10), dim=1)
        discrepancy = behavioral_discrepancy(logprobs, logprobs)
        self.assertLess(abs(discrepancy), 1e-6)

    def test_discrepancy_positive_when_different(self):
        torch.manual_seed(0)
        logprobs_1 = torch.log_softmax(torch.randn(50, 10), dim=1)
        logprobs_2 = torch.log_softmax(torch.randn(50, 10), dim=1)
        discrepancy = behavioral_discrepancy(logprobs_1, logprobs_2)
        self.assertGreater(discrepancy, 0)

    def test_discrepancy_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            behavioral_discrepancy(torch.zeros(2, 3), torch.zeros(2, 4))


if __name__ == "__main__":
    unittest.main()
