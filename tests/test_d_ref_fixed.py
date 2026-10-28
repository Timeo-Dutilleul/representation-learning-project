import unittest

from redundancy_project.src.d_ref import D_REF_PATH, load_d_ref_indices


class DRefFixedTests(unittest.TestCase):
    def test_d_ref_indices_are_persisted_and_reloaded_identically(self):
        first_load = load_d_ref_indices(D_REF_PATH)
        second_load = load_d_ref_indices(D_REF_PATH)

        self.assertEqual(first_load, second_load)
        self.assertEqual(len(first_load), 500)
        self.assertEqual(first_load, list(range(500)))


if __name__ == "__main__":
    unittest.main()
