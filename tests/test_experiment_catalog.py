import tempfile
import unittest
from pathlib import Path

from redundancy_project.experiment_catalog import (
    build_catalog,
    expand_plan,
    load_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentCatalogTests(unittest.TestCase):
    def test_example_plan_expands_to_expected_unique_runs(self):
        plan = load_plan(PROJECT_ROOT / "experiment_design.yaml")
        runs = expand_plan(plan)
        self.assertEqual(len(runs), 32)
        self.assertEqual(len({run["run_id"] for run in runs}), 32)
        self.assertEqual(len({run["condition_id"] for run in runs}), 16)
        self.assertEqual(
            {run["seed"] for run in runs},
            {11, 22},
        )

    def test_catalog_will_not_be_overwritten(self):
        plan_path = PROJECT_ROOT / "experiment_design.yaml"
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "catalog.json"
            build_catalog(plan_path, catalog_path)
            with self.assertRaises(FileExistsError):
                build_catalog(plan_path, catalog_path)

    def test_arbitrary_width_parameter_is_rejected(self):
        plan = load_plan(PROJECT_ROOT / "experiment_design.yaml")
        plan["experiment"]["conditions"][0]["widths"] = [32, 64]
        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            from redundancy_project.experiment_catalog import _validate_plan

            _validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
