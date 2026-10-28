import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import yaml

from redundancy_project import launch_experiment


class LaunchExperimentTests(unittest.TestCase):
    def _write_settings(self, path, **overrides):
        settings = {
            "plan": "plan.yaml",
            "run_limit": 1,
            "temporal_dashboard": True,
            "measurement_every": 1,
            "measurement_samples": 2,
            "output_prefix": "trial",
            "save_each_launch": False,
        }
        settings.update(overrides)
        path.write_text(yaml.safe_dump(settings), encoding="utf-8")
        (path.parent / "plan.yaml").write_text("plan: test\n", encoding="utf-8")
        (path.parent / "analysis_plan.md").write_text(
            "# Test analysis plan\n", encoding="utf-8"
        )

    def _mock_launch_dependencies(self, stack):
        def create_catalog(plan_path, catalog_path):
            Path(catalog_path).write_text("{}", encoding="utf-8")
            return [{"run_id": "run-1"}, {"run_id": "run-2"}]

        stack.enter_context(
            patch.object(
                launch_experiment, "build_catalog", side_effect=create_catalog
            )
        )
        stack.enter_context(
            patch.object(launch_experiment, "write_experiment_dashboard")
        )
        return stack.enter_context(
            patch.object(
                launch_experiment,
                "train_catalog",
                side_effect=lambda **kwargs: kwargs["output_root"].mkdir(
                    parents=True
                ),
            )
        )

    def test_exploratory_launch_replaces_only_its_dedicated_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_path = project_root / "settings.yaml"
            self._write_settings(settings_path)
            output_root = project_root / "results" / "trial-exploratory"
            old_run = output_root / "runs" / "old-run"
            old_run.mkdir(parents=True)
            (old_run / "stale.txt").write_text("old", encoding="utf-8")
            unrelated = project_root / "results" / "keep.txt"
            unrelated.parent.mkdir(parents=True, exist_ok=True)
            unrelated.write_text("keep", encoding="utf-8")

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        launch_experiment, "PROJECT_ROOT", project_root
                    )
                )
                train_catalog = self._mock_launch_dependencies(stack)
                first_output = launch_experiment.launch(settings_path)
                (first_output / "runs" / "new-stale.txt").write_text(
                    "remove on next launch", encoding="utf-8"
                )
                second_output = launch_experiment.launch(settings_path)

            self.assertEqual(first_output, output_root)
            self.assertEqual(second_output, output_root)
            self.assertFalse((output_root / "runs" / "old-run").exists())
            self.assertFalse((output_root / "runs" / "new-stale.txt").exists())
            self.assertTrue((output_root / "experiment_catalog.json").is_file())
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "keep")
            self.assertEqual(train_catalog.call_count, 2)

    def test_timestamped_launch_preserves_previous_launches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_path = project_root / "settings.yaml"
            self._write_settings(settings_path, save_each_launch=True)

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        launch_experiment, "PROJECT_ROOT", project_root
                    )
                )
                self._mock_launch_dependencies(stack)
                first_output = launch_experiment.launch(settings_path)
                second_output = launch_experiment.launch(settings_path)

            self.assertNotEqual(first_output, second_output)
            self.assertTrue(first_output.is_dir())
            self.assertTrue(second_output.is_dir())
            for output in (first_output, second_output):
                provenance = output / "provenance"
                self.assertTrue((provenance / "experiment_launch.yaml").is_file())
                self.assertTrue((provenance / "experiment_design.yaml").is_file())
                self.assertTrue((provenance / "analysis_plan.md").is_file())
                version = yaml.safe_load(
                    (provenance / "code_version.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertIn("sha256_python_sources", version)
                self.assertIn("sha256_protocol_files", version)
                self.assertTrue((output / "launch_metadata.json").is_file())

    def test_custom_fc_subspace_dimensions_are_passed_to_catalog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_path = project_root / "settings.yaml"
            self._write_settings(settings_path, fc_subspace_dims=[2, 7, 20])

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        launch_experiment, "PROJECT_ROOT", project_root
                    )
                )
                train_catalog = self._mock_launch_dependencies(stack)
                launch_experiment.launch(
                    settings_path, run_limit_override=1
                )

            self.assertEqual(
                train_catalog.call_args.kwargs["fc_subspace_dims"],
                (2, 7, 20),
            )
            output = project_root / "results" / "trial-exploratory"
            launch_metadata = yaml.safe_load(
                (output / "launch_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                launch_metadata["effective_settings"]["run_limit"], 1
            )
            self.assertEqual(launch_metadata["run_limit_override"], 1)
            self.assertEqual(launch_metadata["selected_run_count"], 1)

    def test_exact_run_ids_are_archived_and_passed_to_catalog(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_path = project_root / "settings.yaml"
            self._write_settings(
                settings_path, run_limit=0, save_each_launch=True
            )
            selected_ids = ["run-2", "run-1"]

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        launch_experiment, "PROJECT_ROOT", project_root
                    )
                )
                train_catalog = self._mock_launch_dependencies(stack)
                output = launch_experiment.launch(
                    settings_path, run_ids=selected_ids
                )

            self.assertEqual(
                train_catalog.call_args.kwargs["run_ids"], selected_ids
            )
            metadata = json.loads(
                (output / "launch_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["selected_run_ids"], selected_ids)
            self.assertEqual(metadata["selected_run_count"], 2)

    def test_invalid_fc_subspace_dimensions_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_path = project_root / "settings.yaml"
            self._write_settings(settings_path, fc_subspace_dims=[1, 20, 21])

            with patch.object(
                launch_experiment, "PROJECT_ROOT", project_root
            ):
                with self.assertRaisesRegex(
                    ValueError, "hidden-layer width"
                ):
                    launch_experiment.launch(settings_path)

            self.assertFalse((project_root / "results").exists())

    def test_partial_failure_still_writes_dashboard_and_reports_its_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            settings_path = project_root / "settings.yaml"
            self._write_settings(settings_path)

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        launch_experiment, "PROJECT_ROOT", project_root
                    )
                )
                self._mock_launch_dependencies(stack)
                stack.enter_context(
                    patch.object(
                        launch_experiment,
                        "train_catalog",
                        side_effect=RuntimeError("one run failed"),
                    )
                )

                def write_partial_dashboard(output_root):
                    dashboard = Path(output_root) / "dashboard.html"
                    dashboard.write_text("<html></html>", encoding="utf-8")
                    return dashboard

                stack.enter_context(
                    patch.object(
                        launch_experiment,
                        "write_experiment_dashboard",
                        side_effect=write_partial_dashboard,
                    )
                )
                with self.assertRaisesRegex(
                    RuntimeError, "Partial experiment dashboard:"
                ) as error:
                    launch_experiment.launch(settings_path)

            self.assertIn(
                "one run failed",
                str(error.exception),
            )
            self.assertTrue(
                (project_root / "results" / "trial-exploratory" / "dashboard.html").is_file()
            )


if __name__ == "__main__":
    unittest.main()
