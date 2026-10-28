import tempfile
import unittest
from pathlib import Path

from redundancy_project.scripts.export_dashboard import export_dashboard


class ExportDashboardTests(unittest.TestCase):
    def test_exports_only_global_and_per_run_html_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "experiment"
            run_dir = source / "runs" / "run-example"
            run_dir.mkdir(parents=True)
            (source / "dashboard.html").write_text(
                'frame.dataset.dashboard="runs/run-example/dashboard.html"',
                encoding="utf-8",
            )
            (run_dir / "dashboard.html").write_text("run panel", encoding="utf-8")
            (run_dir / "measurements.json").write_text("{}", encoding="utf-8")
            output = root / "docs" / "dashboard"

            exported = export_dashboard(source, output)

            self.assertEqual(
                {path.relative_to(output.resolve()).as_posix() for path in exported},
                {"index.html", "runs/run-example/dashboard.html"},
            )
            self.assertEqual(
                (output / "index.html").read_text(encoding="utf-8"),
                'frame.dataset.dashboard="runs/run-example/dashboard.html"',
            )
            self.assertEqual(
                (output / "runs/run-example/dashboard.html").read_text(
                    encoding="utf-8"
                ),
                "run panel",
            )
            self.assertFalse((output / "runs/run-example/measurements.json").exists())
            self.assertTrue((run_dir / "measurements.json").exists())

    def test_refuses_existing_output_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "experiment"
            run_dir = source / "runs" / "run-example"
            run_dir.mkdir(parents=True)
            (source / "dashboard.html").write_text("new index", encoding="utf-8")
            (run_dir / "dashboard.html").write_text("new run", encoding="utf-8")
            output = root / "docs" / "dashboard"
            output.mkdir(parents=True)
            (output / "index.html").write_text("keep me", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                export_dashboard(source, output)

            self.assertEqual(
                (output / "index.html").read_text(encoding="utf-8"), "keep me"
            )

    def test_overwrite_replaces_matching_html_but_never_deletes_other_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "experiment"
            run_dir = source / "runs" / "run-example"
            run_dir.mkdir(parents=True)
            source_index = source / "dashboard.html"
            source_run_dashboard = run_dir / "dashboard.html"
            source_index.write_text("new index", encoding="utf-8")
            source_run_dashboard.write_text("new run", encoding="utf-8")
            output = root / "docs" / "dashboard"
            old_run_dir = output / "runs" / "old-run"
            old_run_dir.mkdir(parents=True)
            (output / "index.html").write_text("old index", encoding="utf-8")
            (old_run_dir / "dashboard.html").write_text("keep old", encoding="utf-8")

            export_dashboard(source, output, overwrite=True)

            self.assertEqual(
                (output / "index.html").read_text(encoding="utf-8"), "new index"
            )
            self.assertEqual(
                (output / "runs/run-example/dashboard.html").read_text(
                    encoding="utf-8"
                ),
                "new run",
            )
            self.assertEqual(
                (old_run_dir / "dashboard.html").read_text(encoding="utf-8"),
                "keep old",
            )
            self.assertEqual(source_index.read_text(encoding="utf-8"), "new index")
            self.assertEqual(
                source_run_dashboard.read_text(encoding="utf-8"), "new run"
            )


if __name__ == "__main__":
    unittest.main()
