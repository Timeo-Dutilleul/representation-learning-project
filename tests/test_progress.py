import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from redundancy_project.src.progress import format_duration, progress_step


class ProgressStepTests(unittest.TestCase):
    def test_format_duration_handles_seconds_and_minutes(self):
        self.assertEqual(format_duration(12.34), "12.3s")
        self.assertEqual(format_duration(62), "1m 02s")

    def test_success_reports_elapsed_time_on_the_step_line(self):
        output = io.StringIO()
        with patch(
            "redundancy_project.src.progress.time.perf_counter",
            side_effect=(1.0, 3.5),
        ):
            with redirect_stdout(output), progress_step("Training epochs 1-5"):
                pass

        self.assertEqual(
            output.getvalue(),
            "Training epochs 1-5... done in 2.5s\n",
        )

    def test_failure_reports_elapsed_time_and_propagates(self):
        output = io.StringIO()
        with patch(
            "redundancy_project.src.progress.time.perf_counter",
            side_effect=(1.0, 62.0),
        ):
            with redirect_stdout(output):
                with self.assertRaisesRegex(ValueError, "failed"):
                    with progress_step("Temporal measurements"):
                        raise ValueError("failed")

        self.assertEqual(
            output.getvalue(),
            "Temporal measurements... failed after 1m 01s\n",
        )


if __name__ == "__main__":
    unittest.main()
