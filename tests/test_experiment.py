import time
import unittest

from tokenpilot.telemetry.experiment import (
    ExperimentRun,
    QualityMetric,
    RunProfiler,
    RunStatus,
)


class ExperimentRunTests(unittest.TestCase):

    def test_profiler_records_wall_clock(self) -> None:
        run = ExperimentRun(
            task_id="task-001",
            strategy="baseline",
        )

        with RunProfiler(run):
            time.sleep(0.001)

        self.assertEqual(
            run.status,
            RunStatus.SUCCEEDED,
        )

        self.assertIsNotNone(
            run.wall_clock_ms
        )

        self.assertGreater(
            run.wall_clock_ms,
            0,
        )


    def test_quality_metric_is_recorded(self) -> None:
        run = ExperimentRun(
            task_id="task-001",
            strategy="baseline",
        )

        run.set_quality_metric(
            QualityMetric(
                name="task_quality",
                value=0.92,
            )
        )

        metric = run.get_quality_metric(
            "task_quality"
        )

        self.assertIsNotNone(metric)
        self.assertEqual(metric.value, 0.92)


    def test_exception_marks_run_as_error(self) -> None:
        run = ExperimentRun(
            task_id="task-error",
            strategy="baseline",
        )

        try:
            with RunProfiler(run):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

        self.assertEqual(
            run.status,
            RunStatus.ERROR,
        )

        self.assertIn(
            "boom",
            run.error,
        )


    def test_manual_failure_is_preserved(self) -> None:
        run = ExperimentRun(
            task_id="task-fail",
            strategy="candidate",
        )

        with RunProfiler(run):
            run.mark_failed(
                "quality contract failed"
            )

        self.assertEqual(
            run.status,
            RunStatus.FAILED,
        )


if __name__ == "__main__":
    unittest.main()
