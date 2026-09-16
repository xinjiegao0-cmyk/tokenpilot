import unittest
from decimal import Decimal

from tokenpilot.benchmark.compare import (
    compare_runs,
)

from tokenpilot.telemetry.experiment import (
    ExperimentRun,
    QualityMetric, RunStatus,
)

from tokenpilot.telemetry.ledger import (
    ResourceEvent,
)


class ComparisonTests(unittest.TestCase):

    def make_baseline(self):
        run = ExperimentRun(
            task_id="research-001",
            strategy="baseline",
        )

        run.ledger.record(
            ResourceEvent(
                category="model",
                input_tokens=18000,
                output_tokens=2000,
                cost_usd=Decimal("0.200"),
            )
        )

        run.set_quality_metric(
            QualityMetric(
                name="quality",
                value=0.92,
            )
        )

        run.status = RunStatus.SUCCEEDED
        return run


    def make_candidate(self):
        run = ExperimentRun(
            task_id="research-001",
            strategy="tokenpilot",
        )

        run.ledger.record(
            ResourceEvent(
                category="task_model",
                input_tokens=9000,
                output_tokens=1000,
                cost_usd=Decimal("0.110"),
                is_overhead=False,
            )
        )

        run.ledger.record(
            ResourceEvent(
                category="planner",
                input_tokens=800,
                output_tokens=200,
                cost_usd=Decimal("0.030"),
                is_overhead=True,
            )
        )

        run.set_quality_metric(
            QualityMetric(
                name="quality",
                value=0.915,
            )
        )

        run.status = RunStatus.SUCCEEDED
        return run


    def test_cost_comparison_includes_overhead_once(self):
        baseline = self.make_baseline()
        candidate = self.make_candidate()

        result = compare_runs(
            baseline,
            candidate,
            quality_metric="quality",
            max_quality_drop=0.02,
        )

        # baseline = 0.200
        # candidate task = 0.110
        # candidate overhead = 0.030
        # candidate total = 0.140
        #
        # true saving = 0.060
        self.assertEqual(
            result.net_cost_saving_usd,
            Decimal("0.060"),
        )

        self.assertTrue(
            result.quality_gate_passed
        )

        self.assertTrue(
            result.candidate_is_net_better
        )


    def test_quality_failure_blocks_success(self):
        baseline = self.make_baseline()
        candidate = self.make_candidate()

        candidate.set_quality_metric(
            QualityMetric(
                name="quality",
                value=0.80,
            )
        )

        result = compare_runs(
            baseline,
            candidate,
            quality_metric="quality",
            max_quality_drop=0.02,
        )

        self.assertFalse(
            result.quality_gate_passed
        )

        self.assertFalse(
            result.candidate_is_net_better
        )


if __name__ == "__main__":
    unittest.main()
