"""
SIMULATED SMOKE TEST ONLY.

These numbers are synthetic and MUST NOT be used as
TokenPilot benchmark claims or research results.

The purpose is only to validate the accounting and
comparison pipeline before real providers are connected.
"""

import json

from decimal import Decimal

from tokenpilot.benchmark.compare import (
    compare_runs,
)

from tokenpilot.telemetry.experiment import (
    ExperimentRun,
    QualityMetric,
    RunProfiler,
)

from tokenpilot.telemetry.ledger import (
    ResourceEvent,
)


def main():
    baseline = ExperimentRun(
        task_id="simulated-research-001",
        strategy="full-history-baseline",
        metadata={
            "simulation": True,
        },
    )

    with RunProfiler(baseline):
        baseline.ledger.record(
            ResourceEvent(
                category="model_call",
                input_tokens=18000,
                output_tokens=2000,
                cost_usd=Decimal("0.200"),
                latency_ms=2100,
            )
        )

        baseline.set_quality_metric(
            QualityMetric(
                name="quality",
                value=0.92,
            )
        )


    candidate = ExperimentRun(
        task_id="simulated-research-001",
        strategy="tokenpilot-v0.1-simulation",
        metadata={
            "simulation": True,
        },
    )

    with RunProfiler(candidate):
        candidate.ledger.record(
            ResourceEvent(
                category="task_execution",
                input_tokens=9000,
                output_tokens=1000,
                cost_usd=Decimal("0.110"),
                latency_ms=1500,
                is_overhead=False,
            )
        )

        candidate.ledger.record(
            ResourceEvent(
                category="tokenpilot_planner",
                input_tokens=800,
                output_tokens=200,
                cost_usd=Decimal("0.030"),
                latency_ms=300,
                is_overhead=True,
            )
        )

        candidate.set_quality_metric(
            QualityMetric(
                name="quality",
                value=0.915,
            )
        )


    comparison = compare_runs(
        baseline,
        candidate,
        quality_metric="quality",
        max_quality_drop=0.02,
    )


    print(
        json.dumps(
            {
                "warning": (
                    "SIMULATED_SMOKE_TEST_ONLY"
                ),

                "baseline":
                    baseline.to_dict(),

                "candidate":
                    candidate.to_dict(),

                "comparison":
                    comparison.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
