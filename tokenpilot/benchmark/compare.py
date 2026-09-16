from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from tokenpilot.telemetry.experiment import (
    ExperimentRun, RunStatus,
)


def _percent_change(
    original: float,
    difference: float,
) -> float | None:
    if original == 0:
        return None

    return (
        difference / original
    ) * 100.0


@dataclass
class ComparisonResult:
    baseline_run_id: str
    candidate_run_id: str

    baseline_cost_usd: Decimal
    candidate_cost_usd: Decimal
    net_cost_saving_usd: Decimal
    cost_saving_percent: float | None

    baseline_tokens: int
    candidate_tokens: int
    token_saving: int
    token_saving_percent: float | None

    baseline_wall_clock_ms: float | None
    candidate_wall_clock_ms: float | None

    quality_metric: str | None = None
    baseline_quality: float | None = None
    candidate_quality: float | None = None
    quality_delta: float | None = None
    quality_gate_passed: bool | None = None

    candidate_is_net_better: bool = False
    simulation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulation": self.simulation,
            "warning": "SIMULATED_SMOKE_TEST_ONLY" if self.simulation else None,
            "baseline_run_id":
                self.baseline_run_id,

            "candidate_run_id":
                self.candidate_run_id,

            "baseline_cost_usd":
                str(self.baseline_cost_usd),

            "candidate_cost_usd":
                str(self.candidate_cost_usd),

            "net_cost_saving_usd":
                str(self.net_cost_saving_usd),

            "cost_saving_percent":
                self.cost_saving_percent,

            "baseline_tokens":
                self.baseline_tokens,

            "candidate_tokens":
                self.candidate_tokens,

            "token_saving":
                self.token_saving,

            "token_saving_percent":
                self.token_saving_percent,

            "baseline_wall_clock_ms":
                self.baseline_wall_clock_ms,

            "candidate_wall_clock_ms":
                self.candidate_wall_clock_ms,

            "quality_metric":
                self.quality_metric,

            "baseline_quality":
                self.baseline_quality,

            "candidate_quality":
                self.candidate_quality,

            "quality_delta":
                self.quality_delta,

            "quality_gate_passed":
                self.quality_gate_passed,

            "candidate_is_net_better":
                self.candidate_is_net_better,
        }


def compare_runs(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
    *,
    quality_metric: str | None = None,
    max_quality_drop: float | None = None,
) -> ComparisonResult:

    if baseline.task_id != candidate.task_id:
        raise ValueError("cannot compare different tasks")
    simulation = bool(baseline.metadata.get("simulation", False))
    if simulation != bool(candidate.metadata.get("simulation", False)):
        raise ValueError("cannot compare simulated and real runs")
    for key in ("dataset_version", "evaluator_version"):
        if baseline.config.get(key) != candidate.config.get(key):
            raise ValueError("comparison version mismatch: " + key)
    if max_quality_drop is not None:
        if quality_metric is None or not math.isfinite(max_quality_drop) or max_quality_drop < 0:
            raise ValueError("quality threshold requires a metric and finite nonnegative drop")

    baseline_cost = (
        baseline.ledger.total_cost_usd
    )

    candidate_cost = (
        candidate.ledger.total_cost_usd
    )

    # IMPORTANT:
    #
    # candidate.total_cost already includes TokenPilot
    # overhead. Do NOT subtract overhead again.
    net_cost_saving = (
        baseline_cost - candidate_cost
    )

    baseline_tokens = (
        baseline.ledger.total_tokens
    )

    candidate_tokens = (
        candidate.ledger.total_tokens
    )

    token_saving = (
        baseline_tokens - candidate_tokens
    )

    quality_gate_passed: bool | None = None

    baseline_quality: float | None = None
    candidate_quality: float | None = None
    quality_delta: float | None = None

    if quality_metric is not None:
        baseline_metric = (
            baseline.get_quality_metric(
                quality_metric
            )
        )

        candidate_metric = (
            candidate.get_quality_metric(
                quality_metric
            )
        )

        if baseline_metric is None:
            raise ValueError(
                "baseline does not contain "
                f"quality metric: {quality_metric}"
            )

        if candidate_metric is None:
            raise ValueError(
                "candidate does not contain "
                f"quality metric: {quality_metric}"
            )

        if (
            baseline_metric.higher_is_better
            != candidate_metric.higher_is_better
        ):
            raise ValueError(
                "quality metric direction mismatch"
            )

        if baseline_metric.unit != candidate_metric.unit:
            raise ValueError("quality metric unit mismatch")
        if not all(math.isfinite(m.value) for m in (baseline_metric, candidate_metric)):
            raise ValueError("quality values must be finite")

        baseline_quality = (
            baseline_metric.value
        )

        candidate_quality = (
            candidate_metric.value
        )

        quality_delta = (
            candidate_quality
            - baseline_quality
        )

        if max_quality_drop is not None:
            if max_quality_drop < 0:
                raise ValueError(
                    "max_quality_drop "
                    "cannot be negative"
                )

            if baseline_metric.higher_is_better:
                quality_gate_passed = (
                    candidate_quality
                    >= baseline_quality
                    - max_quality_drop
                )

            else:
                quality_gate_passed = (
                    candidate_quality
                    <= baseline_quality
                    + max_quality_drop
                )

    cost_saving_percent = _percent_change(
        float(baseline_cost),
        float(net_cost_saving),
    )

    token_saving_percent = _percent_change(
        float(baseline_tokens),
        float(token_saving),
    )

    quality_ok = (
        quality_gate_passed is True
    )

    candidate_is_net_better = (
        net_cost_saving > 0
        and quality_ok
        and baseline.status == RunStatus.SUCCEEDED
        and candidate.status == RunStatus.SUCCEEDED
        and baseline.metadata.get("accounting_complete", True)
        and candidate.metadata.get("accounting_complete", True)
    )

    return ComparisonResult(
        simulation=simulation,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,

        baseline_cost_usd=baseline_cost,
        candidate_cost_usd=candidate_cost,

        net_cost_saving_usd=net_cost_saving,

        cost_saving_percent=(
            cost_saving_percent
        ),

        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,

        token_saving=token_saving,

        token_saving_percent=(
            token_saving_percent
        ),

        baseline_wall_clock_ms=(
            baseline.wall_clock_ms
        ),

        candidate_wall_clock_ms=(
            candidate.wall_clock_ms
        ),

        quality_metric=quality_metric,
        baseline_quality=baseline_quality,
        candidate_quality=candidate_quality,
        quality_delta=quality_delta,
        quality_gate_passed=(
            quality_gate_passed
        ),

        candidate_is_net_better=(
            candidate_is_net_better
        ),
    )
