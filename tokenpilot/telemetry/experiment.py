from __future__ import annotations

import math
import time
import uuid

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from tokenpilot.telemetry.ledger import ResourceLedger


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class QualityMetric:
    """
    One externally measured quality signal.

    TokenPilot does not decide what "quality" means here.
    Benchmarks provide the measurement.
    """

    name: str
    value: float
    unit: str = "score"
    higher_is_better: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "higher_is_better": self.higher_is_better,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentRun:
    """
    Reproducible record of one benchmark execution.
    """

    task_id: str
    strategy: str

    run_id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    started_at: datetime = field(
        default_factory=utc_now
    )

    finished_at: datetime | None = None

    status: RunStatus = RunStatus.RUNNING

    wall_clock_ms: float | None = None

    ledger: ResourceLedger = field(
        default_factory=ResourceLedger
    )

    quality_metrics: dict[str, QualityMetric] = field(
        default_factory=dict
    )

    config: dict[str, Any] = field(
        default_factory=dict
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    error: str | None = None

    def set_quality_metric(
        self,
        metric: QualityMetric,
    ) -> None:
        if not metric.name.strip():
            raise ValueError(
                "quality metric name cannot be empty"
            )

        if type(metric.value) not in (int, float) or not math.isfinite(metric.value):
            raise ValueError("quality metric value must be finite")
        self.quality_metrics[metric.name] = metric

    def get_quality_metric(
        self,
        name: str,
    ) -> QualityMetric | None:
        return self.quality_metrics.get(name)

    def mark_failed(
        self,
        reason: str,
    ) -> None:
        self.status = RunStatus.FAILED
        self.error = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "strategy": self.strategy,
            "status": self.status.value,

            "started_at": self.started_at.isoformat(),

            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at
                else None
            ),

            "wall_clock_ms": self.wall_clock_ms,

            "quality_metrics": {
                name: metric.to_dict()
                for name, metric
                in self.quality_metrics.items()
            },

            "resources": self.ledger.to_dict(),

            "config": self.config,
            "metadata": self.metadata,
            "error": self.error,
        }


class RunProfiler:
    """
    Measures true wall-clock duration for one experiment run.

    Resource events are still explicitly recorded in the ledger,
    because wall-clock time and accumulated event latency are
    different concepts.
    """

    def __init__(
        self,
        run: ExperimentRun,
    ) -> None:
        self.run = run
        self._start_ns: int | None = None

    def __enter__(self) -> ExperimentRun:
        self._start_ns = time.perf_counter_ns()
        return self.run

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> Literal[False]:
        if self._start_ns is None:
            raise RuntimeError(
                "RunProfiler was not started"
            )

        elapsed_ns = (
            time.perf_counter_ns()
            - self._start_ns
        )

        self.run.wall_clock_ms = (
            elapsed_ns / 1_000_000
        )

        self.run.finished_at = utc_now()

        if exc_value is not None:
            self.run.status = RunStatus.ERROR
            self.run.error = (
                f"{type(exc_value).__name__}: "
                f"{exc_value}"
            )

            return False

        if self.run.status == RunStatus.RUNNING:
            self.run.status = RunStatus.SUCCEEDED

        return False
