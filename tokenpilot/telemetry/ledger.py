from __future__ import annotations

import math

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any


@dataclass
class ResourceEvent:
    """
    One measurable resource-consumption event.

    Normalization rule:
    - input_tokens = total prompt/input tokens
    - cached_input_tokens = subset of input_tokens served from cache
    - output_tokens = generated tokens

    Therefore:
        total_tokens = input_tokens + output_tokens

    cached_input_tokens MUST NOT be added again to total_tokens.
    metadata["usage_breakdown"]["reasoning_tokens"], when present, is an
    output subset and MUST NOT be added again either.
    """

    category: str

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    cost_usd: Decimal = Decimal("0")
    latency_ms: float = 0.0

    provider: str | None = None
    model: str | None = None

    # True when this cost was created by TokenPilot itself,
    # e.g. planning, compression, semantic extraction, verification.
    is_overhead: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return self.input_tokens - self.cached_input_tokens


@dataclass
class ResourceLedger:
    """
    Append-only accounting ledger for TokenPilot.

    Important:
    This component only measures resources.
    It does NOT estimate savings and does NOT judge quality.
    """

    events: list[ResourceEvent] = field(default_factory=list)

    def record(self, event: ResourceEvent) -> None:
        self._validate(event)
        self.events.append(event)

    @staticmethod
    def _validate(event: ResourceEvent) -> None:
        for value in (event.input_tokens, event.output_tokens, event.cached_input_tokens):
            if type(value) is not int:
                raise ValueError("token counts must be integers")
        if not isinstance(event.cost_usd, Decimal) or not event.cost_usd.is_finite():
            raise ValueError("cost_usd must be a finite Decimal")
        if not math.isfinite(event.latency_ms):
            raise ValueError("latency_ms must be finite")
        if not event.category.strip():
            raise ValueError("category cannot be empty")

        if event.input_tokens < 0:
            raise ValueError("input_tokens cannot be negative")

        if event.output_tokens < 0:
            raise ValueError("output_tokens cannot be negative")

        if event.cached_input_tokens < 0:
            raise ValueError("cached_input_tokens cannot be negative")

        if event.cached_input_tokens > event.input_tokens:
            raise ValueError(
                "cached_input_tokens cannot exceed input_tokens"
            )

        if event.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")

        if event.latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")

    @property
    def total_input_tokens(self) -> int:
        return sum(event.input_tokens for event in self.events)

    @property
    def total_output_tokens(self) -> int:
        return sum(event.output_tokens for event in self.events)

    @property
    def total_cached_input_tokens(self) -> int:
        return sum(event.cached_input_tokens for event in self.events)

    @property
    def total_uncached_input_tokens(self) -> int:
        return sum(event.uncached_input_tokens for event in self.events)

    @property
    def total_tokens(self) -> int:
        return sum(event.total_tokens for event in self.events)

    @property
    def task_tokens(self) -> int:
        return sum(
            event.total_tokens
            for event in self.events
            if not event.is_overhead
        )

    @property
    def overhead_tokens(self) -> int:
        return sum(
            event.total_tokens
            for event in self.events
            if event.is_overhead
        )

    @property
    def total_cost_usd(self) -> Decimal:
        return sum(
            (event.cost_usd for event in self.events),
            start=Decimal("0"),
        )

    @property
    def overhead_cost_usd(self) -> Decimal:
        return sum(
            (
                event.cost_usd
                for event in self.events
                if event.is_overhead
            ),
            start=Decimal("0"),
        )

    @property
    def task_cost_usd(self) -> Decimal:
        return self.total_cost_usd - self.overhead_cost_usd

    @property
    def accumulated_event_latency_ms(self) -> float:
        """
        Sum of event latencies.

        This is intentionally NOT called end-to-end latency because
        concurrent events may overlap in wall-clock time.
        """
        return sum(event.latency_ms for event in self.events)

    def summary(self) -> dict[str, Any]:
        return {
            "event_count": len(self.events),

            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,

            "cached_input_tokens": self.total_cached_input_tokens,
            "uncached_input_tokens": self.total_uncached_input_tokens,

            "total_tokens": self.total_tokens,
            "task_tokens": self.task_tokens,
            "overhead_tokens": self.overhead_tokens,

            "task_cost_usd": str(self.task_cost_usd),
            "overhead_cost_usd": str(self.overhead_cost_usd),
            "total_cost_usd": str(self.total_cost_usd),

            "accumulated_event_latency_ms": (
                self.accumulated_event_latency_ms
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        serialized_events: list[dict[str, Any]] = []

        for event in self.events:
            data = asdict(event)
            data["cost_usd"] = str(event.cost_usd)
            serialized_events.append(data)

        return {
            "summary": self.summary(),
            "events": serialized_events,
        }
