"""Explicit, dated estimates. No prices are inferred or downloaded."""
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

from tokenpilot.providers.base import Usage


def money(value: str) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError("price must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class PricingProfile:
    provider: str
    model: str
    as_of: str
    source: str
    input_per_million_usd: Decimal
    output_per_million_usd: Decimal
    cached_input_per_million_usd: Optional[Decimal] = None

    def __post_init__(self):
        if not all(isinstance(v, str) and v.strip()
                   for v in (self.provider, self.model, self.source, self.as_of)):
            raise ValueError("provider, model, date and source are required")
        date.fromisoformat(self.as_of)
        if self.input_per_million_usd is None or self.output_per_million_usd is None:
            raise ValueError('input and output prices are required')
        for value in (self.input_per_million_usd, self.output_per_million_usd,
                      self.cached_input_per_million_usd):
            if value is not None and (not isinstance(value, Decimal)
                                     or not value.is_finite() or value < 0):
                raise ValueError("prices must be finite nonnegative Decimals")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        if data.pop("currency", "USD") != "USD":
            raise ValueError("explicit USD prices are required; no implicit FX conversion")
        for key in ("input_per_million_usd", "output_per_million_usd",
                    "cached_input_per_million_usd"):
            if key in data and data[key] is not None:
                if not isinstance(data[key], str):
                    raise ValueError("JSON prices must be decimal strings")
                data[key] = money(data[key])
        return cls(**data)

    def estimate(self, usage: Usage, *, provider: str, model: str) -> Optional[Decimal]:
        if (provider, model) != (self.provider, self.model):
            raise ValueError("price profile does not match requested provider/model")
        if usage.cached_input_tokens and self.cached_input_per_million_usd is None:
            return None
        cached_rate = self.cached_input_per_million_usd or Decimal("0")
        # Reasoning is already in output; cached input replaces regular input.
        return ((usage.input_tokens - usage.cached_input_tokens) * self.input_per_million_usd
                + usage.cached_input_tokens * cached_rate
                + usage.output_tokens * self.output_per_million_usd) / Decimal("1000000")


@dataclass(frozen=True)
class CostTotals:
    """Total includes local compute valuation and all optimization API calls."""
    task_usd: Optional[Decimal]
    overhead_usd: Optional[Decimal]
    kind: str

    def __post_init__(self):
        if self.kind not in {"measured", "estimated", "simulated", "unknown"}:
            raise ValueError("unsupported cost kind")
        for value in (self.task_usd, self.overhead_usd):
            if value is not None and (not isinstance(value, Decimal)
                                     or not value.is_finite() or value < 0):
                raise ValueError("costs must be finite nonnegative Decimals")

    @property
    def total(self) -> Optional[Decimal]:
        if self.kind == "unknown" or self.task_usd is None or self.overhead_usd is None:
            return None
        return self.task_usd + self.overhead_usd


def cost_metrics(baseline: CostTotals, candidate: CostTotals, *, quality_passed: bool):
    baseline_total, candidate_total = baseline.total, candidate.total
    comparable = (baseline_total is not None and candidate_total is not None
                  and baseline.kind == candidate.kind)
    saving = (baseline_total - candidate_total
              if comparable and baseline_total is not None and candidate_total is not None else None)
    ratio = (candidate.overhead_usd / candidate_total
             if comparable and candidate_total and candidate.overhead_usd is not None else None)
    return {
        "accounting_complete": bool(comparable),
        "billing_kind": baseline.kind if comparable else "unknown",
        "baseline_cost_usd": str(baseline.total) if baseline.total is not None else None,
        "candidate_cost_usd": str(candidate.total) if candidate.total is not None else None,
        "net_saving_usd": str(saving) if saving is not None else None,
        "overhead_ratio": str(ratio) if ratio is not None else None,
        "break_even": saving >= 0 if saving is not None else None,
        "cost_success": bool(comparable and quality_passed and saving is not None and saving > 0),
        "warning": "SIMULATED_SMOKE_TEST_ONLY" if baseline.kind == "simulated" else None,
    }
