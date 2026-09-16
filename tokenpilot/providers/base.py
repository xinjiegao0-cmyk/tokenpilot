"""Provider contract. Credentials belong inside adapters, never experiment records."""
from abc import ABC, abstractmethod
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from tokenpilot.telemetry.ledger import ResourceEvent, ResourceLedger


@dataclass(frozen=True)
class Usage:
    # Cached input is a subset of input, never an additional token bucket.
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    # Optional output breakdown, never added to total_tokens.
    reasoning_tokens: Optional[int] = None

    def __post_init__(self):
        ResourceLedger._validate(ResourceEvent(
            category="usage", input_tokens=self.input_tokens,
            output_tokens=self.output_tokens, cached_input_tokens=self.cached_input_tokens))

        if self.reasoning_tokens is not None and (
            type(self.reasoning_tokens) is not int
            or not 0 <= self.reasoning_tokens <= self.output_tokens
        ):
            raise ValueError("reasoning_tokens must be an integer subset of output_tokens")

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    prompt: str
    max_output_tokens: int
    temperature: Optional[float] = 0.0
    seed: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string")
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if self.temperature is not None and (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be None or a finite number between 0 and 2")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    usage: Usage
    # None means unknown, not free. Adapter must supply documented billing evidence.
    cost_usd: Optional[Decimal] = None
    cost_source: Optional[str] = None
    cost_kind: str = "reported"
    completed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    # False means zero token counts are placeholders, not measured zero usage.
    usage_complete: bool = True


class ProviderError(Exception):
    """Safe diagnostic code, with partial accounting when a response was received."""

    def __init__(self, code: str, *, response: Optional[ProviderResponse] = None,
                 http_status: Optional[int] = None):
        if code not in {
            "network_disabled", "timeout", "transport_error", "http_error",
            "invalid_json", "invalid_usage", "invalid_response", "billing_error",
            "response_too_large", "missing_credential",
        }:
            raise ValueError("invalid provider error code")
        super().__init__(code)
        self.code = code
        self.response = response
        self.http_status = http_status


class BaseProvider(ABC):
    name: str
    simulation: bool = False

    def describe(self) -> Dict[str, Any]:
        """Public reproducibility settings only; never return credentials."""
        return {"provider": self.name}

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """One attempt; adapters must not silently retry or omit billed usage."""
        raise NotImplementedError
