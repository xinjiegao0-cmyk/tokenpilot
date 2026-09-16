"""Explicit capabilities, not inferred equivalence between API-compatible hosts."""
from dataclasses import dataclass
from decimal import Decimal

from tokenpilot.providers.base import ProviderRequest
from tokenpilot.telemetry.pricing import PricingProfile


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    version: str
    token_limit_field: str
    supports_temperature: bool = False
    supports_seed: bool = False
    # Caller-supplied documented upper bound for chat framing token overhead.
    framing_token_allowance: int = 256

    def __post_init__(self):
        if not self.name or not self.version:
            raise ValueError('capability name/version required')
        if self.token_limit_field not in {'max_tokens', 'max_completion_tokens'}:
            raise ValueError('unsupported lowering')
        if type(self.framing_token_allowance) is not int or self.framing_token_allowance < 0:
            raise ValueError('nonnegative framing allowance required')

    def validate(self, request: ProviderRequest):
        if request.temperature is not None and not self.supports_temperature:
            raise ValueError('temperature unsupported by selected capability profile')
        if request.seed is not None and not self.supports_seed:
            raise ValueError('seed unsupported by selected capability profile')

    def input_upper_bound(self, request):
        # This bound only applies to byte-level tokenizers with <=1 token per byte.
        # Provider hidden input, tools and other billing are outside this contract.
        return len(request.prompt.encode('utf-8')) + self.framing_token_allowance


def reserve_api_cost(request, profile: CapabilityProfile, prices: PricingProfile):
    profile.validate(request)
    highest_input_rate = max(prices.input_per_million_usd,
                             prices.cached_input_per_million_usd or Decimal('0'))
    return (Decimal(profile.input_upper_bound(request)) * highest_input_rate
            + Decimal(request.max_output_tokens) * prices.output_per_million_usd) / Decimal('1000000')
