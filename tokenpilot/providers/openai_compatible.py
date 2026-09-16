"""Non-streaming text Chat Completions adapter; all vendor choices are explicit."""
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Callable, Optional

from tokenpilot.providers.base import BaseProvider, ProviderError, ProviderRequest, ProviderResponse, Usage
from tokenpilot.providers.normalization import normalize_chat_usage
from tokenpilot.providers.transport import BaseTransport, UrllibTransport, validate_https_url


@dataclass(frozen=True)
class CostEvidence:
    amount_usd: Decimal
    source: str
    kind: str = "reported"

    def __post_init__(self):
        if (not isinstance(self.amount_usd, Decimal) or not self.amount_usd.is_finite()
                or self.amount_usd < 0):
            raise ValueError("cost must be a finite nonnegative Decimal")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("cost source is required")
        if self.kind not in {"reported", "estimated", "simulated"}:
            raise ValueError("invalid cost kind")


BillingExtractor = Callable[[Mapping], Optional[CostEvidence]]


class OpenAICompatibleProvider(BaseProvider):
    def __init__(self, *, name: str, base_url: str, api_key: Optional[str] = None,
                 transport: Optional[BaseTransport] = None, timeout: float = 30.0,
                 token_limit_field: str = "max_completion_tokens",
                 billing_extractor: Optional[BillingExtractor] = None,
                 billing_version: Optional[str] = None):
        validate_https_url(base_url)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("provider name is required")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if token_limit_field not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError("unsupported token limit field")
        if billing_extractor is not None and (
            not callable(billing_extractor) or not isinstance(billing_version, str)
            or not billing_version.strip()
        ):
            raise ValueError("billing extractor requires a public version identifier")
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()
                                   or not api_key.isascii() or any(c.isspace() for c in api_key)):
            raise ValueError("credential must be a nonempty ASCII token without whitespace")
        self.name = name
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._transport = transport if transport is not None else UrllibTransport()
        self._timeout = timeout
        self._token_limit_field = token_limit_field
        self._billing_extractor = billing_extractor
        self._billing_version = billing_version

    @property
    def simulation(self):
        return self._transport.simulation

    def describe(self):
        return {
            "provider": self.name, "adapter": "chat-completions-v1",
            "endpoint_sha256": hashlib.sha256(self._url.encode()).hexdigest(),
            "token_limit_field": self._token_limit_field, "timeout_seconds": self._timeout,
            "billing_version": self._billing_version,
        }

    def _public_value(self, value):
        # Only a small set of response identifiers is retained; never raw headers/body.
        if not isinstance(value, str) or not re.fullmatch(r"[\w./:-]{1,160}", value):
            return None
        if self._api_key and self._api_key in value:
            return None
        return value

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        if not isinstance(request, ProviderRequest):
            raise ValueError("request must be a ProviderRequest")
        if not self.simulation and not self._api_key:
            raise ProviderError("missing_credential")
        payload = {
            "model": request.model, "messages": [{"role": "user", "content": request.prompt}],
            self._token_limit_field: request.max_output_tokens, "stream": False, "n": 1,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.seed is not None:
            payload["seed"] = request.seed
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        try:
            http = self._transport.post(self._url, headers,
                json.dumps(payload, allow_nan=False).encode("utf-8"), self._timeout)
        except ProviderError:
            raise
        except (TimeoutError,):
            raise ProviderError("timeout") from None
        except Exception:
            raise ProviderError("transport_error") from None
        if http.status_code != 200:
            raise ProviderError("http_error", http_status=http.status_code)
        try:
            raw = json.loads(http.body, parse_float=Decimal)
        except (ValueError, TypeError, UnicodeError):
            raise ProviderError("invalid_json") from None
        if not isinstance(raw, Mapping):
            raise ProviderError("invalid_response")
        usage_complete = True
        try:
            usage = normalize_chat_usage(raw.get("usage"))
        except (ValueError, TypeError):
            usage = Usage(0, 0)
            usage_complete = False
        metadata = {key: self._public_value(raw.get(key))
                    for key in ("id", "model", "system_fingerprint")}
        response = ProviderResponse("", usage, completed=False, metadata=metadata,
                                    usage_complete=usage_complete)
        if self._billing_extractor is not None:
            try:
                evidence = self._billing_extractor(raw)
                if evidence is not None:
                    if not isinstance(evidence, CostEvidence):
                        raise ValueError("invalid billing evidence")
                    source = evidence.source
                    if self._api_key:
                        source = source.replace(self._api_key, "[REDACTED]")
                    response = replace(response, cost_usd=evidence.amount_usd, cost_source=source,
                                       cost_kind="simulated" if self.simulation else evidence.kind)
            except Exception:
                raise ProviderError("billing_error", response=response if usage_complete else None) from None
        if not usage_complete:
            # Billing evidence can be valid independently of the usage payload.
            raise ProviderError("invalid_usage",
                                response=response if response.cost_usd is not None else None)
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise ProviderError("invalid_response", response=response)
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            raise ProviderError("invalid_response", response=response)
        reason = choice.get("finish_reason")
        if not isinstance(reason, str) or reason not in {"stop", "length", "content_filter", "tool_calls", "function_call"}:
            raise ProviderError("invalid_response", response=response)
        metadata["finish_reason"] = reason
        text = message.get("content")
        refused = bool(message.get("refusal"))
        unsupported = bool(message.get("tool_calls") or message.get("function_call"))
        if refused:
            metadata["finish_reason"] = "refusal"
        if unsupported:
            metadata["finish_reason"] = "tool_calls"
        complete = reason == "stop" and not refused and not unsupported
        if complete and not isinstance(text, str):
            raise ProviderError("invalid_response", response=response)
        return replace(response, text=text if isinstance(text, str) else "", completed=complete)
