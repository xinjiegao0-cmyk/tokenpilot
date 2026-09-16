"""Single-case runner: record each known charge before evaluating output."""
import hashlib
import json
import time
from dataclasses import asdict
from decimal import Decimal
from typing import Callable

from tokenpilot.providers.base import BaseProvider, ProviderError, ProviderRequest, ProviderResponse
from tokenpilot.telemetry.experiment import ExperimentRun, QualityMetric, RunProfiler, RunStatus
from tokenpilot.telemetry.ledger import ResourceEvent


class BaselineRunner:
    def __init__(self, provider: BaseProvider):
        self.provider = provider

    def _record(self, run: ExperimentRun, response: ProviderResponse, latency_ms: float):
        has_cost = response.cost_usd is not None and bool(response.cost_source)
        # Estimates and synthetic bills must never become real paid costs.
        kind = "simulated" if self.provider.simulation else response.cost_kind
        known = has_cost and (kind == "reported" or (kind == "simulated" and self.provider.simulation))
        run.ledger.record(ResourceEvent(
            category="model_call", input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            cost_usd=response.cost_usd if known else Decimal("0"),
            latency_ms=latency_ms, provider=self.provider.name, model=run.config["model"],
            metadata={"cost_known": known, "cost_source": response.cost_source,
                      "usage_known": response.usage_complete,
                      "cost_kind": kind if has_cost else "unknown",
                      "unverified_cost_usd": str(response.cost_usd) if has_cost and not known else None},
        ))
        run.metadata["accounting_complete"] = known and response.usage_complete
        run.metadata["response"] = dict(response.metadata)
        if not known:
            run.mark_failed("billing evidence unavailable; cost is incomplete")
            run.metadata["failure_stage"] = "accounting"
        elif not response.usage_complete:
            run.mark_failed("usage unavailable; token accounting is incomplete")
            run.metadata["failure_stage"] = "accounting"

    def run(self, task_id: str, request: ProviderRequest,
            evaluator: Callable[[str], QualityMetric], *, dataset_version: str,
            evaluator_version: str, allow_paid_api: bool = False) -> ExperimentRun:
        if type(allow_paid_api) is not bool:
            raise ValueError("allow_paid_api must be a boolean")
        if not self.provider.simulation and not allow_paid_api:
            raise PermissionError("Real provider execution requires explicit consent")
        if any(not isinstance(v, str) or not v.strip()
               for v in (task_id, dataset_version, evaluator_version)):
            raise ValueError("task, dataset and evaluator versions are required")
        if not isinstance(request, ProviderRequest) or not callable(evaluator):
            raise ValueError("request and evaluator are required")
        fingerprint = hashlib.sha256(json.dumps(
            asdict(request), sort_keys=True, allow_nan=False).encode()).hexdigest()
        run = ExperimentRun(
            task_id=task_id, strategy="full-history-baseline",
            config={"model": request.model, "max_output_tokens": request.max_output_tokens,
                    "temperature": request.temperature, "seed": request.seed,
                    "request_sha256": fingerprint, "dataset_version": dataset_version,
                    "evaluator_version": evaluator_version,
                    "provider_config": self.provider.describe()},
            metadata={"provider": self.provider.name, "simulation": self.provider.simulation,
                      "accounting_complete": False,
                      "warning": "SIMULATED_SMOKE_TEST_ONLY" if self.provider.simulation else None},
        )
        stage = "provider"
        with RunProfiler(run):
            try:
                start = time.perf_counter()
                failure = None
                try:
                    response = self.provider.generate(request)
                except ProviderError as exc:
                    failure = exc
                    response = exc.response
                latency_ms = (time.perf_counter() - start) * 1000
                if response is not None:
                    stage = "accounting"
                    self._record(run, response, latency_ms)
                if failure is not None:
                    stage = "provider"
                    raise failure
                if response is None:
                    raise ValueError("provider returned no response")
                if not response.completed:
                    run.mark_failed("provider completion is incomplete or unsupported")
                    run.metadata["failure_stage"] = "completion"
                else:
                    stage = "evaluation"
                    run.set_quality_metric(evaluator(response.text))
            except Exception as exc:
                run.status = RunStatus.ERROR
                # Raw exceptions may contain credentials, prompts or HTTP headers.
                run.error = stage + ": " + type(exc).__name__
                run.metadata["failure_stage"] = stage
                if isinstance(exc, ProviderError):
                    run.metadata["provider_error_code"] = exc.code
                    run.metadata["http_status"] = exc.http_status
        return run
