"""Auditable batch runner. Saves one redacted record after every attempt."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import platform
import time
from typing import Optional
import uuid

from tokenpilot import __version__
from tokenpilot.benchmark.strategies import select_context
from tokenpilot.benchmark.tasks import build_prompt, evaluate, EVALUATOR_VERSION
from tokenpilot.core.planner import Action, HeuristicPlanner, PlannerState
from tokenpilot.providers.base import ProviderError, ProviderRequest
from tokenpilot.telemetry.pricing import CostTotals, PricingProfile, cost_metrics


@dataclass(frozen=True)
class BatchConfig:
    model: str
    max_output_tokens: int = 1024
    temperature: Optional[float] = None
    seed: Optional[int] = None
    window_bytes: int = 8192
    retrieval_k: int = 8
    local_usd_per_cpu_second: Optional[Decimal] = None
    ablation: Optional[str] = None

    def __post_init__(self):
        ProviderRequest(self.model, '', self.max_output_tokens, self.temperature, self.seed)
        if self.window_bytes <= 0 or self.retrieval_k <= 0:
            raise ValueError('positive context allowances required')
        rate = self.local_usd_per_cpu_second
        if rate is not None and (not isinstance(rate, Decimal) or not rate.is_finite() or rate < 0):
            raise ValueError('local compute rate must be a finite nonnegative Decimal')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def run_case(task, strategy, provider, config, *, dataset_version, dataset_sha256,
             pricing: Optional[PricingProfile] = None, allow_paid_api=False):
    if not provider.simulation and allow_paid_api is not True:
        raise PermissionError('paid API requires explicit flag')
    if pricing is not None and (pricing.provider, pricing.model) != (provider.name, config.model):
        raise ValueError('price profile does not match provider/model')
    started_at, wall_start = utc_now(), time.perf_counter()
    # Thread CPU time excludes provider/network waits and unrelated background threads.
    cpu_start = time.thread_time()
    planning_start = time.perf_counter()
    original_prompt = build_prompt(task, task['documents'])
    state = PlannerState(len(original_prompt.encode('utf-8')), 0)
    planner, trace = HeuristicPlanner(), []
    response, failure, quality = None, None, False
    prompt, selected, provider_latency_ms = '', None, 0.0
    provider_cpu = 0.0
    evaluation_ms = 0.0
    while True:
        plan = planner.next_action(state)
        trace.append(asdict(plan))
        if plan.action == Action.STOP:
            break
        if plan.action == Action.RETRIEVE:
            selected = select_context(task['query'], task['documents'], strategy,
                                      window_bytes=config.window_bytes, retrieval_k=config.retrieval_k,
                                      ablation=config.ablation if strategy == 'tokenpilot' else None)
            prompt = build_prompt(task, selected.documents)
            state.selected_bytes, state.selection_ready = len(prompt.encode('utf-8')), True
            planning_ms = (time.perf_counter() - planning_start) * 1000
        elif plan.action == Action.CALL_MODEL:
            call_start, call_cpu = time.perf_counter(), time.thread_time()
            try:
                response = provider.generate(ProviderRequest(config.model, prompt,
                                             config.max_output_tokens, config.temperature, config.seed))
                if not response.completed:
                    failure, state.failed = 'incomplete_completion', True
                else:
                    state.response_ready = True
            except ProviderError as exc:
                response = exc.response
                failure, state.failed = exc.code, True
            except Exception:
                # Never serialize exception strings: SDK errors can contain secrets.
                failure, state.failed = 'provider_exception', True
            finally:
                provider_latency_ms = (time.perf_counter() - call_start) * 1000
                provider_cpu = time.thread_time() - call_cpu
        elif plan.action == Action.VERIFY:
            evaluation_start = time.perf_counter()
            assert response is not None
            quality = evaluate(response.text, task['expected'])
            evaluation_ms = (time.perf_counter() - evaluation_start) * 1000
            state.verification_done = True
            if not quality:
                failure = 'quality_contract_failed'
    local_cpu_seconds = max(0.0, time.thread_time() - cpu_start - provider_cpu)
    local_cost = (Decimal(str(local_cpu_seconds)) * config.local_usd_per_cpu_second
                  if config.local_usd_per_cpu_second is not None else None)
    task_cost, billing_kind, billing_source = None, 'unknown', None
    usage = asdict(response.usage) if response is not None and response.usage_complete else None
    if response is not None:
        if response.cost_usd is not None and response.cost_source:
            kind = 'simulated' if provider.simulation else response.cost_kind
            if kind in {'reported', 'estimated', 'simulated'} and (kind != 'simulated' or provider.simulation):
                if (isinstance(response.cost_usd, Decimal) and response.cost_usd.is_finite()
                        and response.cost_usd >= 0):
                    task_cost, billing_kind = response.cost_usd, {'reported': 'measured'}.get(kind, kind)
                    billing_source = response.cost_source
                else:
                    failure = failure or 'invalid_billing_evidence'
        if (task_cost is None and usage is not None and pricing is not None
                and not provider.simulation and failure != 'invalid_billing_evidence'):
            task_cost = pricing.estimate(response.usage, provider=provider.name, model=config.model)
            if task_cost is not None:
                billing_kind, billing_source = 'estimated', {'source': pricing.source, 'as_of': pricing.as_of}
    # Local CPU dollars are a configured estimate. Even a measured model bill plus
    # estimated local overhead yields an estimated net-compute total.
    total_kind = billing_kind
    if billing_kind == 'measured' and local_cost is not None:
        total_kind = 'estimated'
    total = CostTotals(task_cost, local_cost, total_kind)
    complete = total.total is not None and usage is not None
    public_config = asdict(config)
    public_config['local_usd_per_cpu_second'] = (str(config.local_usd_per_cpu_second)
                                                if config.local_usd_per_cpu_second is not None else None)
    assert selected is not None
    return {
        'schema_version': 'tokenpilot-run-v1', 'run_id': str(uuid.uuid4()),
        'task_id': task['id'], 'tier': task['tier'], 'strategy': strategy,
        'started_at': started_at, 'finished_at': utc_now(),
        'simulation': bool(provider.simulation),
        'warning': 'SIMULATED_SMOKE_TEST_ONLY' if provider.simulation else None,
        'dataset_version': dataset_version, 'dataset_sha256': dataset_sha256,
        'evaluator_version': EVALUATOR_VERSION, 'tokenpilot_version': __version__,
        'python_version': platform.python_version(), 'provider': provider.describe(),
        'config': public_config, 'pricing_profile': ({key: str(value) if isinstance(value, Decimal) else value
                                                   for key, value in asdict(pricing).items()} if pricing else None),
        'request_sha256': digest(prompt),
        'response_sha256': digest(response.text) if response else None,
        'response_metadata': dict(response.metadata) if response else {},
        'status': 'succeeded' if quality and complete and failure is None else 'failed',
        'failure_reason': failure or ('accounting_incomplete' if not complete else None),
        'quality_passed': quality, 'quality_method': 'strict JSON answer and ground-truth citation set',
        'accounting_complete': complete, 'usage': usage,
        'billing': {'task_cost_usd': str(task_cost) if task_cost is not None else None,
                    'kind': billing_kind, 'source': billing_source,
                    'local_cost_usd': str(local_cost) if local_cost is not None else None,
                    'total_kind': total_kind,
                    'total_cost_usd': str(total.total) if complete else None},
        'overhead': {'local_cpu_seconds': local_cpu_seconds, 'planning_ms': planning_ms,
                     'evaluation_ms': evaluation_ms, 'model_calls': 0,
                     'scope': 'context selection, prompt construction and verification; all strategies'},
        'latency_ms': {'provider': provider_latency_ms, 'wall': (time.perf_counter() - wall_start) * 1000},
        'context': {'original_bytes': state.original_bytes, 'selected_bytes': state.selected_bytes,
                    'selected_ids': [doc['id'] for doc in selected.documents],
                    'bypassed': selected.bypassed, 'reason': selected.reason},
        'plan_trace': trace,
    }


def compare_records(baseline, candidate):
    for key in ('task_id', 'dataset_sha256', 'evaluator_version', 'simulation', 'pricing_profile'):
        if baseline[key] != candidate[key]:
            raise ValueError('incomparable records: ' + key)
    for key in ('model', 'temperature', 'seed', 'max_output_tokens', 'local_usd_per_cpu_second'):
        if baseline['config'][key] != candidate['config'][key]:
            raise ValueError('incomparable generation/accounting config: ' + key)
    if baseline['provider'] != candidate['provider']:
        raise ValueError('incomparable providers')

    def totals(record):
        billing = record['billing']
        def decimal_or_none(value):
            return Decimal(value) if value is not None else None
        return CostTotals(decimal_or_none(billing['task_cost_usd']),
                          decimal_or_none(billing['local_cost_usd']),
                          billing['total_kind'] if record['accounting_complete'] else 'unknown')
    result = cost_metrics(totals(baseline), totals(candidate), quality_passed=(
        baseline['quality_passed'] and candidate['quality_passed']
        and baseline['status'] == 'succeeded' and candidate['status'] == 'succeeded'))
    result.update({'baseline_run_id': baseline['run_id'], 'candidate_run_id': candidate['run_id'],
                   'task_id': baseline['task_id'], 'baseline_strategy': baseline['strategy'],
                   'candidate_strategy': candidate['strategy'],
                   'simulation': baseline['simulation']})
    return result
