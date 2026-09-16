"""Reproducible structured benchmark; default mode is strictly offline."""
import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import sys
import uuid

from benchmarks.smoke_live_provider import load_config
from tokenpilot.benchmark.batch import BatchConfig, compare_records, run_case
from tokenpilot.benchmark.strategies import ABLATIONS, STRATEGIES, select_context
from tokenpilot.benchmark.tasks import build_prompt, load_dataset
from tokenpilot.providers.base import BaseProvider, ProviderRequest
from tokenpilot.providers.capabilities import CapabilityProfile, reserve_api_cost
from tokenpilot.providers.fixture import FixtureProvider
from tokenpilot.providers.openai_compatible import OpenAICompatibleProvider
from tokenpilot.providers.transport import UrllibTransport
from tokenpilot.telemetry.pricing import PricingProfile, money


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--live', action='store_true')
    result.add_argument('--allow-paid-api', action='store_true')
    result.add_argument('--allow-unknown-cost', action='store_true',
                        help='usage sanity only: <=2 calls, <=2048 input byte-bound, <=512 output tokens; costs unknown')
    result.add_argument('--env-file', type=Path, default=Path('.env'))
    result.add_argument('--model')
    result.add_argument('--pricing', type=Path)
    result.add_argument('--capabilities', type=Path,
                        help='explicit JSON capability profile for the selected provider')
    result.add_argument('--max-calls', type=int, default=4)
    result.add_argument('--max-api-budget-usd', type=money)
    result.add_argument('--accept-estimated-budget', action='store_true',
                        help='acknowledge price/tokenizer assumptions; not a provider billing hard cap')
    result.add_argument('--local-usd-per-cpu-second', type=money)
    result.add_argument('--max-output-tokens', type=int, default=1024)
    result.add_argument('--task', action='append', help='task id (repeatable); default all tasks')
    result.add_argument('--strategy', action='append', choices=STRATEGIES)
    result.add_argument('--ablation', choices=ABLATIONS)
    result.add_argument('--window-bytes', type=int, default=8192)
    result.add_argument('--retrieval-k', type=int, default=8)
    result.add_argument('--output-dir', type=Path, default=Path('benchmarks/results'))
    result.add_argument('--dry-run', action='store_true', help='plan and reserve only; no key read or calls')
    return result


def execute(args):
    if args.allow_unknown_cost and not args.live:
        raise ValueError('unknown-cost sanity applies only to live execution')
    if args.live and not args.allow_paid_api and not args.dry_run:
        raise ValueError('live execution requires --allow-paid-api')
    dataset, dataset_hash = load_dataset()
    tasks = [task for task in dataset['tasks'] if args.task is None or task['id'] in args.task]
    if not tasks or (args.task and set(args.task) != {task['id'] for task in tasks}):
        raise ValueError('unknown task id')
    strategies = args.strategy or list(STRATEGIES)
    if len(strategies) != len(set(strategies)):
        raise ValueError('duplicate strategy')
    if args.ablation and 'tokenpilot' not in strategies:
        raise ValueError('ablation requires tokenpilot strategy')
    calls = len(tasks) * len(strategies)
    prices, capabilities = None, None
    reserve = None
    if args.live:
        if args.max_calls <= 0 or calls > args.max_calls:
            raise ValueError('planned calls exceed explicit max-calls limit')
        if not args.model or not args.capabilities:
            raise ValueError('live mode requires model and capability file')
        capabilities = CapabilityProfile(**json.loads(args.capabilities.read_text()))
        if args.allow_unknown_cost:
            if calls > 2 or args.max_output_tokens > 512:
                raise ValueError('usage sanity permits at most two calls and 512 output tokens per call')
            if args.pricing or args.max_api_budget_usd is not None:
                raise ValueError('unknown-cost sanity cannot assert a dollar budget or price')
        else:
            if not args.pricing:
                raise ValueError('live benchmark requires explicit pricing')
            prices = PricingProfile.from_dict(json.loads(args.pricing.read_text()))
            if prices.provider != 'moonshot' or prices.model != args.model:
                raise ValueError('pricing must match moonshot and requested model exactly')
            if args.max_api_budget_usd is None or args.max_api_budget_usd <= 0:
                raise ValueError('positive --max-api-budget-usd required')
            if not args.accept_estimated_budget:
                raise ValueError('explicit --accept-estimated-budget required')
    config = BatchConfig(args.model or 'record-interpreter-v1', args.max_output_tokens,
                         window_bytes=args.window_bytes, retrieval_k=args.retrieval_k,
                         local_usd_per_cpu_second=args.local_usd_per_cpu_second,
                         ablation=args.ablation)
    if args.live:
        assert capabilities is not None
        reserve = None if args.allow_unknown_cost else Decimal('0')
        for task in tasks:
            for strategy in strategies:
                selection = select_context(task['query'], task['documents'], strategy,
                                           window_bytes=config.window_bytes,
                                           retrieval_k=config.retrieval_k,
                                           ablation=config.ablation if strategy == 'tokenpilot' else None)
                request = ProviderRequest(config.model, build_prompt(task, selection.documents),
                                          config.max_output_tokens, temperature=None)
                capabilities.validate(request)
                if args.allow_unknown_cost:
                    if capabilities.input_upper_bound(request) > 2048:
                        raise ValueError('usage sanity input exceeds 2048 byte-token bound')
                else:
                    assert reserve is not None and prices is not None
                    reserve += reserve_api_cost(request, capabilities, prices)
        if reserve is not None and reserve > args.max_api_budget_usd:
            raise ValueError('all-attempt API reserve exceeds budget; no requests sent')
    plan = {'simulation': not args.live, 'planned_calls': calls,
            'api_reserve_usd': str(reserve) if reserve is not None else None,
            'budget_kind': ('unknown cost; token-bounded sanity only' if args.allow_unknown_cost else
                            'estimated upper bound under configured prices and tokenizer assumptions'),
            'dataset_sha256': dataset_hash, 'tasks': [task['id'] for task in tasks],
            'strategies': strategies, 'config': asdict(config)}
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    provider: BaseProvider = FixtureProvider()
    if args.live:
        # Only now read credentials: all validation/preflight has passed.
        assert capabilities is not None
        local = load_config(args.env_file)
        if not local.get('MOONSHOT_API_KEY') or not local.get('MOONSHOT_BASE_URL'):
            raise ValueError('local Moonshot credential and base URL required')
        provider = OpenAICompatibleProvider(
            name='moonshot', base_url=local['MOONSHOT_BASE_URL'], api_key=local['MOONSHOT_API_KEY'],
            transport=UrllibTransport(allow_network=True), token_limit_field=capabilities.token_limit_field)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = str(uuid.uuid4())
    records_path = args.output_dir / (batch_id + '.jsonl')
    records = []
    comparisons: list[dict] = []
    with records_path.open('x', encoding='utf-8') as output:
        for task in tasks:
            task_records = []
            for strategy in strategies:
                record = run_case(task, strategy, provider, config, dataset_version=dataset['version'],
                                  dataset_sha256=dataset_hash, pricing=prices, allow_paid_api=args.allow_paid_api)
                record['batch_id'] = batch_id
                record['capabilities'] = asdict(capabilities) if capabilities else None
                output.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
                output.flush()
                records.append(record)
                task_records.append(record)
            candidate = next((record for record in task_records if record['strategy'] == 'tokenpilot'), None)
            if candidate:
                comparisons.extend(compare_records(record, candidate) for record in task_records
                                   if record['strategy'] != 'tokenpilot')
    summary = {'schema_version': 'tokenpilot-batch-v1', 'batch_id': batch_id, 'plan': plan,
               'record_count': len(records), 'quality_pass_count': sum(r['quality_passed'] for r in records),
               'accounting_complete_count': sum(r['accounting_complete'] for r in records),
               'failure_count': sum(r['status'] != 'succeeded' for r in records),
               'warning': None if args.live else 'SIMULATED_SMOKE_TEST_ONLY',
               'limitations': dataset['limitations'], 'comparisons': comparisons,
               'records_file': records_path.name}
    summary_path = args.output_dir / (batch_id + '.summary.json')
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + '\n', encoding='utf-8')
    print(json.dumps({'summary': str(summary_path), 'records': str(records_path),
                      'record_count': len(records), 'quality_pass_count': summary['quality_pass_count'],
                      'accounting_complete_count': summary['accounting_complete_count'],
                      'warning': summary['warning']}, indent=2))
    # Expected baseline quality failures are data, not a crashed batch.
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, TypeError, OSError, KeyError, ArithmeticError):
        print('CONFIG_ERROR: validate task, budget, pricing, capabilities and output permissions.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
