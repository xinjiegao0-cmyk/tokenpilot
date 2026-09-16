from decimal import Decimal as D
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.run import main
from tokenpilot.benchmark.batch import BatchConfig, compare_records, run_case
from tokenpilot.benchmark.strategies import select_context
from tokenpilot.benchmark.tasks import build_prompt, evaluate, load_dataset
from tokenpilot.providers.base import ProviderError, ProviderResponse, Usage
from tokenpilot.providers.capabilities import CapabilityProfile, reserve_api_cost
from tokenpilot.providers.fixture import FixtureProvider
from tokenpilot.telemetry.pricing import PricingProfile


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset, cls.digest = load_dataset()

    def run_task(self, task=None, strategy='tokenpilot', provider=None, config=None, **kwargs):
        return run_case(task or self.dataset['tasks'][0], strategy, provider or FixtureProvider(),
                        config or BatchConfig('fixture-model', local_usd_per_cpu_second=D('0')),
                        dataset_version=self.dataset['version'], dataset_sha256=self.digest, **kwargs)

    def test_all_full_and_candidate_cases_satisfy_ground_truth(self):
        self.assertEqual(len(self.dataset['tasks']), 12)
        for task in self.dataset['tasks']:
            for strategy in ('full-history', 'tokenpilot'):
                with self.subTest(task=task['id'], strategy=strategy):
                    record = self.run_task(task, strategy)
                    self.assertTrue(record['quality_passed'])
                    self.assertEqual(record['status'], 'succeeded')
                    self.assertEqual(record['warning'], 'SIMULATED_SMOKE_TEST_ONLY')
                    self.assertEqual(record['plan_trace'][-1]['action'], 'stop')

    def test_short_bypass_and_long_reduction(self):
        for task in self.dataset['tasks']:
            record = self.run_task(task)
            if task['tier'] == 'short':
                self.assertTrue(record['context']['bypassed'])
            if task['tier'] == 'long':
                self.assertLess(record['context']['selected_bytes'], record['context']['original_bytes'])

    def test_strict_json_citations_and_types(self):
        expected = {'answer': 1, 'citations': ['d1']}
        for actual in ('{"answer":true,"citations":["d1"]}',
                       '{"answer":1,"citations":["d1","d1"]}',
                       '{"answer":1,"citations":["d2"]}',
                       '{"answer":1,"citations":["d1"],"extra":0}',
                       '{"answer":0,"answer":1,"citations":["d1"]}', 'not json'):
            self.assertFalse(evaluate(actual, expected))
        self.assertTrue(evaluate(json.dumps(expected), expected))

    def test_selector_has_no_answer_access(self):
        task = self.dataset['tasks'][4]
        selected = select_context(task['query'], task['documents'], 'tokenpilot')
        poisoned = dict(task, expected={'answer': 'wrong', 'citations': []})
        other = select_context(poisoned['query'], poisoned['documents'], 'tokenpilot')
        self.assertEqual(selected, other)
        self.assertEqual(build_prompt(task, selected.documents), build_prompt(poisoned, other.documents))

    def test_equal_values_different_provenance_not_deduplicated(self):
        docs = [{'id': 'd1', 'key': 'x', 'value': 7}, {'id': 'd2', 'key': 'x', 'value': 7}]
        selection = select_context({'key': 'x', 'operation': 'sum'}, docs, 'tokenpilot', ablation='no-bypass')
        self.assertEqual(len(selection.documents), 2)

    def test_dependency_page_in_ablation_exposes_quality_loss(self):
        task = next(t for t in self.dataset['tasks'] if t['id'] == 'long-join')
        record = self.run_task(task, config=BatchConfig('fixture-model', ablation='no-page-in'))
        self.assertFalse(record['quality_passed'])
        self.assertEqual(record['failure_reason'], 'quality_contract_failed')

    def test_no_pruning_ablation_retains_full_input(self):
        task = next(t for t in self.dataset['tasks'] if t['id'] == 'long-sum')
        record = self.run_task(task, config=BatchConfig('fixture-model', ablation='no-pruning'))
        self.assertEqual(record['context']['original_bytes'], record['context']['selected_bytes'])
        self.assertTrue(record['quality_passed'])

    def test_unknown_local_overhead_blocks_comparison(self):
        a = self.run_task(strategy='full-history', config=BatchConfig('fixture-model'))
        b = self.run_task(config=BatchConfig('fixture-model'))
        self.assertTrue(a['quality_passed'])
        result = compare_records(a, b)
        self.assertIsNone(result['net_saving_usd'])
        self.assertFalse(result['cost_success'])

    def test_config_mismatch_rejected(self):
        a = self.run_task(strategy='full-history')
        b = self.run_task(config=BatchConfig('other-model', local_usd_per_cpu_second=D('0')))
        with self.assertRaises(ValueError):
            compare_records(a, b)

    def test_failure_is_retained_without_raw_error(self):
        class Broken(FixtureProvider):
            def generate(self, request):
                raise RuntimeError('SECRET-CREDENTIAL')
        record = self.run_task(provider=Broken())
        self.assertEqual(record['failure_reason'], 'provider_exception')
        self.assertIsNone(record['usage'])
        self.assertNotIn('SECRET-CREDENTIAL', json.dumps(record))

    def test_paid_flag_checked_before_call(self):
        class Real(FixtureProvider):
            simulation = False
            def generate(self, request):
                raise AssertionError('must not call')
        with self.assertRaises(PermissionError):
            self.run_task(provider=Real())

    def test_partial_failure_preserves_usage_and_estimate(self):
        class Partial(FixtureProvider):
            simulation = False
            name = 'fixture'
            def generate(self, request):
                raise ProviderError('invalid_response', response=ProviderResponse('', Usage(10, 5), completed=False))
        prices = PricingProfile('fixture', 'fixture-model', '2026-09-16', 'synthetic rates', D('1'), D('2'))
        record = self.run_task(provider=Partial(), pricing=prices, allow_paid_api=True)
        self.assertEqual(record['usage']['input_tokens'], 10)
        self.assertEqual(record['billing']['task_cost_usd'], '0.00002')
        self.assertEqual(record['billing']['kind'], 'estimated')
        self.assertFalse(record['quality_passed'])
        self.assertEqual(record['failure_reason'], 'invalid_response')

    def test_offline_cli_writes_one_record_per_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(main(['--task', 'short-lookup', '--output-dir', folder]), 0)
            records = list(Path(folder).glob('*.jsonl'))
            self.assertEqual(len(records), 1)
            self.assertEqual(len(records[0].read_text().splitlines()), 4)
            self.assertNotIn('"records": [', records[0].read_text())

    def test_live_and_dry_run_guards_do_not_read_credentials(self):
        with patch('benchmarks.run.load_config', side_effect=AssertionError('no key reads')):
            self.assertEqual(main(['--live']), 2)
            self.assertEqual(main(['--dry-run', '--task', 'short-lookup']), 0)
            self.assertEqual(main(['--live', '--allow-paid-api']), 2)


class BudgetTests(unittest.TestCase):
    def test_reserve_uses_higher_cache_rate_and_output_limit(self):
        from tokenpilot.providers.base import ProviderRequest
        profile = CapabilityProfile('test', 'v1', 'max_tokens', framing_token_allowance=2)
        prices = PricingProfile('test', 'm', '2026-09-16', 'synthetic', D('1'), D('3'), D('2'))
        request = ProviderRequest('m', 'abc', 10, temperature=None)
        self.assertEqual(reserve_api_cost(request, profile, prices), D('0.00004'))

    def test_unsupported_parameters_rejected(self):
        from tokenpilot.providers.base import ProviderRequest
        profile = CapabilityProfile('test', 'v1', 'max_tokens')
        with self.assertRaises(ValueError):
            profile.validate(ProviderRequest('m', 'abc', 10))
