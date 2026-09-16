import json
import unittest
from decimal import Decimal
from tokenpilot.providers.base import BaseProvider, ProviderRequest, ProviderResponse, Usage
from tokenpilot.providers.normalization import normalize_chat_usage
from tokenpilot.benchmark.runner import BaselineRunner
from tokenpilot.telemetry.experiment import QualityMetric, RunStatus
from tokenpilot.telemetry.ledger import ResourceLedger, ResourceEvent
from tests import test_compare
from tokenpilot.benchmark.compare import compare_runs


class FakeProvider(BaseProvider):
    name = 'fixture'
    simulation = True
    def generate(self, request):
        return ProviderResponse('answer', Usage(100, 20, 60), Decimal('0.01'), 'synthetic fixture')


class RunnerTests(unittest.TestCase):
    def run_fixture(self, provider=None, evaluator=None):
        return BaselineRunner(provider or FakeProvider()).run('task',
            ProviderRequest('fixture-model', 'private prompt', 50),
            evaluator or (lambda text: QualityMetric('quality', 1.0)),
            dataset_version='fixture-v1', evaluator_version='exact-v1')

    def test_success_and_reproducibility(self):
        a, b = self.run_fixture(), self.run_fixture()
        self.assertEqual(a.status, RunStatus.SUCCEEDED)
        self.assertEqual(a.config, b.config)
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertEqual(a.ledger.total_tokens, 120)
        self.assertEqual(a.ledger.total_cost_usd, Decimal('0.01'))
        self.assertTrue(a.metadata['simulation'])
        self.assertNotIn('private prompt', json.dumps(a.to_dict()))

    def test_real_provider_blocked_before_call(self):
        p = FakeProvider()
        p.simulation = False
        p.generate = lambda request: self.fail('must not call')
        with self.assertRaises(PermissionError): self.run_fixture(p)

    def test_unknown_cost_fails(self):
        p = FakeProvider()
        p.generate = lambda request: ProviderResponse('answer', Usage(100, 20))
        run = self.run_fixture(p)
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertFalse(run.metadata['accounting_complete'])
        self.assertEqual(run.ledger.total_tokens, 120)

    def test_failure_trace_hides_exception_message(self):
        p = FakeProvider()
        def fail(request): raise RuntimeError('secret-api-key')
        p.generate = fail
        run = self.run_fixture(p)
        self.assertEqual(run.status, RunStatus.ERROR)
        self.assertEqual(run.metadata['failure_stage'], 'provider')
        self.assertNotIn('secret-api-key', json.dumps(run.to_dict()))
        self.assertIsNotNone(run.finished_at)

    def test_evaluation_error_retains_cost(self):
        def fail(text): raise ValueError('bad evaluation')
        run = self.run_fixture(evaluator=fail)
        self.assertEqual(run.status, RunStatus.ERROR)
        self.assertEqual(run.ledger.total_cost_usd, Decimal('0.01'))
        self.assertEqual(run.metadata['failure_stage'], 'evaluation')

    def test_normalization(self):
        usage = normalize_chat_usage({'prompt_tokens': 100, 'completion_tokens': 20,
            'prompt_tokens_details': {'cached_tokens': 60}, 'total_tokens': 120})
        self.assertEqual(usage.total_tokens, 120)

    def test_invalid_usage(self):
        for raw in (None, {}, {'prompt_tokens': -1, 'completion_tokens': 0},
                    {'prompt_tokens': True, 'completion_tokens': 0},
                    {'prompt_tokens': 1, 'completion_tokens': 0, 'total_tokens': 2},
                    {'prompt_tokens': 1, 'completion_tokens': 0, 'prompt_tokens_details': {'cached_tokens': 2}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError): normalize_chat_usage(raw)

    def test_invalid_accounting(self):
        for kwargs in ({'cost_usd': Decimal('NaN')}, {'cost_usd': Decimal('Infinity')},
                       {'latency_ms': float('nan')}, {'input_tokens': 1.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResourceLedger().record(ResourceEvent(category='test', **kwargs))

    def test_comparison_requires_quality_and_success(self):
        fixture = test_compare.ComparisonTests()
        a, b = fixture.make_baseline(), fixture.make_candidate()
        self.assertFalse(compare_runs(a, b).candidate_is_net_better)
        b.status = RunStatus.ERROR
        self.assertFalse(compare_runs(a, b, quality_metric='quality', max_quality_drop=.02).candidate_is_net_better)

    def test_comparison_validation(self):
        fixture = test_compare.ComparisonTests()
        a, b = fixture.make_baseline(), fixture.make_candidate()
        with self.assertRaises(ValueError): compare_runs(a, b, max_quality_drop=.1)
        b.task_id = 'different'
        with self.assertRaises(ValueError): compare_runs(a, b)

    def test_incomplete_accounting_blocks_success(self):
        fixture = test_compare.ComparisonTests()
        a, b = fixture.make_baseline(), fixture.make_candidate()
        b.metadata['accounting_complete'] = False
        self.assertFalse(compare_runs(a, b, quality_metric='quality', max_quality_drop=.02).candidate_is_net_better)

    def test_lower_is_better_and_zero_baseline(self):
        fixture = test_compare.ComparisonTests()
        a, b = fixture.make_baseline(), fixture.make_candidate()
        a.set_quality_metric(QualityMetric('error', .2, higher_is_better=False))
        b.set_quality_metric(QualityMetric('error', .1, higher_is_better=False))
        self.assertTrue(compare_runs(a, b, quality_metric='error', max_quality_drop=0).quality_gate_passed)
        a.ledger = ResourceLedger()
        self.assertIsNone(compare_runs(a, b).cost_saving_percent)
