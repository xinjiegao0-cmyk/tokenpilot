import json
import unittest
from decimal import Decimal
from unittest.mock import Mock

from benchmarks.smoke_provider import FixtureTransport, fixture_billing, run_demo
from tokenpilot.benchmark.compare import compare_runs
from tokenpilot.benchmark.runner import BaselineRunner
from tokenpilot.providers.base import ProviderRequest, ProviderResponse, Usage
from tokenpilot.providers.openai_compatible import OpenAICompatibleProvider
from tokenpilot.providers.transport import HttpResponse
from tokenpilot.telemetry.experiment import QualityMetric, RunStatus


class BaselineAdapterTests(unittest.TestCase):
    def provider(self, transform=None, billing=fixture_billing):
        transport = FixtureTransport()
        raw = json.loads(transport.post('', {}, b'', 1).body)
        if transform:
            transform(raw)
        transport.post = Mock(return_value=HttpResponse(200, json.dumps(raw).encode()))
        return OpenAICompatibleProvider(name='fixture', base_url='https://fixture.invalid/v1',
            transport=transport, billing_extractor=billing,
            billing_version='fixture-v1' if billing else None)

    def run_fixture(self, provider, evaluator=None):
        return BaselineRunner(provider).run('case', ProviderRequest('fixture-model', 'sensitive prompt', 32),
            evaluator or (lambda text: QualityMetric('quality', 1.0)),
            dataset_version='fixture-v1', evaluator_version='exact-v1')

    def test_end_to_end_demo(self):
        run = run_demo()
        self.assertEqual(run.status, RunStatus.SUCCEEDED)
        self.assertEqual(run.ledger.total_tokens, 120)
        self.assertEqual(run.ledger.total_cost_usd, Decimal('0.00012'))
        self.assertEqual(run.ledger.events[0].metadata['cost_kind'], 'simulated')
        self.assertEqual(run.metadata['warning'], 'SIMULATED_SMOKE_TEST_ONLY')
        self.assertEqual(run.metadata['response']['model'], 'fixture-model-v1')
        self.assertNotIn('What is 2 + 2?', json.dumps(run.to_dict()))

    def test_missing_bill_fails_but_retains_usage(self):
        run = self.run_fixture(self.provider(billing=None))
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertEqual(run.ledger.total_tokens, 120)
        self.assertFalse(run.metadata['accounting_complete'])
        self.assertEqual(run.ledger.events[0].metadata['cost_kind'], 'unknown')

    def test_malformed_choices_preserve_known_charge(self):
        run = self.run_fixture(self.provider(lambda raw: raw.update(choices=[])))
        self.assertEqual(run.status, RunStatus.ERROR)
        self.assertEqual(run.metadata['provider_error_code'], 'invalid_response')
        self.assertEqual(run.ledger.total_cost_usd, Decimal('0.00012'))
        self.assertEqual(run.ledger.total_tokens, 120)

    def test_billing_error_preserves_tokens(self):
        def billing(raw):
            raise ValueError('secret credential in exception')
        run = self.run_fixture(self.provider(billing=billing))
        self.assertEqual(run.status, RunStatus.ERROR)
        self.assertEqual(run.ledger.total_tokens, 120)
        self.assertFalse(run.metadata['accounting_complete'])
        self.assertEqual(run.metadata['provider_error_code'], 'billing_error')
        self.assertNotIn('secret credential', json.dumps(run.to_dict()))

    def test_invalid_usage_preserves_independent_known_bill(self):
        for usage in (None, {}, {'prompt_tokens': -1, 'completion_tokens': 1}):
            with self.subTest(usage=usage):
                evaluator = Mock(side_effect=AssertionError('must not evaluate'))
                run = self.run_fixture(self.provider(lambda raw: raw.update(usage=usage)), evaluator)
                self.assertEqual(run.status, RunStatus.ERROR)
                self.assertEqual(run.metadata['provider_error_code'], 'invalid_usage')
                self.assertEqual(run.ledger.total_cost_usd, Decimal('0.00012'))
                self.assertFalse(run.metadata['accounting_complete'])
                self.assertFalse(run.ledger.events[0].metadata['usage_known'])
                self.assertTrue(run.ledger.events[0].metadata['cost_known'])
                evaluator.assert_not_called()

    def test_truncation_and_refusal_skip_evaluator_and_keep_cost(self):
        for reason in ('length', 'content_filter', 'tool_calls'):
            with self.subTest(reason=reason):
                evaluator = Mock(side_effect=AssertionError('should not evaluate'))
                provider = self.provider(lambda raw: raw['choices'][0].update(finish_reason=reason))
                run = self.run_fixture(provider, evaluator)
                self.assertEqual(run.status, RunStatus.FAILED)
                self.assertEqual(run.ledger.total_cost_usd, Decimal('0.00012'))
                evaluator.assert_not_called()
        provider = self.provider(lambda raw: raw['choices'][0]['message'].update(content=None, refusal='no'))
        self.assertEqual(self.run_fixture(provider).status, RunStatus.FAILED)

    def test_nonfinite_quality_fails_with_cost_preserved(self):
        run = self.run_fixture(self.provider(), lambda _: QualityMetric('quality', float('nan')))
        self.assertEqual(run.status, RunStatus.ERROR)
        self.assertEqual(run.metadata['failure_stage'], 'evaluation')
        self.assertEqual(run.ledger.total_cost_usd, Decimal('0.00012'))
        json.dumps(run.to_dict(), allow_nan=False)

    def test_estimate_never_becomes_reported_cost(self):
        # No network: this fake provider only returns an in-memory response.
        provider = self.provider()
        provider._transport.simulation = False
        provider.generate = Mock(return_value=ProviderResponse('4', Usage(100, 20),
            Decimal('.01'), 'fictional rate card', cost_kind='estimated'))
        run = BaselineRunner(provider).run('case', ProviderRequest('fixture', 'prompt', 32),
            lambda _: QualityMetric('quality', 1.), dataset_version='v1', evaluator_version='v1',
            allow_paid_api=True)
        self.assertEqual(run.status, RunStatus.FAILED)
        self.assertFalse(run.metadata['accounting_complete'])
        self.assertEqual(run.ledger.total_cost_usd, 0)
        self.assertEqual(run.ledger.events[0].metadata['unverified_cost_usd'], '0.01')

    def test_comparison_provenance(self):
        a, b = run_demo(), run_demo()
        result = compare_runs(a, b, quality_metric='exact_match', max_quality_drop=0)
        self.assertTrue(result.simulation)
        self.assertEqual(result.to_dict()['warning'], 'SIMULATED_SMOKE_TEST_ONLY')
        b.metadata['simulation'] = False
        with self.assertRaises(ValueError): compare_runs(a, b)
        b.metadata['simulation'] = True
        for key in ('dataset_version', 'evaluator_version'):
            with self.subTest(key=key):
                b.config = dict(a.config, **{key: 'mismatch'})
                with self.assertRaises(ValueError): compare_runs(a, b)

    def test_real_runner_consent_blocks_transport(self):
        provider = OpenAICompatibleProvider(name='disabled', base_url='https://fixture.invalid/v1')
        provider.generate = Mock(side_effect=AssertionError('must not execute'))
        with self.assertRaises(PermissionError): self.run_fixture(provider)
        provider.generate.assert_not_called()
