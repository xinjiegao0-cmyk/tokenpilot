"""Sanitized/synthetic usage fixtures; no network or real credentials."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.smoke_live_provider import main, load_config
from tokenpilot.providers.base import Usage, ProviderRequest
from tokenpilot.providers.normalization import normalize_chat_usage
from tokenpilot.providers.openai_compatible import OpenAICompatibleProvider
from tokenpilot.benchmark.runner import BaselineRunner
from tokenpilot.telemetry.experiment import QualityMetric
from tests.test_openai_compatible import FixtureTransport


def fixture(name):
    return json.loads(Path(__file__).with_name('fixtures').joinpath(name + '.json').read_text())


class UsageNormalizationTests(unittest.TestCase):
    def test_fixtures_reach_serialized_ledger_without_double_counting(self):
        for name, total, reasoning, cached in [('moonshot_length', 36, 19, 0),
                                               ('moonshot_cached', 120, 12, 60),
                                               ('openai_reasoning', 120, 12, 60)]:
            with self.subTest(name=name):
                provider = OpenAICompatibleProvider(name='fixture', base_url='https://example.invalid/v1',
                    transport=FixtureTransport(fixture(name)))
                run = BaselineRunner(provider).run('fixture', ProviderRequest('fixture', 'hello', 32),
                    lambda text: QualityMetric('exact', 1.0), dataset_version='v1', evaluator_version='v1')
                event = run.to_dict()['resources']['events'][0]
                self.assertEqual(event['metadata']['usage_breakdown'], {'reasoning_tokens': reasoning})
                self.assertEqual(event['cached_input_tokens'], cached)
                self.assertEqual(run.ledger.total_tokens, total)
                self.assertFalse(run.metadata['accounting_complete'])
                self.assertFalse(event['metadata']['cost_known'])
                self.assertEqual(bool(run.quality_metrics), name != 'moonshot_length')

    def test_input_output_aliases_and_absent_details(self):
        usage = normalize_chat_usage({'input_tokens':100, 'output_tokens':20,
            'input_tokens_details':{'cached_tokens':60}, 'output_tokens_details':{'reasoning_tokens':12}})
        self.assertEqual(usage, Usage(100, 20, 60, 12))
        self.assertIsNone(normalize_chat_usage({'prompt_tokens':1,'completion_tokens':0}).reasoning_tokens)
        for value in (None, {}):
            self.assertIsNone(normalize_chat_usage({'prompt_tokens':1,'completion_tokens':0,
                'completion_tokens_details':value}).reasoning_tokens)
        self.assertEqual(normalize_chat_usage({'prompt_tokens':1,'completion_tokens':0,
            'completion_tokens_details':{'reasoning_tokens':0}}).reasoning_tokens, 0)

    def test_invalid_breakdowns_and_alias_conflicts(self):
        variants = [{'completion_tokens_details':{'reasoning_tokens':v}}
                    for v in (-1, True, 1.5, '1', None, 21)]
        variants += [{'completion_tokens_details':v} for v in ([], False, 0, '')]
        variants += [{'input_tokens':99}, {'output_tokens':21},
                     {'cached_tokens':101}, {'cached_tokens':True},
                     {'cached_tokens':2,'prompt_tokens_details':{'cached_tokens':3}},
                     {'completion_tokens_details':{'reasoning_tokens':1},
                      'output_tokens_details':{'reasoning_tokens':2}}, {'total_tokens':139}]
        for update in variants:
            with self.subTest(update=update), self.assertRaises(ValueError):
                normalize_chat_usage(dict({'prompt_tokens':100,'completion_tokens':20}, **update))

    def test_matching_aliases_are_not_added(self):
        usage = normalize_chat_usage({'prompt_tokens':100, 'input_tokens':100,
            'completion_tokens':20, 'output_tokens':20, 'cached_tokens':60,
            'prompt_tokens_details':{'cached_tokens':60}})
        self.assertEqual(usage.total_tokens, 120)
        self.assertEqual(usage.cached_input_tokens, 60)


class LiveSmokeTests(unittest.TestCase):
    def test_default_does_not_read_config_or_create_transport(self):
        with patch('benchmarks.smoke_live_provider.load_config') as config, \
             patch('benchmarks.smoke_live_provider.UrllibTransport') as transport, \
             patch('sys.stdout', new_callable=io.StringIO):
            self.assertEqual(main([]), 0)
        config.assert_not_called()
        transport.assert_not_called()

    def test_env_quotes_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text('# local\nexport MOONSHOT_API_KEY="synthetic-key"\nMOONSHOT_MODEL=local\nIGNORED=value\n')
            with patch.dict(os.environ, {'MOONSHOT_MODEL':'override'}, clear=True):
                self.assertEqual(load_config(path), {'MOONSHOT_API_KEY':'synthetic-key', 'MOONSHOT_MODEL':'override'})

    def test_opt_in_one_call_records_usage_even_with_unknown_cost(self):
        transport = FixtureTransport(fixture('moonshot_cached'))
        transport.simulation = False
        with patch('benchmarks.smoke_live_provider.load_config', return_value={
                'MOONSHOT_API_KEY':'SYNTHETIC_SECRET', 'MOONSHOT_BASE_URL':'https://example.invalid/v1',
                'MOONSHOT_MODEL':'configured-model'}), \
             patch('benchmarks.smoke_live_provider.UrllibTransport', return_value=transport) as factory, \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--allow-paid-api']), 0)
        factory.assert_called_once_with(allow_network=True)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][2]['model'], 'configured-model')
        self.assertNotIn('temperature', transport.calls[0][2])
        result = json.loads(output.getvalue())
        self.assertEqual(result['experiment_status'], 'failed')
        self.assertFalse(result['accounting_complete'])
        self.assertEqual(result['ledger']['summary']['total_tokens'], 120)
        self.assertNotIn('SYNTHETIC_SECRET', output.getvalue())

    def test_truncated_response_is_nonzero_but_keeps_usage(self):
        transport = FixtureTransport(fixture('moonshot_length'))
        transport.simulation = False
        with patch('benchmarks.smoke_live_provider.load_config', return_value={
                'MOONSHOT_API_KEY':'synthetic', 'MOONSHOT_BASE_URL':'https://example.invalid/v1'}), \
             patch('benchmarks.smoke_live_provider.UrllibTransport', return_value=transport), \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--allow-paid-api', '--max-output-tokens', '20']), 1)
        self.assertEqual(json.loads(output.getvalue())['ledger']['summary']['total_tokens'], 36)
        self.assertEqual(len(transport.calls), 1)

    def test_invalid_config_does_not_call_network_or_leak_values(self):
        with patch('benchmarks.smoke_live_provider.load_config', return_value={
                'MOONSHOT_API_KEY':'SYNTHETIC_SECRET', 'MOONSHOT_BASE_URL':'invalid-secret-url'}), \
             patch('tokenpilot.providers.transport.build_opener') as opener, \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            self.assertEqual(main(['--allow-paid-api']), 2)
        opener.assert_not_called()
        self.assertNotIn('SECRET', output.getvalue())
        self.assertNotIn('invalid-secret-url', output.getvalue())
