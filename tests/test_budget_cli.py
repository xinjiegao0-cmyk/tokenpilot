from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.run import main


class LiveBudgetCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pricing = self.root / 'pricing.json'
        self.capabilities = self.root / 'capabilities.json'
        self.pricing.write_text(json.dumps({'provider': 'moonshot', 'model': 'fixture-model',
            'as_of': '2026-09-16', 'source': 'synthetic unit test prices, not real prices',
            'input_per_million_usd': '1', 'output_per_million_usd': '2'}))
        self.capabilities.write_text(json.dumps({'name': 'test', 'version': 'v1',
            'token_limit_field': 'max_tokens'}))
        self.args = ['--live', '--allow-paid-api', '--task', 'short-lookup',
            '--model', 'fixture-model', '--pricing', str(self.pricing),
            '--capabilities', str(self.capabilities), '--accept-estimated-budget',
            '--max-api-budget-usd', '1', '--output-dir', str(self.root / 'results')]

    def run_silent(self, args):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main(args)

    def test_budget_rejection_before_credentials_or_network(self):
        args = list(self.args)
        args[args.index('--max-api-budget-usd') + 1] = '0.00000001'
        with patch('benchmarks.run.load_config', side_effect=AssertionError('no credential reads')):
            self.assertEqual(self.run_silent(args), 2)
        self.assertFalse((self.root / 'results').exists())

    def test_call_limit_rejection_before_credentials(self):
        with patch('benchmarks.run.load_config', side_effect=AssertionError('no credential reads')):
            self.assertEqual(self.run_silent(self.args + ['--max-calls', '3']), 2)

    def test_live_dry_run_validates_without_reading_secret(self):
        with patch('benchmarks.run.load_config', side_effect=AssertionError('no credential reads')):
            self.assertEqual(self.run_silent(self.args + ['--dry-run']), 0)

    def test_no_flag_and_no_budget_ack_cannot_run(self):
        for flag in ('--allow-paid-api', '--accept-estimated-budget'):
            args = [value for value in self.args if value != flag]
            with patch('benchmarks.run.load_config', side_effect=AssertionError('no credential reads')):
                self.assertEqual(self.run_silent(args), 2)

    def test_wrong_model_profile_cannot_run(self):
        args = list(self.args)
        args[args.index('--model') + 1] = 'wrong'
        with patch('benchmarks.run.load_config', side_effect=AssertionError('no credential reads')):
            self.assertEqual(self.run_silent(args), 2)

    def test_real_mode_uses_exactly_bounded_calls_with_mock_transport(self):
        from tokenpilot.providers.base import ProviderResponse, Usage
        def reply(request):
            return ProviderResponse('{"answer":"harbor-17","citations":["short-lookup-e1"]}', Usage(100, 30))
        with patch('benchmarks.run.load_config', return_value={
                'MOONSHOT_API_KEY': 'unit-test-credential', 'MOONSHOT_BASE_URL': 'https://fixture.invalid/v1'}), \
             patch('tokenpilot.providers.openai_compatible.OpenAICompatibleProvider.generate', side_effect=reply) as generate:
            self.assertEqual(self.run_silent(self.args), 0)
            self.assertEqual(generate.call_count, 4)
        records = list((self.root / 'results').glob('*.jsonl'))
        rows = [json.loads(line) for line in records[0].read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row['billing']['kind'] == 'estimated' for row in rows))
        self.assertTrue(all(not row['accounting_complete'] for row in rows))
        self.assertNotIn('unit-test-credential', records[0].read_text())
