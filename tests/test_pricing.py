import unittest
from decimal import Decimal as D

from tokenpilot.providers.base import Usage
from tokenpilot.telemetry.pricing import CostTotals, PricingProfile, cost_metrics


class PricingTests(unittest.TestCase):
    def profile(self, cached=D('0.2')):
        return PricingProfile('fixture', 'model', '2026-09-16', 'synthetic test rates',
                              D('1'), D('2'), cached)

    def test_breakdowns_not_double_counted(self):
        cost = self.profile().estimate(Usage(100, 50, 20, 40), provider='fixture', model='model')
        self.assertEqual(cost, D('0.000184'))

    def test_unknown_cache_rate_is_unknown(self):
        self.assertIsNone(self.profile(None).estimate(
            Usage(100, 50, 20), provider='fixture', model='model'))

    def test_mismatched_model_rejected(self):
        with self.assertRaises(ValueError):
            self.profile().estimate(Usage(1, 1), provider='fixture', model='other')

    def test_unknown_overhead_cannot_claim_success(self):
        result = cost_metrics(CostTotals(D('2'), D('0'), 'estimated'),
                              CostTotals(D('1'), None, 'estimated'), quality_passed=True)
        self.assertIsNone(result['net_saving_usd'])
        self.assertFalse(result['cost_success'])

    def test_overhead_can_reverse_gross_savings(self):
        result = cost_metrics(CostTotals(D('2'), D('0'), 'estimated'),
                              CostTotals(D('1'), D('1.5'), 'estimated'), quality_passed=True)
        self.assertEqual(result['net_saving_usd'], '-0.5')
        self.assertFalse(result['cost_success'])

    def test_mixed_evidence_cannot_compare(self):
        result = cost_metrics(CostTotals(D('2'), D('0'), 'measured'),
                              CostTotals(D('1'), D('0'), 'estimated'), quality_passed=True)
        self.assertFalse(result['accounting_complete'])

    def test_invalid_values(self):
        for value in (D('NaN'), D('Infinity'), D('-1')):
            with self.assertRaises(ValueError):
                self.profile(value)

    def test_quality_required(self):
        result = cost_metrics(CostTotals(D('2'), D('0'), 'estimated'),
                              CostTotals(D('1'), D('0.1'), 'estimated'), quality_passed=False)
        self.assertFalse(result['cost_success'])


class LegacyComparisonTests(unittest.TestCase):
    def test_unknown_ledger_placeholder_not_exported_as_zero(self):
        from tokenpilot.benchmark.compare import compare_runs
        from tokenpilot.telemetry.experiment import ExperimentRun
        baseline = ExperimentRun('task', 'baseline', metadata={'accounting_complete': False})
        candidate = ExperimentRun('task', 'candidate', metadata={'accounting_complete': False})
        result = compare_runs(baseline, candidate).to_dict()
        self.assertIsNone(result['baseline_cost_usd'])
        self.assertIsNone(result['candidate_cost_usd'])
        self.assertIsNone(result['net_cost_saving_usd'])
        self.assertIsNone(result['cost_saving_percent'])
        self.assertFalse(result['candidate_is_net_better'])
