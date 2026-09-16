import unittest
from decimal import Decimal

from tokenpilot.telemetry.ledger import (
    ResourceEvent,
    ResourceLedger,
)


class ResourceLedgerTests(unittest.TestCase):

    def test_records_basic_usage(self) -> None:
        ledger = ResourceLedger()

        ledger.record(
            ResourceEvent(
                category="model_call",
                input_tokens=1000,
                output_tokens=250,
                cost_usd=Decimal("0.015"),
                latency_ms=800.0,
                provider="example",
                model="model-a",
            )
        )

        self.assertEqual(ledger.total_input_tokens, 1000)
        self.assertEqual(ledger.total_output_tokens, 250)
        self.assertEqual(ledger.total_tokens, 1250)

        self.assertEqual(
            ledger.total_cost_usd,
            Decimal("0.015"),
        )


    def test_cached_tokens_are_not_double_counted(self) -> None:
        ledger = ResourceLedger()

        ledger.record(
            ResourceEvent(
                category="model_call",
                input_tokens=1000,
                cached_input_tokens=700,
                output_tokens=200,
                cost_usd=Decimal("0.010"),
            )
        )

        self.assertEqual(
            ledger.total_cached_input_tokens,
            700,
        )

        self.assertEqual(
            ledger.total_uncached_input_tokens,
            300,
        )

        # 1000 input + 200 output
        # cached tokens are already part of input tokens.
        self.assertEqual(
            ledger.total_tokens,
            1200,
        )


    def test_separates_task_and_overhead_cost(self) -> None:
        ledger = ResourceLedger()

        ledger.record(
            ResourceEvent(
                category="task_model_call",
                input_tokens=1000,
                output_tokens=200,
                cost_usd=Decimal("0.020"),
                is_overhead=False,
            )
        )

        ledger.record(
            ResourceEvent(
                category="planner",
                input_tokens=200,
                output_tokens=50,
                cost_usd=Decimal("0.004"),
                is_overhead=True,
            )
        )

        self.assertEqual(
            ledger.task_cost_usd,
            Decimal("0.020"),
        )

        self.assertEqual(
            ledger.overhead_cost_usd,
            Decimal("0.004"),
        )

        self.assertEqual(
            ledger.total_cost_usd,
            Decimal("0.024"),
        )

        self.assertEqual(
            ledger.task_tokens,
            1200,
        )

        self.assertEqual(
            ledger.overhead_tokens,
            250,
        )


    def test_rejects_negative_tokens(self) -> None:
        ledger = ResourceLedger()

        with self.assertRaises(ValueError):
            ledger.record(
                ResourceEvent(
                    category="invalid",
                    input_tokens=-1,
                )
            )


    def test_rejects_cache_larger_than_input(self) -> None:
        ledger = ResourceLedger()

        with self.assertRaises(ValueError):
            ledger.record(
                ResourceEvent(
                    category="invalid",
                    input_tokens=100,
                    cached_input_tokens=101,
                )
            )


    def test_rejects_empty_category(self) -> None:
        ledger = ResourceLedger()

        with self.assertRaises(ValueError):
            ledger.record(
                ResourceEvent(
                    category="   ",
                )
            )


    def test_serialization_is_json_friendly(self) -> None:
        ledger = ResourceLedger()

        ledger.record(
            ResourceEvent(
                category="planner",
                input_tokens=100,
                output_tokens=20,
                cost_usd=Decimal("0.00125"),
                is_overhead=True,
                metadata={
                    "experiment": "v0.1",
                },
            )
        )

        data = ledger.to_dict()

        self.assertEqual(
            data["events"][0]["cost_usd"],
            "0.00125",
        )

        self.assertEqual(
            data["summary"]["total_cost_usd"],
            "0.00125",
        )


if __name__ == "__main__":
    unittest.main()
