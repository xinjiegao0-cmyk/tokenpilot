"""Offline adapter demonstration. Synthetic data cannot support benchmark claims."""
import json
from decimal import Decimal
from pathlib import Path

from tokenpilot.benchmark.runner import BaselineRunner
from tokenpilot.providers.base import ProviderRequest
from tokenpilot.providers.openai_compatible import CostEvidence, OpenAICompatibleProvider
from tokenpilot.providers.transport import BaseTransport, HttpResponse
from tokenpilot.telemetry.experiment import QualityMetric, RunStatus


class FixtureTransport(BaseTransport):
    simulation = True

    def post(self, url, headers, body, timeout):
        # No socket, HTTP client, environment variable, or credential access.
        return HttpResponse(200, Path(__file__).with_name("fixtures").joinpath("chat_completion.json").read_bytes())


def fixture_billing(raw):
    # This is a fixture extension, NOT a standard Chat Completions billing field.
    return CostEvidence(Decimal(raw["_fixture_cost_usd"]), "synthetic fixture v1", "simulated")


def run_demo():
    provider = OpenAICompatibleProvider(
        name="offline-fixture", base_url="https://fixture.invalid/v1",
        transport=FixtureTransport(), billing_extractor=fixture_billing,
        billing_version="synthetic-billing-v1",
    )
    return BaselineRunner(provider).run(
        "arithmetic-fixture-001", ProviderRequest("fixture-model-v1", "What is 2 + 2?", 32, seed=7),
        lambda text: QualityMetric("exact_match", float(text.strip() == "4")),
        dataset_version="arithmetic-synthetic-v1", evaluator_version="exact-match-v1",
    )


def main():
    run = run_demo()
    print(json.dumps({"warning": "SIMULATED_SMOKE_TEST_ONLY", "run": run.to_dict()},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if run.status == RunStatus.SUCCEEDED else 1


if __name__ == "__main__":
    raise SystemExit(main())
