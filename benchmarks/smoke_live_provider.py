"""Opt-in, one-request Moonshot smoke; never a cost/savings benchmark."""
import argparse
import json
import os
from pathlib import Path

from tokenpilot.benchmark.runner import BaselineRunner
from tokenpilot.providers.base import ProviderRequest
from tokenpilot.providers.openai_compatible import OpenAICompatibleProvider
from tokenpilot.providers.transport import UrllibTransport
from tokenpilot.telemetry.experiment import QualityMetric


def load_config(path):
    # Read only supported keys; no shell execution, expansion, or environment mutation.
    keys = {"MOONSHOT_API_KEY", "MOONSHOT_BASE_URL", "MOONSHOT_MODEL"}
    config = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, separator, value = line.partition("=")
            if separator and key.strip() in keys:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                config[key.strip()] = value
    config.update({key: os.environ[key] for key in keys if key in os.environ})
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid-api", action="store_true",
                        help="Explicitly allow one real, potentially paid API request")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", help="Override MOONSHOT_MODEL (default: kimi-k2.6)")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Includes internal reasoning; a cap is not a monetary budget")
    args = parser.parse_args(argv)
    if not args.allow_paid_api:
        print("NETWORK_DISABLED: use --allow-paid-api to make one paid API request.")
        return 0
    try:
        config = load_config(args.env_file)
        key = config.get("MOONSHOT_API_KEY")
        base_url = config.get("MOONSHOT_BASE_URL")
        if not key or not base_url:
            raise ValueError("missing config")
        provider = OpenAICompatibleProvider(
            name="moonshot", base_url=base_url, api_key=key,
            transport=UrllibTransport(allow_network=True), token_limit_field="max_tokens",
        )
        request = ProviderRequest(args.model or config.get("MOONSHOT_MODEL", "kimi-k2.6"),
                                  "Reply with exactly: TOKENPILOT_OK",
                                  args.max_output_tokens, temperature=None)
        run = BaselineRunner(provider).run(
            "provider-connectivity-smoke", request,
            lambda text: QualityMetric("exact_match", float(text.strip() == "TOKENPILOT_OK")),
            dataset_version="connectivity-smoke-v1", evaluator_version="exact-match-v1",
            allow_paid_api=True,
        )
        # Do not print model identifiers, response text, credentials, or raw errors.
        metric = run.quality_metrics.get("exact_match")
        passed = metric is not None and metric.value == 1.0
        print(json.dumps({
            "smoke_ok": passed,
            "experiment_status": run.status.value,
            "accounting_complete": run.metadata["accounting_complete"],
            "note": "Usage smoke only; missing billing evidence remains unknown, not free.",
            "provider_error_code": run.metadata.get("provider_error_code"),
            "finish_reason": run.metadata.get("response", {}).get("finish_reason"),
            "ledger": {"summary": run.ledger.summary(), "usage_breakdowns": [
                event.metadata["usage_breakdown"] for event in run.ledger.events]},
        }, indent=2))
        return 0 if passed else 1
    except (ValueError, OSError, UnicodeError):
        print("CONFIG_ERROR: check local credential, HTTPS base URL, model and token limit.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
