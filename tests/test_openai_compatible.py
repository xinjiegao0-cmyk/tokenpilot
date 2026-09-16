"""Offline adapter contracts. Every response and credential below is synthetic."""
import io
import json
import socket
import unittest
from decimal import Decimal
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from tokenpilot.providers.base import ProviderError, ProviderRequest
from tokenpilot.providers.openai_compatible import CostEvidence, OpenAICompatibleProvider
from tokenpilot.providers.transport import BaseTransport, HttpResponse, UrllibTransport


def synthetic_response():
    return {
        "id": "synthetic-response-1",
        "model": "synthetic-model-v1",
        "system_fingerprint": "synthetic-fingerprint-v1",
        "choices": [{"message": {"role": "assistant", "content": "synthetic answer"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                  "total_tokens": 120, "prompt_tokens_details": {"cached_tokens": 60}},
    }


class FixtureTransport(BaseTransport):
    simulation = True

    def __init__(self, raw=None, *, body=None, status=200, error=None):
        self.body = body if body is not None else json.dumps(
            synthetic_response() if raw is None else raw).encode("utf-8")
        self.status = status
        self.error = error
        self.calls = []

    def post(self, url, headers, body, timeout):
        self.calls.append((url, dict(headers), json.loads(body), timeout))
        if self.error is not None:
            raise self.error
        return HttpResponse(self.status, self.body)


def synthetic_billing(raw):
    return CostEvidence(Decimal("0.0123"), "synthetic billing fixture v1", "simulated")


class OpenAICompatibleTests(unittest.TestCase):
    def request(self, **kwargs):
        values = {"model": "synthetic-model", "prompt": "synthetic prompt 中文", "max_output_tokens": 80}
        values.update(kwargs)
        return ProviderRequest(**values)

    def provider(self, transport=None, **kwargs):
        return OpenAICompatibleProvider(name="synthetic-provider",
            base_url="https://example.invalid/v1/",
            transport=transport if transport is not None else FixtureTransport(), **kwargs)

    def test_payload_maps_explicit_limit_and_sampling_parameters(self):
        transport = FixtureTransport()
        provider = self.provider(transport, timeout=4.5)
        response = provider.generate(self.request(temperature=0.25, seed=0))
        self.assertEqual(len(transport.calls), 1)
        url, headers, payload, timeout = transport.calls[0]
        self.assertEqual(url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(payload, {"model": "synthetic-model",
            "messages": [{"role": "user", "content": "synthetic prompt 中文"}],
            "max_completion_tokens": 80, "stream": False, "n": 1, "temperature": 0.25, "seed": 0})
        self.assertEqual(timeout, 4.5)
        self.assertNotIn("Authorization", headers)
        self.assertTrue(provider.simulation)
        self.assertEqual(response.text, "synthetic answer")
        self.assertTrue(response.completed)
        self.assertEqual(response.usage.total_tokens, 120)
        self.assertEqual(response.usage.cached_input_tokens, 60)

    def test_legacy_limit_and_optional_parameters_are_not_sent_together(self):
        transport = FixtureTransport()
        self.provider(transport, token_limit_field="max_tokens").generate(
            self.request(temperature=None, seed=None))
        payload = transport.calls[0][2]
        self.assertEqual(payload["max_tokens"], 80)
        for omitted in ("max_completion_tokens", "temperature", "seed"):
            self.assertNotIn(omitted, payload)

    def test_invalid_request_is_rejected_before_transport(self):
        invalid = ({"model": ""}, {"prompt": None}, {"max_output_tokens": True},
                   {"max_output_tokens": 0}, {"temperature": float("nan")},
                   {"temperature": True}, {"temperature": 2.1}, {"seed": False})
        transport = FixtureTransport()
        provider = self.provider(transport)
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                provider.generate(self.request(**values))
        self.assertEqual(transport.calls, [])

    def test_absent_billing_is_unknown_not_free(self):
        raw = synthetic_response()
        raw["cost_usd"] = "0.0001"  # Vendor fields require an explicit billing extractor.
        response = self.provider(FixtureTransport(raw)).generate(self.request())
        self.assertIsNone(response.cost_usd)
        self.assertIsNone(response.cost_source)
        self.assertEqual(response.usage.total_tokens, 120)

    def test_billing_none_keeps_usage_and_unknown_cost(self):
        response = self.provider(billing_extractor=lambda raw: None,
            billing_version="synthetic-none-v1").generate(self.request())
        self.assertIsNone(response.cost_usd)
        self.assertEqual(response.usage.total_tokens, 120)

    def test_decimal_billing_is_exact_and_fixture_is_always_simulated(self):
        body = json.dumps(synthetic_response())[:-1] + ', "synthetic_cost": 0.01234567890123456789}'
        response = self.provider(FixtureTransport(body=body.encode()),
            billing_extractor=lambda raw: CostEvidence(raw["synthetic_cost"], "synthetic invoice", "reported"),
            billing_version="synthetic-decimal-v1").generate(self.request())
        self.assertEqual(response.cost_usd, Decimal("0.01234567890123456789"))
        self.assertEqual(response.cost_kind, "simulated")

    def test_billing_error_preserves_known_usage_and_hides_exception(self):
        def fail(raw):
            raise ValueError("SYNTHETIC_PRIVATE_BILLING_VALUE")
        provider = self.provider(billing_extractor=fail, billing_version="synthetic-failure-v1")
        with self.assertRaises(ProviderError) as caught:
            provider.generate(self.request())
        self.assertEqual(caught.exception.code, "billing_error")
        self.assertEqual(caught.exception.response.usage.total_tokens, 120)
        self.assertIsNone(caught.exception.response.cost_usd)
        self.assertNotIn("SYNTHETIC_PRIVATE_BILLING_VALUE", str(caught.exception))

    def test_invalid_output_preserves_known_usage_and_cost(self):
        for choices in (None, [], [{"message": None}], [1], [{
            "message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]):
            raw = synthetic_response()
            raw["choices"] = choices
            provider = self.provider(FixtureTransport(raw), billing_extractor=synthetic_billing,
                billing_version="synthetic-billing-v1")
            with self.subTest(choices=choices), self.assertRaises(ProviderError) as caught:
                provider.generate(self.request())
            self.assertEqual(caught.exception.code, "invalid_response")
            self.assertEqual(caught.exception.response.usage.total_tokens, 120)
            self.assertEqual(caught.exception.response.cost_usd, Decimal("0.0123"))
            self.assertFalse(caught.exception.response.completed)

    def test_malformed_finish_reason_is_safe_partial_error(self):
        for reason in ([], {}, 17, None, "unknown"):
            raw = synthetic_response()
            raw["choices"][0]["finish_reason"] = reason
            with self.subTest(reason=reason):
                with self.assertRaises(ProviderError) as caught:
                    self.provider(FixtureTransport(raw)).generate(self.request())
                self.assertEqual(caught.exception.code, "invalid_response")
                self.assertEqual(caught.exception.response.usage.total_tokens, 120)

    def test_incomplete_and_refused_outputs_keep_accounting(self):
        variants = (("length", {}, "length"), ("content_filter", {"content": None}, "content_filter"),
                    ("stop", {"refusal": "synthetic refusal", "content": None}, "refusal"),
                    ("tool_calls", {"tool_calls": [{"synthetic": True}], "content": None}, "tool_calls"))
        for reason, message, expected in variants:
            raw = synthetic_response()
            raw["choices"][0]["finish_reason"] = reason
            raw["choices"][0]["message"].update(message)
            with self.subTest(reason=reason, expected=expected):
                response = self.provider(FixtureTransport(raw), billing_extractor=synthetic_billing,
                    billing_version="synthetic-billing-v1").generate(self.request())
                self.assertFalse(response.completed)
                self.assertEqual(response.metadata["finish_reason"], expected)
                self.assertEqual(response.usage.total_tokens, 120)
                self.assertEqual(response.cost_usd, Decimal("0.0123"))

    def test_invalid_usage_rejected_including_falsy_details(self):
        invalid = [None, {}, {"prompt_tokens": True, "completion_tokens": 1},
                   {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 9},
                   {"prompt_tokens": 2, "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 3}}]
        invalid.extend({"prompt_tokens": 2, "completion_tokens": 1, "prompt_tokens_details": value}
                       for value in ([], 0, False, ""))
        for usage in invalid:
            raw = synthetic_response()
            raw["usage"] = usage
            with self.subTest(usage=usage), self.assertRaises(ProviderError) as caught:
                self.provider(FixtureTransport(raw)).generate(self.request())
            self.assertEqual(caught.exception.code, "invalid_usage")
            self.assertIsNone(caught.exception.response)

    def test_absent_or_null_cache_details_mean_zero(self):
        for details in (None, {}):
            raw = synthetic_response()
            raw["usage"]["prompt_tokens_details"] = details
            response = self.provider(FixtureTransport(raw)).generate(self.request())
            self.assertEqual(response.usage.cached_input_tokens, 0)
            self.assertEqual(response.usage.total_tokens, 120)
        raw["usage"].pop("prompt_tokens_details")
        self.assertEqual(self.provider(FixtureTransport(raw)).generate(self.request()).usage.cached_input_tokens, 0)

    def test_invalid_json_and_root_shape_are_safe_errors(self):
        for body, expected in ((b"SYNTHETIC_PRIVATE_INVALID_JSON", "invalid_json"),
                               (b"\xff", "invalid_json"), (b"[]", "invalid_response")):
            with self.subTest(body=body), self.assertRaises(ProviderError) as caught:
                self.provider(FixtureTransport(body=body)).generate(self.request())
            self.assertEqual(caught.exception.code, expected)
            self.assertNotIn("SYNTHETIC_PRIVATE", str(caught.exception))

    def test_transport_failures_and_http_errors_are_never_retried(self):
        cases = ((TimeoutError("synthetic private timeout"), "timeout"),
                 (RuntimeError("synthetic private error"), "transport_error"),
                 (ProviderError("response_too_large"), "response_too_large"))
        for error, expected in cases:
            transport = FixtureTransport(error=error)
            with self.subTest(error=error), self.assertRaises(ProviderError) as caught:
                self.provider(transport).generate(self.request())
            self.assertEqual(caught.exception.code, expected)
            self.assertEqual(len(transport.calls), 1)
            self.assertNotIn("private", str(caught.exception))
        transport = FixtureTransport(status=429, body=b"synthetic private error")
        with self.assertRaises(ProviderError) as caught:
            self.provider(transport).generate(self.request())
        self.assertEqual(caught.exception.code, "http_error")
        self.assertEqual(caught.exception.http_status, 429)
        self.assertEqual(len(transport.calls), 1)

    def test_credentials_only_enter_authorization_and_not_public_metadata(self):
        token = "SYNTHETIC_TEST_TOKEN_DO_NOT_USE"
        raw = synthetic_response()
        raw["id"] = token
        raw["model"] = "synthetic model with whitespace"
        transport = FixtureTransport(raw)
        provider = self.provider(transport, api_key=token,
            billing_extractor=lambda raw: CostEvidence(Decimal("0.01"), "synthetic " + token),
            billing_version="synthetic-redaction-v1")
        response = provider.generate(self.request())
        self.assertEqual(transport.calls[0][1]["Authorization"], "Bearer " + token)
        self.assertIsNone(response.metadata["id"])
        self.assertIsNone(response.metadata["model"])
        self.assertNotIn(token, response.cost_source)
        self.assertNotIn(token, json.dumps(provider.describe()))
        self.assertNotIn(token, repr(provider))

    def test_real_transport_without_explicit_key_never_discovers_credentials(self):
        transport = FixtureTransport()
        transport.simulation = False
        with patch.dict("os.environ", {"OPENAI_API_KEY": "SYNTHETIC_ENV_TOKEN_DO_NOT_USE"}):
            with self.assertRaises(ProviderError) as caught:
                self.provider(transport).generate(self.request())
        self.assertEqual(caught.exception.code, "missing_credential")
        self.assertEqual(transport.calls, [])

    def test_invalid_endpoint_or_config_is_rejected(self):
        invalid = ({"base_url": "http://example.invalid/v1"},
                   {"base_url": "https://user:synthetic@example.invalid/v1"},
                   {"base_url": "https://example.invalid/v1?token=synthetic"},
                   {"base_url": "https://example.invalid/v1#fragment"},
                   {"timeout": 0}, {"timeout": float("inf")},
                   {"token_limit_field": "invented_limit"}, {"api_key": "synthetic\r\nheader"},
                   {"billing_extractor": synthetic_billing})
        for values in invalid:
            options = {"name": "synthetic-provider", "base_url": "https://example.invalid/v1"}
            options.update(values)
            with self.subTest(values=values), self.assertRaises(ValueError):
                OpenAICompatibleProvider(**options)


class UrllibTransportTests(unittest.TestCase):
    url = "https://example.invalid/v1/chat/completions"

    def test_default_network_guard_prevents_opener_creation(self):
        with patch("tokenpilot.providers.transport.build_opener") as build:
            with self.assertRaises(ProviderError) as caught:
                UrllibTransport().post(self.url, {}, b"{}", 5)
        self.assertEqual(caught.exception.code, "network_disabled")
        build.assert_not_called()

    def test_mocked_success_forwards_timeout_and_disables_redirects(self):
        response = Mock()
        response.status = 200
        response.read.return_value = b"{}"
        opened = Mock()
        opened.__enter__ = Mock(return_value=response)
        opened.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = opened
        with patch("tokenpilot.providers.transport.build_opener", return_value=opener) as build:
            result = UrllibTransport(allow_network=True).post(self.url, {}, b"{}", 7)
        self.assertEqual(result, HttpResponse(200, b"{}"))
        self.assertEqual(opener.open.call_count, 1)
        self.assertEqual(opener.open.call_args[1]["timeout"], 7)
        proxy_handler, redirect_handler = build.call_args[0]
        self.assertEqual(proxy_handler.proxies, {})
        self.assertIsNone(redirect_handler.redirect_request(None, None, 302, None, None,
            "https://other.invalid/"))

    def test_mocked_transport_errors_are_classified_without_retry(self):
        cases = ((socket.timeout("synthetic private timeout"), "timeout", None),
                 (URLError(socket.timeout("synthetic")), "timeout", None),
                 (URLError("synthetic private transport error"), "transport_error", None),
                 (HTTPError(self.url, 429, "synthetic private HTTP error", {}, io.BytesIO(b"synthetic body")),
                  "http_error", 429))
        for error, expected, status in cases:
            opener = Mock()
            opener.open.side_effect = error
            with self.subTest(error=error):
                with patch("tokenpilot.providers.transport.build_opener", return_value=opener):
                    with self.assertRaises(ProviderError) as caught:
                        UrllibTransport(allow_network=True).post(self.url, {}, b"{}", 5)
                self.assertEqual(caught.exception.code, expected)
                self.assertEqual(caught.exception.http_status, status)
                self.assertEqual(opener.open.call_count, 1)
                self.assertNotIn("private", str(caught.exception))

    def test_mocked_oversized_response_is_rejected(self):
        response = Mock()
        response.read.return_value = b"12345"
        opened = Mock()
        opened.__enter__ = Mock(return_value=response)
        opened.__exit__ = Mock(return_value=False)
        opener = Mock()
        opener.open.return_value = opened
        with patch("tokenpilot.providers.transport.build_opener", return_value=opener):
            with self.assertRaises(ProviderError) as caught:
                UrllibTransport(allow_network=True, max_response_bytes=4).post(self.url, {}, b"{}", 5)
        self.assertEqual(caught.exception.code, "response_too_large")
        response.read.assert_called_once_with(5)
        self.assertEqual(opener.open.call_count, 1)


if __name__ == "__main__":
    unittest.main()
