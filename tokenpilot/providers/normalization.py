"""Normalize compatible usage counts; cache/reasoning are subsets, not additions."""
from collections.abc import Mapping
from tokenpilot.providers.base import Usage


def _consistent(values, default=None):
    if not values:
        return default
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("token counts must be nonnegative integers")
    if any(value != values[0] for value in values[1:]):
        raise ValueError("conflicting usage aliases")
    return values[0]


def normalize_chat_usage(raw):
    if not isinstance(raw, Mapping):
        raise ValueError("usage is missing or invalid")
    counts = []
    for aliases in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
        values = [raw[key] for key in aliases if key in raw]
        if not values:
            raise ValueError("required usage fields missing")
        counts.append(_consistent(values))
    details = {}
    for key in ("prompt_tokens_details", "input_tokens_details",
                "completion_tokens_details", "output_tokens_details"):
        value = raw.get(key)
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise ValueError("invalid token details")
        details[key] = value
    cached = [raw["cached_tokens"]] if "cached_tokens" in raw else []
    cached += [details[key]["cached_tokens"]
               for key in ("prompt_tokens_details", "input_tokens_details")
               if "cached_tokens" in details[key]]
    reasoning = [details[key]["reasoning_tokens"]
                 for key in ("completion_tokens_details", "output_tokens_details")
                 if "reasoning_tokens" in details[key]]
    usage = Usage(counts[0], counts[1], _consistent(cached, 0), _consistent(reasoning))
    if "total_tokens" in raw and (type(raw["total_tokens"]) is not int
                                  or raw["total_tokens"] != usage.total_tokens):
        raise ValueError("inconsistent total_tokens")
    return usage
