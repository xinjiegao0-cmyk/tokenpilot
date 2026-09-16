"""Strict normalization of the Chat Completions usage shape; no network client."""
from collections.abc import Mapping
from tokenpilot.providers.base import Usage


def normalize_chat_usage(raw):
    if not isinstance(raw, Mapping):
        raise ValueError("usage is missing or invalid")
    if "prompt_tokens" not in raw or "completion_tokens" not in raw:
        raise ValueError("required usage fields missing")
    details = raw.get("prompt_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, Mapping):
        raise ValueError("invalid prompt token details")
    usage = Usage(raw["prompt_tokens"], raw["completion_tokens"], details.get("cached_tokens", 0))
    if "total_tokens" in raw and (type(raw["total_tokens"]) is not int or raw["total_tokens"] != usage.total_tokens):
        raise ValueError("inconsistent total_tokens")
    return usage
