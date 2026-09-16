"""Small injectable HTTP boundary. No retries, redirects, or credential discovery."""
import math
import socket
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from tokenpilot.providers.base import ProviderError


def validate_https_url(url: str) -> None:
    try:
        parts = urlsplit(url)
        valid = (parts.scheme == "https" and parts.hostname and not parts.username
                 and not parts.password and not parts.query and not parts.fragment
                 and parts.port != 0 and not any(c.isspace() for c in url))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("endpoint must be HTTPS without credentials, query or fragment")


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes


class BaseTransport(ABC):
    simulation: bool = False

    @abstractmethod
    def post(self, url: str, headers: Mapping[str, str], body: bytes,
             timeout: float) -> HttpResponse:
        """Perform exactly one attempt. Offline fixtures must set simulation=True."""
        raise NotImplementedError


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UrllibTransport(BaseTransport):
    def __init__(self, *, allow_network: bool = False, max_response_bytes: int = 2_000_000):
        if type(allow_network) is not bool:
            raise ValueError("allow_network must be a boolean")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self._allow_network = allow_network
        self._max_response_bytes = max_response_bytes

    def post(self, url, headers, body, timeout):
        if not self._allow_network:
            raise ProviderError("network_disabled")
        validate_https_url(url)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        request = Request(url, data=body, headers=dict(headers), method="POST")
        # No environment proxy discovery; redirects must not forward Authorization.
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout) as response:
                data = response.read(self._max_response_bytes + 1)
                if len(data) > self._max_response_bytes:
                    raise ProviderError("response_too_large")
                return HttpResponse(response.status, data)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            # Error bodies and URLs can contain secrets. Never retain them.
            raise ProviderError("http_error", http_status=status) from None
        except (TimeoutError, socket.timeout):
            raise ProviderError("timeout") from None
        except URLError as exc:
            code = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "transport_error"
            raise ProviderError(code) from None
        except ProviderError:
            raise
        except Exception:
            raise ProviderError("transport_error") from None
