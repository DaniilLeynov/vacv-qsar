from __future__ import annotations

import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .auth import build_headers, build_query, secret_query_parameters
from .errors import HttpError
from .manifest import ErrorHandler, Requester


@dataclass
class Response:
    url: str
    status: int
    headers: dict[str, str]
    content: bytes

    @property
    def content_type(self) -> str:
        for name, value in self.headers.items():
            if name.lower() == "content-type":
                return value
        return ""


def build_url(
    requester: Requester,
    extra_parameters: dict[str, str] | None = None,
) -> str:
    base = requester.url_base.rstrip("/")
    path = requester.path.strip("/")
    url = f"{base}/{path}" if path else base

    parameters: dict[str, str] = dict(requester.request_parameters)
    parameters.update(build_query(requester.authenticator))
    if extra_parameters:
        parameters.update({k: str(v) for k, v in extra_parameters.items()})

    if parameters:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{urlencode(parameters)}"
    return url


def build_headers_for(requester: Requester) -> dict[str, str]:
    headers = dict(requester.headers)
    headers.update(build_headers(requester.authenticator))
    return headers


def mask_url(url: str, secret_parameters: set[str]) -> str:
    if not secret_parameters:
        return url
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    masked = [
        (name, "REDACTED" if name in secret_parameters else value)
        for name, value in pairs
    ]
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(masked), parts.fragment)
    )


def mask_for(requester: Requester, url: str) -> str:
    return mask_url(url, secret_query_parameters(requester.authenticator))


def _content_length(headers: dict[str, str]) -> int | None:
    for name, value in headers.items():
        if name.lower() == "content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def http_get(
    url: str,
    headers: dict[str, str],
    error_handler: ErrorHandler,
    opener: Callable = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
    on_download: Callable[[int, int | None], None] | None = None,
    on_retry: Callable[[int, int | None], None] | None = None,
) -> Response:
    attempt = 0
    while True:
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with opener(request, timeout=error_handler.timeout_seconds) as raw:
                response_headers = {str(k): str(v) for k, v in raw.headers.items()}
                if on_download is None:
                    content = raw.read()
                else:
                    total = _content_length(response_headers)
                    chunks = []
                    received = 0
                    while True:
                        chunk = raw.read(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        received += len(chunk)
                        on_download(received, total)
                    content = b"".join(chunks)
                return Response(
                    url=raw.geturl(),
                    status=getattr(raw, "status", 200),
                    headers=response_headers,
                    content=content,
                )
        except urllib.error.HTTPError as exc:
            if (
                exc.code in error_handler.retry_statuses
                and attempt < error_handler.max_retries
            ):
                attempt += 1
                if on_retry:
                    on_retry(attempt, exc.code)
                sleep(error_handler.backoff_seconds * attempt)
                continue
            raise HttpError(exc.code, url, exc.reason or "") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt < error_handler.max_retries:
                attempt += 1
                if on_retry:
                    on_retry(attempt, None)
                sleep(error_handler.backoff_seconds * attempt)
                continue
            reason = getattr(exc, "reason", exc)
            raise HttpError(0, url, str(reason)) from exc
