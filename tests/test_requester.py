from urllib.parse import parse_qs, urlsplit

import pytest

from src.fetch.manifest import Authenticator, Requester
from src.fetch.requester import build_headers_for, build_url, mask_url


def requester(**overrides):
    data = {
        "url_base": "https://example.org/api/",
        "path": "items.json",
        "request_parameters": {"organism": "Vaccinia virus"},
        "headers": {"Accept": "application/json"},
    }
    data.update(overrides)
    return Requester(**data)


def query_of(url):
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


def test_build_url_joins_base_path_and_parameters():
    url = build_url(requester())
    parts = urlsplit(url)
    assert url.startswith("https://example.org/api/items.json?")
    assert query_of(url)["organism"] == ["Vaccinia virus"]
    assert parts.path == "/api/items.json"


def test_build_url_adds_extra_parameters():
    url = build_url(requester(), {"offset": 100, "limit": 50})
    assert query_of(url)["offset"] == ["100"]
    assert query_of(url)["limit"] == ["50"]


def test_build_url_without_path():
    url = build_url(requester(path=""))
    assert url.startswith("https://example.org/api?") or url == (
        "https://example.org/api?organism=Vaccinia+virus"
    )


def test_api_key_query_is_added_and_masked(monkeypatch):
    monkeypatch.setenv("MY_KEY", "super-secret")
    auth = Authenticator(type="ApiKeyQuery", env="MY_KEY", query_parameter="api_key")
    url = build_url(requester(authenticator=auth))
    assert query_of(url)["api_key"] == ["super-secret"]

    masked = mask_url(url, {"api_key"})
    assert "super-secret" not in masked
    assert query_of(masked)["api_key"] == ["REDACTED"]


def test_mask_url_leaves_other_parameters():
    url = "https://example.org/x?api_key=secret&organism=VV"
    masked = mask_url(url, {"api_key"})
    assert query_of(masked)["organism"] == ["VV"]
    assert "secret" not in masked


def test_mask_url_without_secrets_is_noop():
    url = "https://example.org/x?a=1"
    assert mask_url(url, set()) == url


def test_headers_include_auth(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "token")
    auth = Authenticator(type="Bearer", env="MY_TOKEN")
    headers = build_headers_for(requester(authenticator=auth))
    assert headers["Accept"] == "application/json"
    assert headers["Authorization"] == "Bearer token"


class FakeRaw:
    def __init__(self, data, headers, url, status=200):
        self._data = data
        self._position = 0
        self.headers = headers
        self.status = status
        self._url = url

    def read(self, size=-1):
        if size is None or size < 0:
            chunk = self._data[self._position :]
            self._position = len(self._data)
            return chunk
        chunk = self._data[self._position : self._position + size]
        self._position += len(chunk)
        return chunk

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_http_get_streams_with_progress():
    from src.fetch.manifest import ErrorHandler
    from src.fetch.requester import http_get

    payload = b"x" * 200000
    calls = []

    def opener(request, timeout):
        return FakeRaw(
            payload,
            {"Content-Length": str(len(payload))},
            "https://example.org/file",
        )

    response = http_get(
        "https://example.org/file",
        {},
        ErrorHandler(),
        opener=opener,
        sleep=lambda seconds: None,
        on_download=lambda received, total: calls.append((received, total)),
    )

    assert response.content == payload
    assert calls[0] == (65536, len(payload))
    assert calls[-1] == (len(payload), len(payload))


def test_http_get_retries_on_server_error():
    import urllib.error

    from src.fetch.manifest import ErrorHandler
    from src.fetch.requester import http_get

    attempts = []
    retries = []

    def opener(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(
                "https://example.org/x", 500, "server error", {}, None
            )
        return FakeRaw(b"ok", {}, "https://example.org/x")

    response = http_get(
        "https://example.org/x",
        {},
        ErrorHandler(max_retries=3, backoff_seconds=0),
        opener=opener,
        sleep=lambda seconds: None,
        on_retry=lambda attempt, status: retries.append((attempt, status)),
    )

    assert response.content == b"ok"
    assert len(attempts) == 3
    assert retries == [(1, 500), (2, 500)]
