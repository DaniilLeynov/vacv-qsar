import base64

import pytest

from src.fetch.auth import build_headers, build_query, secret_query_parameters
from src.fetch.errors import AuthError
from src.fetch.manifest import Authenticator


def test_no_auth():
    assert build_headers(Authenticator(type="NoAuth")) == {}
    assert build_query(Authenticator(type="NoAuth")) == {}


def test_bearer(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret-token")
    headers = build_headers(Authenticator(type="Bearer", env="MY_TOKEN"))
    assert headers == {"Authorization": "Bearer secret-token"}


def test_bearer_without_env_var(monkeypatch):
    monkeypatch.delenv("MY_TOKEN", raising=False)
    with pytest.raises(AuthError, match="MY_TOKEN"):
        build_headers(Authenticator(type="Bearer", env="MY_TOKEN"))


def test_api_key_header(monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc")
    auth = Authenticator(type="ApiKeyHeader", env="MY_KEY", header_name="X-API-Key")
    assert build_headers(auth) == {"X-API-Key": "abc"}


def test_api_key_header_with_prefix(monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc")
    auth = Authenticator(
        type="ApiKeyHeader", env="MY_KEY", header_name="X-API-Key", prefix="Token"
    )
    assert build_headers(auth) == {"X-API-Key": "Token abc"}


def test_basic(monkeypatch):
    monkeypatch.setenv("MY_BASIC", "user:pass")
    headers = build_headers(Authenticator(type="Basic", env="MY_BASIC"))
    expected = base64.b64encode(b"user:pass").decode("ascii")
    assert headers == {"Authorization": f"Basic {expected}"}


def test_basic_requires_user_colon_password(monkeypatch):
    monkeypatch.setenv("MY_BASIC", "no-colon")
    with pytest.raises(AuthError, match="user:password"):
        build_headers(Authenticator(type="Basic", env="MY_BASIC"))


def test_api_key_query(monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc")
    auth = Authenticator(type="ApiKeyQuery", env="MY_KEY", query_parameter="api_key")
    assert build_headers(auth) == {}
    assert build_query(auth) == {"api_key": "abc"}
    assert secret_query_parameters(auth) == {"api_key"}


def test_api_key_query_requires_parameter(monkeypatch):
    monkeypatch.setenv("MY_KEY", "abc")
    with pytest.raises(AuthError, match="query_parameter"):
        build_query(Authenticator(type="ApiKeyQuery", env="MY_KEY"))
