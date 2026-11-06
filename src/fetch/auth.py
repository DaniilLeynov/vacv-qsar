from __future__ import annotations

import base64
import os

from .errors import AuthError
from .manifest import Authenticator


def _secret(auth: Authenticator) -> str:
    if not auth.env:
        raise AuthError(f"{auth.type}: environment variable name is not set")
    value = os.environ.get(auth.env)
    if not value:
        raise AuthError(
            f"{auth.type}: environment variable {auth.env} is not set or empty"
        )
    return value


def build_headers(auth: Authenticator) -> dict[str, str]:
    if auth.type == "NoAuth":
        return {}
    if auth.type == "Bearer":
        return {"Authorization": f"Bearer {_secret(auth)}"}
    if auth.type == "ApiKeyHeader":
        if not auth.header_name:
            raise AuthError("ApiKeyHeader: header name is not set")
        value = _secret(auth)
        if auth.prefix:
            value = f"{auth.prefix} {value}"
        return {auth.header_name: value}
    if auth.type == "Basic":
        raw = _secret(auth)
        if ":" not in raw:
            raise AuthError("Basic: env value must be in user:password format")
        encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return {"Authorization": f"Basic {encoded}"}
    if auth.type == "ApiKeyQuery":
        return {}
    raise AuthError(f"unknown auth type {auth.type!r}")


def build_query(auth: Authenticator) -> dict[str, str]:
    if auth.type != "ApiKeyQuery":
        return {}
    if not auth.query_parameter:
        raise AuthError("ApiKeyQuery: query_parameter is not set")
    return {auth.query_parameter: _secret(auth)}


def secret_query_parameters(auth: Authenticator) -> set[str]:
    if auth.type == "ApiKeyQuery" and auth.query_parameter:
        return {auth.query_parameter}
    return set()
