from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ManifestError

SOURCE_TYPES = {"DeclarativeSource", "FileSource"}
AUTH_TYPES = {"NoAuth", "Bearer", "ApiKeyHeader", "ApiKeyQuery", "Basic"}
PAGINATOR_TYPES = {"NoPagination", "OffsetIncrement", "PageIncrement", "NextUrl"}
ERROR_HANDLER_TYPES = {"DefaultErrorHandler"}

ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

SECRET_KEYS = {"token", "api_key", "apikey", "password", "secret", "authorization"}

TOP_KEYS = {
    "version",
    "id",
    "type",
    "description",
    "docs_url",
    "license",
    "requester",
    "paginator",
    "error_handler",
    "output",
    "download",
}
REQUESTER_KEYS = {
    "type",
    "url_base",
    "path",
    "http_method",
    "request_parameters",
    "headers",
    "authenticator",
}
AUTH_KEYS = {"type", "env", "header_name", "prefix", "query_parameter"}
PAGINATOR_KEYS = {
    "type",
    "page_size",
    "offset_parameter",
    "limit_parameter",
    "page_parameter",
    "start_page",
    "next_url_path",
    "max_pages",
    "stop_when_empty",
}
ERROR_HANDLER_KEYS = {
    "type",
    "max_retries",
    "backoff_seconds",
    "retry_statuses",
    "timeout_seconds",
}
OUTPUT_KEYS = {"extension"}
DOWNLOAD_KEYS = {"url", "filename", "sha256"}


@dataclass
class Authenticator:
    type: str = "NoAuth"
    env: str | None = None
    header_name: str | None = None
    prefix: str | None = None
    query_parameter: str | None = None


@dataclass
class Requester:
    url_base: str
    path: str = ""
    http_method: str = "GET"
    request_parameters: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    authenticator: Authenticator = field(default_factory=Authenticator)


@dataclass
class Paginator:
    type: str = "NoPagination"
    page_size: int = 100
    offset_parameter: str = "offset"
    limit_parameter: str | None = "limit"
    page_parameter: str = "page"
    start_page: int = 1
    next_url_path: str | None = None
    max_pages: int = 10
    stop_when_empty: str | None = None


@dataclass
class ErrorHandler:
    type: str = "DefaultErrorHandler"
    max_retries: int = 3
    backoff_seconds: float = 2.0
    retry_statuses: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    timeout_seconds: float = 30.0


@dataclass
class Output:
    extension: str = "json"


@dataclass
class Download:
    url: str
    filename: str
    sha256: str | None = None


@dataclass
class Manifest:
    id: str
    type: str
    version: str = "0.1"
    description: str = ""
    docs_url: str = ""
    license: str = ""
    requester: Requester | None = None
    paginator: Paginator | None = None
    error_handler: ErrorHandler | None = None
    output: Output | None = None
    download: Download | None = None

    @property
    def error_handler_or_default(self) -> ErrorHandler:
        return self.error_handler or ErrorHandler()

    @property
    def paginator_or_default(self) -> Paginator:
        return self.paginator or Paginator()

    @property
    def output_or_default(self) -> Output:
        return self.output or Output()

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fail(message: str) -> None:
    raise ManifestError(message)


def _unknown_keys(data: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        _fail(f"{where}: unknown fields {unknown}; allowed: {sorted(allowed)}")


def _require(data: dict, key: str, where: str) -> Any:
    if key not in data or data[key] is None:
        _fail(f"{where}: required field '{key}' is missing")
    return data[key]


def _str(data: dict, key: str, where: str, default: str | None = None) -> str:
    if key not in data or data[key] is None:
        if default is None:
            _fail(f"{where}: field '{key}' must be a string")
        return default
    value = data[key]
    if not isinstance(value, str):
        _fail(f"{where}: field '{key}' must be a string")
    return value


def _int(data: dict, key: str, where: str, default: int | None = None) -> int:
    if key not in data or data[key] is None:
        if default is None:
            _fail(f"{where}: field '{key}' must be an integer")
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{where}: field '{key}' must be an integer")
    return value


def _float(data: dict, key: str, where: str, default: float | None = None) -> float:
    if key not in data or data[key] is None:
        if default is None:
            _fail(f"{where}: field '{key}' must be a number")
        return default
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{where}: field '{key}' must be a number")
    return float(value)


def _dict(data: dict, key: str, where: str) -> dict:
    if key not in data or data[key] is None:
        return {}
    value = data[key]
    if not isinstance(value, dict):
        _fail(f"{where}: field '{key}' must be an object")
    return value


def _scalar_map(data: dict, key: str, where: str) -> dict[str, str]:
    raw = _dict(data, key, where)
    result: dict[str, str] = {}
    for name, value in raw.items():
        if isinstance(value, (dict, list)) or value is None:
            _fail(f"{where}.{key}.{name}: only strings and numbers are allowed")
        result[str(name)] = str(value)
    return result


def _check_secrets(data: Any, where: str = "manifest") -> None:
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).lower() in SECRET_KEYS and isinstance(value, str) and value:
                _fail(
                    f"{where}.{key}: secrets must not be stored in the manifest, "
                    "use the 'env' field and an environment variable"
                )
            _check_secrets(value, f"{where}.{key}")
    elif isinstance(data, list):
        for index, value in enumerate(data):
            _check_secrets(value, f"{where}[{index}]")


def _parse_authenticator(data: dict, where: str) -> Authenticator:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object")
    _unknown_keys(data, AUTH_KEYS, where)
    auth_type = _str(data, "type", where, "NoAuth")
    if auth_type not in AUTH_TYPES:
        _fail(f"{where}.type: unknown type {auth_type!r}; allowed: {sorted(AUTH_TYPES)}")

    auth = Authenticator(type=auth_type)
    auth.env = _str(data, "env", where, None) if "env" in data else None
    auth.header_name = (
        _str(data, "header_name", where, None) if "header_name" in data else None
    )
    auth.prefix = _str(data, "prefix", where, None) if "prefix" in data else None
    auth.query_parameter = (
        _str(data, "query_parameter", where, None)
        if "query_parameter" in data
        else None
    )

    if auth_type in {"Bearer", "ApiKeyHeader", "ApiKeyQuery", "Basic"} and not auth.env:
        _fail(f"{where}: auth type {auth_type} requires the 'env' field")
    if auth_type == "ApiKeyHeader" and not auth.header_name:
        _fail(f"{where}: ApiKeyHeader requires the 'header_name' field")
    if auth_type == "ApiKeyQuery" and not auth.query_parameter:
        _fail(f"{where}: ApiKeyQuery requires the 'query_parameter' field")
    return auth


def _parse_requester(data: dict, where: str) -> Requester:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object")
    _unknown_keys(data, REQUESTER_KEYS, where)

    req_type = _str(data, "type", where, "HttpRequester")
    if req_type != "HttpRequester":
        _fail(f"{where}.type: expected HttpRequester")

    url_base = _str(data, "url_base", where)
    if not url_base.startswith(("http://", "https://")):
        _fail(f"{where}.url_base: expected an http(s) URL")

    method = _str(data, "http_method", where, "GET").upper()
    if method != "GET":
        _fail(f"{where}.http_method: expected GET")

    requester = Requester(
        url_base=url_base,
        path=_str(data, "path", where, ""),
        http_method=method,
        request_parameters=_scalar_map(data, "request_parameters", where),
        headers=_scalar_map(data, "headers", where),
    )
    if "authenticator" in data and data["authenticator"] is not None:
        requester.authenticator = _parse_authenticator(
            data["authenticator"], f"{where}.authenticator"
        )
    return requester


def _parse_paginator(data: dict, where: str) -> Paginator:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object")
    _unknown_keys(data, PAGINATOR_KEYS, where)

    pag_type = _str(data, "type", where, "NoPagination")
    if pag_type not in PAGINATOR_TYPES:
        _fail(f"{where}.type: unknown type {pag_type!r}; allowed: {sorted(PAGINATOR_TYPES)}")

    paginator = Paginator(
        type=pag_type,
        page_size=_int(data, "page_size", where, 100),
        offset_parameter=_str(data, "offset_parameter", where, "offset"),
        limit_parameter=(
            _str(data, "limit_parameter", where, None)
            if "limit_parameter" in data
            else "limit"
        ),
        page_parameter=_str(data, "page_parameter", where, "page"),
        start_page=_int(data, "start_page", where, 1),
        next_url_path=(
            _str(data, "next_url_path", where, None)
            if "next_url_path" in data
            else None
        ),
        max_pages=_int(data, "max_pages", where, 10),
        stop_when_empty=(
            _str(data, "stop_when_empty", where, None)
            if "stop_when_empty" in data
            else None
        ),
    )

    if paginator.page_size <= 0:
        _fail(f"{where}.page_size: must be greater than zero")
    if paginator.max_pages <= 0:
        _fail(f"{where}.max_pages: must be greater than zero")
    if paginator.start_page < 0:
        _fail(f"{where}.start_page: must not be negative")
    if pag_type == "OffsetIncrement" and not paginator.offset_parameter:
        _fail(f"{where}.offset_parameter: required for OffsetIncrement")
    if pag_type == "PageIncrement" and not paginator.page_parameter:
        _fail(f"{where}.page_parameter: required for PageIncrement")
    if pag_type == "NextUrl" and not paginator.next_url_path:
        _fail(f"{where}.next_url_path: required for NextUrl")
    return paginator


def _parse_error_handler(data: dict, where: str) -> ErrorHandler:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object")
    _unknown_keys(data, ERROR_HANDLER_KEYS, where)

    handler_type = _str(data, "type", where, "DefaultErrorHandler")
    if handler_type not in ERROR_HANDLER_TYPES:
        _fail(f"{where}.type: expected DefaultErrorHandler")

    statuses = data.get("retry_statuses", [429, 500, 502, 503, 504])
    if not isinstance(statuses, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in statuses
    ):
        _fail(f"{where}.retry_statuses: expected a list of integers")

    handler = ErrorHandler(
        type=handler_type,
        max_retries=_int(data, "max_retries", where, 3),
        backoff_seconds=_float(data, "backoff_seconds", where, 2.0),
        retry_statuses=list(statuses),
        timeout_seconds=_float(data, "timeout_seconds", where, 30.0),
    )
    if handler.max_retries < 0:
        _fail(f"{where}.max_retries: must not be negative")
    if handler.backoff_seconds < 0:
        _fail(f"{where}.backoff_seconds: must not be negative")
    if handler.timeout_seconds <= 0:
        _fail(f"{where}.timeout_seconds: must be greater than zero")
    return handler


def _parse_output(data: dict, where: str) -> Output:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object")
    _unknown_keys(data, OUTPUT_KEYS, where)
    extension = _str(data, "extension", where, "json").lstrip(".")
    if not re.fullmatch(r"[A-Za-z0-9]+", extension):
        _fail(f"{where}.extension: expected a plain extension such as json")
    return Output(extension=extension)


def _parse_download(data: dict, where: str) -> Download:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object")
    _unknown_keys(data, DOWNLOAD_KEYS, where)

    url = _str(data, "url", where)
    if not url.startswith(("http://", "https://")):
        _fail(f"{where}.url: expected an http(s) URL")

    filename = _str(data, "filename", where)
    if Path(filename).name != filename or filename in {"", ".", ".."}:
        _fail(f"{where}.filename: expected a plain file name without directories")

    sha256 = _str(data, "sha256", where, None) if "sha256" in data else None
    if sha256 is not None and not SHA256_RE.match(sha256):
        _fail(f"{where}.sha256: expected a 64-character hex string")
    return Download(url=url, filename=filename, sha256=sha256)


def parse_manifest(data: dict, where: str = "manifest") -> Manifest:
    if not isinstance(data, dict):
        _fail(f"{where}: expected an object at the manifest root")
    _unknown_keys(data, TOP_KEYS, where)
    _check_secrets(data, where)

    manifest_id = _str(data, "id", where)
    if not ID_RE.match(manifest_id):
        _fail(f"{where}.id: allowed characters are letters, digits, dot, dash and underscore")

    source_type = _str(data, "type", where)
    if source_type not in SOURCE_TYPES:
        _fail(f"{where}.type: unknown type {source_type!r}; allowed: {sorted(SOURCE_TYPES)}")

    version_raw = data.get("version", "0.1")
    if isinstance(version_raw, (int, float)) and not isinstance(version_raw, bool):
        version = str(version_raw)
    elif isinstance(version_raw, str):
        version = version_raw
    else:
        _fail(f"{where}.version: expected a string or a number")

    manifest = Manifest(
        id=manifest_id,
        type=source_type,
        version=version,
        description=_str(data, "description", where, ""),
        docs_url=_str(data, "docs_url", where, ""),
        license=_str(data, "license", where, ""),
    )

    if source_type == "DeclarativeSource":
        manifest.requester = _parse_requester(
            _require(data, "requester", where), f"{where}.requester"
        )
        if "paginator" in data and data["paginator"] is not None:
            manifest.paginator = _parse_paginator(
                data["paginator"], f"{where}.paginator"
            )
        if "error_handler" in data and data["error_handler"] is not None:
            manifest.error_handler = _parse_error_handler(
                data["error_handler"], f"{where}.error_handler"
            )
        if "output" in data and data["output"] is not None:
            manifest.output = _parse_output(data["output"], f"{where}.output")
    else:
        manifest.download = _parse_download(
            _require(data, "download", where), f"{where}.download"
        )

    return manifest


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    if not path.exists():
        _fail(f"manifest not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _fail(f"{path}: invalid YAML: {exc}")
    if raw is None:
        _fail(f"{path}: empty manifest")
    return parse_manifest(raw, where=str(path))


def find_manifests(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    result = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if path.name.startswith("_") or ".mapping." in path.name:
            continue
        result.append(path)
    return result


def resolve_source(spec: str, directory: str | Path) -> Manifest:
    candidate = Path(spec)
    if candidate.suffix.lower() in {".yaml", ".yml"} and candidate.exists():
        return load_manifest(candidate)

    directory = Path(directory)
    for suffix in (".yaml", ".yml"):
        path = directory / f"{spec}{suffix}"
        if path.exists():
            return load_manifest(path)
    _fail(f"unknown source {spec!r}: no file in {directory}")
