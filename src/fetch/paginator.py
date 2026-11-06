from __future__ import annotations

import json
from typing import Any
from urllib.parse import urljoin

from .errors import PaginationError
from .manifest import Paginator, Requester
from .requester import Response, build_url


def json_path(data: Any, path: str) -> Any:
    cleaned = path.strip()
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    cleaned = cleaned.lstrip(".")
    if not cleaned:
        return data
    current = data
    for part in cleaned.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, tuple, dict)):
        return len(value) == 0
    return False


class PageFetcher:
    def __init__(self, config: Paginator, requester: Requester):
        self.config = config
        self.requester = requester
        self.page = 0

    def first_url(self) -> str:
        self.page = 1
        return self._url_for_page(self.page)

    def next_url(self, response: Response) -> str | None:
        if self.config.type == "NoPagination":
            return None
        if self.page >= self.config.max_pages:
            return None
        if self._must_stop(response):
            return None
        self.page += 1
        return self._url_for_page(self.page)

    def _url_for_page(self, page: int) -> str:
        kind = self.config.type

        if kind == "NoPagination":
            return build_url(self.requester)

        if kind == "OffsetIncrement":
            offset = (page - 1) * self.config.page_size
            parameters = {self.config.offset_parameter: str(offset)}
            if self.config.limit_parameter:
                parameters[self.config.limit_parameter] = str(self.config.page_size)
            return build_url(self.requester, parameters)

        if kind == "PageIncrement":
            number = self.config.start_page + page - 1
            return build_url(self.requester, {self.config.page_parameter: str(number)})

        if kind == "NextUrl":
            return build_url(self.requester)

        raise PaginationError(f"unknown paginator type {kind!r}")

    def _must_stop(self, response: Response) -> bool:
        if self.config.type != "NextUrl" and not response.content.strip():
            return True
        if not self.config.stop_when_empty:
            return False
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise PaginationError(
                f"failed to parse JSON for stop_when_empty: {exc}"
            ) from exc
        return _is_empty(json_path(data, self.config.stop_when_empty))


class NextUrlFetcher(PageFetcher):
    def next_url(self, response: Response) -> str | None:
        if self.config.type == "NoPagination":
            return None
        if self.page >= self.config.max_pages:
            return None
        if self._must_stop(response):
            return None
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise PaginationError(
                f"failed to parse JSON for next_url_path: {exc}"
            ) from exc
        next_value = json_path(data, self.config.next_url_path or "")
        if _is_empty(next_value) or not isinstance(next_value, str):
            return None
        self.page += 1
        return urljoin(response.url, next_value)


def make_paginator(config: Paginator, requester: Requester) -> PageFetcher:
    if config.type == "NextUrl":
        return NextUrlFetcher(config, requester)
    return PageFetcher(config, requester)
