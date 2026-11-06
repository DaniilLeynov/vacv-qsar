import json
from urllib.parse import parse_qs, urlsplit

import pytest

from src.fetch.errors import PaginationError
from src.fetch.manifest import Paginator, Requester
from src.fetch.paginator import make_paginator
from src.fetch.requester import Response

SOURCE_URL = "https://example.org/api/items.json?limit=20"


def requester():
    return Requester(
        url_base="https://example.org/api/",
        path="items.json",
        request_parameters={"limit": "20"},
    )


def response(payload=None, url=SOURCE_URL, content=None):
    body = content if content is not None else json.dumps(payload).encode("utf-8")
    return Response(
        url=url,
        status=200,
        headers={"Content-Type": "application/json"},
        content=body,
    )


def query_of(url):
    return parse_qs(urlsplit(url).query, keep_blank_values=True)


def test_no_pagination_returns_single_url():
    fetcher = make_paginator(Paginator(type="NoPagination"), requester())
    first = fetcher.first_url()
    assert query_of(first)["limit"] == ["20"]
    assert fetcher.next_url(response({"items": [1, 2]})) is None


def test_offset_increment_sequence():
    config = Paginator(
        type="OffsetIncrement",
        page_size=100,
        offset_parameter="offset",
        limit_parameter="limit",
        max_pages=5,
        stop_when_empty="$.items",
    )
    fetcher = make_paginator(config, requester())

    first = fetcher.first_url()
    assert query_of(first)["offset"] == ["0"]
    assert query_of(first)["limit"] == ["100"]

    second = fetcher.next_url(response({"items": [1, 2]}))
    assert query_of(second)["offset"] == ["100"]

    third = fetcher.next_url(response({"items": [3]}))
    assert query_of(third)["offset"] == ["200"]

    assert fetcher.next_url(response({"items": []})) is None


def test_offset_increment_respects_max_pages():
    config = Paginator(
        type="OffsetIncrement", page_size=10, max_pages=2, stop_when_empty="$.items"
    )
    fetcher = make_paginator(config, requester())
    fetcher.first_url()
    second = fetcher.next_url(response({"items": [1]}))
    assert query_of(second)["offset"] == ["10"]
    assert fetcher.next_url(response({"items": [2]})) is None


def test_offset_increment_stops_on_empty_body():
    config = Paginator(type="OffsetIncrement", page_size=10, max_pages=5)
    fetcher = make_paginator(config, requester())
    fetcher.first_url()
    assert fetcher.next_url(response(content=b"")) is None


def test_page_increment_sequence():
    config = Paginator(
        type="PageIncrement", page_parameter="page", start_page=1, max_pages=5,
        stop_when_empty="$.items",
    )
    fetcher = make_paginator(config, requester())
    assert query_of(fetcher.first_url())["page"] == ["1"]
    assert query_of(fetcher.next_url(response({"items": [1]})))["page"] == ["2"]
    assert query_of(fetcher.next_url(response({"items": [2]})))["page"] == ["3"]


def test_next_url_resolves_relative_link():
    config = Paginator(
        type="NextUrl", next_url_path="$.page_meta.next", max_pages=5
    )
    fetcher = make_paginator(config, requester())
    assert fetcher.first_url() == SOURCE_URL

    payload = {"page_meta": {"next": "/api/items.json?offset=20"}}
    next_url = fetcher.next_url(response(payload))
    assert next_url == "https://example.org/api/items.json?offset=20"


def test_next_url_stops_when_null():
    config = Paginator(type="NextUrl", next_url_path="$.page_meta.next", max_pages=5)
    fetcher = make_paginator(config, requester())
    fetcher.first_url()
    assert fetcher.next_url(response({"page_meta": {"next": None}})) is None


def test_next_url_stops_when_absent():
    config = Paginator(type="NextUrl", next_url_path="$.page_meta.next", max_pages=5)
    fetcher = make_paginator(config, requester())
    fetcher.first_url()
    assert fetcher.next_url(response({"other": 1})) is None


def test_next_url_rejects_non_json():
    config = Paginator(type="NextUrl", next_url_path="$.next", max_pages=5)
    fetcher = make_paginator(config, requester())
    fetcher.first_url()
    with pytest.raises(PaginationError, match="JSON"):
        fetcher.next_url(response(content=b"<html>"))


def test_stop_when_empty_rejects_non_json():
    config = Paginator(
        type="OffsetIncrement", page_size=10, max_pages=5, stop_when_empty="$.items"
    )
    fetcher = make_paginator(config, requester())
    fetcher.first_url()
    with pytest.raises(PaginationError, match="JSON"):
        fetcher.next_url(response(content=b"not json"))
