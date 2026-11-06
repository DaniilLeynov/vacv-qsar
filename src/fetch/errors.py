class FetchError(Exception):
    """Base error of the transport layer."""


class ManifestError(FetchError):
    """Manifest is missing or failed validation."""


class AuthError(FetchError):
    """Authentication could not be built."""


class HttpError(FetchError):
    """HTTP request failed."""

    def __init__(self, status: int, url: str, message: str = ""):
        self.status = status
        self.url = url
        self.message = message
        detail = f"HTTP {status}: {url}"
        if message:
            detail += f" ({message})"
        super().__init__(detail)


class PaginationError(FetchError):
    """Paginator could not determine the next page."""
