"""Public exception hierarchy for the CyborgDB Python SDK."""

__all__ = [
    "CyborgError",
    "AuthenticationError",
    "ServiceUnavailableError",
    "ConnectionTimeoutError",
]


class CyborgError(Exception):
    """Base class for all CyborgDB SDK exceptions."""


class AuthenticationError(CyborgError):
    """Raised on HTTP 401/403; check your API key and do not retry."""


class ServiceUnavailableError(CyborgError):
    """Raised on HTTP 5xx or connection failure (MaxRetryError, NewConnectionError); retry with exponential backoff."""


class ConnectionTimeoutError(CyborgError):
    """Raised on urllib3 TimeoutError; retry with a shorter timeout or backoff."""
