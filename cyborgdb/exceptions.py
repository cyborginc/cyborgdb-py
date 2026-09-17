"""Public exception hierarchy for the CyborgDB Python SDK."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

__all__ = [
    "CyborgDBError",
    "ValidationError",
    "AuthenticationError",
    "NotFoundError",
    "ConflictError",
    "RateLimitError",
    "ServiceError",
    "TransportError",
]

logger = logging.getLogger(__name__)


class CyborgDBError(ValueError):
    """Base class for every error this SDK raises.

    Catch this to handle any CyborgDB failure; catch a subclass to handle one
    kind. Every instance carries the context below, and the originating
    exception stays reachable through ``__cause__``.

    Attributes:
        status_code: HTTP status, or ``None`` for transport failures and for
            arguments rejected before the request was sent.
        request_id: Service correlation id, or ``None`` when the service did
            not supply one.
        detail: The service's own message, or ``None`` when there was no
            response.
        retry_after: Seconds from the ``Retry-After`` header, or ``None``.
        retryable: Whether backing off and retrying can succeed. Fixed per
            type — retry loops branch on this rather than re-deriving a status
            table.
    """

    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        detail: Optional[str] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.detail = detail
        self.retry_after = retry_after


class ValidationError(CyborgDBError):
    """HTTP 400 and 422, and arguments rejected before the request was sent.

    ``status_code`` is ``None`` when the SDK caught the problem itself.
    """

    retryable = False


class AuthenticationError(CyborgDBError):
    """HTTP 401 and 403. Check the API key and its permissions.

    Inspect ``status_code`` to tell the two apart.
    """

    retryable = False


class NotFoundError(CyborgDBError):
    """HTTP 404 — the collection or item does not exist."""

    retryable = False


class ConflictError(CyborgDBError):
    """HTTP 409 — a state conflict, such as training while training is running.

    Poll for the terminal state; a backoff loop must not retry a 409.
    """

    retryable = False


class RateLimitError(CyborgDBError):
    """HTTP 429. Honor ``retry_after`` when it is set.

    The service does not rate-limit yet (cyborgdb-core#2386); this type exists
    so callers can write the handler once.
    """

    retryable = True


class ServiceError(CyborgDBError):
    """Any 5xx.

    Not "ServiceUnavailable": a 500 is a server bug, not unavailability.
    """

    retryable = True


class TransportError(CyborgDBError):
    """No HTTP response reached the client.

    DNS failure, connection refused, TLS failure, or timeout. ``status_code``
    is ``None``. Timeouts land here too: a timeout and a refused connection
    have the same caller action, and the difference is diagnostic — read the
    message.
    """

    retryable = True


# Status -> exception class. Mirrors the `mapping` block of the taxonomy file.
_STATUS_MAP = {
    400: ValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    409: ConflictError,
    422: ValidationError,
    429: RateLimitError,
}


def _header(headers: Any, name: str) -> Optional[str]:
    """Read one header, tolerating the several shapes urllib3 hands back."""
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value)
    try:
        lowered = name.lower()
        for key, value in dict(headers).items():
            if str(key).lower() == lowered:
                return str(value)
    except (TypeError, ValueError):
        pass
    return None


def _detail(body: Any) -> Optional[str]:
    """Pull the service's own message out of a response body."""
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            return body or None
    if isinstance(body, dict) and "detail" in body:
        value = body["detail"]
        return value if isinstance(value, str) else json.dumps(value)
    return None


def _retry_after(headers: Any) -> Optional[float]:
    """Parse ``Retry-After``, seconds form only.

    The HTTP-date form is not used by the service and is reported as absent.
    """
    raw = _header(headers, "Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def translate_api_error(exc: Exception, context: str) -> Exception:
    """Map a generated-client or urllib3 exception onto the public taxonomy.

    This is the single translation point for the SDK: the client modules call
    it instead of each carrying its own cascade, so the mapping cannot drift
    between methods.

    Returns the exception to raise — it never raises on its own. Statuses the
    taxonomy does not name keep the previous ``ValueError``, so those paths are
    unchanged for callers.

    Args:
        exc: The originating exception.
        context: Short description of the operation, used in the message.
    """
    # Imported here: cyborgdb.exceptions must stay importable even if the
    # generated client is absent (it is regenerated, not vendored by hand).
    from cyborgdb.openapi_client.exceptions import ApiException
    import urllib3.exceptions

    if isinstance(exc, urllib3.exceptions.HTTPError):
        logger.error("%s: %s", context, exc)
        return TransportError(f"{context}: {exc}", detail=str(exc))

    if not isinstance(exc, ApiException):
        return exc

    status = getattr(exc, "status", None)
    headers = getattr(exc, "headers", None)
    detail = _detail(getattr(exc, "body", None))
    kwargs = {
        "status_code": status,
        "request_id": _header(headers, "X-Request-Id"),
        "detail": detail,
        "retry_after": _retry_after(headers),
    }
    message = f"{context}: {exc}"

    error_cls = _STATUS_MAP.get(status)
    if error_cls is None and isinstance(status, int) and status >= 500:
        error_cls = ServiceError
    if error_cls is None:
        # Not named by the taxonomy — preserve the existing behavior.
        logger.error(message)
        return ValueError(message)

    logger.error(message)
    return error_cls(message, **kwargs)
