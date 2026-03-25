"""
Custom exceptions for CyborgDB client.

This module defines custom exception classes for better error handling
and more descriptive error messages in the CyborgDB Python SDK.
"""

from typing import Optional


class CyborgDBError(Exception):
    """Base exception class for all CyborgDB client errors."""


class CyborgDBConnectionError(CyborgDBError):
    """Raised when connection to CyborgDB service fails."""

    def __init__(self, message: str, base_url: Optional[str] = None):
        super().__init__(message)
        self.base_url = base_url


class CyborgDBValidationError(CyborgDBError, ValueError):
    """Raised when input validation fails."""


class CyborgDBAuthenticationError(CyborgDBError):
    """Raised when authentication fails (invalid API key or unauthorized)."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class CyborgDBNotFoundError(CyborgDBError):
    """Raised when a requested resource (e.g., index) is not found."""

    def __init__(self, message: str, resource_name: Optional[str] = None):
        super().__init__(message)
        self.resource_name = resource_name


class CyborgDBIndexError(CyborgDBError):
    """Raised when an index-related operation fails."""

    def __init__(self, message: str, index_name: Optional[str] = None):
        super().__init__(message)
        self.index_name = index_name


class CyborgDBInvalidKeyError(CyborgDBValidationError):
    """Raised when an invalid encryption key is provided."""

    def __init__(self, message: str = "index_key must be a 32-byte bytes object"):
        super().__init__(message)


class CyborgDBInvalidURLError(CyborgDBValidationError):
    """Raised when an invalid base URL is provided."""

    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(message)
        self.url = url
