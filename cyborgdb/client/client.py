"""
CyborgDB REST Client

This module provides a Python client for interacting with the CyborgDB REST API.
"""

from pathlib import Path
from typing import Dict, List, Optional
import secrets
import logging
import binascii
from pydantic import ValidationError

# Import from the OpenAPI generated models
from cyborgdb.openapi_client.models import (
    CreateIndexRequest as _OpenAPICreateIndexRequest,
)

# Import the OpenAPI generated client
try:
    from cyborgdb.openapi_client.api_client import ApiClient, Configuration
    from cyborgdb.openapi_client.api.default_api import DefaultApi

    from cyborgdb.openapi_client.exceptions import ApiException
except ImportError:
    raise ImportError(
        "Failed to import openapi_client. Make sure the OpenAPI client library is properly installed."
    )

from cyborgdb.client.encrypted_index import EncryptedIndex

logger = logging.getLogger(__name__)

__all__ = [
    "Client",
    "EncryptedIndex",
]

CreateIndexRequest = _OpenAPICreateIndexRequest


def _validate_index_key(index_key: bytes) -> None:
    """Raise ValueError unless ``index_key`` is a 32-byte ``bytes`` object."""
    if not isinstance(index_key, bytes) or len(index_key) != 32:
        raise ValueError("index_key must be a 32-byte bytes object")


class Client:
    """
    Client for interacting with CyborgDB via REST API.

    This class provides methods for creating, loading, and managing encrypted indexes.
    """

    def __init__(self, base_url, api_key, verify_ssl=None):
        # If base_url is http, disable SSL verification
        if base_url.startswith("http://"):
            verify_ssl = False

        # Set up the OpenAPI client configuration
        self.config = Configuration()
        self.config.host = base_url

        # Configure SSL verification
        if verify_ssl is None:
            # Auto-detect: disable SSL verification for localhost/127.0.0.1 (development)
            if "localhost" in base_url or "127.0.0.1" in base_url:
                self.config.verify_ssl = False
                # Disable SSL warnings for localhost
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                logger.info(
                    "SSL verification disabled for localhost (development mode)"
                )
            else:
                self.config.verify_ssl = True
        else:
            self.config.verify_ssl = verify_ssl
            if not verify_ssl:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                logger.warning(
                    "SSL verification is disabled. Not recommended for production."
                )

        # Add authentication if provided
        if api_key:
            self.config.api_key = {"X-API-Key": api_key}

        # Create the API client
        try:
            self.api_client = ApiClient(self.config)
            self.api = DefaultApi(self.api_client)

            # If API key was provided, also set it directly in default headers
            if api_key:
                self.api_client.default_headers["X-API-Key"] = api_key

        except Exception as e:
            error_msg = f"Failed to initialize client: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def generate_key(save: bool = False) -> bytes:
        """
        Generate a secure 32-byte key for use with CyborgDB indexes.

        Args:
            save (bool): If True, save the key to a file in the user's home directory
                         for reuse. Not recommended for production use.
        Returns:
            bytes: A cryptographically secure 32-byte key.
        """
        if not save:
            return secrets.token_bytes(32)

        key_path = Path.home() / ".cyborgdb" / "index_key"
        key_path.parent.mkdir(parents=True, exist_ok=True)

        if key_path.exists():
            if key_path.stat().st_size == 32:
                logger.warning(
                    f"Loading existing index key from '{key_path}'.\nSaving keys is not recommended for production use."
                )
                return key_path.read_bytes()

        key = secrets.token_bytes(32)
        key_path.write_bytes(key)
        logger.warning(
            f"Generated new index key and saved to '{key_path}'.\nSaving keys is not recommended for production use."
        )
        return key

    def list_indexes(self) -> List[str]:
        """
        Get a list of all encrypted index names accessible via the client.

        Returns:
            A list of index names.

        Raises:
            ValueError: If the list of indexes could not be retrieved.
        """
        try:
            response = self.api.list_indexes_v1_indexes_list_get()
            return response.indexes
        except ApiException as e:
            error_msg = f"Failed to list indexes: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def create_index(
        self,
        index_name: str,
        index_key: Optional[bytes] = None,
        kms_name: Optional[str] = None,
        dimension: Optional[int] = None,
        embedding_model: Optional[str] = None,
        metric: Optional[str] = None,
        storage_precision: Optional[str] = None,
    ) -> EncryptedIndex:
        """
        Create and return a new encrypted DiskIVF index.

        At least one of ``index_key`` or ``kms_name`` must be provided.

        - ``index_key`` only — SDK supplies the 32-byte key; the service treats
          it as the DEK and does no KMS round-trips.
        - ``kms_name`` only — the service generates a fresh DEK and wraps it
          under the named ``kms.registry`` entry; the SDK never sees the DEK.
        - ``index_key`` + ``kms_name`` — only valid when ``kms_name`` references
          a ``provider: none`` registry entry, in which case ``index_key`` is
          the wrapping KEK. Passing both against a real-KMS slot
          (``provider: aws-kms`` / ``aws``) is rejected by the service with
          a 400.
        """
        if index_key is None and kms_name is None:
            raise ValueError(
                "create_index requires index_key, kms_name, or both"
            )

        if index_key is not None:
            _validate_index_key(index_key)

        try:
            key_hex = (
                binascii.hexlify(index_key).decode("ascii")
                if index_key is not None
                else None
            )

            request = CreateIndexRequest(
                index_name=index_name,
                index_key=key_hex,
                kms_name=kms_name,
                dimension=dimension,
                embedding_model=embedding_model,
                metric=metric,
                storage_precision=storage_precision,
            )

            self.api.create_index_v1_indexes_create_post(
                create_index_request=request,
                _headers={
                    "X-API-Key": self.config.api_key["X-API-Key"],
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            return EncryptedIndex(
                index_name=index_name,
                index_key=index_key,
                api=self.api,
                api_client=self.api_client,
            )

        except ApiException as e:
            error_msg = f"Failed to create index: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except ValidationError as ve:
            error_msg = f"Validation error while creating index: {ve}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def load_index(
        self,
        index_name: str,
        index_key: Optional[bytes] = None,
    ) -> EncryptedIndex:
        """
        Load an existing encrypted index by name.

        ``index_key`` is required for ``provider: none`` indexes (the SDK owns
        the KEK). For KMS-backed indexes the service resolves the DEK via the
        stored ``KMSBlob``, so ``index_key`` can be omitted.
        """
        if index_key is not None:
            _validate_index_key(index_key)

        try:
            index = EncryptedIndex(
                index_name=index_name,
                index_key=index_key,
                api=self.api,
                api_client=self.api_client,
            )

            _ = index.index_type  # Access for validation; value not used.

            return index

        except ApiException as e:
            error_msg = f"Failed to load index '{index_name}': {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except ValidationError as ve:
            error_msg = f"Validation error while loading index '{index_name}': {ve}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_health(self) -> Dict[str, str]:
        """
        Get the health status of the CyborgDB instance.

        Returns:
            A dictionary containing health status information.

        Raises:
            ValueError: If the health status could not be retrieved.
        """
        try:
            return self.api.health_check_v1_health_get()
        except ApiException as e:
            error_msg = f"Failed to get health status: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
