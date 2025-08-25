"""
CyborgDB REST Client

This module provides a Python client for interacting with the CyborgDB REST API.
"""

from typing import Dict, List, Optional, Union
import secrets
import logging
import binascii
from pydantic import ValidationError

# Import from the OpenAPI generated models
from cyborgdb.openapi_client.models import (
    IndexIVFModel as _OpenAPIIndexIVFModel,
    IndexIVFPQModel as _OpenAPIIndexIVFPQModel,
    IndexIVFFlatModel as _OpenAPIIndexIVFFlatModel,
    IndexConfig as _OpenAPIIndexConfig,
    CreateIndexRequest as _OpenAPICreateIndexRequest,
)

# Import the OpenAPI generated client
try:
    from cyborgdb.openapi_client.api_client import ApiClient, Configuration
    from cyborgdb.openapi_client.api.default_api import DefaultApi

    # Note: Model imports removed as they're accessed through the API client
    from cyborgdb.openapi_client.models.index_ivf_flat_model import IndexIVFFlatModel
    from cyborgdb.openapi_client.models.index_ivf_model import IndexIVFModel
    from cyborgdb.openapi_client.models.index_ivfpq_model import IndexIVFPQModel
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
    "IndexConfig",
    "IndexIVF",
    "IndexIVFPQ",
    "IndexIVFFlat",
    "generate_key",
]


def generate_key() -> bytes:
    """
    Generate a secure 32-byte key for use with CyborgDB indexes.

    Returns:
        bytes: A cryptographically secure 32-byte key.
    """
    return secrets.token_bytes(32)


# Re-export with your preferred names
IndexIVF = _OpenAPIIndexIVFModel
IndexIVFPQ = _OpenAPIIndexIVFPQModel
IndexIVFFlat = _OpenAPIIndexIVFFlatModel
IndexConfig = _OpenAPIIndexConfig
CreateIndexRequest = _OpenAPICreateIndexRequest


class Client:
    """
    Client for interacting with CyborgDB via REST API.

    This class provides methods for creating, loading, and managing encrypted indexes.
    """

    def __init__(self, api_url, api_key, verify_ssl=None):
        # If api_url is http, disable SSL verification
        if api_url.startswith("http://"):
            verify_ssl = False

        # Set up the OpenAPI client configuration
        self.config = Configuration()
        self.config.host = api_url

        # Configure SSL verification
        if verify_ssl is None:
            # Auto-detect: disable SSL verification for localhost/127.0.0.1 (development)
            if "localhost" in api_url or "127.0.0.1" in api_url:
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
        index_key: bytes,
        index_config: Union[IndexIVFModel, IndexIVFPQModel, IndexIVFFlatModel],
        embedding_model: Optional[str] = None,
    ) -> EncryptedIndex:
        """
        Create and return a new encrypted index based on the provided configuration.
        """
        # Validate index_key
        if not isinstance(index_key, bytes) or len(index_key) != 32:
            raise ValueError("index_key must be a 32-byte bytes object")

        try:
            # Convert binary key to hex string
            key_hex = binascii.hexlify(index_key).decode("ascii")

            # Create an IndexConfig instance with the appropriate model
            index_config_obj = IndexConfig(index_config)

            # Create the complete request object
            request = CreateIndexRequest(
                index_name=index_name,
                index_key=key_hex,
                index_config=index_config_obj,
                embedding_model=embedding_model,
            )

            # Call the generated API method
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

    def load_index(self, index_name: str, index_key: bytes) -> EncryptedIndex:
        """
        Load an existing encrypted index by name and key.
        """

        # Validate index_key
        if not isinstance(index_key, bytes) or len(index_key) != 32:
            raise ValueError("index_key must be a 32-byte bytes object")

        try:
            # Convert binary key to hex string

            index = EncryptedIndex(
                index_name=index_name,
                index_key=index_key,
                api=self.api,
                api_client=self.api_client,
            )

            # Attempt to access index.index_type to validate existence.
            # This will raise an exception if the index does not exist.
            _ = index.index_type  # Access for validation; value not used.

            # Create the EncryptedIndex instance
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
