"""
CyborgDB REST Client

This module provides a Python client for interacting with the CyborgDB REST API.
"""

from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import os
import json
import secrets
import numpy as np
import logging
from enum import Enum
from pathlib import Path
import binascii
from pydantic import ValidationError

# Import the OpenAPI generated client
try:
    from cyborgdb.openapi_client.api_client import ApiClient, Configuration
    from cyborgdb.openapi_client.api.default_api import DefaultApi
    from cyborgdb.openapi_client.models.index_config import IndexConfig as ApiIndexConfig
    from cyborgdb.openapi_client.models.create_index_request import CreateIndexRequest as IndexCreateRequest
    from cyborgdb.openapi_client.models.query_request import QueryRequest
    from cyborgdb.openapi_client.models.batch_query_request import BatchQueryRequest
    from cyborgdb.openapi_client.models.upsert_request import UpsertRequest
    from cyborgdb.openapi_client.models.delete_request import DeleteRequest
    from cyborgdb.openapi_client.models.train_request import TrainRequest
    from cyborgdb.openapi_client.models.vector_item import VectorItem as Item
    from cyborgdb.openapi_client.models.index_ivf_flat_model import IndexIVFFlatModel
    from cyborgdb.openapi_client.models.index_ivf_model import IndexIVFModel
    from cyborgdb.openapi_client.models.index_ivfpq_model import IndexIVFPQModel
    from cyborgdb.openapi_client.exceptions import ApiException
    from cyborgdb.openapi_client.models.query_result_item import QueryResultItem
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
    "generate_key"
]


def generate_key() -> bytes:
    """
    Generate a secure 32-byte key for use with CyborgDB indexes.
    
    Returns:
        bytes: A cryptographically secure 32-byte key.
    """
    return secrets.token_bytes(32)

# Import from the OpenAPI generated models
from cyborgdb.openapi_client.models import (
    IndexIVFModel as _OpenAPIIndexIVFModel,
    IndexIVFPQModel as _OpenAPIIndexIVFPQModel,
    IndexIVFFlatModel as _OpenAPIIndexIVFFlatModel,
    IndexConfig as _OpenAPIIndexConfig,
    CreateIndexRequest as _OpenAPICreateIndexRequest
)

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
    
    def __init__(self, api_url, api_key):
        # Set up the OpenAPI client configuration
        self.config = Configuration()
        self.config.host = api_url
        
        # Add authentication if provided
        if api_key:
            self.config.api_key = {'X-API-Key': api_key}
        
        # Create the API client
        try:
            self.api_client = ApiClient(self.config)
            self.api = DefaultApi(self.api_client)
            
            # If API key was provided, also set it directly in default headers
            if api_key:
                self.api_client.default_headers['X-API-Key'] = api_key
            
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
        max_cache_size: int = 0
    ) -> EncryptedIndex:
        """
        Create and return a new encrypted index based on the provided configuration.
        """
        # Validate index_key
        if not isinstance(index_key, bytes) or len(index_key) != 32:
            raise ValueError("index_key must be a 32-byte bytes object")

        try:
            # Convert binary key to hex string
            key_hex = binascii.hexlify(index_key).decode('ascii')
                
            # Create an IndexConfig instance with the appropriate model
            index_config_obj = IndexConfig(index_config)

            # Create the complete request object
            request = CreateIndexRequest(
                index_name=index_name,
                index_key=key_hex,
                index_config=index_config_obj,
                embedding_model=embedding_model
            )

            # Call the generated API method
            response = self.api.create_index_v1_indexes_create_post(
                create_index_request=request,
                _headers={
                    'X-API-Key': self.config.api_key['X-API-Key'],
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            return EncryptedIndex(
                index_name=index_name,
                index_key=index_key,
                api=self.api,
                api_client=self.api_client,
                max_cache_size=max_cache_size
            )

        except ApiException as e:
            error_msg = f"Failed to create index: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except ValidationError as ve:
            error_msg = f"Validation error while creating index: {ve}"
            logger.error(error_msg)
            raise ValueError(error_msg)