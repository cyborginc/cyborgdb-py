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
from cyborgdb.openapi_client.models.create_index_request import CreateIndexRequest


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

class EncryptedIndex:
    """
    Provides access to an encrypted vector index via the REST API.
    
    This class handles operations on an encrypted vector index, including
    adding/updating vectors, searching, and managing index metadata.
    """
    
    def __init__(
        self, 
        index_name: str, 
        index_key: bytes, 
        api: DefaultApi,
        api_client: ApiClient,
        max_cache_size: int = 0
    ):
        """
        Initialize with API access to an index.
        
        Args:
            index_name: Name of the index
            index_key: Encryption key for the index
            api: API client instance
            api_client: The lower-level API client
            max_cache_size: Maximum cache size
        """
        self._index_name = index_name
        self._index_key = index_key
        self._api = api
        self._api_client = api_client
        self._max_cache_size = max_cache_size
        self._index_config = None
    
    @property
    def index_name(self) -> str:
        """Get the name of the index."""
        return self._index_name
    
    @property
    def index_type(self) -> str:
        """Get the type of the index."""
        # Retrieve index info if not already cached
        if not hasattr(self, '_index_type_cached'):
            try:
                response = self._api.get_index_info_v1_indexes_describe_post(
                    self._index_name, 
                    key=self._key_to_hex()
                )
                self._index_type_cached = response.index_type
            except ApiException as e:
                logger.error(f"Failed to retrieve index type: {e}")
                self._index_type_cached = "unknown"
                
        return self._index_type_cached
    
    @property
    def index_config(self) -> Dict[str, Any]:
        """Get the configuration of the index as a dictionary."""
        # Retrieve index info if not already cached
        if not self._index_config:
            try:
                response = self._api.get_index_info_v1_indexes_describe_post(
                    self._index_name, 
                    key=self._key_to_hex()
                )
                self._index_config = response.config
            except ApiException as e:
                logger.error(f"Failed to retrieve index config: {e}")
                self._index_config = {}
                
        return self._index_config
    
    def is_trained(self) -> bool:
        """
        Check if the index has been trained.
        
        Returns:
            bool: True if the index is trained, otherwise False.
        """
        try:
            response = self._api.get_index_status(
                self._index_name,
                key=self._key_to_hex()
            )
            return response.trained
        except ApiException as e:
            logger.error(f"Failed to get index training status: {e}")
            return False
    
    def delete_index(self) -> None:
        """
        Delete the current index and all its associated data.
        
        Warning:
            This action is irreversible.
            
        Raises:
            ValueError: If the index could not be deleted.
        """
        try:
            self._api.delete_index_v1_indexes_delete_post(
                self._index_name,
                key=self._key_to_hex()
            )
        except ApiException as e:
            error_msg = f"Failed to delete index: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def get(
        self, 
        ids: List[str], 
        include: List[str] = ["vector", "contents", "metadata"]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve and decrypt items associated with the specified IDs.
        
        Args:
            ids: IDs to retrieve.
            include: Item fields to return. Can include 'vector', 'contents', and 'metadata'.
                Default is ['vector', 'contents', 'metadata'].
                
        Returns:
            A list of dictionaries representing the items with the requested fields.
            IDs will always be included in the returned items.
            
        Raises:
            ValueError: If the items could not be retrieved or decrypted.
        """
        try:
            from cyborgdb.openapi_client.models import GetRequest, Request
            
            # Create the proper request objects
            get_request = GetRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                ids=ids,
                include=include
            )
            print("get_request: ", get_request)
            response = self._api.get_vectors_v1_vectors_get_post(
                get_request=get_request,
                _headers={
                    'X-API-Key': self._api_client.configuration.api_key['X-API-Key'],
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
            # Convert API response to our format
            items = []
            if hasattr(response, 'results'):
                for item in response.results:
                    item_dict = {"id": item.id}
                    
                    if "vector" in include and hasattr(item, "vector"):
                        item_dict["vector"] = item.vector
                        
                    if "contents" in include and hasattr(item, "contents"):
                        item_dict["contents"] = item.contents
                        
                    if "metadata" in include and hasattr(item, "metadata"):
                        # Convert metadata string to dict if needed
                        if isinstance(item.metadata, str):
                            try:
                                item_dict["metadata"] = json.loads(item.metadata)
                            except json.JSONDecodeError:
                                item_dict["metadata"] = {}
                        else:
                            item_dict["metadata"] = item.metadata
                            
                    items.append(item_dict)
                    
            return items
        except Exception as e:
            error_msg = f"Get operation failed: {str(e)}"
            logger.error(error_msg)
            raise
        except ApiException as e:
            error_msg = f"Failed to retrieve items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def train(
        self, 
        batch_size: int = 2048, 
        max_iters: int = 100, 
        tolerance: float = 1e-6,
        max_memory: int = 0
    ) -> None:
        """
        Build the index using the specified training configuration.
        
        Prior to calling this, all queries will be conducted using encrypted exhaustive search.
        After, they will be conducted using encrypted ANN search.
        
        Args:
            batch_size: Size of each batch for training. Default is 2048.
            max_iters: Maximum iterations for training. Default is 100.
            tolerance: Convergence tolerance for training. Default is 1e-6.
            max_memory: Maximum memory (MB) usage during training. Default is 0 (no limit).
            
        Note:
            There must be at least 2 * n_lists vector embeddings in the index prior to calling
            this function.
            
        Raises:
            ValueError: If there are not enough vector embeddings in the index for training,
                or if the index could not be trained.
        """
        try:
            request = TrainRequest(
                batch_size=batch_size,
                index_name=self._index_name,
                max_iters=max_iters,
                tolerance=tolerance,
                max_memory=max_memory,
                index_key=self._key_to_hex()
            )

            self._api.train_index_v1_indexes_train_post(train_request=request)
        except ApiException as e:
            error_msg = f"Failed to train index: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    def upsert(
        self, 
        arg1: Union[List[Dict[str, Any]], List[str], np.ndarray],
        arg2: Optional[np.ndarray] = None
    ) -> None:
        """
        Add or update vector embeddings in the index.
        
        If an item already exists at the specified ID, it will be overwritten.
        
        This method can be called in one of two ways:
        1. With a list of dictionaries, each containing 'id', 'vector', and optional 'contents'
        and 'metadata'.
        - If the index was created with an embedding model and 'vector' is not provided,
            'contents' will be automatically embedded.
        2. With separate IDs and vectors arrays.
        
        Args:
            arg1: Either a list of dictionaries or a list/array of IDs.
            arg2: If arg1 is a list of IDs, this should be an array of vector embeddings.
            
        Raises:
            ValueError: If vector dimensions are incompatible with the index configuration,
                if index was not created or loaded yet, if there is a mismatch between
                the number of vectors and IDs, or if the vectors could not be upserted.
            TypeError: If the arguments do not match expected types.
        """
        try:
            items = []
            
            # Case 1: arg1 is a list of dictionaries
            if arg2 is None:
                if not isinstance(arg1, list) or not all(isinstance(item, dict) for item in arg1):
                    raise TypeError("When arg2 is None, arg1 must be a list of dictionaries")
                    
                # Convert each dict to an Item
                for item_dict in arg1:
                    if "id" not in item_dict:
                        raise ValueError("Each item dictionary must contain an 'id' field")
                        
                    item = {
                        "id": item_dict["id"]
                    }
                    
                    if "vector" in item_dict:
                        item["vector"] = item_dict["vector"]
                        
                    if "contents" in item_dict:
                        item["contents"] = item_dict["contents"]
                        
                    if "metadata" in item_dict:
                        # Convert dict metadata to JSON string if needed
                        if isinstance(item_dict["metadata"], dict):
                            item["metadata"] = item_dict["metadata"]#json.dumps(item_dict["metadata"])
                        else:
                            item["metadata"] = item_dict["metadata"]
                            
                    items.append(item)
            
            # Case 2: arg1 is a list of IDs, arg2 is a matrix of vectors
            else:
                if not isinstance(arg1, list):
                    raise TypeError("arg1 must be a list of IDs")
                    
                # Convert numpy array to list if needed
                vectors = arg2
                if isinstance(vectors, np.ndarray):
                    vectors = vectors.tolist()
                    
                if len(arg1) != len(vectors):
                    raise ValueError("Number of IDs must match number of vectors")
                    
                # Create items from IDs and vectors
                for id_val, vector in zip(arg1, vectors):
                    items.append({
                        "id": str(id_val),
                        "vector": vector
                    })
            
            # Import the UpsertRequest model from the OpenAPI-generated code
            from cyborgdb.openapi_client.models import UpsertRequest
            
            # Create the upsert request with all required fields
            request = UpsertRequest(
                items=items,
                index_key=self._key_to_hex(),
                index_name=self._index_name
            )
            
            # Make the API call with the correct parameter
            self._api.upsert_vectors_v1_vectors_upsert_post(
                upsert_request=request,  # This is the only required parameter
                _headers={
                    'X-API-Key': self._api_client.configuration.api_key['X-API-Key'],
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )
            
        except ApiException as e:
            error_msg = f"Failed to upsert items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise

    def delete(self, ids: List[str]) -> None:
        """
        Delete the specified encrypted items stored in the index.
        
        Removes all associated fields (vector, contents, metadata) for the given IDs.
        
        Warning:
            This action is irreversible.
            
        Args:
            ids: IDs to delete.
            
        Raises:
            ValueError: If the items could not be deleted.
        """
        try:

            delete_request = DeleteRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                ids=ids)
            self._api.delete_vectors_v1_vectors_delete_post(
                delete_request=delete_request
            )
        except ApiException as e:
            error_msg = f"Failed to delete items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def query(
        self,
        query_vector: Optional[Union[List[float], np.ndarray]] = None,
        query_vectors: Optional[Union[np.ndarray, List[List[float]]]] = None,
        query_contents: Optional[str] = None,
        top_k: int = 100,
        n_probes: int = 1,
        filters: Optional[Dict[str, Any]] = None,
        include: List[str] = ["distance", "metadata"],
        greedy: bool = False
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Retrieve the nearest neighbors for given query vectors.
        Supports both single vector (1D) and batched vectors (2D).
        """
        try:
            from cyborgdb.openapi_client.models import QueryRequest, Request
            import numpy as np

            if filters is None:
                filters = {}

            # Determine the correct vector input
            vector_list = None

            if query_vector is not None:
                if isinstance(query_vector, np.ndarray):
                    if query_vector.ndim != 1:
                        raise ValueError("Expected 1D NumPy array for `query_vector`.")
                    vector_list = query_vector.tolist()
                elif isinstance(query_vector, list):
                    if query_vector and isinstance(query_vector[0], (list, np.ndarray)):
                        raise ValueError("Received nested list in `query_vector`; did you mean to use `query_vectors`?")
                    vector_list = list(map(float, query_vector))  # Ensure float type

            elif query_vectors is not None:
                if isinstance(query_vectors, list):
                    query_vectors = np.array(query_vectors, dtype=np.float32)
                if isinstance(query_vectors, np.ndarray):
                    if query_vectors.ndim != 2:
                        raise ValueError("Expected 2D NumPy array or list of lists for `query_vectors`.")
                    vector_list = query_vectors.tolist()
                else:
                    raise ValueError("Invalid type for `query_vectors`")

            elif query_contents is None:
                raise ValueError("You must provide `query_vector`, `query_vectors`, or `query_contents`.")

            # Construct query request
            print("top k =", top_k)
            print("vector_list:", vector_list)
            query_request = BatchQueryRequest(
                index_key=self._key_to_hex(),
                index_name=self._index_name,
                query_vectors=vector_list,
                query_contents=query_contents,
                top_k=top_k,
                n_probes=n_probes,
                greedy=greedy,
                filters=filters,
                include=include,
            )

            request = Request(query_request)

            # Execute query via REST
            print("calling query_vectors_v1_vectors_query_post")
            try:
                # Get raw response instead of deserialized object
                raw_response = self._api.query_vectors_v1_vectors_query_post_without_preload_content(
                    request=request,
                    _headers={
                        'X-API-Key': self._api_client.configuration.api_key['X-API-Key'],
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )
                
                # Parse raw JSON response manually
                response_text = raw_response.data.decode('utf-8')
                response_json = json.loads(response_text)
                
                # Process the results as plain dictionaries
                results = []
                if 'results' in response_json:
                    # Check if the results is a list of lists or just a list
                    if response_json['results'] and isinstance(response_json['results'][0], list):
                        # It's a list of lists (batch query results)
                        for query_result in response_json['results']:
                            query_items = []
                            for item in query_result:
                                result_item = {"id": item["id"]}
                                if "distance" in include and "distance" in item:
                                    result_item["distance"] = item["distance"]
                                if "metadata" in include and "metadata" in item:
                                    result_item["metadata"] = item["metadata"]
                                query_items.append(result_item)
                            results.append(query_items)
                    else:
                        # It's a flat list (single query results)
                        query_items = []
                        for item in response_json['results']:
                            result_item = {"id": item["id"]}
                            if "distance" in include and "distance" in item:
                                result_item["distance"] = item["distance"]
                            if "metadata" in include and "metadata" in item:
                                result_item["metadata"] = item["metadata"]
                            query_items.append(result_item)
                        results.append(query_items)
                
                return results
            except Exception as e:
                error_msg = f"Unexpected error in query: {str(e)}"
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
                raise
        except ApiException as e:
            error_msg = f"Query failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error in query: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error

    def _key_to_hex(self) -> str:
        """Convert the binary key to a hex string for API calls."""
        return binascii.hexlify(self._index_key).decode('ascii')


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
                
                print(f"Headers set: {self.api_client.default_headers}")
            
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
