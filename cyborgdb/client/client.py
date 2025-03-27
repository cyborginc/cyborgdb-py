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
    from cyborgdb.openapi_client.models.upsert_request import UpsertRequest
    from cyborgdb.openapi_client.models.train_request import TrainRequest
    from cyborgdb.openapi_client.models.vector_item import VectorItem as Item
    from cyborgdb.openapi_client.models.index_ivf_flat_model import IndexIVFFlatModel
    from cyborgdb.openapi_client.models.index_ivf_model import IndexIVFModel
    from cyborgdb.openapi_client.models.index_ivfpq_model import IndexIVFPQModel
    from cyborgdb.openapi_client.exceptions import ApiException
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
    "DBLocation",
    "generate_key"
]


def generate_key() -> bytes:
    """
    Generate a secure 32-byte key for use with CyborgDB indexes.
    
    Returns:
        bytes: A cryptographically secure 32-byte key.
    """
    return secrets.token_bytes(32)


class DBLocation(str, Enum):
    """Storage location for database components."""
    
    REDIS = "redis"
    POSTGRES = "postgres"


class DBConfig:
    """
    Configuration for a database storage component.
    
    Attributes:
        location (str): Storage location type ('redis' or 'postgres').
        table_name (Optional[str]): Name of the table (for SQL databases).
        connection_string (Optional[str]): Connection string for the database.
    """
    
    def __init__(
        self, 
        location: Union[str, DBLocation], 
        table_name: Optional[str] = None,
        connection_string: Optional[str] = None
    ):
        """
        Initialize a DBConfig object.
        
        Args:
            location: Storage location ('redis' or 'postgres').
            table_name: Name of the table (for SQL databases).
            connection_string: Connection string for the database.
            
        Raises:
            ValueError: If location is not one of the supported types.
        """
        if isinstance(location, DBLocation):
            location = location.value
            
        if location not in ["redis", "postgres"]:
            raise ValueError(
                f"Invalid location: {location}. Must be one of: 'redis', 'postgres'"
            )
            
        self.location = location
        self.table_name = table_name
        self.connection_string = connection_string


class IndexConfig:
    """
    Base class for index configurations.
    
    This is an abstract base class that provides common properties and methods
    for all index configuration types.
    """
    
    def __init__(self, dimension: int, metric: str, index_type: str, n_lists: int):
        """
        Initialize with index configuration parameters.
        
        Args:
            dimension: The dimensionality of the vectors.
            metric: The distance metric to use.
            index_type: The type of index.
            n_lists: The number of lists (coarse clusters).
        """
        self._dimension = dimension
        self._metric = metric
        self._index_type = index_type
        self._n_lists = n_lists
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the vectors in the index."""
        return self._dimension
    
    @property
    def metric(self) -> str:
        """Get the distance metric used in the index."""
        return self._metric
    
    @property
    def index_type(self) -> str:
        """Get the type of the index."""
        return self._index_type
    
    def n_lists(self) -> int:
        """Get the number of lists (coarse clusters) in the index."""
        return self._n_lists
    
    def pq_dim(self) -> int:
        """Get the Product Quantization dimension, if applicable."""
        return 0
    
    def pq_bits(self) -> int:
        """Get the number of bits per PQ code, if applicable."""
        return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary representation."""
        result = {
            "dimension": self.dimension,
            "metric": self.metric,
            "type": self.index_type,
            "n_lists": self.n_lists()
        }
        
        # Add PQ params if applicable
        if self.pq_dim() > 0:
            result["pqDim"] = self.pq_dim()
            result["pqBits"] = self.pq_bits()
            
        return result


class IndexIVF(IndexConfig):
    """
    Configuration for an IVF (Inverted File) index.
    
    IVF performs coarse clustering for accelerated search.
    """
    
    def __init__(self, dimension: int, n_lists: int, metric: str = "euclidean"):
        """
        Initialize an IVF index configuration.
        
        Args:
            dimension: Dimensionality of the vectors.
            n_lists: Number of coarse clusters (lists).
            metric: Distance metric to use ("euclidean", "cosine", or "squared_euclidean").
                Default is "euclidean".
                
        Raises:
            ValueError: If metric is not one of the supported types.
        """
        if metric not in ["euclidean", "cosine", "squared_euclidean"]:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: 'euclidean', 'cosine', 'squared_euclidean'"
            )
            
        super().__init__(dimension, metric, "ivf", n_lists)


class IndexIVFPQ(IndexConfig):
    """
    Configuration for an IVFPQ (Inverted File with Product Quantization) index.
    
    IVFPQ performs coarse clustering followed by product quantization for more efficient storage.
    """
    
    def __init__(
        self, 
        dimension: int, 
        n_lists: int, 
        pq_dim: int, 
        pq_bits: int,
        metric: str = "euclidean"
    ):
        """
        Initialize an IVFPQ index configuration.
        
        Args:
            dimension: Dimensionality of the vectors.
            n_lists: Number of coarse clusters (lists).
            pq_dim: Dimensionality for product quantization.
            pq_bits: Number of bits per quantizer.
            metric: Distance metric to use ("euclidean", "cosine", or "squared_euclidean").
                Default is "euclidean".
                
        Raises:
            ValueError: If metric is not one of the supported types.
        """
        if metric not in ["euclidean", "cosine", "squared_euclidean"]:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: 'euclidean', 'cosine', 'squared_euclidean'"
            )
            
        super().__init__(dimension, metric, "ivfpq", n_lists)
        self._pq_dim = pq_dim
        self._pq_bits = pq_bits
    
    def pq_dim(self) -> int:
        """Get the Product Quantization dimension."""
        return self._pq_dim
    
    def pq_bits(self) -> int:
        """Get the number of bits per PQ code."""
        return self._pq_bits


class IndexIVFFlat(IndexConfig):
    """
    Configuration for an IVFFlat (Inverted File with Flat Quantization) index.
    
    IVFFlat performs coarse clustering but stores the original vectors.
    """
    
    def __init__(self, dimension: int, n_lists: int, metric: str = "euclidean"):
        """
        Initialize an IVFFlat index configuration.
        
        Args:
            dimension: Dimensionality of the vectors.
            n_lists: Number of coarse clusters (lists).
            metric: Distance metric to use ("euclidean", "cosine", or "squared_euclidean").
                Default is "euclidean".
                
        Raises:
            ValueError: If metric is not one of the supported types.
        """
        if metric not in ["euclidean", "cosine", "squared_euclidean"]:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: 'euclidean', 'cosine', 'squared_euclidean'"
            )
            
        super().__init__(dimension, metric, "ivfflat", n_lists)


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
            response = self._api.get_vectors_v1_vectors_get_post(
                self._index_name,
                ids=ids,
                key=self._key_to_hex(),
                include=include
            )
            
            # Convert API response to our format
            items = []
            for item in response.items:
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
                max_iters=max_iters,
                tolerance=tolerance,
                max_memory=max_memory,
                key=self._key_to_hex()
            )
            
            self._api.train_index_v1_indexes_train_post(self._index_name, request)
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
                            item["metadata"] = json.dumps(item_dict["metadata"])
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
            
            # Create the upsert request
            request = UpsertRequest(
                items=items,
                key=self._key_to_hex()
            )
            
            # Make the API call
            self._api.upsert_vectors_v1_vectors_upsert_post(self._index_name, request)
            
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
            self._api.delete_vectors_v1_vectors_delete_post(
                self._index_name,
                ids=ids,
                key=self._key_to_hex()
            )
        except ApiException as e:
            error_msg = f"Failed to delete items: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def query(
        self,
        query_vector: Optional[Union[List[float], np.ndarray, List[List[float]]]] = None,
        query_contents: Optional[str] = None,
        top_k: int = 100,
        n_probes: int = 1,
        filters: Optional[Dict[str, Any]] = None,
        include: List[str] = ["distance", "metadata"],
        greedy: bool = False
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Retrieve the nearest neighbors for given query vectors.
        
        Args:
            query_vector: Query vectors to search. Can be a 1D array for a single query
                or a 2D array for multiple queries.
            query_contents: Text contents to search if auto-embedding is enabled.
            top_k: Number of nearest neighbors to return for each query. Default is 100.
            n_probes: Number of lists to probe during the query. Default is 1.
            filters: JSON-like dictionary specifying metadata filters. Default is None.
            include: Fields to include in results. Can contain "distance", "metadata".
                Default is ["distance", "metadata"].
            greedy: Whether to use greedy search. Default is False.
            
        Returns:
            For a single query, returns a list of dictionaries where each dictionary contains
            'id', 'distance', and optionally 'metadata'. For multiple queries, returns a list
            of such lists.
            
        Raises:
            ValueError: If the query vectors have incompatible dimensions with the index,
                if the index was not created or loaded yet, or if the query could not be executed.
            TypeError: If query_vectors is not a valid type.
            
        Note:
            If this function is called on an index where train() has not been executed, the query will
            use encrypted exhaustive search, which may be slower.
        """
        try:
            # Process query vectors if provided
            query_vectors_list = None
            if query_vector is not None:
                if isinstance(query_vector, np.ndarray):
                    # Handle 1D and 2D arrays differently
                    if query_vector.ndim == 1:
                        query_vectors_list = [query_vector.tolist()]
                    else:
                        query_vectors_list = query_vector.tolist()
                elif isinstance(query_vector, list):
                    # Check if it's a list of lists or a single list
                    if query_vector and isinstance(query_vector[0], (list, np.ndarray)):
                        query_vectors_list = [v if isinstance(v, list) else v.tolist() for v in query_vector]
                    else:
                        query_vectors_list = [query_vector]
                else:
                    raise TypeError("query_vector must be a numpy array or a list of vectors")
            
            # Prepare filters
            filters_json = None
            if filters:
                filters_json = json.dumps(filters)
            
            # Create query request
            request = QueryRequest(
                vectors=query_vectors_list,
                contents=query_contents,
                top_k=top_k,
                n_probes=n_probes,
                filters=filters_json,
                include=include,
                greedy=greedy,
                key=self._key_to_hex()
            )
            
            # Execute query
            response = self._api.query_vectors_v1_vectors_query_post(self._index_name, request)
            
            # Process results
            results = []
            for query_results in response.results:
                query_items = []
                for item in query_results:
                    result_item = {
                        "id": item.id
                    }
                    
                    if "distance" in include:
                        result_item["distance"] = item.distance
                        
                    if "metadata" in include and hasattr(item, "metadata"):
                        # Parse metadata JSON if needed
                        if isinstance(item.metadata, str):
                            try:
                                result_item["metadata"] = json.loads(item.metadata)
                            except json.JSONDecodeError:
                                result_item["metadata"] = {}
                        else:
                            result_item["metadata"] = item.metadata
                            
                    query_items.append(result_item)
                    
                results.append(query_items)
            
            # For single query, return just the results list instead of list of lists
            return results[0] if len(results) == 1 else results
            
        except ApiException as e:
            error_msg = f"Query failed: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except (TypeError, ValueError) as e:
            logger.error(str(e))
            raise
    
    def _key_to_hex(self) -> str:
        """Convert the binary key to a hex string for API calls."""
        return binascii.hexlify(self._index_key).decode('ascii')


class Client:
    """
    Client for interacting with CyborgDB via REST API.
    
    This class provides methods for creating, loading, and managing encrypted indexes.
    """
    
    def __init__(self, api_url, index_location, config_location, items_location=None, api_key=None, timeout=60, cpu_threads=0, gpu_accelerate=False):
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
            
            # Save DB configurations
            self.index_location = index_location
            self.config_location = config_location
            self.items_location = items_location
            
            # Save compute options
            self.cpu_threads = cpu_threads
            self.gpu_accelerate = gpu_accelerate
            
        except Exception as e:
            error_msg = f"Failed to initialize client: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    def _convert_to_api_config(self, index_config: IndexConfig) -> ApiIndexConfig:
        """Convert a local IndexConfig object to the API's configuration format."""
        config_dict = {
            "dimension": index_config.dimension,
            "metric": index_config.metric,
            "type": index_config.index_type,
            "n_lists": index_config.n_lists()
        }
        print("config_dict: ", config_dict)
        
        # Add PQ parameters if applicable
        if hasattr(index_config, 'pq_dim') and callable(index_config.pq_dim):
            pq_dim = index_config.pq_dim()
            if pq_dim > 0:
                config_dict["pq_dim"] = pq_dim
                config_dict["pq_bits"] = index_config.pq_bits()
                
        return config_dict
    
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

    # def create_index(
    #     self,
    #     index_name: str,
    #     index_key: bytes,
    #     index_config: Union[IndexIVFModel, IndexIVFPQModel, IndexIVFFlatModel],
    #     embedding_model: Optional[str] = None,
    #     max_cache_size: int = 0
    # ) -> EncryptedIndex:
    #     """
    #     Create and return a new encrypted index based on the provided configuration.
    #     """
    #     # Validate index_key
    #     if not isinstance(index_key, bytes) or len(index_key) != 32:
    #         raise ValueError("index_key must be a 32-byte bytes object")

    #     try:
    #         # Convert binary key to hex string
    #         key_hex = binascii.hexlify(index_key).decode('ascii')
            
    #         # Get the dictionary from the index_config
    #         if hasattr(index_config, 'model_dump'):
    #             index_config_dict = index_config.model_dump()
    #         elif hasattr(index_config, 'to_dict'):
    #             index_config_dict = index_config.to_dict()
    #         else:
    #             # If it's already a dictionary, use it as is
    #             index_config_dict = index_config
                
    #         # Convert camelCase to snake_case for the API
    #         converted_config = {}
    #         for key, value in index_config_dict.items():
    #             # Convert camelCase to snake_case
    #             if key == "indexType":
    #                 converted_config["index_type"] = value
    #             elif key == "nLists":
    #                 converted_config["n_lists"] = value
    #             elif key == "pqDim":
    #                 converted_config["pq_dim"] = value
    #             elif key == "pqBits":
    #                 converted_config["pq_bits"] = value
    #             else:
    #                 converted_config[key] = value
            
    #         # Get the index type (ivf, ivfpq, etc.)
    #         index_type = converted_config.pop("index_type", "ivf")
            
    #         # Create the appropriate flat structure without nesting under model type
    #         request_data = {
    #             "index_name": index_name,
    #             "index_key": key_hex,
    #             "index_config": {
    #                 # Include required fields directly
    #                 "dimension": converted_config.get("dimension", 128),
    #                 "metric": converted_config.get("metric", "euclidean"),
    #                 "index_type": index_type,
    #                 "n_lists": converted_config.get("n_lists", 10)
    #             }
    #         }
            
    #         # Add PQ-specific fields if needed
    #         if index_type == "ivfpq":
    #             request_data["index_config"]["pq_dim"] = converted_config.get("pq_dim", 8)
    #             request_data["index_config"]["pq_bits"] = converted_config.get("pq_bits", 8)
            
    #         if embedding_model:
    #             request_data["embedding_model"] = embedding_model
            
    #         # Import necessary modules
    #         import requests
            
    #         # Get the base URL from the API client
    #         host = self.api_client.configuration.host
            
    #         # Construct the full URL
    #         url = f"{host}/v1/indexes/create"
            
    #         # Make the request directly
    #         headers = {
    #             'X-API-Key': self.config.api_key['X-API-Key'],
    #             'Content-Type': 'application/json',
    #             'Accept': 'application/json'
    #         }
            
    #         print("Sending request data:", json.dumps(request_data, indent=2))
            
    #         response = requests.post(
    #             url,
    #             json=request_data,
    #             headers=headers
    #         )
            
    #         # Check for successful response
    #         if not response.ok:
    #             raise ApiException(
    #                 status=response.status_code,
    #                 reason=f"API returned error: {response.text}"
    #             )
            
    #         return EncryptedIndex(
    #             index_name=index_name,
    #             index_key=index_key,
    #             api=self.api,
    #             api_client=self.api_client,
    #             max_cache_size=max_cache_size
    #         )

    #     except ApiException as e:
    #         error_msg = f"Failed to create index: {e}"
    #         logger.error(error_msg)
    #         raise ValueError(error_msg)
    #     except Exception as e:
    #         error_msg = f"Unexpected error creating index: {e}"
    #         logger.error(error_msg)
    #         raise ValueError(error_msg)
        
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
            
            # Extract configuration from the index_config object
            if hasattr(index_config, 'model_dump'):
                config_dict = index_config.model_dump()
            elif hasattr(index_config, 'to_dict'):
                config_dict = index_config.to_dict()
            else:
                config_dict = index_config
                
            # Determine which model type we need
            index_type = config_dict.get("indexType", "ivf")
            print("index_type: ", index_type)
            
            # Create the appropriate model object from your OpenAPI generated classes
            # We'll need to import these classes - adjust the imports as needed
            from cyborgdb.openapi_client.models import (
                CreateIndexRequest, 
                IndexConfig,
                IndexIVFModel,
                IndexIVFPQModel, 
                IndexIVFFlatModel
            )
            
            # Create the specific model instance based on the index type
            if index_type == "ivf":
                model = IndexIVFModel(
                    dimension=config_dict.get("dimension", 128),
                    metric=config_dict.get("metric", "euclidean"),
                    n_lists=config_dict.get("nLists", 10)
                )
            elif index_type == "ivfpq":
                model = IndexIVFPQModel(
                    dimension=config_dict.get("dimension", 128),
                    metric=config_dict.get("metric", "euclidean"),
                    n_lists=config_dict.get("nLists", 10),
                    pq_dim=config_dict.get("pqDim", 8),
                    pq_bits=config_dict.get("pqBits", 8)
                )
            elif index_type == "ivfflat":
                model = IndexIVFFlatModel(
                    dimension=config_dict.get("dimension", 128),
                    metric=config_dict.get("metric", "euclidean"),
                    n_lists=config_dict.get("nLists", 10)
                )
            else:
                # Default to IVF
                model = IndexIVFModel(
                    dimension=config_dict.get("dimension", 128),
                    metric=config_dict.get("metric", "euclidean"),
                    n_lists=config_dict.get("nLists", 10)
                )

            print("model: ", model)
                
            # Create an IndexConfig instance with the appropriate model
            index_config_obj = IndexConfig(model)
        
            print("index_config_obj: ", index_config_obj)
            
            # Create the complete request object
            request = CreateIndexRequest(
                index_name=index_name,
                index_key=key_hex,
                index_config=index_config_obj,
                embedding_model=embedding_model
            )
            print("index_config: ", index_config)
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
