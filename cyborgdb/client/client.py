"""
CyborgDB Client Wrapper

This module provides a Python wrapper around the CyborgDB C++ library.
It offers a high-level API for interacting with the encrypted vector database.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import json
import secrets
import numpy as np
from enum import Enum
from pathlib import Path

# Import the C++ bindings
try:
    import cyborgdb_core as _core
except ImportError:
    raise ImportError(
        "Failed to import cyborgdb_core. Make sure the C++ library is properly installed."
    )

__all__ = [
    "Client", 
    "EncryptedIndex",
    "IndexConfig",
    "IndexIVF",
    "IndexIVFPQ",
    "IndexIVFFlat",
    "DBConfig",
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
    
    MEMORY = "memory"
    REDIS = "redis"
    POSTGRES = "postgres"
    NONE = "none"


class DBConfig:
    """
    Configuration for a database storage component.
    
    Attributes:
        location (str): Storage location type ('memory', 'redis', 'postgres', or 'none').
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
            location: Storage location ('memory', 'redis', 'postgres', or 'none').
            table_name: Name of the table (for SQL databases).
            connection_string: Connection string for the database.
            
        Raises:
            ValueError: If location is not one of the supported types.
        """
        if isinstance(location, DBLocation):
            location = location.value
            
        if location not in ["memory", "redis", "postgres", "none"]:
            raise ValueError(
                f"Invalid location: {location}. Must be one of: 'memory', 'redis', 'postgres', 'none'"
            )
            
        self._config = _core.DBConfig(location, table_name, connection_string)
    
    @property
    def core_config(self):
        """Get the underlying C++ DBConfig object."""
        return self._config


class IndexConfig:
    """
    Base class for index configurations.
    
    This is an abstract class that provides common properties and methods
    for all index configuration types.
    """
    
    def __init__(self, core_config):
        """
        Initialize with a core IndexConfig object.
        
        Args:
            core_config: The C++ IndexConfig object.
        """
        self._config = core_config
    
    @property
    def dimension(self) -> int:
        """Get the dimensionality of the vectors in the index."""
        return self._config.dimension
    
    @property
    def metric(self) -> str:
        """Get the distance metric used in the index."""
        return self._config.metric
    
    @property
    def index_type(self) -> str:
        """Get the type of the index."""
        return self._config.index_type
    
    def n_lists(self) -> int:
        """Get the number of lists (coarse clusters) in the index."""
        return self._config.n_lists()
    
    def pq_dim(self) -> int:
        """Get the Product Quantization dimension, if applicable."""
        return self._config.pq_dim()
    
    def pq_bits(self) -> int:
        """Get the number of bits per PQ code, if applicable."""
        return self._config.pq_bits()
    
    @property
    def core_config(self):
        """Get the underlying C++ IndexConfig object."""
        return self._config


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
            
        core_config = _core.IndexIVF(dimension, n_lists, metric)
        super().__init__(core_config)


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
            
        core_config = _core.IndexIVFPQ(dimension, n_lists, pq_dim, pq_bits, metric)
        super().__init__(core_config)


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
            
        core_config = _core.IndexIVFFlat(dimension, n_lists, metric)
        super().__init__(core_config)


class EncryptedIndex:
    """
    Provides access to an encrypted vector index.
    
    This class handles operations on an encrypted vector index, including
    adding/updating vectors, searching, and managing index metadata.
    """
    
    def __init__(self, core_index):
        """
        Initialize with a core EncryptedIndex object.
        
        Args:
            core_index: The C++ EncryptedIndex object.
        """
        self._index = core_index
    
    @property
    def index_name(self) -> str:
        """Get the name of the index."""
        return self._index.index_name()
    
    @property
    def index_type(self) -> str:
        """Get the type of the index."""
        return self._index.index_type()
    
    @property
    def index_config(self) -> Dict[str, Any]:
        """Get the configuration of the index as a dictionary."""
        return self._index.index_config()
    
    def is_trained(self) -> bool:
        """
        Check if the index has been trained.
        
        Returns:
            bool: True if the index is trained, otherwise False.
        """
        return self._index.is_trained()
    
    def delete_index(self) -> None:
        """
        Delete the current index and all its associated data.
        
        Warning:
            This action is irreversible.
            
        Raises:
            ValueError: If the index could not be deleted.
        """
        self._index.delete_index()
    
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
        return self._index.get(ids, include)
    
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
        self._index.train(batch_size, max_iters, tolerance, max_memory)
    
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
        self._index.upsert(arg1, arg2)
    
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
        self._index.delete(ids)
    
    def query(
        self,
        query_vector: Optional[Union[List[float], np.ndarray, List[List[float]]]] = None,
        query_contents: Optional[str] = None,
        top_k: int = 100,
        n_probes: int = 1,
        filters: Dict[str, Any] = None,
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
        filters = filters or {}
        return self._index.query(
            query_vector, query_contents, top_k, n_probes, filters, include, greedy
        )


class Client:
    """
    Client for interacting with CyborgDB.
    
    This class provides methods for creating, loading, and managing encrypted indexes.
    """
    
    def __init__(
        self,
        index_location: Union[DBConfig, Dict[str, Any]],
        config_location: Union[DBConfig, Dict[str, Any]],
        items_location: Optional[Union[DBConfig, Dict[str, Any]]] = None,
        cpu_threads: int = 0,
        gpu_accelerate: bool = False,
        working_dir: Optional[str] = None
    ):
        """
        Initialize a new instance of Client.
        
        Args:
            index_location: Configuration for index storage location.
            config_location: Configuration for index metadata storage.
            items_location: Configuration for future item storage. Default is None.
            cpu_threads: Number of CPU threads to use (0 = all cores). Default is 0.
            gpu_accelerate: Whether to enable GPU acceleration (requires CUDA). Default is False.
            working_dir: Optional working directory for license files. Default is None.
            
        Raises:
            ValueError: If cpu_threads is less than 0, if any DBConfig is invalid,
                if GPU is unavailable when gpu_accelerate is True, if the backing store
                is not available, or if the Client could not be initialized.
        """
        # Set working directory if provided
        if working_dir:
            _core.set_working_dir(working_dir)
            
        # Convert dict to DBConfig if necessary
        if isinstance(index_location, dict):
            index_location = DBConfig(
                index_location.get("location", "memory"),
                index_location.get("table_name"),
                index_location.get("connection_string")
            )
        
        if isinstance(config_location, dict):
            config_location = DBConfig(
                config_location.get("location", "memory"),
                config_location.get("table_name"),
                config_location.get("connection_string")
            )
            
        if items_location is None:
            items_location = DBConfig(DBLocation.NONE)
        elif isinstance(items_location, dict):
            items_location = DBConfig(
                items_location.get("location", "none"),
                items_location.get("table_name"),
                items_location.get("connection_string")
            )
            
        self._client = _core.Client(
            index_location.core_config,
            config_location.core_config,
            items_location.core_config,
            cpu_threads,
            gpu_accelerate
        )
    
    def list_indexes(self) -> List[str]:
        """
        Get a list of all encrypted index names accessible via the client.
        
        Returns:
            A list of index names.
            
        Raises:
            ValueError: If the list of indexes could not be retrieved.
        """
        return self._client.list_indexes()
    
    def create_index(
        self,
        index_name: str,
        index_key: bytes,
        index_config: Union[IndexConfig, IndexIVF, IndexIVFPQ, IndexIVFFlat],
        embedding_model: Optional[str] = None,
        max_cache_size: int = 0
    ) -> EncryptedIndex:
        """
        Create and return a new encrypted index based on the provided configuration.
        
        Args:
            index_name: Name of the index to create (must be unique).
            index_key: 32-byte encryption key for the index, used to secure index data.
            index_config: Configuration for the index type.
            embedding_model: Name of the SentenceTransformer model to use for text embeddings.
                Default is None.
            max_cache_size: Maximum size for the local cache. Default is 0.
            
        Returns:
            An EncryptedIndex instance for the newly created index.
            
        Raises:
            ValueError: If the index name is not unique, if the index configuration is invalid,
                or if the index could not be created.
        """
        # Convert index_key to byte array if it's not already
        if not isinstance(index_key, bytes) or len(index_key) != 32:
            raise ValueError("index_key must be a 32-byte bytes object")
        
        # Convert to bytearray of length 32 as expected by the C++ code
        key_array = bytearray(index_key)
        
        # Create the index
        core_index = self._client.create_index(
            index_name,
            key_array,
            index_config.core_config,
            embedding_model,
            max_cache_size
        )
        
        return EncryptedIndex(core_index)
    
    def load_index(
        self,
        index_name: str,
        index_key: bytes,
        max_cache_size: int = 0
    ) -> EncryptedIndex:
        """
        Load an existing encrypted index and return an instance of EncryptedIndex.
        
        Args:
            index_name: Name of the index to load.
            index_key: 32-byte encryption key for the index, used to secure index data.
            max_cache_size: Maximum size for the local cache. Default is 0.
            
        Returns:
            An EncryptedIndex instance for the loaded index.
            
        Raises:
            ValueError: If the index name does not exist, if the index could not be loaded
                or decrypted.
        """
        # Convert index_key to byte array if it's not already
        if not isinstance(index_key, bytes) or len(index_key) != 32:
            raise ValueError("index_key must be a 32-byte bytes object")
        
        # Convert to bytearray of length 32 as expected by the C++ code
        key_array = bytearray(index_key)
        
        # Load the index
        core_index = self._client.load_index(index_name, key_array, max_cache_size)
        
        return EncryptedIndex(core_index)