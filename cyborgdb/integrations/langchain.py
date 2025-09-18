"""
LangChain integration for CyborgDB-py REST API.

This module provides a LangChain VectorStore implementation for CyborgDB,
enabling seamless integration with LangChain applications.

Requirements:
    pip install cyborgdb-py[langchain]
"""

import uuid
import json
import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Iterable

try:
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    # Import CyborgDB components
    from cyborgdb import Client, EncryptedIndex, IndexIVF, IndexIVFPQ, IndexIVFFlat

    class CyborgVectorStore(VectorStore):
        """
        CyborgDB vector store for LangChain.

        This class provides a LangChain-compatible interface to CyborgDB's
        encrypted vector storage, supporting all standard vector store operations
        including similarity search, metadata filtering, and document management.

        Note: Document content is currently stored in the metadata field with a
        reserved key "_content" because the CyborgDB contents field expects a
        specific object format. This approach ensures compatibility while keeping
        the original metadata separate and accessible.

        Attributes:
            index_name: Name of the CyborgDB index
            index: The underlying EncryptedIndex instance
            client: CyborgDB client instance
        """

        @staticmethod
        def generate_key() -> bytes:
            """
            Generate a secure 32-byte key for use with CyborgDB indexes.

            Returns:
                bytes: A cryptographically secure 32-byte key.
            """
            import secrets

            return secrets.token_bytes(32)

        @staticmethod
        def _validate_index_key(index_key: bytes) -> None:
            """
            Validate that index_key is a proper 32-byte encryption key.

            Args:
                index_key: The encryption key to validate

            Raises:
                TypeError: If index_key is not bytes type
                ValueError: If index_key is not exactly 32 bytes
            """
            if not isinstance(index_key, bytes):
                raise TypeError(
                    f"index_key must be bytes, got {type(index_key).__name__}. "
                    f"Use Client.generate_key() to create a secure 32-byte key."
                )

            if len(index_key) != 32:
                raise ValueError(
                    f"index_key must be exactly 32 bytes, got {len(index_key)} bytes. "
                    f"Use Client.generate_key() to create a secure 32-byte key."
                )

        def __init__(
            self,
            index_name: str,
            index_key: bytes,
            api_key: str,
            base_url: str,
            embedding: Union[str, Embeddings, SentenceTransformer],
            index_type: str = "ivfflat",
            index_config_params: Optional[Dict[str, Any]] = None,
            dimension: Optional[int] = None,
            metric: str = "cosine",
            verify_ssl: Optional[bool] = None,
        ) -> None:
            """
            Initialize a CyborgVectorStore.

            Args:
                index_name: Name for the index
                index_key: 32-byte encryption key for the index
                api_key: API key for CyborgDB authentication
                base_url: URL of the CyborgDB API server
                embedding: Embedding model - can be:
                    - String model name (for SentenceTransformer)
                    - SentenceTransformer instance
                    - LangChain Embeddings instance
                index_type: Type of index - "ivfflat", "ivf", or "ivfpq"
                index_config_params: Additional index configuration parameters
                dimension: Embedding dimension (auto-detected if not provided)
                metric: Distance metric - "cosine", "euclidean", or "squared_euclidean"
                verify_ssl: SSL verification (None for auto-detect, True/False to override)

            Raises:
                ValueError: If required parameters are invalid or index creation fails
            """
            self._validate_index_key(index_key)

            self.index_name = index_name
            self.index_key = index_key

            # Set up embedding model
            self._setup_embedding_model(embedding)

            # Create client
            self.client = Client(
                base_url=base_url, api_key=api_key, verify_ssl=verify_ssl
            )

            # Initialize or load index
            self._initialize_index(
                index_type=index_type,
                index_config_params=index_config_params or {},
                dimension=dimension,
                metric=metric,
            )

        def _setup_embedding_model(
            self, embedding: Union[str, Embeddings, SentenceTransformer]
        ) -> None:
            """Configure the embedding model."""
            if isinstance(embedding, str):
                self.embedding_model_name = embedding
                self.embedding_model = None  # Lazy load
            else:
                self.embedding_model = embedding
                self.embedding_model_name = (
                    getattr(embedding, "model_name", "")
                    if hasattr(embedding, "model_name")
                    else ""
                )

        def _initialize_index(
            self,
            index_type: Optional[str] = None,
            index_config_params: Optional[Dict[str, Any]] = None,
            dimension: Optional[int] = None,
            metric: Optional[str] = None,
        ) -> None:
            """Initialize or load the CyborgDB index."""
            # Check if index already exists
            try:
                existing_indexes = self.client.list_indexes()
                index_exists = self.index_name in existing_indexes
            except Exception as e:
                index_exists = False
                warnings.warn(f"Could not check if index exists: {e}", RuntimeWarning)

            if index_exists:
                # Load existing index
                self._load_existing_index()
            else:
                # Create new index
                self._create_new_index(
                    index_type, index_config_params, dimension, metric
                )

        def _load_existing_index(self) -> None:
            """Load an existing index."""
            self.index = EncryptedIndex(
                index_name=self.index_name,
                index_key=self.index_key,
                api=self.client.api,
                api_client=self.client.api_client,
            )

        def _create_new_index(
            self,
            index_type: Optional[str] = None,
            index_config_params: Optional[Dict[str, Any]] = None,
            dimension: Optional[int] = None,
            metric: Optional[str] = None,
        ) -> None:
            """Create a new index."""
            # Determine embedding dimension
            if dimension is None:
                dimension = self._detect_embedding_dimension()

            # Create index configuration
            if index_type is not None:
                config = self._create_index_config(
                    index_type, dimension, index_config_params or {}
                )
            else:
                config = None

            # Create the index
            self.index = self.client.create_index(
                index_name=self.index_name,
                index_key=self.index_key,
                index_config=config,
                embedding_model=self.embedding_model_name
                if self.embedding_model_name
                else None,
                metric=metric,
            )

        def _detect_embedding_dimension(self) -> int:
            """Helper to detect the embedding dimension from the model."""
            # Lazy load if needed
            if self.embedding_model is None and self.embedding_model_name:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            if self.embedding_model is None:
                raise RuntimeError(
                    "No embedding model provided and dimension not specified"
                )

            # Try different methods to get dimension
            if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
                return self.embedding_model.get_sentence_embedding_dimension()
            elif hasattr(self.embedding_model, "embed_query"):
                dummy = self.embedding_model.embed_query("dimension check")
                return (
                    len(dummy)
                    if isinstance(dummy, list)
                    else np.asarray(dummy).shape[0]
                )
            else:
                # Try encode method
                dummy = self.embedding_model.encode(["dimension check"])[0]
                return (
                    len(dummy)
                    if isinstance(dummy, list)
                    else np.asarray(dummy).shape[0]
                )

        def _create_index_config(
            self, index_type: str, dimension: int, params: Dict[str, Any]
        ) -> Union[IndexIVF, IndexIVFPQ, IndexIVFFlat]:
            """Create the appropriate index configuration."""
            if index_type == "ivf":
                return IndexIVF(dimension=dimension)
            elif index_type == "ivfpq":
                pq_dim = params.get("pq_dim", 8)
                pq_bits = params.get("pq_bits", 8)
                return IndexIVFPQ(
                    dimension=dimension,
                    pq_dim=pq_dim,
                    pq_bits=pq_bits,
                )
            elif index_type == "ivfflat":
                return IndexIVFFlat(dimension=dimension)
            else:
                raise ValueError(
                    f"Invalid index type: {index_type}. Must be 'ivf', 'ivfpq', or 'ivfflat'"
                )

        def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
            """
            Helper to generate embeddings for the given texts.

            Args:
                texts: Single text string or list of texts

            Returns:
                1D array for single text, 2D array for multiple texts
            """
            # Lazy load model if needed
            if self.embedding_model is None and self.embedding_model_name:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            if self.embedding_model is None:
                raise RuntimeError("No embedding model available")

            is_single = isinstance(texts, str)
            texts_list = [texts] if is_single else texts

            # Generate embeddings based on model type
            if hasattr(self.embedding_model, "encode") and hasattr(
                self.embedding_model, "get_sentence_embedding_dimension"
            ):
                # SentenceTransformer
                embeddings = self.embedding_model.encode(
                    texts_list, convert_to_numpy=True
                )
            elif hasattr(self.embedding_model, "embed_documents") and hasattr(
                self.embedding_model, "embed_query"
            ):
                # LangChain Embeddings
                if is_single:
                    raw = self.embedding_model.embed_query(texts)
                    embeddings = np.array(raw, dtype=np.float32)[None, :]
                else:
                    raw = self.embedding_model.embed_documents(texts_list)
                    embeddings = np.array(raw, dtype=np.float32)
            elif callable(self.embedding_model):
                # Generic callable
                embeddings = self.embedding_model(texts_list)
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings, dtype=np.float32)
            else:
                raise TypeError(
                    f"Unsupported embedding model type: {type(self.embedding_model)}"
                )

            return embeddings[0] if is_single else embeddings

        def add_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs,
        ) -> List[str]:
            """
            Add texts to the vector store.

            Args:
                texts: Texts to add
                metadatas: Optional metadata for each text
                ids: Optional IDs for each text (generated if not provided)
                **kwargs: Additional arguments:
                    - vectors: Optional pre-computed embeddings (List or array of shape [num_texts, embedding_dim])

            Returns:
                List of IDs for the added texts

            Raises:
                ValueError: If lengths of texts, metadatas, ids, or vectors don't match
            """
            texts_list = list(texts)
            num_texts = len(texts_list)

            if num_texts == 0:
                return []

            # Validate or generate IDs
            if ids is not None:
                if len(ids) != num_texts:
                    raise ValueError("Length of ids must match length of texts")
                id_list = list(ids)
            else:
                id_list = [str(uuid.uuid4()) for _ in range(num_texts)]

            # Validate metadata
            if metadatas is not None and len(metadatas) != num_texts:
                raise ValueError("Length of metadatas must match length of texts")

            # Generate or use provided embeddings
            if kwargs.get("vectors", None) is not None:
                embeddings = kwargs["vectors"]
                # Validate vectors shape/length
                if hasattr(embeddings, "__len__"):
                    if len(embeddings) != num_texts:
                        raise ValueError(f"Length of vectors ({len(embeddings)}) must match length of texts ({num_texts})")
                else:
                    raise ValueError("vectors must be a list or array-like object")
            else:
                embeddings = self.get_embeddings(texts_list)

            # Build items for upsert
            items = []
            for i in range(num_texts):
                # Handle both numpy arrays and lists for embeddings
                if hasattr(embeddings, "shape"):
                    vector = (
                        embeddings[i].tolist()
                        if len(embeddings.shape) > 1
                        else embeddings.tolist()
                    )
                else:
                    # embeddings is likely a list of lists or a list
                    vector = (
                        embeddings[i]
                        if isinstance(embeddings[i], list)
                        else [embeddings[i]]
                    )
                item = {"id": id_list[i], "vector": vector}

                # Store text in metadata with document content
                # Note: The contents field expects a specific object format, not plain strings
                # Until this is fully supported, we store text in metadata with a reserved key
                if metadatas is not None and metadatas[i]:
                    item["metadata"] = metadatas[i].copy()
                    item["metadata"]["_content"] = texts_list[i]
                else:
                    item["metadata"] = {"_content": texts_list[i]}

                items.append(item)

            # Upsert to index
            self.index.upsert(items)

            return id_list

        def add_documents(
            self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs
        ) -> List[str]:
            """
            Add documents to the vector store.

            Args:
                documents: Documents to add
                ids: Optional IDs for documents
                **kwargs: Additional arguments
                    - vectors: Optional pre-computed embeddings (List or array of shape [num_texts, embedding_dim])

            Returns:
                List of IDs for the added documents
            """
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return self.add_texts(texts, metadatas, ids=ids, **kwargs)

        def delete(
            self, ids: Optional[List[str]] = None, delete_index: bool = False
        ) -> bool:
            """
            Delete documents or the entire index.

            Args:
                ids: IDs of documents to delete (if None and delete_index=False, returns False)
                delete_index: If True, deletes the entire index

            Returns:
                True if operation succeeded, False otherwise
            """
            try:
                if delete_index:
                    self.index.delete_index()
                    return True
                elif ids is not None and len(ids) > 0:
                    self.index.delete(ids)
                    return True
                else:
                    return False
            except Exception as e:
                warnings.warn(f"Delete operation failed: {e}")
                return False

        def get(self, ids: List[str]) -> List[Document]:
            """
            Retrieve documents by their IDs.

            Args:
                ids: List of document IDs to retrieve

            Returns:
                List of Documents
            """
            items = self.index.get(ids, include=["metadata"])

            docs = []
            for item in items:
                metadata = item.get("metadata", {})

                # Parse metadata if it's a string
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata}

                # Extract content from metadata
                metadata_copy = metadata.copy() if isinstance(metadata, dict) else {}
                content = metadata_copy.pop("_content", "")

                docs.append(Document(page_content=content, metadata=metadata_copy))

            return docs

        def list_ids(self):
            """
            List all document IDs in the vector store.

            Returns:
                List of document IDs
            """
            return self.index.list_ids()

        def _execute_query(
            self,
            query: Union[str, List[float]],
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            n_probes: Optional[int] = None,
        ) -> List[Dict[str, Any]]:
            """
            Execute a search query.

            Args:
                query: Query text or embedding vector
                k: Number of results to return
                filter: Optional metadata filter
                n_probes: Number of probes for search

            Returns:
                List of result dictionaries
            """
            filter = filter or {}

            if isinstance(query, str):
                # Text query - generate embedding
                embedding = self.get_embeddings(query)
                results = self.index.query(
                    query_vectors=embedding,
                    top_k=k,
                    n_probes=n_probes,
                    filters=filter,
                    include=["distance", "metadata"],
                    greedy=False,
                )
            else:
                # Vector query
                results = self.index.query(
                    query_vectors=query,
                    top_k=k,
                    n_probes=n_probes,
                    filters=filter,
                    include=["distance", "metadata"],
                    greedy=False,
                )

            # Handle batch query results
            if (
                isinstance(results, list)
                and len(results) > 0
                and isinstance(results[0], list)
            ):
                results = results[0]

            return results if results else []

        def similarity_search(
            self,
            query: str,
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            **kwargs,
        ) -> List[Document]:
            """
            Search for documents similar to the query.

            Args:
                query: Query text
                k: Number of results to return
                filter: Optional metadata filter
                **kwargs: Additional arguments (n_probes can be specified)

            Returns:
                List of similar documents
            """
            n_probes = kwargs.get("n_probes", None)
            results = self._execute_query(query, k, filter, n_probes)

            docs = []
            for item in results:
                metadata = item.get("metadata", {})

                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata}

                # Extract content from metadata
                metadata_copy = metadata.copy() if isinstance(metadata, dict) else {}
                content = metadata_copy.pop("_content", "")

                docs.append(Document(page_content=content, metadata=metadata_copy))

            return docs

        def similarity_search_with_score(
            self,
            query: str,
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            **kwargs,
        ) -> List[Tuple[Document, float]]:
            """
            Search for documents with similarity scores.

            Args:
                query: Query text
                k: Number of results to return
                filter: Optional metadata filter
                **kwargs: Additional arguments

            Returns:
                List of (document, score) tuples
            """
            n_probes = kwargs.get("n_probes", None)
            results = self._execute_query(query, k, filter, n_probes)

            docs_with_scores = []
            for item in results:
                metadata = item.get("metadata", {})

                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata}

                # Extract content from metadata (consistent with other methods)
                metadata_copy = metadata.copy() if isinstance(metadata, dict) else {}
                content = metadata_copy.pop("_content", "")

                doc = Document(page_content=content, metadata=metadata_copy)

                # Convert distance to similarity score
                distance = item.get("distance", 0.0)
                similarity = self._normalize_score(distance)

                docs_with_scores.append((doc, similarity))

            return docs_with_scores

        def similarity_search_by_vector(
            self,
            embedding: List[float],
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            **kwargs,
        ) -> List[Document]:
            """
            Search for documents similar to an embedding vector.

            Args:
                embedding: Query embedding vector
                k: Number of results to return
                filter: Optional metadata filter
                **kwargs: Additional arguments

            Returns:
                List of similar documents
            """
            n_probes = kwargs.get("n_probes", None)
            results = self._execute_query(embedding, k, filter, n_probes)

            docs = []
            for item in results:
                metadata = item.get("metadata", {})

                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {"raw": metadata}

                # Extract content from metadata
                metadata_copy = metadata.copy() if isinstance(metadata, dict) else {}
                content = metadata_copy.pop("_content", "")

                docs.append(Document(page_content=content, metadata=metadata_copy))

            return docs

        def _normalize_score(self, distance: float) -> float:
            """
            Convert distance to similarity score [0, 1].

            Args:
                distance: Distance value from query

            Returns:
                Normalized similarity score
            """
            # Get metric from index config
            try:
                config = self.index.index_config
                metric = (
                    config.get("metric", "cosine")
                    if isinstance(config, dict)
                    else getattr(config, "metric", "cosine")
                )
            except Exception:
                metric = "cosine"

            if metric == "cosine":
                # Cosine distance: 0 (identical) to 2 (opposite)
                return max(0.0, 1.0 - (distance / 2.0))
            elif metric == "euclidean":
                # Euclidean: exponential decay
                return np.exp(-distance)
            elif metric == "squared_euclidean":
                # Squared Euclidean: exponential decay with sqrt
                return np.exp(-np.sqrt(distance))
            else:
                # Default: inverse distance
                return 1.0 / (1.0 + distance)

        def as_retriever(
            self,
            search_type: Optional[str] = None,
            search_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> VectorStoreRetriever:
            """
            Return a LangChain retriever for this vector store.

            Args:
                search_type: Type of search (default: "similarity")
                search_kwargs: Keyword arguments for search
                **kwargs: Additional retriever arguments

            Returns:
                VectorStoreRetriever instance
            """
            return VectorStoreRetriever(
                vectorstore=self,
                search_type=search_type or "similarity",
                search_kwargs=search_kwargs or {},
                **kwargs,
            )

        # Async methods for compatibility
        async def aadd_texts(
            self,
            texts: Iterable[str],
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs,
        ) -> List[str]:
            """Async version of add_texts."""
            import asyncio

            return await asyncio.to_thread(
                self.add_texts, texts, metadatas=metadatas, ids=ids, **kwargs
            )

        async def aadd_documents(
            self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs
        ) -> List[str]:
            """Async version of add_documents."""
            import asyncio

            return await asyncio.to_thread(
                self.add_documents, documents, ids=ids, **kwargs
            )

        async def adelete(
            self, ids: Optional[List[str]] = None, delete_index: bool = False
        ) -> bool:
            """Async version of delete."""
            import asyncio

            return await asyncio.to_thread(
                self.delete, ids=ids, delete_index=delete_index
            )

        async def asimilarity_search(
            self,
            query: str,
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            **kwargs,
        ) -> List[Document]:
            """Async version of similarity_search."""
            import asyncio

            return await asyncio.to_thread(
                self.similarity_search, query, k, filter, **kwargs
            )

        async def asimilarity_search_with_score(
            self,
            query: str,
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            **kwargs,
        ) -> List[Tuple[Document, float]]:
            """Async version of similarity_search_with_score."""
            import asyncio

            return await asyncio.to_thread(
                self.similarity_search_with_score, query, k, filter, **kwargs
            )

        async def asimilarity_search_by_vector(
            self,
            embedding: List[float],
            k: Optional[int] = None,
            filter: Optional[Dict] = None,
            **kwargs,
        ) -> List[Document]:
            """Async version of similarity_search_by_vector."""
            import asyncio

            return await asyncio.to_thread(
                self.similarity_search_by_vector, embedding, k, filter, **kwargs
            )

        @classmethod
        def from_texts(
            cls,
            texts: List[str],
            embedding: Union[str, Embeddings, SentenceTransformer],
            metadatas: Optional[List[Dict]] = None,
            **kwargs,
        ) -> "CyborgVectorStore":
            """
            Create a vector store from a list of texts.

            Args:
                texts: List of texts to add
                embedding: Embedding model
                metadatas: Optional metadata for each text
                **kwargs: Additional arguments including:
                    - index_name: Name for the index
                    - index_key: 32-byte encryption key
                    - api_key: CyborgDB API key
                    - base_url: CyborgDB API URL
                    - ids: Optional IDs for texts
                    - index_type: Type of index
                    - metric: Distance metric
                    - and more...

            Returns:
                CyborgVectorStore instance

            Raises:
                ValueError: If required parameters are missing
            """
            # Extract parameters
            ids = kwargs.pop("ids", None)
            index_name = kwargs.pop("index_name", "langchain_index")
            index_key = kwargs.pop("index_key", None)
            api_key = kwargs.pop("api_key", None)
            base_url = kwargs.pop("base_url", "http://localhost:8000")

            cls._validate_index_key(index_key)
            if api_key is None:
                raise ValueError("api_key must be provided for CyborgDB.")

            # Extract optional parameters
            index_type = kwargs.pop("index_type", "ivfflat")
            metric = kwargs.pop("metric", "cosine")
            dimension = kwargs.pop("dimension", None)
            verify_ssl = kwargs.pop("verify_ssl", None)

            # Handle index config
            index_config_params = kwargs.pop("index_config_params", {})
            for key in {"pq_dim", "pq_bits"}:
                if key in kwargs:
                    index_config_params[key] = kwargs.pop(key)

            # Create vector store
            store = cls(
                index_name=index_name,
                index_key=index_key,
                api_key=api_key,
                base_url=base_url,
                embedding=embedding,
                index_type=index_type,
                index_config_params=index_config_params,
                dimension=dimension,
                metric=metric,
                verify_ssl=verify_ssl,
            )

            # Add texts if provided
            if texts:
                store.add_texts(texts, metadatas, ids=ids)

            if not store.index.is_trained():
                warnings.warn("Not enough data to train index.")

            return store

        @classmethod
        def from_documents(
            cls,
            documents: List[Document],
            embedding: Union[str, Embeddings, SentenceTransformer],
            **kwargs,
        ) -> "CyborgVectorStore":
            """
            Create a vector store from documents.

            Args:
                documents: List of documents to add
                embedding: Embedding model
                **kwargs: Additional arguments (see from_texts)

            Returns:
                CyborgVectorStore instance
            """
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return cls.from_texts(texts, embedding, metadatas, **kwargs)

    __all__ = ["CyborgVectorStore"]

except ImportError as e:
    # Handle missing dependencies gracefully
    CyborgVectorStore = None
    __all__ = []

    _original_error = str(e)

    def _missing_dependency_error():
        raise ImportError(
            f"To use the LangChain integration with cyborgdb-py, "
            f"please install the required dependencies: pip install cyborgdb-py[langchain]\n"
            f"Original error: {_original_error}"
        )

    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            _missing_dependency_error()

        def __getattr__(self, name):
            _missing_dependency_error()

    CyborgVectorStore = _MissingDependency
