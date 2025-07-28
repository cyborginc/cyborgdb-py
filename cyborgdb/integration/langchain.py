"""
LangChain integration for CyborgDB-py REST API.

This module requires the langchain-core package:
    pip install cyborgdb-py[langchain]
"""

import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Iterable
import warnings

try:
    from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    # Import from your cyborgdb-py SDK
    from cyborgdb import (
        Client,
        EncryptedIndex,
        IndexIVF,
        IndexIVFPQ, 
        IndexIVFFlat,
        generate_key
    )

    class CyborgVectorStore(VectorStore):
        """
        CyborgDB vector store for use with LangChain.
        
        This class implements the LangChain VectorStore interface to provide
        vector storage and retrieval with CyborgDB's encrypted index.
        """

        def __init__(self, 
                    index_name: str, 
                    index_key: bytes, 
                    api_key: str,
                    api_url: str,
                    embedding: Union[str, Embeddings, SentenceTransformer], 
                    index_type: str = "ivfflat", 
                    index_config_params: Optional[Dict[str, Any]] = None,
                    dimension: Optional[int] = None, 
                    metric: str = "cosine",
                    max_cache_size: int = 0) -> None:
            """
            Initialize a new CyborgVectorStore.
            
            Args:
                index_name: Name of the index
                index_key: 32-byte encryption key
                api_key: API key for CyborgDB
                api_url: URL of the CyborgDB API server
                embedding: Embedding model or function (string model name, SentenceTransformer, or LangChain Embeddings)
                index_type: Type of index ("ivfflat", "ivf", or "ivfpq")
                index_config_params: Additional parameters for index configuration
                dimension: Dimension of embeddings (if not provided, inferred from model)
                metric: Distance metric to use ("cosine", "euclidean", "squared_euclidean")
                max_cache_size: Maximum cache size for the index
            """
            
            # Store parameters
            self.index_name = index_name
            self.index_key = index_key
            self.max_cache_size = max_cache_size
            
            # Handle embedding model
            if isinstance(embedding, str):
                self.embedding_model_name = embedding
                self.embedding_model = None
            else:
                self.embedding_model = embedding
                self.embedding_model_name = getattr(embedding, 'model_name', '') if hasattr(embedding, 'model_name') else ''
            
            # Create configs
            index_config_params = index_config_params or {}
            
            # Create the client
            self.client = Client(
                api_url=api_url,
                api_key=api_key
            )
            
            # Check if index exists
            index_exists = False
            try:
                existing_indexes = self.client.list_indexes()
                index_exists = index_name in existing_indexes
            except Exception as e:
                # Fallback in case list_indexes isn't supported
                index_exists = False
                warnings.warn(
                    f"Could not verify if index '{index_name}' exists: {e}",
                    RuntimeWarning
                )
            
            if index_exists:
                # Try to load existing index
                try:
                    # Create a temporary encrypted index instance
                    from cyborgdb.client.encrypted_index import EncryptedIndex
                    self.index = EncryptedIndex(
                        index_name=index_name,
                        index_key=index_key,
                        api=self.client.api,
                        api_client=self.client.api_client,
                        max_cache_size=max_cache_size
                    )
                except Exception as e:
                    raise ValueError(f"Failed to load existing index: {e}")
            else:
                # Create new index
                # Determine embedding dimension
                if dimension is not None:
                    embedding_dim = dimension
                else:
                    if self.embedding_model is None and self.embedding_model_name:
                        self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    
                    if self.embedding_model is None:
                        raise RuntimeError("No embedding model provided and no dimension specified")
                    
                    # Determine embedding dimension from model
                    if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
                        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    elif hasattr(self.embedding_model, "embed_query"):
                        dummy = self.embedding_model.embed_query("dimension check")
                        embedding_dim = len(dummy) if isinstance(dummy, list) else np.asarray(dummy).shape[0]
                    else:
                        # Try to encode a dummy text
                        dummy = self.embedding_model.encode(["dimension check"])[0]
                        embedding_dim = len(dummy) if isinstance(dummy, list) else np.asarray(dummy).shape[0]

                # Build the index config
                if index_type not in ["ivfflat", "ivf", "ivfpq"]:
                    raise ValueError(f"Invalid index type: {index_type}. Must be one of ['ivfflat', 'ivf', 'ivfpq']")
                
                # Create the appropriate index config
                if index_type == "ivf":
                    n_lists = index_config_params.get("n_lists", 1024)
                    config = IndexIVF(dimension=embedding_dim, n_lists=n_lists, metric=metric)
                elif index_type == "ivfpq":
                    n_lists = index_config_params.get("n_lists", 1024)
                    pq_dim = index_config_params.get("pq_dim", 8)
                    pq_bits = index_config_params.get("pq_bits", 8)
                    config = IndexIVFPQ(dimension=embedding_dim, n_lists=n_lists, pq_dim=pq_dim, pq_bits=pq_bits, metric=metric)
                else:  # ivfflat
                    n_lists = index_config_params.get("n_lists", 1024)
                    config = IndexIVFFlat(dimension=embedding_dim, n_lists=n_lists, metric=metric)

                # Create the index
                self.index = self.client.create_index(
                    index_name=index_name,
                    index_key=index_key,
                    index_config=config,
                    embedding_model=self.embedding_model_name if self.embedding_model_name else None,
                    max_cache_size=max_cache_size
                )
            
        def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
            """
            Generate embeddings for the given texts.
            """
            # Lazy load by name if needed
            if self.embedding_model is None and self.embedding_model_name:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            if self.embedding_model is None:
                raise RuntimeError("No embedding model available")

            is_single = isinstance(texts, str)
            texts_list = [texts] if is_single else texts

            # 1) SentenceTransformer path
            if hasattr(self.embedding_model, 'encode') and hasattr(self.embedding_model, 'get_sentence_embedding_dimension'):
                embeddings = self.embedding_model.encode(texts_list, convert_to_numpy=True)

            # 2) LangChain Embeddings path
            elif hasattr(self.embedding_model, 'embed_documents') and hasattr(self.embedding_model, 'embed_query'):
                if is_single:
                    raw = self.embedding_model.embed_query(texts)
                    embeddings = np.array(raw, dtype=np.float32)[None, :]
                else:
                    raw = self.embedding_model.embed_documents(texts_list)
                    embeddings = np.array(raw, dtype=np.float32)
            
            # 3) Generic callable
            elif callable(self.embedding_model):
                embeddings = self.embedding_model(texts_list)
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings, dtype=np.float32)
            else:
                raise TypeError(
                    f"Unsupported embedding model type: {type(self.embedding_model)}. "
                    "Must be SentenceTransformer, LangChain Embeddings, or callable."
                )

            # If single-text, return 1-D array
            if is_single:
                return embeddings[0]
            return embeddings

        def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None, **kwargs) -> List[str]:
            """
            Add texts to the vector store.
            """
            texts_list = list(texts)
            num_texts = len(texts_list)
            
            # Return early if no texts
            if num_texts == 0:
                return []
            
            # Validate or generate IDs
            if ids is not None:
                if len(ids) != num_texts:
                    raise ValueError("Length of ids must match length of texts.")
                id_list = list(ids)
            else:
                id_list = [str(uuid.uuid4()) for _ in range(num_texts)]
            
            # Validate metadata length if provided
            if metadatas is not None and len(metadatas) != num_texts:
                raise ValueError("Length of metadatas must match length of texts.")
            
            # Generate embeddings
            embeddings = self.get_embeddings(texts_list)
            
            # Process metadata
            items = []
            for i, (text, doc_id) in enumerate(zip(texts_list, id_list)):
                item = {
                    "id": doc_id,
                    "contents": text,
                    "vector": embeddings[i].tolist() if len(embeddings.shape) > 1 else embeddings.tolist()
                }
                
                if metadatas is not None and metadatas[i]:
                    item["metadata"] = metadatas[i]
                
                items.append(item)
            
            # Upsert items to the index
            self.index.upsert(items)
            
            return id_list
        
        def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs) -> List[str]:
            """
            Add documents to the vector store.
            """
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            return self.add_texts(texts, metadatas, ids=ids, **kwargs)
        
        def delete(self, ids: Optional[List[str]] = None, delete_index: bool = False) -> bool:
            """
            Delete documents from the vector store or delete the entire index.
            """
            try:
                if delete_index:
                    self.index.delete_index()
                elif ids is not None and len(ids) > 0:
                    self.index.delete(ids)
                else:
                    return False
                return True
            except Exception as e:
                warnings.warn(f"Delete operation failed: {e}")
                return False

        def _execute_query(self, query, k=4, filter=None, n_probes=1) -> List[Dict[str, Any]]:
            """Helper to execute a search query and process the results"""
            filter = filter or {}
            
            if isinstance(query, str):
                # Text query - get embeddings first
                embedding = self.get_embeddings(query)
                results = self.index.query(
                    query_vector=embedding, 
                    top_k=k,
                    n_probes=n_probes,
                    filters=filter,
                    include=["distance", "metadata", "contents"]
                )
            else:
                # Embedding query
                results = self.index.query(
                    query_vector=query, 
                    top_k=k,
                    n_probes=n_probes,
                    filters=filter,
                    include=["distance", "metadata", "contents"]
                )
            
            # Handle the results format
            if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
                # Batch query result - take the first result
                results = results[0]
            
            return results if results else []
        
        def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
            """
            Return documents most similar to query.
            """
            n_probes = kwargs.get('n_probes', 1)
            results = self._execute_query(query, k, filter, n_probes)
            
            # Convert to Documents
            docs = []
            for item in results:
                content = item.get("contents", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                    
                # Handle metadata
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {"raw": metadata}
                        
                docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            return docs
        
        def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Tuple[Document, float]]:
            """
            Return documents most similar to query along with relevance scores.
            """
            n_probes = kwargs.get('n_probes', 1)
            results = self._execute_query(query, k, filter, n_probes)
            
            # Convert to Documents with scores
            docs_with_scores = []
            for item in results:
                content = item.get("contents", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                
                # Handle metadata
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {"raw": metadata}
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                # Get distance and normalize to similarity
                distance = item.get("distance", 0.0)
                similarity = self._normalize_score(distance)
                
                docs_with_scores.append((doc, similarity))
            
            return docs_with_scores
        
        def _normalize_score(self, distance: float) -> float:
            """
            Normalize a distance score to a similarity score in the range [0, 1].
            """
            # Get the metric from the index config
            try:
                config = self.index.index_config
                if isinstance(config, dict):
                    metric = config.get('metric', 'cosine')
                else:
                    metric = getattr(config, 'metric', 'cosine')
            except:
                metric = 'cosine'
            
            if metric == "cosine":
                # Cosine distance: 0 (identical) to 2 (opposite)
                return max(0.0, 1.0 - (distance / 2.0))
            elif metric == "euclidean":
                # Euclidean: use exponential decay
                return np.exp(-distance)
            elif metric == "squared_euclidean":
                # Squared Euclidean: use exponential decay with sqrt
                return np.exp(-np.sqrt(distance))
            else:
                # Default normalization
                return 1.0 / (1.0 + distance)
        
        def similarity_search_by_vector(self, embedding: List[float], k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
            """
            Return documents most similar to embedding vector.
            """
            n_probes = kwargs.get('n_probes', 1)
            results = self._execute_query(embedding, k, filter, n_probes)
            
            # Convert to Documents
            docs = []
            for item in results:
                content = item.get("contents", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                    
                # Handle metadata
                metadata = item.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {"raw": metadata}
                        
                docs.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            return docs
        
        def as_retriever(
            self,
            search_type: Optional[str] = None,
            search_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> VectorStoreRetriever:
            """Return a retriever object for this vectorstore."""
            
            return VectorStoreRetriever(
                vectorstore=self,
                search_type=search_type or "similarity",
                search_kwargs=search_kwargs or {},
                **kwargs,
            )        
        
        @classmethod
        def from_texts(cls, texts: List[str], embedding: Union[str, Embeddings, SentenceTransformer], metadatas: Optional[List[Dict]] = None, **kwargs) -> "CyborgVectorStore":
            """
            Create a vector store from texts.
            """
            ids = kwargs.pop("ids", None)
            
            # Extract required parameters
            index_name = kwargs.pop("index_name", "langchain_index")
            index_key = kwargs.pop("index_key", None)
            api_key = kwargs.pop("api_key", None)
            api_url = kwargs.pop("api_url", "https://api.cyborgdb.com")

            if index_key is None:
                raise ValueError(
                    "index_key must be provided for CyborgVectorStore. "
                    "Use generate_key() to generate a secure 32-byte key."
                )
            
            if api_key is None:
                raise ValueError("api_key must be provided for CyborgVectorStore.")
            
            # Extract other parameters
            index_type = kwargs.pop("index_type", "ivfflat")
            metric = kwargs.pop("metric", "cosine")
            dimension = kwargs.pop("dimension", None)
            max_cache_size = kwargs.pop("max_cache_size", 0)
            
            # Handle index config parameters
            index_config_params = kwargs.pop("index_config_params", {})
            for key in {"n_lists", "pq_dim", "pq_bits"}:
                if key in kwargs:
                    index_config_params[key] = kwargs.pop(key)
            
            # Create the vector store
            store = cls(
                index_name=index_name,
                index_key=index_key,
                api_key=api_key,
                api_url=api_url,
                embedding=embedding,
                index_type=index_type,
                index_config_params=index_config_params,
                dimension=dimension,
                metric=metric,
                max_cache_size=max_cache_size
            )
            
            # Add texts if provided
            if texts:
                store.add_texts(texts, metadatas, ids=ids)
            
            # Train the index if needed and there are enough documents
            n_lists = index_config_params.get("n_lists", 1024)
            try:
                if not store.index.is_trained() and len(texts) >= 2 * n_lists:
                    store.index.train()
            except Exception as e:
                warnings.warn(f"Could not train index: {e}")
            
            return store
        
        @classmethod
        def from_documents(cls, documents: List[Document], embedding: Union[str, Embeddings, SentenceTransformer], **kwargs) -> "CyborgVectorStore":
            """
            Create a vector store from documents.
            """
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return cls.from_texts(texts, embedding, metadatas, **kwargs)

        # Async variants for compatibility
        async def aadd_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None, **kwargs) -> List[str]:
            """Async version of add_texts"""
            import asyncio
            return await asyncio.to_thread(self.add_texts, texts, metadatas=metadatas, ids=ids, **kwargs)

        async def asimilarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
            """Async version of similarity_search"""
            import asyncio
            return await asyncio.to_thread(self.similarity_search, query, k, filter, **kwargs)
        
        async def asimilarity_search_with_score(self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Tuple[Document, float]]:
            """Async version of similarity_search_with_score"""
            import asyncio
            return await asyncio.to_thread(self.similarity_search_with_score, query, k, filter, **kwargs)
        
        async def aadd_documents(self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs) -> List[str]:
            """Async version of add_documents"""
            import asyncio
            return await asyncio.to_thread(self.add_documents, documents, ids=ids, **kwargs)

        async def adelete(self, ids: Optional[List[str]] = None, delete_index: bool = False) -> bool:
            """Async version of delete"""
            import asyncio
            return await asyncio.to_thread(self.delete, ids=ids, delete_index=delete_index)

        async def asimilarity_search_by_vector(self, embedding: List[float], k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
            """Async version of similarity_search_by_vector"""
            import asyncio
            return await asyncio.to_thread(self.similarity_search_by_vector, embedding, k, filter, **kwargs)

    __all__ = ['CyborgVectorStore']

except ImportError as e:
    CyborgVectorStore = None
    __all__ = []
    
    # Store the original error
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
    
    # Replace with a class that raises a helpful error
    CyborgVectorStore = _MissingDependency