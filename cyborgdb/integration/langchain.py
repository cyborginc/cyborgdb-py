"""
LangChain integration for CyborgDB-py.

This module requires the langchain-core package:
    pip install cyborgdb-py[langchain]
"""

import uuid
import sys
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
        IndexConfig,
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
                    embedding: Union[str, Embeddings, SentenceTransformer], 
                    index_location: str, 
                    config_location: str, 
                    items_location: Optional[str] = None, 
                    index_type: str = "ivfflat", 
                    index_config_params: Optional[Dict[str, Any]] = None,
                    dimension: Optional[int] = None, 
                    metric: str = "cosine") -> None:
            """
            Initialize a new CyborgVectorStore.
            
            Args:
                index_name: Name of the index
                index_key: 32-byte encryption key
                api_key: API key for CyborgDB
                embedding: Embedding model or function (string model name, SentenceTransformer, or LangChain Embeddings)
                index_location: Location for index data
                config_location: Location for index configuration
                items_location: Optional location for item data
                index_type: Type of index ("ivfflat", "ivf", or "ivfpq")
                index_config_params: Additional parameters for index configuration
                dimension: Dimension of embeddings (if not provided, inferred from model)
                metric: Distance metric to use ("cosine", "euclidean", "squared_euclidean")
            """
            
            # Store parameters
            self.index_name = index_name
            self.index_key = index_key
            
            # Handle embedding model
            if isinstance(embedding, str):
                self.embedding_model_name = embedding
                self.embedding_model = None
            else:
                self.embedding_model = embedding
                self.embedding_model_name = ""
            
            # Create configs
            index_config_params = index_config_params or {}
            
            # Create the client and index
            self.client = Client(
                api_key=api_key,
                index_location=index_location,
                config_location=config_location,
                items_location=items_location or "memory"
            )
            
            # Check if index exists
            index_exists = False
            try:
                existing_indexes = self.client.list_indexes()
                index_exists = index_name in existing_indexes
            except Exception:
                # Fallback in case ListIndexes isn't supported by the backend
                index_exists = False
                warnings.warn(
                    f"Could not verify if index '{index_name}' exists. Proceeding to create a new one.",
                    RuntimeWarning
                )
            
            if index_exists:
                self.index = self.client.load_index(index_name, index_key)
                # Try to extract model name from index config if available
                try:
                    config_model_name = self.index.get_config().get("embedding_model")
                    if config_model_name:
                        self.embedding_model_name = config_model_name
                        if self.embedding_model is not None and hasattr(self.embedding_model, "__str__"):
                            if str(self.embedding_model) != self.embedding_model_name:
                                self.embedding_model = None
                except:
                    pass
            else:
                # Determine embedding dimension
                if dimension is not None and embedding is None:
                    embedding_dim = dimension
                else:
                    if self.embedding_model is None and self.embedding_model_name:
                        self.embedding_model = SentenceTransformer(self.embedding_model_name)
                    
                    if self.embedding_model is None:
                        raise RuntimeError("No embedding model provided and no dimension specified")
                    
                    # Determine embedding dimension from model
                    if hasattr(self.embedding_model, "get_sentence_embedding_dimension"):
                        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    else:
                        dummy = self.embedding_model.embed_query("dimension check")
                        embedding_dim = len(dummy) if isinstance(dummy, list) else np.asarray(dummy).shape[0]

                # Build the index config
                if index_type not in ["ivfflat", "ivf", "ivfpq"]:
                    raise ValueError(f"Invalid index type: {index_type}. Must be one of ['ivfflat', 'ivf', 'ivfpq']")
                
                # Create the appropriate index config
                if index_type == "ivf":
                    n_lists = index_config_params.get("n_lists", 1024)
                    config = IndexIVF(embedding_dim, n_lists, metric)
                elif index_type == "ivfpq":
                    n_lists = index_config_params.get("n_lists", 1024)
                    pq_dim = index_config_params.get("pq_dim", 8)
                    pq_bits = index_config_params.get("pq_bits", 8)
                    config = IndexIVFPQ(embedding_dim, n_lists, pq_dim, pq_bits, metric)
                else:  # ivfflat
                    n_lists = index_config_params.get("n_lists", 1024)
                    config = IndexIVFFlat(embedding_dim, n_lists, metric)

                # Create the index
                self.index = self.client.create_index(
                    index_name=index_name,
                    index_key=index_key,
                    index_config=config,
                    embedding_model=self.embedding_model_name
                )
            
        def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
            """
            Generate embeddings for the given texts using either:
            - A SentenceTransformer
            - Any LangChain Embeddings implementation
            """
            # Lazy load by name if needed
            if self.embedding_model is None and self.embedding_model_name:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)

            if self.embedding_model is None:
                raise RuntimeError("No embedding model available")

            is_single = isinstance(texts, str)
            texts_list = [texts] if is_single else texts

            # 1) SentenceTransformer path
            if isinstance(self.embedding_model, SentenceTransformer):
                # SentenceTransformer.encode always returns NumPy if requested
                embeddings = self.embedding_model.encode(texts_list, convert_to_numpy=True)

            # 2) LangChain Embeddings path
            elif isinstance(self.embedding_model, Embeddings):
                # embed_query or embed_documents returns List[float] or List[List[float]]
                if is_single:
                    raw = self.embedding_model.embed_query(texts)
                    embeddings = np.array(raw, dtype=np.float32)[None, :]
                else:
                    raw = self.embedding_model.embed_documents(texts_list)
                    embeddings = np.array(raw, dtype=np.float32)

            else:
                raise TypeError(
                    f"Unsupported embedding model type: {type(self.embedding_model)}. "
                    "Must be SentenceTransformer or a LangChain Embeddings."
                )

            # If single-text, return 1-D array
            if is_single:
                return embeddings[0]
            return embeddings

        def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, ids: Optional[List[str]] = None, **kwargs) -> List[str]:
            """
            Add texts to the vector store.
            
            Args:
                texts: Iterable of strings to add
                metadatas: Optional list of metadata dictionaries
                ids: Optional list of document IDs
                
            Returns:
                List of IDs of the added texts
            """
            # Convert texts to a list if it's not already
            texts_list = list(texts)
            num_texts = len(texts_list)
            
            # Validate or generate IDs
            if ids is not None:
                if len(ids) != num_texts:
                    raise ValueError("Length of ids must match length of texts.")
                id_list = list(ids)
            else:
                id_list = [str(uuid.uuid4()) for _ in range(num_texts)]
            
            # Generate embeddings
            embeddings = self.get_embeddings(texts_list)
            
            # Process metadata if provided
            items = []
            for i, (text, doc_id) in enumerate(zip(texts_list, id_list)):
                item = {
                    "id": doc_id,
                    "contents": text,
                    "vector": embeddings[i].tolist() if len(embeddings.shape) > 1 else embeddings.tolist()
                }
                
                if metadatas is not None and i < len(metadatas):
                    item["metadata"] = metadatas[i] or {}
                
                items.append(item)
            
            # Upsert items to the index
            self.index.upsert(items)
            
            return id_list
        
        def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs) -> List[str]:
            """
            Add documents to the vector store.
            
            Args:
                documents: List of Document objects
                ids: Optional list of document IDs
                
            Returns:
                List of IDs of the added documents
            """
            # Extract texts and metadata from documents
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Call add_texts
            return self.add_texts(texts, metadatas, ids=ids, **kwargs)
        
        def delete(self, ids: Optional[List[str]] = None, delete_index: bool = False) -> bool:
            """
            Delete documents from the vector store or delete the entire index.
            
            Args:
                ids: Optional list of IDs to delete
                delete_index: If True, deletes the entire index regardless of `ids`
                
            Returns:
                Success of deletion
            """
            try:
                if delete_index:
                    # Delete the entire index
                    self.index.delete_index()
                elif ids is not None:
                    # Delete specified documents
                    self.index.delete(ids)
                else:
                    # No action if delete_index is False and no ids provided
                    return False
                return True
            except Exception:
                return False

        def _execute_query(self, query, k=4, filter=None) -> Tuple[List[Dict[str, Any]], List[float]]:
            """Helper to execute a search query and process the results"""
            filter = filter or {}
            
            if isinstance(query, str):
                # Text query - get embeddings first
                embedding = self.get_embeddings(query)
                results = self.index.query(
                    query_vectors=embedding, 
                    top_k=k,
                    filters=filter,
                    include=["distance", "metadata", "contents"]
                )
            else:
                # Embedding query
                results = self.index.query(
                    query_vectors=query, 
                    top_k=k,
                    filters=filter,
                    include=["distance", "metadata", "contents"]
                )
            
            # If no results, return empty
            if not results or len(results) == 0:
                return [], []
            
            # Get contents for results if needed
            result_ids = [r["id"] for r in results]
            
            # Get full items from the index if contents are not in the results
            if "contents" not in results[0]:
                items = self.index.get(result_ids, include=["contents", "metadata"])
                # Map items by ID for easy lookup
                items_by_id = {item["id"]: item for item in items}
                
                # Add contents to results
                for r in results:
                    if r["id"] in items_by_id:
                        r["contents"] = items_by_id[r["id"]].get("contents", "")
                        r["metadata"] = items_by_id[r["id"]].get("metadata", {})
            
            # Process the results
            items = []
            distances = []
            
            for r in results:
                items.append(r)
                distances.append(r.get("distance", 0.0))
            
            return items, distances
        
        def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
            """
            Return documents most similar to query.
            
            Args:
                query: Query text
                k: Number of documents to return
                filter: Optional metadata filters
                
            Returns:
                List of Documents
            """
            items, _ = self._execute_query(query, k, filter)
            
            # Convert to Documents
            docs = []
            for item in items:
                content = item.get("contents", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                docs.append(Document(
                    page_content=content,
                    metadata=item.get("metadata", {})
                ))
            
            return docs
        
        def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Tuple[Document, float]]:
            """
            Return documents most similar to query along with relevance scores.
            
            Args:
                query: Query text
                k: Number of documents to return
                filter: Optional metadata filters
                
            Returns:
                List of (Document, score) tuples
            """
            items, distances = self._execute_query(query, k, filter)
            
            # Convert to Documents with scores
            docs_with_scores = []
            for i, item in enumerate(items):
                content = item.get("contents", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                
                doc = Document(
                    page_content=content,
                    metadata=item.get("metadata", {})
                )
                
                # Normalize the distance to a similarity score
                similarity = self._normalize_score(distances[i])
                
                docs_with_scores.append((doc, similarity))
            
            return docs_with_scores
        
        def _normalize_score(self, distance: float) -> float:
            """
            Normalize a distance score to a similarity score in the range [0, 1].
            
            Args:
                distance: Raw distance score from the index
                
            Returns:
                Normalized score where 1 is most similar and 0 is least similar
            """
            # Get the metric used by the index - adjust this based on your API
            try:
                metric = getattr(self.index, 'metric', 'cosine')
            except:
                metric = 'cosine'  # default fallback
            
            if metric == "cosine":
                # Cosine distance: 0 (identical) to 2 (opposite), convert to 1 to 0
                return 1.0 - (distance / 2.0)
            elif metric == "euclidean":
                # Euclidean: normalize based on embedding size
                max_distance = 2.0  # For normalized embeddings
                return 1.0 - min(distance / max_distance, 1.0)
            elif metric == "squared_euclidean":
                # Euclidean: normalize based on embedding size
                max_distance = 4.0  # For normalized embeddings
                return 1.0 - min(distance / max_distance, 1.0)
            else:
                # Default normalization, assuming 0 = identical, higher = more different
                return 1.0 / (1.0 + distance)
        
        def similarity_search_by_vector(self, embedding: List[float], k: int = 4, filter: Optional[Dict] = None, **kwargs) -> List[Document]:
            """
            Return documents most similar to embedding vector.
            
            Args:
                embedding: Embedding vector
                k: Number of documents to return
                filter: Optional metadata filters
                
            Returns:
                List of Documents
            """
            items, _ = self._execute_query(embedding, k, filter)
            
            # Convert to Documents
            docs = []
            for item in items:
                content = item.get("contents", "")
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                docs.append(Document(
                    page_content=content,
                    metadata=item.get("metadata", {})
                ))
            
            return docs
        
        def as_retriever(
            self,
            search_type: Optional[str] = None,
            search_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> VectorStoreRetriever:
            """Return a retriever object for this vectorstore.

            Args:
                search_type: Type of search to perform. Defaults to "similarity".
                search_kwargs: Keyword arguments to pass to the search function.
                **kwargs: Additional keyword arguments to pass to the retriever.

            Returns:
                A retriever object.
            """
            
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
            
            Args:
                texts: List of strings
                embedding: Embedding function
                metadatas: Optional list of metadata dictionaries
                **kwargs: Additional arguments for vector store creation
                
            Returns:
                CyborgVectorStore
            """
            ids = kwargs.pop("ids", None)
            
            # Extract required parameters
            index_name = kwargs.pop("index_name", "langchain_index")
            index_key = kwargs.pop("index_key", None)
            api_key = kwargs.pop("api_key", None)

            if index_key is None:
                raise ValueError(
                    "index_key must be provided for CyborgVectorStore. "
                    "Use generate_key() to generate a secure 32-byte key."
                )
            
            index_location = kwargs.pop("index_location")
            config_location = kwargs.pop("config_location")
            index_type = kwargs.pop("index_type", "ivfflat")
            metric = kwargs.pop("metric", "cosine")
            items_location = kwargs.pop("items_location", None)
            dimension = kwargs.pop("dimension", None)
            
            # Handle index config parameters
            index_config_params = kwargs.pop("index_config_params", {})
            for key in {"n_lists", "pq_dim", "pq_bits"}:
                if key in kwargs:
                    index_config_params[key] = kwargs.pop(key)
            
            # Create documents if metadatas provided
            if metadatas is None:
                metadatas = [{} for _ in texts]
            
            # Create the vector store
            store = cls(
                index_name=index_name,
                index_key=index_key,
                api_key=api_key,
                embedding=embedding,
                index_location=index_location,
                config_location=config_location,
                items_location=items_location,
                index_type=index_type,
                index_config_params=index_config_params,
                dimension=dimension,
                metric=metric
            )
            
            # Add texts
            store.add_texts(texts, metadatas, ids=ids)
            
            # Train the index if needed and there are enough documents
            n_lists = index_config_params.get("n_lists", 1024)
            try:
                if not store.index.is_trained() and len(texts) >= 2 * n_lists:
                    store.index.train()
            except:
                pass  # Training might not be available in all versions
            
            return store
        
        @classmethod
        def from_documents(cls, documents: List[Document], embedding: Union[str, Embeddings, SentenceTransformer], **kwargs) -> "CyborgVectorStore":
            """
            Create a vector store from documents.
            
            Args:
                documents: List of Document objects
                embedding: Embedding function
                **kwargs: Additional arguments for vector store creation
                
            Returns:
                CyborgVectorStore
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