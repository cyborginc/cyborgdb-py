"""
LangChain integration tests for CyborgDB-py.

This module tests the LangChain VectorStore implementation for CyborgDB.
"""

import unittest
import os
import json
import numpy as np
import asyncio
from typing import List, Dict, Any

# Test imports
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import CyborgDB components
import cyborgdb
from cyborgdb.integration.langchain import CyborgVectorStore


# Mock embedding class for testing
class MockEmbeddings(Embeddings):
    """Mock embeddings for testing that generates deterministic vectors."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text content
            hash_val = hash(text) % 1000000
            np.random.seed(hash_val)
            embedding = np.random.randn(self.dimension).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Generate deterministic embedding for a single query."""
        return self.embed_documents([text])[0]


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "LangChain dependencies not available")
class TestLangChainIntegration(unittest.TestCase):
    """Test suite for CyborgDB LangChain integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Test parameters
        cls.dimension = 384
        cls.api_url = "https://localhost:8000"
        cls.api_key = os.getenv("CYBORGDB_API_KEY", "cyborg_5e5be271a8884c10a6c96caa68870e74")
        
        # Test data
        cls.test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Vector databases are optimized for similarity search.",
            "Embeddings represent text as dense numerical vectors.",
            "Transformer models have revolutionized NLP tasks.",
            "RAG combines retrieval with language generation.",
            "LangChain simplifies building LLM applications."
        ]
        
        cls.test_metadata = [
            {"category": "animals", "source": "proverb"},
            {"category": "AI", "source": "textbook"},
            {"category": "programming", "source": "tutorial"},
            {"category": "AI", "source": "research"},
            {"category": "AI", "source": "textbook"},
            {"category": "database", "source": "documentation"},
            {"category": "AI", "source": "research"},
            {"category": "AI", "source": "paper"},
            {"category": "AI", "source": "blog"},
            {"category": "programming", "source": "documentation"}
        ]
        
        # Create test documents
        cls.test_documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(cls.test_texts, cls.test_metadata)
        ]
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        pass
    
    def setUp(self):
        """Set up for each test."""
        self.index_key = cyborgdb.generate_key()
        self.index_names_to_cleanup = []
    
    def tearDown(self):
        """Clean up after each test."""
        # Clean up any created indexes
        try:
            client = cyborgdb.Client(
                api_url=self.api_url,
                api_key=self.api_key
            )
            for index_name in self.index_names_to_cleanup:
                try:
                    # Create a temporary encrypted index instance to delete it
                    from cyborgdb.client.encrypted_index import EncryptedIndex
                    index = EncryptedIndex(
                        index_name=index_name,
                        index_key=self.index_key,
                        api=client.api,
                        api_client=client.api_client,
                        max_cache_size=0
                    )
                    index.delete_index()
                except:
                    pass
        except:
            pass
    
    def test_01_create_vectorstore_with_mock_embeddings(self):
        """Test creating a vector store with mock embeddings."""
        index_name = "langchain_test_mock_embeddings"
        self.index_names_to_cleanup.append(index_name)
        
        # Create vector store with mock embeddings
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="cosine",
            index_config_params={"n_lists": 10}
        )
        
        # Add texts
        ids = vectorstore.add_texts(
            texts=self.test_texts[:5],
            metadatas=self.test_metadata[:5]
        )
        
        self.assertEqual(len(ids), 5)
        
        # Test similarity search
        results = vectorstore.similarity_search("artificial intelligence", k=3)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], Document)
    
    def test_02_create_vectorstore_with_sentence_transformer(self):
        """Test creating a vector store with SentenceTransformer."""
        index_name = "langchain_test_sentence_transformer"
        self.index_names_to_cleanup.append(index_name)
        
        # Use a small pre-trained model for faster testing
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Create vector store with SentenceTransformer model name
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=model_name,
            index_type="ivfflat",
            metric="cosine",
            index_config_params={"n_lists": 10}
        )
        
        # Add documents
        ids = vectorstore.add_documents(self.test_documents[:5])
        self.assertEqual(len(ids), 5)
        
        # Test similarity search with score
        results_with_scores = vectorstore.similarity_search_with_score(
            "What is machine learning?", 
            k=3
        )
        
        self.assertEqual(len(results_with_scores), 3)
        for doc, score in results_with_scores:
            self.assertIsInstance(doc, Document)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_03_metadata_filtering(self):
        """Test metadata filtering in searches."""
        index_name = "langchain_test_metadata_filter"
        self.index_names_to_cleanup.append(index_name)
        
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="euclidean",
            index_config_params={"n_lists": 10}
        )
        
        # Add all test documents
        vectorstore.add_documents(self.test_documents)
        
        # Search with metadata filter
        ai_results = vectorstore.similarity_search(
            "artificial intelligence",
            k=10,
            filter={"category": "AI"}
        )
        
        # Verify all results have the correct category
        for doc in ai_results:
            self.assertEqual(doc.metadata.get("category"), "AI")
        
        # Search with multiple metadata conditions
        research_results = vectorstore.similarity_search(
            "neural networks",
            k=10,
            filter={"category": "AI", "source": "research"}
        )
        
        for doc in research_results:
            self.assertEqual(doc.metadata.get("category"), "AI")
            self.assertEqual(doc.metadata.get("source"), "research")
    
    def test_04_similarity_search_by_vector(self):
        """Test similarity search using a vector directly."""
        index_name = "langchain_test_vector_search"
        self.index_names_to_cleanup.append(index_name)
        
        embeddings = MockEmbeddings(self.dimension)
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=embeddings,
            index_type="ivfflat",
            metric="cosine"
        )
        
        # Add documents
        vectorstore.add_documents(self.test_documents)
        
        # Get embedding for a query
        query_embedding = embeddings.embed_query("machine learning algorithms")
        
        # Search by vector
        results = vectorstore.similarity_search_by_vector(
            embedding=query_embedding,
            k=5
        )
        
        self.assertEqual(len(results), 5)
        for doc in results:
            self.assertIsInstance(doc, Document)
    
    def test_05_delete_operations(self):
        """Test delete operations."""
        index_name = "langchain_test_delete"
        self.index_names_to_cleanup.append(index_name)
        
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat"
        )
        
        # Add texts with specific IDs
        ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        vectorstore.add_texts(
            texts=self.test_texts[:5],
            metadatas=self.test_metadata[:5],
            ids=ids
        )
        
        # Delete specific documents
        success = vectorstore.delete(ids=["doc2", "doc4"])
        self.assertTrue(success)
        
        # Verify deletion by searching
        results = vectorstore.similarity_search("machine learning", k=10)
        result_texts = [doc.page_content for doc in results]
        
        # doc2 text should not be in results
        self.assertNotIn(self.test_texts[1], result_texts)
    
    def test_06_from_texts_classmethod(self):
        """Test creating vector store from texts using class method."""
        index_name = "langchain_test_from_texts"
        self.index_names_to_cleanup.append(index_name)
        
        vectorstore = CyborgVectorStore.from_texts(
            texts=self.test_texts,
            embedding=MockEmbeddings(self.dimension),
            metadatas=self.test_metadata,
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            index_type="ivfflat",
            metric="cosine"
        )
        
        # Verify the store was created and populated
        results = vectorstore.similarity_search("programming", k=3)
        self.assertGreater(len(results), 0)
    
    def test_07_from_documents_classmethod(self):
        """Test creating vector store from documents using class method."""
        index_name = "langchain_test_from_documents"
        self.index_names_to_cleanup.append(index_name)
        
        vectorstore = CyborgVectorStore.from_documents(
            documents=self.test_documents,
            embedding=MockEmbeddings(self.dimension),
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            index_type="ivfflat",
            n_lists=10
        )
        
        # Verify the store was created and populated
        results = vectorstore.similarity_search("database", k=3)
        self.assertGreater(len(results), 0)
    
    def test_08_as_retriever(self):
        """Test using vector store as a retriever."""
        index_name = "langchain_test_retriever"
        self.index_names_to_cleanup.append(index_name)
        
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat"
        )
        
        # Add documents
        vectorstore.add_documents(self.test_documents)
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Use retriever
        docs = retriever.get_relevant_documents("What is deep learning?")
        
        self.assertEqual(len(docs), 3)
        for doc in docs:
            self.assertIsInstance(doc, Document)
    
    def test_09_async_operations(self):
        """Test async operations."""
        index_name = "langchain_test_async"
        self.index_names_to_cleanup.append(index_name)
        
        async def run_async_tests():
            vectorstore = CyborgVectorStore(
                index_name=index_name,
                index_key=self.index_key,
                api_key=self.api_key,
                api_url=self.api_url,
                embedding=MockEmbeddings(self.dimension),
                index_type="ivfflat"
            )
            
            # Async add texts
            ids = await vectorstore.aadd_texts(
                texts=self.test_texts[:5],
                metadatas=self.test_metadata[:5]
            )
            self.assertEqual(len(ids), 5)
            
            # Async similarity search
            results = await vectorstore.asimilarity_search(
                "machine learning",
                k=3
            )
            self.assertEqual(len(results), 3)
            
            # Async similarity search with score
            results_with_scores = await vectorstore.asimilarity_search_with_score(
                "artificial intelligence",
                k=2
            )
            self.assertEqual(len(results_with_scores), 2)
            
            # Async delete
            success = await vectorstore.adelete(ids=[ids[0]])
            self.assertTrue(success)
        
        # Run async tests
        asyncio.run(run_async_tests())
    
    def test_10_train_index(self):
        """Test training the index when enough vectors are present."""
        index_name = "langchain_test_train"
        self.index_names_to_cleanup.append(index_name)
        
        # Create with enough data to train
        n_lists = 4
        min_vectors_for_training = 2 * n_lists
        
        # Generate more test data
        additional_texts = [f"Test document number {i}" for i in range(20)]
        additional_metadata = [{"index": i} for i in range(20)]
        
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            index_config_params={"n_lists": n_lists}
        )
        
        # Add enough documents to train
        all_texts = self.test_texts + additional_texts
        all_metadata = self.test_metadata + additional_metadata
        
        vectorstore.add_texts(
            texts=all_texts[:min_vectors_for_training + 5],
            metadatas=all_metadata[:min_vectors_for_training + 5]
        )
        
        # Train the index
        vectorstore.index.train()
        
        # Verify it's trained
        self.assertTrue(vectorstore.index.is_trained())
        
        # Test search on trained index
        results = vectorstore.similarity_search("test document", k=5)
        self.assertGreater(len(results), 0)
    
    def test_11_edge_cases(self):
        """Test edge cases and error handling."""
        index_name = "langchain_test_edge_cases"
        self.index_names_to_cleanup.append(index_name)
        
        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat"
        )
        
        # Test empty search results
        results = vectorstore.similarity_search("xyz123abc456", k=5)
        self.assertIsInstance(results, list)
        
        # Test adding empty list
        ids = vectorstore.add_texts(texts=[], metadatas=[])
        self.assertEqual(len(ids), 0)
        
        # Test mismatched texts and metadata lengths
        with self.assertRaises(Exception):
            vectorstore.add_texts(
                texts=["text1", "text2"],
                metadatas=[{"meta": 1}],  # Only one metadata for two texts
                ids=["id1", "id2"]  # Correct number of IDs
            )
        
        # Test delete with no IDs
        success = vectorstore.delete(ids=None, delete_index=False)
        self.assertFalse(success)


if __name__ == '__main__':
    unittest.main()