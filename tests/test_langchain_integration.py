"""
LangChain integration tests for CyborgDB-py.

This module tests the LangChain VectorStore implementation for CyborgDB.
"""

import unittest
import os
import json
import numpy as np
import asyncio
from typing import List
from dotenv import load_dotenv
import cyborgdb
from cyborgdb.integrations.langchain import CyborgVectorStore

# Load environment variables from .env.local
load_dotenv(".env.local")

# Test imports
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Define a dummy Embeddings class if langchain is not available
    class Embeddings:
        pass


# Mock embedding class for testing
if LANGCHAIN_AVAILABLE:

    class MockEmbeddings(Embeddings):
        """Mock embeddings for testing that generates semantically meaningful vectors."""

        def __init__(self, dimension: int = 384):
            self.dimension = dimension
            # Create a simple vocabulary for keyword extraction
            self.vocab = {}
            self.vocab_size = 0

        def _text_to_vector(self, text: str) -> List[float]:
            """Convert text to a vector with semantic meaning."""
            # Tokenize and normalize
            words = text.lower().split()

            # Build vocabulary on the fly
            for word in words:
                if word not in self.vocab:
                    self.vocab[word] = self.vocab_size
                    self.vocab_size += 1

            # Create a sparse vector representation
            vector = np.zeros(self.dimension)

            # Use TF representation with position encoding
            for i, word in enumerate(words):
                if word in self.vocab:
                    # Use multiple dimensions per word to avoid collisions
                    word_idx = self.vocab[word]
                    # Spread the word representation across multiple dimensions
                    base_idx = (word_idx * 7) % self.dimension  # 7 is a prime number

                    # Add term frequency
                    vector[base_idx] += 1.0

                    # Add position encoding to neighboring dimensions
                    if base_idx + 1 < self.dimension:
                        vector[base_idx + 1] += 0.5 / (i + 1)  # Position weight
                    if base_idx + 2 < self.dimension:
                        vector[base_idx + 2] += 0.3  # Word presence indicator

            # Add some deterministic noise based on full text to make vectors unique
            # but preserve similarity
            np.random.seed(hash(text) % 1000000)
            noise = np.random.randn(self.dimension) * 0.1  # Small noise
            vector += noise

            # Normalize to unit length (important for cosine similarity)
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

            return vector.tolist()

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """Generate embeddings that capture semantic similarity."""
            return [self._text_to_vector(text) for text in texts]

        def embed_query(self, text: str) -> List[float]:
            """Generate embedding for a single query."""
            return self._text_to_vector(text)


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "LangChain dependencies not available")
class TestLangChainIntegration(unittest.TestCase):
    """Test suite for CyborgDB LangChain integration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Test parameters
        cls.dimension = 384
        cls.api_url = "http://localhost:8000"
        cls.api_key = os.getenv("CYBORGDB_API_KEY")

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
            "LangChain simplifies building LLM applications.",
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
            {"category": "programming", "source": "documentation"},
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
            client = cyborgdb.Client(api_url=self.api_url, api_key=self.api_key)
            for index_name in self.index_names_to_cleanup:
                try:
                    # Create a temporary encrypted index instance to delete it
                    from cyborgdb.client.encrypted_index import EncryptedIndex

                    index = EncryptedIndex(
                        index_name=index_name,
                        index_key=self.index_key,
                        api=client.api,
                        api_client=client.api_client,
                        max_cache_size=0,
                    )
                    index.delete_index()
                except Exception:
                    pass
        except Exception:
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
            index_config_params={"n_lists": 10},
        )

        # Add texts
        ids = vectorstore.add_texts(
            texts=self.test_texts[:5], metadatas=self.test_metadata[:5]
        )

        self.assertEqual(len(ids), 5)

        # Test similarity search
        results = vectorstore.similarity_search("artificial intelligence", k=3)
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], Document)

    def test_02_verify_content_storage_in_metadata(self):
        """Test that text is stored in metadata with _content key."""
        index_name = "langchain_test_content_storage"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="cosine",
            index_config_params={"n_lists": 10},
        )

        # Add test documents with specific IDs
        test_ids = ["test_id_1", "test_id_2", "test_id_3"]
        test_texts = self.test_texts[:3]
        test_metadatas = self.test_metadata[:3]

        vectorstore.add_texts(texts=test_texts, metadatas=test_metadatas, ids=test_ids)

        # Use the underlying index to verify data storage
        raw_items = vectorstore.index.get(test_ids, include=["metadata"])

        for i, item in enumerate(raw_items):
            # Verify that text is stored in metadata with _content key
            metadata = item["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)

            # Check that _content key exists and has correct value
            self.assertIn("_content", metadata)
            self.assertEqual(metadata["_content"], test_texts[i])

            # Verify that original metadata is preserved
            for key, value in test_metadatas[i].items():
                self.assertEqual(metadata.get(key), value)

    def test_03_get_documents_by_id(self):
        """Test retrieving documents by ID."""
        index_name = "langchain_test_get_by_id"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
        )

        # Add documents with specific IDs
        test_ids = [f"doc_{i}" for i in range(5)]
        vectorstore.add_texts(
            texts=self.test_texts[:5], metadatas=self.test_metadata[:5], ids=test_ids
        )

        # Get specific documents
        retrieved_docs = vectorstore.get(["doc_1", "doc_3"])

        self.assertEqual(len(retrieved_docs), 2)

        # Verify the content and metadata
        doc_map = {doc.page_content: doc for doc in retrieved_docs}

        # Check that we got the right documents
        self.assertIn(self.test_texts[1], doc_map)
        self.assertIn(self.test_texts[3], doc_map)

        # Verify metadata for each document
        doc1 = doc_map[self.test_texts[1]]
        self.assertEqual(doc1.metadata["category"], "AI")
        self.assertEqual(doc1.metadata["source"], "textbook")

        doc3 = doc_map[self.test_texts[3]]
        self.assertEqual(doc3.metadata["category"], "AI")
        self.assertEqual(doc3.metadata["source"], "research")

    def test_04_metadata_filtering(self):
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
            index_config_params={"n_lists": 10},
        )

        # Add all test documents
        vectorstore.add_documents(self.test_documents)

        # Search with metadata filter
        ai_results = vectorstore.similarity_search(
            "artificial intelligence", k=10, filter={"category": "AI"}
        )

        # Verify all results have the correct category
        for doc in ai_results:
            self.assertEqual(doc.metadata.get("category"), "AI")

        # Search with multiple metadata conditions
        research_results = vectorstore.similarity_search(
            "neural networks", k=10, filter={"category": "AI", "source": "research"}
        )

        for doc in research_results:
            self.assertEqual(doc.metadata.get("category"), "AI")
            self.assertEqual(doc.metadata.get("source"), "research")

    def test_05_similarity_search_by_vector(self):
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
            metric="cosine",
        )

        # Add documents
        vectorstore.add_documents(self.test_documents)

        # Get embedding for a query
        query_embedding = embeddings.embed_query("machine learning algorithms")

        # Search by vector
        results = vectorstore.similarity_search_by_vector(
            embedding=query_embedding, k=5
        )

        self.assertEqual(len(results), 5)
        for doc in results:
            self.assertIsInstance(doc, Document)

    def test_06_delete_operations(self):
        """Test delete operations."""
        index_name = "langchain_test_delete"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
        )

        # Add texts with specific IDs
        ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        vectorstore.add_texts(
            texts=self.test_texts[:5], metadatas=self.test_metadata[:5], ids=ids
        )

        # Delete specific documents
        success = vectorstore.delete(ids=["doc2", "doc4"])
        self.assertTrue(success)

        # Verify deletion by searching
        results = vectorstore.similarity_search("machine learning", k=10)
        result_texts = [doc.page_content for doc in results]

        # doc2 text should not be in results
        self.assertNotIn(self.test_texts[1], result_texts)

    def test_07_from_texts_classmethod(self):
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
            metric="cosine",
        )

        # Verify the store was created and populated
        results = vectorstore.similarity_search("programming", k=3)
        self.assertGreater(len(results), 0)

    def test_08_from_documents_classmethod(self):
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
            n_lists=10,
        )

        # Verify the store was created and populated
        results = vectorstore.similarity_search("database", k=3)
        self.assertGreater(len(results), 0)

    def test_09_as_retriever(self):
        """Test using vector store as a retriever."""
        index_name = "langchain_test_retriever"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
        )

        # Add documents
        vectorstore.add_documents(self.test_documents)

        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        # Use retriever
        docs = retriever.invoke("What is deep learning?")

        self.assertEqual(len(docs), 3)
        for doc in docs:
            self.assertIsInstance(doc, Document)

    def test_10_async_operations(self):
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
                index_type="ivfflat",
            )

            # Async add texts
            ids = await vectorstore.aadd_texts(
                texts=self.test_texts[:5], metadatas=self.test_metadata[:5]
            )
            self.assertEqual(len(ids), 5)

            # Async similarity search
            results = await vectorstore.asimilarity_search("machine learning", k=3)
            self.assertEqual(len(results), 3)

            # Async similarity search with score
            results_with_scores = await vectorstore.asimilarity_search_with_score(
                "artificial intelligence", k=2
            )
            self.assertEqual(len(results_with_scores), 2)

            # Async delete
            success = await vectorstore.adelete(ids=[ids[0]])
            self.assertTrue(success)

        # Run async tests
        asyncio.run(run_async_tests())

    def test_11_train_index(self):
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
            index_config_params={"n_lists": n_lists},
        )

        # Add enough documents to train
        all_texts = self.test_texts + additional_texts
        all_metadata = self.test_metadata + additional_metadata

        vectorstore.add_texts(
            texts=all_texts[: min_vectors_for_training + 5],
            metadatas=all_metadata[: min_vectors_for_training + 5],
        )

        # Train the index
        vectorstore.index.train()

        # Verify it's trained
        self.assertTrue(vectorstore.index.is_trained())

        # Test search on trained index
        results = vectorstore.similarity_search("test document", k=5)
        self.assertGreater(len(results), 0)

    def test_12_edge_cases(self):
        """Test edge cases and error handling."""
        index_name = "langchain_test_edge_cases"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
        )

        # Test empty search results
        results = vectorstore.similarity_search("xyz123abc456", k=5)
        self.assertIsInstance(results, list)

        # Test adding empty list
        ids = vectorstore.add_texts(texts=[], metadatas=[])
        self.assertEqual(len(ids), 0)

        # Test mismatched texts and metadata lengths
        with self.assertRaises(ValueError):
            vectorstore.add_texts(
                texts=["text1", "text2"],
                metadatas=[{"meta": 1}],  # Only one metadata for two texts
                ids=["id1", "id2"],  # Correct number of IDs
            )

        # Test delete with no IDs
        success = vectorstore.delete(ids=None, delete_index=False)
        self.assertFalse(success)

    def test_13_content_preserved_through_search(self):
        """Test that content is properly preserved through search operations via metadata."""
        index_name = "langchain_test_content_search"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            api_url=self.api_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="cosine",
        )

        # Add documents with distinct content
        test_texts = [
            "This is a very specific test sentence about quantum computing.",
            "Another unique sentence about blockchain technology and cryptocurrencies.",
            "A third sentence discussing artificial general intelligence (AGI).",
        ]
        test_metadata = [
            {"topic": "quantum"},
            {"topic": "blockchain"},
            {"topic": "AGI"},
        ]

        vectorstore.add_texts(texts=test_texts, metadatas=test_metadata)

        # Search and verify exact content is returned
        results = vectorstore.similarity_search("quantum computing", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].page_content, test_texts[0])
        self.assertEqual(results[0].metadata["topic"], "quantum")

        # Search with score and verify
        results_with_score = vectorstore.similarity_search_with_score("blockchain", k=1)
        self.assertEqual(len(results_with_score), 1)
        doc, score = results_with_score[0]
        self.assertEqual(doc.page_content, test_texts[1])
        self.assertEqual(doc.metadata["topic"], "blockchain")
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
