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
        cls.base_url = "http://localhost:8000"
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
        ] * 2000  # Replicate to increase data size

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
        ] * 2000  # Replicate to match texts

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
        self.index_key = CyborgVectorStore.generate_key()
        self.index_names_to_cleanup = []

    def tearDown(self):
        """Clean up after each test."""
        # Clean up any created indexes
        try:
            client = cyborgdb.Client(base_url=self.base_url, api_key=self.api_key)

            for index_name in self.index_names_to_cleanup:
                try:
                    # Create a temporary encrypted index instance to delete it
                    from cyborgdb.client.encrypted_index import EncryptedIndex

                    index = EncryptedIndex(
                        index_name=index_name,
                        index_key=self.index_key,
                        api=client.api,
                        api_client=client.api_client,
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
            base_url=self.base_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="cosine",
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
            base_url=self.base_url,
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
            base_url=self.base_url,
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
            base_url=self.base_url,
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
            base_url=self.base_url,
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
            base_url=self.base_url,
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
            base_url=self.base_url,
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
            base_url=self.base_url,
            index_type="ivfflat",
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
            base_url=self.base_url,
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

        # Use base test data without replication for this test
        base_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for data science.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
        ]
        base_metadata = [
            {"category": "animals", "source": "proverb"},
            {"category": "AI", "source": "textbook"},
            {"category": "programming", "source": "tutorial"},
            {"category": "AI", "source": "research"},
            {"category": "AI", "source": "textbook"},
        ]

        async def run_async_tests():
            vectorstore = CyborgVectorStore(
                index_name=index_name,
                index_key=self.index_key,
                api_key=self.api_key,
                base_url=self.base_url,
                embedding=MockEmbeddings(self.dimension),
                index_type="ivfflat",
            )

            # Async add texts
            ids = await vectorstore.aadd_texts(
                texts=base_texts, metadatas=base_metadata
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

    # def test_11_train_index(self):
    #     """Test training the index when enough vectors are present."""
    #     index_name = "langchain_test_train"
    #     self.index_names_to_cleanup.append(index_name)

    #     # Create with enough data to train
    #     n_lists = 4
    #     min_vectors_for_training = 2 * n_lists

    #     # Generate more test data
    #     additional_texts = [f"Test document number {i}" for i in range(20)]
    #     additional_metadata = [{"index": i} for i in range(20)]

    #     vectorstore = CyborgVectorStore(
    #         index_name=index_name,
    #         index_key=self.index_key,
    #         api_key=self.api_key,
    #         base_url=self.base_url,
    #         embedding=MockEmbeddings(self.dimension),
    #         index_type="ivfflat",
    #         index_config_params={"n_lists": n_lists},
    #     )

    #     # Add enough documents to train
    #     all_texts = self.test_texts + additional_texts
    #     all_metadata = self.test_metadata + additional_metadata

    #     vectorstore.add_texts(
    #         texts=all_texts[: min_vectors_for_training + 5],
    #         metadatas=all_metadata[: min_vectors_for_training + 5],
    #     )

    #     # Train the index
    #     vectorstore.index.train(
    #         n_lists=n_lists,
    #         batch_size=2048,
    #         max_iters=100,
    #         tolerance=1e-4
    #     )

    #     # Verify it's trained
    #     self.assertTrue(vectorstore.index.is_trained())

    #     # Test search on trained index
    #     results = vectorstore.similarity_search("test document", k=5)
    #     self.assertGreater(len(results), 0)

    def test_12_edge_cases(self):
        """Test edge cases and error handling."""
        index_name = "langchain_test_edge_cases"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.base_url,
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
            base_url=self.base_url,
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

    def test_14_add_texts_with_precomputed_embeddings(self):
        """Test adding texts with pre-computed embeddings."""
        index_name = "langchain_test_precomputed_embeddings"
        self.index_names_to_cleanup.append(index_name)

        # Generate pre-computed embeddings
        mock_embedder = MockEmbeddings(self.dimension)
        test_texts = [
            "First text about science.",
            "Second text about technology.",
            "Third text about mathematics.",
        ]
        test_metadata = [{"id": 1}, {"id": 2}, {"id": 3}]
        test_ids = ["emb_1", "emb_2", "emb_3"]

        # Pre-compute the embeddings (could be from any source)
        precomputed_embeddings = mock_embedder.embed_documents(test_texts)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding=mock_embedder,  # Still needed for queries
            index_type="ivfflat",
            metric="cosine",
        )

        # Add texts with pre-computed embeddings
        ids = vectorstore.add_texts(
            texts=test_texts,
            metadatas=test_metadata,
            ids=test_ids,
            embeddings=precomputed_embeddings,  # Use pre-computed embeddings
        )

        self.assertEqual(len(ids), 3)
        self.assertEqual(ids, test_ids)

        # Verify documents were added and can be retrieved
        retrieved_docs = vectorstore.get(test_ids)
        self.assertEqual(len(retrieved_docs), 3)

        # Verify the content was stored correctly
        retrieved_texts = {doc.page_content for doc in retrieved_docs}
        self.assertEqual(retrieved_texts, set(test_texts))

        # Verify metadata was preserved
        for doc in retrieved_docs:
            self.assertIn("id", doc.metadata)
            self.assertIn(doc.metadata["id"], [1, 2, 3])

    def test_15_add_texts_with_numpy_embeddings(self):
        """Test adding texts with pre-computed numpy array embeddings."""
        index_name = "langchain_test_numpy_embeddings"
        self.index_names_to_cleanup.append(index_name)

        test_texts = [
            "Numpy vector test one.",
            "Numpy vector test two.",
            "Numpy vector test three.",
        ]
        test_metadata = [{"idx": 1}, {"idx": 2}, {"idx": 3}]

        # Create numpy array of embeddings
        precomputed_embeddings = np.random.randn(3, self.dimension).astype(np.float32)
        # Normalize for cosine similarity
        precomputed_embeddings = precomputed_embeddings / np.linalg.norm(
            precomputed_embeddings, axis=1, keepdims=True
        )

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="cosine",
        )

        # Add texts with numpy embeddings
        ids = vectorstore.add_texts(
            texts=test_texts, metadatas=test_metadata, embeddings=precomputed_embeddings
        )

        self.assertEqual(len(ids), 3)

        # Verify documents were added
        retrieved_docs = vectorstore.get(ids[:2])
        self.assertEqual(len(retrieved_docs), 2)

    def test_16_add_documents_with_embeddings(self):
        """Test adding documents with pre-computed embeddings."""
        index_name = "langchain_test_docs_with_embeddings"
        self.index_names_to_cleanup.append(index_name)

        # Create test documents
        test_docs = [
            Document(
                page_content="Document about physics.", metadata={"subject": "physics"}
            ),
            Document(
                page_content="Document about chemistry.",
                metadata={"subject": "chemistry"},
            ),
            Document(
                page_content="Document about biology.", metadata={"subject": "biology"}
            ),
        ]
        test_ids = ["doc_physics", "doc_chemistry", "doc_biology"]

        # Pre-compute embeddings (could be from any model/source)
        mock_embedder = MockEmbeddings(self.dimension)
        texts = [doc.page_content for doc in test_docs]
        precomputed_embeddings = mock_embedder.embed_documents(texts)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding=mock_embedder,
            index_type="ivfflat",
            metric="cosine",
        )

        # Add documents with pre-computed embeddings
        ids = vectorstore.add_documents(
            documents=test_docs, ids=test_ids, embeddings=precomputed_embeddings
        )

        self.assertEqual(len(ids), 3)
        self.assertEqual(ids, test_ids)

        # Verify documents were added and can be retrieved
        retrieved_docs = vectorstore.get(test_ids)
        self.assertEqual(len(retrieved_docs), 3)

        # Verify content and metadata were stored correctly
        retrieved_contents = {doc.page_content for doc in retrieved_docs}
        expected_contents = {doc.page_content for doc in test_docs}
        self.assertEqual(retrieved_contents, expected_contents)

        # Verify metadata was preserved
        subjects_found = {doc.metadata["subject"] for doc in retrieved_docs}
        self.assertEqual(subjects_found, {"physics", "chemistry", "biology"})

    def test_17_from_texts_with_embeddings(self):
        """Test creating vector store from texts with pre-computed embeddings."""
        index_name = "langchain_test_from_texts_embeddings"
        self.index_names_to_cleanup.append(index_name)

        test_texts = [
            "Create from texts one.",
            "Create from texts two.",
            "Create from texts three.",
        ]
        test_metadata = [{"num": 1}, {"num": 2}, {"num": 3}]

        # Pre-compute embeddings
        mock_embedder = MockEmbeddings(self.dimension)
        precomputed_embeddings = mock_embedder.embed_documents(test_texts)

        # Create vector store with pre-computed embeddings
        vectorstore = CyborgVectorStore.from_texts(
            texts=test_texts,
            embedding=mock_embedder,
            metadatas=test_metadata,
            embeddings=precomputed_embeddings,  # Pass pre-computed embeddings
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.base_url,
            index_type="ivfflat",
            metric="cosine",
        )

        # Verify the store was created and populated
        results = vectorstore.similarity_search("Create from texts", k=3)
        self.assertEqual(len(results), 3)

    def test_18_embeddings_validation(self):
        """Test validation of pre-computed embeddings."""
        index_name = "langchain_test_embeddings_validation"
        self.index_names_to_cleanup.append(index_name)

        vectorstore = CyborgVectorStore(
            index_name=index_name,
            index_key=self.index_key,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding=MockEmbeddings(self.dimension),
            index_type="ivfflat",
            metric="cosine",
        )

        test_texts = ["Text one.", "Text two.", "Text three."]

        # Test mismatched number of embeddings
        wrong_size_embeddings = [
            [0.1] * self.dimension,
            [0.2] * self.dimension,
        ]  # Only 2 embeddings for 3 texts

        with self.assertRaises(ValueError) as context:
            vectorstore.add_texts(texts=test_texts, embeddings=wrong_size_embeddings)

        self.assertIn("must match number of texts", str(context.exception))

    def test_19_async_with_embeddings(self):
        """Test async operations with pre-computed embeddings."""
        index_name = "langchain_test_async_embeddings"
        self.index_names_to_cleanup.append(index_name)

        async def run_async_tests():
            mock_embedder = MockEmbeddings(self.dimension)
            test_texts = ["Async text one.", "Async text two.", "Async text three."]
            precomputed_embeddings = mock_embedder.embed_documents(test_texts)

            vectorstore = CyborgVectorStore(
                index_name=index_name,
                index_key=self.index_key,
                api_key=self.api_key,
                base_url=self.base_url,
                embedding=mock_embedder,
                index_type="ivfflat",
            )

            # Async add texts with pre-computed embeddings
            ids = await vectorstore.aadd_texts(
                texts=test_texts, embeddings=precomputed_embeddings
            )
            self.assertEqual(len(ids), 3)

            # Async add documents with embeddings
            test_docs = [Document(page_content=text) for text in test_texts]
            doc_ids = await vectorstore.aadd_documents(
                documents=test_docs, embeddings=precomputed_embeddings
            )
            self.assertEqual(len(doc_ids), 3)

        # Run async tests
        asyncio.run(run_async_tests())


if __name__ == "__main__":
    unittest.main()
