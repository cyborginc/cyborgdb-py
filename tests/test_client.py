import os
import unittest
import numpy as np
import time
from dotenv import load_dotenv
from cyborgdb import Client, EncryptedIndex, IndexIVFFlat

# Load environment variables from .env.local
load_dotenv(".env.local")


class ClientIntegrationTest(unittest.TestCase):
    """Integration tests for the CyborgDB client with full backend."""

    def setUp(self):
        """Set up the test environment."""
        # Create real client (no mocking)
        self.client = Client(
            base_url="http://localhost:8000", api_key=os.getenv("CYBORGDB_API_KEY", "")
        )

        # Create a test key
        self.test_key = Client.generate_key()

        # Create a test key with the client's member function
        self.test_key = self.client.generate_key()

        # Create a test index
        self.index_name = f"test_index_{int(time.time())}"
        self.index_config = IndexIVFFlat(dimension=128)

        # try:
        self.index = self.client.create_index(
            self.index_name, self.test_key, self.index_config, metric="euclidean"
        )

    def tearDown(self):
        """Clean up after tests."""
        try:
            self.index.delete_index()
        except Exception:
            pass

    def test_upsert_and_query(self):
        """Test upserting vectors and querying them."""
        # Create some test vectors
        num_vectors = 100
        dimension = 128
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"test_{i}" for i in range(num_vectors)]

        # Upsert vectors
        self.index.upsert(ids, vectors)

        # Query a vector
        query_vector = np.random.rand(dimension).astype(np.float32)
        results = self.index.query(query_vectors=query_vector, top_k=10)

        # Check results - results is a flat list, not nested
        self.assertEqual(len(results), 10)
        self.assertTrue("id" in results[0])
        self.assertTrue("distance" in results[0])

    def test_health_check(self):
        """Test the health check endpoint."""
        health = self.client.get_health()
        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertEqual(health["status"], "healthy", "API is not healthy")

    def test_load_index(self):
        """Test loading an existing index."""
        # Load the index using the same name and key
        loaded_index = self.client.load_index(self.index_name, self.test_key)

        # Check if the loaded index is the same as the original
        self.assertEqual(loaded_index.index_name, self.index_name)

        # Check if the index type is correct
        self.assertIsInstance(loaded_index, EncryptedIndex)

    def test_upsert_with_numpy_array(self):
        """Test upserting vectors using numpy array (binary format)."""
        num_vectors = 50
        dimension = 128
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"numpy_{i}" for i in range(num_vectors)]

        # Upsert with numpy array - should use binary format internally
        self.index.upsert(ids, vectors)

        # Verify vectors were inserted by querying
        query_vector = vectors[0]  # Use first vector as query
        results = self.index.query(query_vectors=query_vector, top_k=5)

        self.assertEqual(len(results), 5)
        # First result should be the same vector we queried with
        self.assertEqual(results[0]["id"], "numpy_0")
        self.assertAlmostEqual(results[0]["distance"], 0.0, places=5)

    def test_upsert_with_numpy_float64_conversion(self):
        """Test upserting vectors with float64 numpy array (should auto-convert)."""
        num_vectors = 20
        dimension = 128
        # Create float64 array - should be auto-converted to float32
        vectors = np.random.rand(num_vectors, dimension).astype(np.float64)
        ids = [f"float64_{i}" for i in range(num_vectors)]

        # Should not raise an error - auto-converts to float32
        self.index.upsert(ids, vectors)

        # Verify vectors were inserted
        query_vector = vectors[0].astype(np.float32)
        results = self.index.query(query_vectors=query_vector, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "float64_0")

    def test_upsert_binary_direct(self):
        """Test calling upsert_binary directly."""
        num_vectors = 30
        dimension = 128
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"direct_{i}" for i in range(num_vectors)]

        # Call upsert_binary directly
        self.index.upsert_binary(ids, vectors)

        # Verify vectors were inserted
        query_vector = vectors[0]
        results = self.index.query(query_vectors=query_vector, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["id"], "direct_0")

    def test_query_with_2d_numpy_array(self):
        """Test querying with 2D numpy array (batch query, binary format)."""
        # First insert some vectors
        num_vectors = 100
        dimension = 128
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"batch_{i}" for i in range(num_vectors)]
        self.index.upsert(ids, vectors)

        # Query with multiple vectors at once (2D array)
        query_vectors = vectors[:5]  # Use first 5 vectors as queries
        results = self.index.query(query_vectors=query_vectors, top_k=3)

        # Should return a list of lists (one list per query)
        self.assertEqual(len(results), 5)
        for i, result_list in enumerate(results):
            self.assertEqual(len(result_list), 3)
            # First result for each query should be itself
            self.assertEqual(result_list[0]["id"], f"batch_{i}")
            # Distance should be very small (near zero) - allow for float precision
            self.assertLess(result_list[0]["distance"], 0.01)

    def test_query_binary_direct(self):
        """Test calling query_binary directly."""
        # First insert some vectors
        num_vectors = 50
        dimension = 128
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"qbin_{i}" for i in range(num_vectors)]
        self.index.upsert(ids, vectors)

        # Call query_binary directly
        query_vectors = vectors[:3].copy()  # Use first 3 vectors
        results = self.index.query_binary(query_vectors=query_vectors, top_k=5)

        self.assertEqual(len(results), 3)
        for i, result_list in enumerate(results):
            self.assertEqual(len(result_list), 5)
            self.assertEqual(result_list[0]["id"], f"qbin_{i}")
