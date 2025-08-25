import os
import unittest
import numpy as np
import time
from dotenv import load_dotenv
from cyborgdb import Client, EncryptedIndex, IndexIVF

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
        self.index_config = IndexIVF(dimension=128, n_lists=10, metric="euclidean")

        try:
            self.index = self.client.create_index(
                self.index_name, self.test_key, self.index_config
            )
        except Exception as e:
            # If IndexIVF fails (likely using lite backend), skip these tests
            if "cyborgdb_lite" in str(e) or "IndexIVF" in str(e):
                self.skipTest(
                    "Server is using lite backend, skipping full version tests"
                )
            raise

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

        # Check results
        self.assertEqual(len(results[0]), 10)
        self.assertTrue("id" in results[0][0])
        self.assertTrue("distance" in results[0][0])

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
