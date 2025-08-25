import os
import unittest
import numpy as np
import time
import requests
from dotenv import load_dotenv
from cyborgdb import Client, IndexIVFFlat

# Load environment variables from .env.local
load_dotenv(".env.local")


class ClientLiteIntegrationTest(unittest.TestCase):
    """Integration tests for the CyborgDB client with Lite backend."""

    @classmethod
    def setUpClass(cls):
        """Check if server is using lite version."""
        try:
            # Check server health endpoint
            response = requests.get("http://localhost:8000/v1/health")
            if response.status_code != 200:
                raise unittest.SkipTest("Server not available")
        except Exception:
            raise unittest.SkipTest("Server not available")

    def setUp(self):
        """Set up the test environment."""
        # Create real client (no mocking)
        self.client = Client(
            base_url="http://localhost:8000",
            api_key=os.getenv("CYBORGDB_API_KEY", "test-api-key"),
        )

        # Create a test key
        self.test_key = self.client.generate_key()

        # Create a test index using IndexIVFFlat which should work with both versions
        self.index_name = f"test_index_lite_{int(time.time())}"
        self.index_config = IndexIVFFlat(dimension=128, n_lists=10, metric="euclidean")

        try:
            self.index = self.client.create_index(
                self.index_name, self.test_key, self.index_config
            )
        except Exception as e:
            # If IndexIVFFlat also fails, skip these tests
            self.skipTest(f"Cannot create index with lite backend: {e}")

    def tearDown(self):
        """Clean up after tests."""
        try:
            if hasattr(self, "index"):
                self.index.delete_index()
        except Exception:
            pass

    def test_health_check(self):
        """Test that the server is healthy."""
        response = requests.get("http://localhost:8000/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_upsert_and_query(self):
        """Test upserting vectors and querying them."""
        # Create some test vectors
        num_vectors = 50  # Use fewer vectors for lite version
        dimension = 128
        vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
        ids = [f"test_{i}" for i in range(num_vectors)]

        # Upsert vectors
        self.index.upsert(ids, vectors)

        # Query a vector
        query_vector = np.random.rand(dimension).astype(np.float32)
        results = self.index.query(query_vectors=query_vector, top_k=5)

        # Check results
        self.assertGreater(len(results[0]), 0)
        self.assertLessEqual(len(results[0]), 5)

    def test_load_index(self):
        """Test loading an existing index."""
        # Load the index that was created in setUp
        loaded_index = self.client.load_index(self.index_name, self.test_key)

        # Verify it's the same index
        self.assertEqual(loaded_index.index_name, self.index_name)

        # Add a vector to the loaded index
        test_vector = np.random.rand(128).astype(np.float32)
        loaded_index.upsert(["load_test"], [test_vector])

        # Query to verify it works
        results = loaded_index.query(query_vectors=test_vector, top_k=1)
        self.assertGreater(len(results[0]), 0)


if __name__ == "__main__":
    unittest.main()
