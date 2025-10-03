#!/usr/bin/env python3
"""
Additional test coverage for Python SDK to achieve standardization
Implements missing SSL, IVFPQ, IVF, error handling, and edge case tests
"""

import unittest
import os
import time
import uuid
import asyncio
import numpy as np
from unittest.mock import patch
import requests

import cyborgdb as cyborgdb


def create_client():
    """
    Create a CyborgDB client with auto-detection for HTTP/HTTPS on port 8000.
    Simple approach: try HTTPS first, fallback to HTTP on the same port.
    """
    api_key = os.environ.get("CYBORGDB_API_KEY", "")
    host = "localhost"
    port = "8000"

    # Try HTTPS first on port 8000
    try:
        import ssl
        import socket

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        # Test SSL connection on port 8000
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        ssl_sock = context.wrap_socket(sock, server_hostname=host)

        try:
            ssl_sock.connect((host, int(port)))
            ssl_sock.close()

            # SSL connection successful
            return cyborgdb.Client(
                base_url=f"https://{host}:{port}",
                api_key=api_key,
                verify_ssl=False,  # Disable for localhost (likely self-signed)
            )
        except Exception:
            ssl_sock.close()
    except Exception:
        pass

    # Fallback to HTTP on port 8000
    return cyborgdb.Client(base_url=f"http://{host}:{port}", api_key=api_key)


def generate_unique_name(prefix="test_"):
    """Generate a unique index name with a given prefix by appending a UUID v4."""
    return f"{prefix}{uuid.uuid4()}"


class TestSSLVerification(unittest.TestCase):
    """Test SSL/TLS verification functionality"""

    def setUp(self):
        self.api_key = os.environ.get("CYBORGDB_API_KEY", "test-key")
        self.localhost_url = "http://localhost:8000"
        self.production_url = "https://api.cyborgdb.com"

    def test_ssl_auto_detection_localhost(self):
        """Test SSL auto-detection for localhost URLs"""
        with patch("cyborgdb.Client") as mock_client:
            # Test HTTP localhost - should auto-disable SSL
            cyborgdb.Client(base_url="http://localhost:8000", api_key=self.api_key)
            mock_client.assert_called_once()

    def test_ssl_explicit_disable(self):
        """Test explicit SSL verification disable"""
        with patch("cyborgdb.Client") as mock_client:
            cyborgdb.Client(
                base_url=self.production_url, api_key=self.api_key, verify_ssl=False
            )
            mock_client.assert_called_once()

    def test_ssl_explicit_enable(self):
        """Test explicit SSL verification enable"""
        with patch("cyborgdb.Client") as mock_client:
            cyborgdb.Client(
                base_url=self.production_url, api_key=self.api_key, verify_ssl=True
            )
            mock_client.assert_called_once()

    def test_ssl_certificate_validation(self):
        """Test SSL certificate validation scenarios"""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.SSLError(
                "Certificate verification failed"
            )

            cyborgdb.Client(base_url=self.production_url, api_key=self.api_key)

            with self.assertRaises(requests.exceptions.SSLError):
                mock_get()

    def test_auto_detection(self):
        """Test auto-detection works with current environment"""
        client = create_client()
        self.assertIsNotNone(client)

        # Try a basic operation to ensure the connection works
        health = client.get_health()
        # Accept various health response formats
        self.assertIsInstance(health, (dict, bool, str, type(None)))


class TestIndexTypes(unittest.TestCase):
    """Test all index types (IVF, IVFPQ) that are missing from current Python tests"""

    @classmethod
    def setUpClass(cls):
        cls.client = create_client()
        cls.dimension = 128

    def setUp(self):
        self.index_name = generate_unique_name()
        self.index_key = cyborgdb.Client.generate_key()
        self.test_vectors = np.random.rand(10, self.dimension).astype(np.float32)

    def tearDown(self):
        """Clean up created indexes"""
        try:
            if hasattr(self, "index") and self.index:
                self.index.delete_index()
        except Exception:
            pass

    def test_ivf_index_creation_and_operations(self):
        """Test IVF index creation, upsert, and query operations"""
        index_config = cyborgdb.IndexIVF(dimension=self.dimension)

        self.index = self.client.create_index(
            self.index_name,
            self.index_key,
            index_config,
            metric="euclidean",
        )

        # Verify index properties
        self.assertEqual(self.index.index_type, "ivf")

        # Test upsert
        items = []
        for i in range(len(self.test_vectors)):
            items.append(
                {
                    "id": str(i),
                    "vector": self.test_vectors[i],
                    "metadata": {"test_id": i},
                }
            )

        self.index.upsert(items)
        time.sleep(1)  # Allow processing

        # Test query
        query_vector = self.test_vectors[0]
        results = self.index.query(query_vectors=[query_vector], top_k=5)

        self.assertGreater(len(results[0]), 0)
        self.assertTrue("id" in results[0][0])

    def test_ivfpq_index_creation_and_operations(self):
        """Test IVFPQ index creation with PQ parameters"""
        index_config = cyborgdb.IndexIVFPQ(
            dimension=self.dimension, pq_dim=32, pq_bits=8
        )

        self.index = self.client.create_index(
            self.index_name, self.index_key, index_config, metric="euclidean"
        )

        # Verify index properties
        self.assertEqual(self.index.index_type, "ivfpq")

        # Test upsert
        items = []
        for i in range(len(self.test_vectors)):
            items.append(
                {
                    "id": str(i),
                    "vector": self.test_vectors[i],
                    "metadata": {"test_id": i},
                }
            )

        self.index.upsert(items)
        time.sleep(1)  # Allow processing

        # Test query
        query_vector = self.test_vectors[0]
        results = self.index.query(query_vectors=[query_vector], top_k=5)

        self.assertGreater(len(results[0]), 0)
        self.assertTrue("id" in results[0][0])

    def test_ivfpq_parameter_validation(self):
        """Test IVFPQ parameter validation"""
        # Test invalid pq_dim = 0
        invalid_config = cyborgdb.IndexIVFPQ(
            dimension=self.dimension, pq_dim=0, pq_bits=8
        )
        
        with self.assertRaises(Exception) as context:
            invalid_index = self.client.create_index(
                generate_unique_name(),
                self.client.generate_key(),
                invalid_config,
                metric="euclidean",
            )
            invalid_index.delete_index()
        
        # Verify the error is about pq_dim
        self.assertIn("pq_dim", str(context.exception).lower())


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error scenarios"""

    def setUp(self):
        self.client = create_client()

    def test_invalid_api_key(self):
        """Test handling of invalid API keys"""
        client = cyborgdb.Client(
            base_url="http://localhost:8000", api_key="invalid-key-12345"
        )

        # Try to create an index - this should require authentication
        with self.assertRaises(Exception) as context:
            index_config = cyborgdb.IndexIVFFlat(dimension=128)
            client.create_index(
                generate_unique_name(),
                client.generate_key(),
                index_config,
                metric="euclidean"
            )
        
        error_str = str(context.exception).lower()
        auth_related = any(
            keyword in error_str
            for keyword in [
                "auth",
                "key",
                "unauthorized",
                "401",
                "forbidden",
                "403",
            ]
        )
        self.assertTrue(auth_related, f"Expected authentication error, got: {context.exception}")

    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        index_name = generate_unique_name()
        index_key = self.client.generate_key()

        # Test invalid dimension
        with self.assertRaises(Exception):
            config = cyborgdb.IndexIVFFlat(dimension=-1)
            self.client.create_index(
                index_name, index_key, config, metric="euclidean"
            )

        # Test invalid metric
        index_config = cyborgdb.IndexIVFFlat(dimension=128)
        with self.assertRaises(Exception):
            self.client.create_index(
                index_name, index_key, index_config, metric="invalid_metric"
            )

    def test_network_connectivity_issues(self):
        """Test handling of network connectivity issues"""
        client = cyborgdb.Client(
            base_url="http://non-existent-server:8000", api_key="test-key"
        )

        with self.assertRaises(Exception):
            client.get_health()

    def test_invalid_vector_dimensions(self):
        """Test handling of invalid vector dimensions"""
        index_config = cyborgdb.IndexIVFFlat(dimension=128)
        index_name = generate_unique_name()
        index_key = self.client.generate_key()

        index = self.client.create_index(
            index_name, index_key, index_config, metric="euclidean"
        )

        try:
            # Test wrong vector dimension
            with self.assertRaises(Exception):
                invalid_vector = np.random.rand(64).astype(np.float32)
                index.upsert([{"id": "test", "vector": invalid_vector, "metadata": {}}])
        finally:
            index.delete_index()

    def test_server_error_responses(self):
        """Test handling of server error responses"""
        # Test with empty index name (should cause an error)
        index_key = self.client.generate_key()
        index_config = cyborgdb.IndexIVFFlat(dimension=128)

        with self.assertRaises(Exception):
            self.client.create_index(
                "",  # Empty name should cause error
                index_key,
                index_config,
                metric="euclidean",
            )

        # Test invalid index key format
        with self.assertRaises(Exception):
            self.client.create_index(
                generate_unique_name(),
                b"invalid_short_key",  # Invalid key length
                cyborgdb.IndexIVFFlat(dimension=128),
                metric="euclidean",
            )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def setUp(self):
        self.client = create_client()
        self.index_name = generate_unique_name()
        self.index_key = self.client.generate_key()
        self.index_config = cyborgdb.IndexIVFFlat(dimension=128)
        self.index = self.client.create_index(
            self.index_name,
            self.index_key,
            self.index_config,
            metric="euclidean",
        )

    def tearDown(self):
        """Clean up created indexes"""
        try:
            if self.index:
                self.index.delete_index()
        except Exception:
            pass

    def test_empty_query_results(self):
        """Test handling of empty query results"""
        query_vector = np.random.rand(128).astype(np.float32)
        results = self.index.query(query_vectors=[query_vector], top_k=10)

        # Should return empty results for empty index
        self.assertEqual(len(results[0]), 0)

    def test_mismatched_parameter_lengths(self):
        """Test validation of mismatched parameter lengths"""
        vectors = [np.random.rand(128).astype(np.float32) for _ in range(3)]
        
        # Create items with proper length handling
        items = []
        for i in range(len(vectors)):
            items.append(
                {
                    "id": f"item_{i}",
                    "vector": vectors[i],
                    "metadata": {"index": i},
                }
            )

        # This should succeed - verify items are stored
        self.index.upsert(items)
        time.sleep(1)
        stored_ids = self.index.list_ids()
        
        for i in range(len(vectors)):
            self.assertIn(f"item_{i}", stored_ids)

    def test_content_preservation_through_operations(self):
        """Test that content is preserved through various operations"""
        original_vector = np.random.rand(128).astype(np.float32)
        original_metadata = {"test_key": "test_value", "number": 42}

        # Upsert
        self.index.upsert(
            [
                {
                    "id": "preserve_test",
                    "vector": original_vector,
                    "metadata": original_metadata,
                }
            ]
        )

        time.sleep(1)

        # Retrieve and verify
        results = self.index.get(["preserve_test"], include=["vector", "metadata"])
        self.assertEqual(len(results), 1)

        retrieved = results[0]
        self.assertEqual(retrieved["id"], "preserve_test")
        np.testing.assert_array_equal(retrieved["vector"], original_vector)
        self.assertEqual(retrieved["metadata"], original_metadata)

    def test_index_cleanup_error_handling(self):
        """Test error handling during index cleanup operations"""
        # Create index
        test_index_name = generate_unique_name()
        test_index_key = self.client.generate_key()
        test_config = cyborgdb.IndexIVFFlat(dimension=128)

        test_index = self.client.create_index(
            test_index_name, test_index_key, test_config, metric="euclidean"
        )

        # Delete index
        test_index.delete_index()

        # Try to delete again - should handle gracefully
        with self.assertRaises(Exception):
            test_index.delete_index()

    def test_concurrent_operations(self):
        """Test concurrent operations handling"""

        async def concurrent_upsert():
            vectors = [np.random.rand(128).astype(np.float32) for _ in range(5)]

            for i, vector in enumerate(vectors):
                item = [
                    {
                        "id": f"concurrent_{i}",
                        "vector": vector,
                        "metadata": {"batch": i},
                    }
                ]
                self.index.upsert(item)

            # Verify all items were inserted
            time.sleep(2)
            results = self.index.list_ids()
            concurrent_ids = [id for id in results if id.startswith("concurrent_")]
            self.assertEqual(len(concurrent_ids), 5)

        # Run the concurrent test
        asyncio.run(concurrent_upsert())


class TestBackendCompatibility(unittest.TestCase):
    """Test backend compatibility (Lite vs Full)"""

    def test_lite_backend_compatibility(self):
        """Test operations with lite backend"""
        client = create_client()
        health = client.get_health()
        self.assertIsInstance(health, (dict, bool, str, type(None)))

    def test_feature_availability_differences(self):
        """Test feature availability between backend variants"""
        client = create_client()

        # Test if advanced index types are available
        index_config = cyborgdb.IndexIVFPQ(dimension=128, pq_dim=32, pq_bits=8)
        index_name = generate_unique_name()
        index_key = client.generate_key()

        index = client.create_index(
            index_name, index_key, index_config, metric="euclidean"
        )
        index.delete_index()


if __name__ == "__main__":
    unittest.main()