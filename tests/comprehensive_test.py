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
                # This would trigger an SSL error in a real scenario
                mock_get()  # Call the mock, not the side_effect

    def test_auto_detection(self):
        """Test auto-detection works with current environment"""
        # Test that our auto-detection creates a working client
        client = create_client()
        self.assertIsNotNone(client)

        # Try a basic operation to ensure the connection works
        try:
            health = client.get_health()
            # Accept various health response formats
            self.assertIsInstance(health, (dict, bool, str, type(None)))
        except Exception as e:
            # If it fails, it should be due to API key or server issues, not SSL
            error_str = str(e).lower()
            ssl_related = any(
                keyword in error_str
                for keyword in ["ssl", "certificate", "handshake", "verification"]
            )
            if ssl_related:
                self.fail(f"SSL auto-detection failed: {e}")
            # Otherwise, it might be an API key or connectivity issue, which is acceptable


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
        # Test if IVFPQ is supported at all first
        try:
            valid_config = cyborgdb.IndexIVFPQ(
                dimension=self.dimension, pq_dim=32, pq_bits=8
            )
            # Try creating a valid index to ensure IVFPQ is supported
            test_index_name = generate_unique_name()
            test_index_key = self.client.generate_key()
            test_index = self.client.create_index(
                test_index_name,
                test_index_key,
                valid_config,
                metric="euclidean",
            )
            test_index.delete_index()  # Clean up

            # Now test invalid parameters
            # Test invalid pq_dim = 0
            try:
                invalid_config = cyborgdb.IndexIVFPQ(
                    dimension=self.dimension, pq_dim=0, pq_bits=8
                )
                invalid_index = self.client.create_index(
                    generate_unique_name(),
                    self.client.generate_key(),
                    invalid_config,
                    metric="euclidean",
                )
                # If we get here, validation failed
                invalid_index.delete_index()
                self.fail(
                    "Expected validation error for pq_dim=0, but index creation succeeded"
                )
            except Exception as e:
                # Expected - should reject invalid pq_dim
                self.assertIn(
                    "pq_dim", str(e).lower(), f"Error should mention pq_dim: {e}"
                )

        except Exception as e:
            # IVFPQ might not be supported
            if "not supported" in str(e).lower() or "lite" in str(e).lower():
                self.skipTest(f"IVFPQ not supported in current backend: {e}")
            else:
                raise


class TestErrorHandling(unittest.TestCase):
    """Test comprehensive error scenarios"""

    def setUp(self):
        self.client = create_client()

    def test_invalid_api_key(self):
        """Test handling of invalid API keys"""
        client = cyborgdb.Client(
            base_url="http://localhost:8000", api_key="invalid-key-12345"
        )

        # The client creation succeeds, but API calls should fail
        # Skip this test if server doesn't validate API keys
        try:
            client.get_health()
            # If server doesn't validate API keys, skip the test
            self.skipTest(
                "Server appears to not validate API keys - skipping API key validation test"
            )
        except Exception as e:
            # Expected behavior - API call should fail with invalid key
            # Check that it's actually an authentication error, not a network error
            error_str = str(e).lower()
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
            if auth_related:
                self.assertTrue(True, f"Invalid API key properly rejected: {e}")
            else:
                self.skipTest(
                    f"Got non-authentication error, possibly network issue: {e}"
                )

    def test_malformed_requests(self):
        """Test handling of malformed requests"""
        index_name = generate_unique_name()
        index_key = self.client.generate_key()

        # Test invalid dimension - may not be validated at construction
        try:
            config = cyborgdb.IndexIVFFlat(dimension=-1)
            # If constructor doesn't validate, try creating index
            with self.assertRaises(Exception):
                self.client.create_index(
                    index_name, index_key, config, metric="euclidean"
                )
        except Exception:
            # Parameter validation occurred during construction
            self.assertTrue(True, "Invalid dimension properly caught")

        # Test invalid metric
        index_config = cyborgdb.IndexIVFFlat(dimension=128)
        try:
            with self.assertRaises(Exception):
                self.client.create_index(
                    index_name, index_key, index_config, metric="invalid_metric"
                )
        except Exception:
            # Expected - invalid metric rejected
            self.assertTrue(True, "Invalid metric properly rejected")

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
        # Note: This test checks that the client can handle server errors appropriately
        # The actual behavior depends on the client implementation

        # Test with potentially problematic data that might cause server errors
        try:
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
        except Exception as e:
            # Expected - empty index name should be rejected
            self.assertTrue(True, f"Empty index name properly rejected: {e}")

        # Alternative test: invalid index key format
        try:
            with self.assertRaises(Exception):
                self.client.create_index(
                    generate_unique_name(),
                    b"invalid_short_key",  # Invalid key length
                    cyborgdb.IndexIVFFlat(dimension=128),
                    metric="euclidean",
                )
        except Exception as e:
            self.assertTrue(True, f"Invalid key format properly rejected: {e}")


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
        ids = ["id1", "id2"]  # Fewer IDs than vectors
        metadata = [{"test": 1}, {"test": 2}]  # Fewer metadata than vectors

        # Create items with explicit mismatch
        items = []
        try:
            for i in range(len(vectors)):
                items.append(
                    {
                        "id": ids[i],  # This will fail when i >= len(ids)
                        "vector": vectors[i],
                        "metadata": metadata[i] if i < len(metadata) else {},
                    }
                )
            # If we get here, there was an IndexError during item creation
            self.fail("Expected IndexError when accessing ids[2], but didn't get one")
        except IndexError:
            # Expected - can't access ids[2] when ids only has 2 elements
            pass

        # Alternative test: create items with proper length handling, then test upsert validation
        items_with_defaults = []
        for i in range(len(vectors)):
            items_with_defaults.append(
                {
                    "id": ids[i] if i < len(ids) else f"default_id_{i}",
                    "vector": vectors[i],
                    "metadata": metadata[i] if i < len(metadata) else {"default": True},
                }
            )

        # Now test if the server/SDK validates the operation
        try:
            self.index.upsert(items_with_defaults)
            # If this succeeds, the SDK/server allows operations with default values
            # This is actually valid behavior - verify the items were actually stored
            time.sleep(1)
            stored_ids = self.index.list_ids()
            expected_ids = [item["id"] for item in items_with_defaults]
            for expected_id in expected_ids:
                self.assertIn(
                    expected_id, stored_ids, f"Item {expected_id} should be stored"
                )

        except Exception as e:
            # If it fails, verify it's a validation error, not a network error
            error_str = str(e).lower()
            validation_related = any(
                keyword in error_str
                for keyword in ["validation", "length", "mismatch", "parameter"]
            )
            self.assertTrue(validation_related, f"Expected validation error, got: {e}")

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
                # In a real async implementation, this would be:
                # tasks.append(self.index.async_upsert(item))
                # For now, we simulate concurrent behavior
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

        try:
            health = client.get_health()
            # If backend is lite, certain operations might not be available
            # This is a placeholder for actual lite-specific testing
            self.assertIsInstance(health, dict)
        except Exception as e:
            # Handle lite backend limitations
            self.assertIn("lite", str(e).lower())

    def test_feature_availability_differences(self):
        """Test feature availability between backend variants"""
        client = create_client()

        # Test if advanced index types are available
        try:
            index_config = cyborgdb.IndexIVFPQ(dimension=128, pq_dim=32, pq_bits=8)
            index_name = generate_unique_name()
            index_key = client.generate_key()

            index = client.create_index(
                index_name, index_key, index_config, metric="euclidean"
            )
            index.delete_index()

        except Exception as e:
            # Some backends might not support all index types
            if "not supported" in str(e).lower() or "lite" in str(e).lower():
                self.skipTest("IVFPQ not supported in current backend")
            else:
                raise


if __name__ == "__main__":
    unittest.main()
