import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from cyborgdb.demo import get_demo_api_key


class TestGetDemoApiKey(unittest.TestCase):
    """Unit tests for the get_demo_api_key function."""

    def setUp(self):
        """Set up test environment."""
        # Store original env var to restore later
        self.original_env = os.environ.get("CYBORGDB_DEMO_ENDPOINT")

    def tearDown(self):
        """Clean up test environment."""
        # Restore original env var
        if self.original_env is not None:
            os.environ["CYBORGDB_DEMO_ENDPOINT"] = self.original_env
        elif "CYBORGDB_DEMO_ENDPOINT" in os.environ:
            del os.environ["CYBORGDB_DEMO_ENDPOINT"]

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_success(self, mock_post):
        """Test successful demo API key generation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "apiKey": "demo_test_key_12345",
            "expiresAt": datetime.now(timezone.utc).timestamp() + 3600,
        }
        mock_post.return_value = mock_response

        # Call the function
        api_key = get_demo_api_key()

        # Verify the result
        self.assertEqual(api_key, "demo_test_key_12345")

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(
            call_args.kwargs["json"], {"description": "Temporary demo API key"}
        )
        self.assertEqual(
            call_args.kwargs["headers"],
            {"Content-Type": "application/json", "Accept": "application/json"},
        )

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_with_custom_description(self, mock_post):
        """Test demo API key generation with custom description."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "apiKey": "demo_test_key_67890",
        }
        mock_post.return_value = mock_response

        # Call the function with custom description
        custom_description = "My custom demo key"
        api_key = get_demo_api_key(description=custom_description)

        # Verify the result
        self.assertEqual(api_key, "demo_test_key_67890")

        # Verify the custom description was used
        call_args = mock_post.call_args
        self.assertEqual(call_args.kwargs["json"], {"description": custom_description})

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_uses_default_endpoint(self, mock_post):
        """Test that default endpoint is used when env var is not set."""
        # Ensure env var is not set
        if "CYBORGDB_DEMO_ENDPOINT" in os.environ:
            del os.environ["CYBORGDB_DEMO_ENDPOINT"]

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "apiKey": "demo_test_key_default",
        }
        mock_post.return_value = mock_response

        # Call the function
        api_key = get_demo_api_key()

        # Verify default endpoint was used
        call_args = mock_post.call_args
        self.assertEqual(
            call_args.args[0],
            "https://api.cyborgdb.co/v1/api-key/manage/create-demo-key",
        )
        self.assertEqual(api_key, "demo_test_key_default")

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_uses_env_endpoint(self, mock_post):
        """Test that custom endpoint from env var is used."""
        # Set custom endpoint
        custom_endpoint = "https://custom.api.example.com/demo-key"
        os.environ["CYBORGDB_DEMO_ENDPOINT"] = custom_endpoint

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "apiKey": "demo_test_key_custom",
        }
        mock_post.return_value = mock_response

        # Call the function
        api_key = get_demo_api_key()

        # Verify custom endpoint was used
        call_args = mock_post.call_args
        self.assertEqual(call_args.args[0], custom_endpoint)
        self.assertEqual(api_key, "demo_test_key_custom")

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_missing_api_key_in_response(self, mock_post):
        """Test handling of response missing apiKey field."""
        # Mock response without apiKey
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
        }
        mock_post.return_value = mock_response

        # Call the function and expect ValueError
        with self.assertRaises(ValueError) as context:
            get_demo_api_key()

        self.assertIn("Demo API key not found in response", str(context.exception))

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_http_error(self, mock_post):
        """Test handling of HTTP errors."""
        # Mock HTTP error
        import requests

        mock_post.side_effect = requests.exceptions.HTTPError("404 Not Found")

        # Call the function and expect ValueError
        with self.assertRaises(ValueError) as context:
            get_demo_api_key()

        self.assertIn("Failed to generate demo API key", str(context.exception))

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_connection_error(self, mock_post):
        """Test handling of connection errors."""
        # Mock connection error
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to connect")

        # Call the function and expect ValueError
        with self.assertRaises(ValueError) as context:
            get_demo_api_key()

        self.assertIn("Failed to generate demo API key", str(context.exception))

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        # Mock timeout error
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        # Call the function and expect ValueError
        with self.assertRaises(ValueError) as context:
            get_demo_api_key()

        self.assertIn("Failed to generate demo API key", str(context.exception))

    @patch("cyborgdb.demo.requests.post")
    def test_get_demo_api_key_with_expiration_info(self, mock_post):
        """Test that expiration info is logged correctly."""
        # Mock successful response with expiration
        mock_response = MagicMock()
        mock_response.status_code = 200
        future_timestamp = datetime.now(timezone.utc).timestamp() + 7200  # 2 hours
        mock_response.json.return_value = {
            "apiKey": "demo_test_key_expires",
            "expiresAt": future_timestamp,
        }
        mock_post.return_value = mock_response

        # Call the function
        with self.assertLogs("cyborgdb.demo", level="INFO") as log_context:
            api_key = get_demo_api_key()

        # Verify the result
        self.assertEqual(api_key, "demo_test_key_expires")

        # Verify expiration was logged
        self.assertTrue(
            any("Demo API key will expire in" in log for log in log_context.output)
        )


if __name__ == "__main__":
    unittest.main()
