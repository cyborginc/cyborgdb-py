"""
Demo API key generation for CyborgDB.

This module provides functionality to generate temporary demo API keys
from the CyborgDB demo API service.
"""

from datetime import datetime, timezone, timedelta
import os
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)


def get_demo_api_key(description: Optional[str] = None) -> str:
    """
    Generate a temporary demo API key from the CyborgDB demo API service.

    This function generates a temporary API key that can be used for demo purposes.
    The endpoint can be configured via the CYBORGDB_DEMO_ENDPOINT environment variable.

    Args:
        description: Optional description for the demo API key.
            Defaults to "Temporary demo API key" if not provided.

    Returns:
        str: The generated demo API key.

    Raises:
        ValueError: If the demo API key could not be generated.

    Example:
        >>> import cyborgdb
        >>> demo_key = cyborgdb.get_demo_api_key()
        >>> client = cyborgdb.Client("https://your-instance.com", demo_key)
    """

    # Use environment variable if set, otherwise use default endpoint
    endpoint = os.getenv(
        "CYBORGDB_DEMO_ENDPOINT",
        "https://api.cyborgdb.co/v1/api-key/manage/create-demo-key",
    )

    # Set default description if not provided
    if description is None:
        description = "Temporary demo API key"

    # Prepare the request payload
    payload = {"description": description}

    # Prepare headers
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    try:
        # Make the POST request (no authentication required)
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)

        # Check if request was successful
        response.raise_for_status()

        # Parse the response
        data = response.json()

        # Extract the API key
        api_key = data.get("apiKey", None)
        if not api_key:
            error_msg = "Demo API key not found in response."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log expiration info if available
        expires_at = data.get("expiresAt", None)
        if expires_at:
            # Calculate time left until expiration
            expires_at_dt = datetime.fromtimestamp(expires_at, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            time_left = expires_at_dt - now

            # Remove microseconds for cleaner display
            time_left = time_left - time_left % timedelta(seconds=1)
            logger.info("Demo API key will expire in %s", time_left)

        return api_key

    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to generate demo API key: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)
