"""Client module for CyborgDB."""

# Import all items from the client module for easier access
from cyborgdb.client.client import (
    Client,
    EncryptedIndex,
    IndexConfig,
    IndexIVF,
    IndexIVFPQ,
    IndexIVFFlat,
    generate_key
)

__all__ = [
    "Client",
    "EncryptedIndex",
    "IndexConfig",
    "IndexIVF",
    "IndexIVFPQ",
    "IndexIVFFlat",
    "generate_key"
]