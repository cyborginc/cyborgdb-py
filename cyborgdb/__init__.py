# cyborgdb/__init__.py

"""CyborgDB: A vector database platform."""

# Re-export classes from client module
from .client.client import (
    Client,
    IndexConfig,
    IndexIVF,
    IndexIVFPQ, 
    IndexIVFFlat,
    generate_key
)

# Re-export from encrypted_index.py
from .client.encrypted_index import EncryptedIndex

__all__ = [
    "Client",
    "EncryptedIndex",
    "IndexConfig",
    "IndexIVF",
    "IndexIVFPQ",
    "IndexIVFFlat",
    "generate_key"
]