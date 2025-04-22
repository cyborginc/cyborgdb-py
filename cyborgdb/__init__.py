"""CyborgDB: A vector database platform."""

# Re-export classes from client module
from cyborgdb.client.client import (
    Client,
    IndexConfig,
    IndexIVF,
    IndexIVFPQ, 
    IndexIVFFlat,
    generate_key
)

# Re-export from encrypted_index.py
from cyborgdb.client.encrypted_index import EncryptedIndex

__all__ = [
    "Client",
    "EncryptedIndex",
    "IndexConfig",
    "IndexIVF",
    "IndexIVFPQ",
    "IndexIVFFlat",
    "generate_key"
]