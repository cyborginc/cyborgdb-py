# cyborgdb/__init__.py

"""CyborgDB: A vector database platform."""

# Re-export classes from client module
from .client.client import (
    Client,
    IndexConfig,
    IndexIVF,
    IndexIVFPQ,
    IndexIVFFlat
    IndexIVFFlat

)

# Re-export from encrypted_index.py
from .client.encrypted_index import EncryptedIndex

# Try to import LangChain integration (optional dependency)
try:
    from .integrations.langchain import CyborgVectorStore
except ImportError:
    # Create a placeholder that raises a helpful error when accessed
    class CyborgVectorStore:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CyborgVectorStore requires LangChain dependencies. "
                "Please install them with: pip install cyborgdb[langchain]"
            )

        def __class_getitem__(cls, item):
            raise ImportError(
                "CyborgVectorStore requires LangChain dependencies. "
                "Please install them with: pip install cyborgdb[langchain]"
            )


__all__ = [
    "Client",
    "EncryptedIndex",
    "IndexConfig",
    "IndexIVF",
    "IndexIVFPQ",
    "IndexIVFFlat",
    "CyborgVectorStore"
]
