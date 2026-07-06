# cyborgdb/__init__.py

"""CyborgDB: A vector database platform."""

# Re-export classes from client module
from .client.client import Client

# Re-export from encrypted_index.py
from .client.encrypted_index import EncryptedIndex

# Re-export demo functionality
from .demo import get_demo_api_key

# Re-export sample dataset loader
from .datasets import (
    DEFAULT_SAMPLE_DATASET,
    SAMPLE_DATASETS_BASE_URL,
    SampleDataset,
    load_sample_dataset,
)

from importlib.metadata import PackageNotFoundError, version

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


try:
    __version__ = version("cyborgdb")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "Client",
    "EncryptedIndex",
    "CyborgVectorStore",
    "get_demo_api_key",
    "load_sample_dataset",
    "SampleDataset",
    "SAMPLE_DATASETS_BASE_URL",
    "DEFAULT_SAMPLE_DATASET",
    "__version__",
]
