"""
Sample dataset loader for CyborgDB.

Fetches a small reference dataset hosted on S3 on demand and caches it locally,
so quickstart and test code can populate an index without bundling data into
the SDK. Hosting the dataset out-of-band keeps the SDK lean and lets us iterate
the dataset without cutting an SDK release.

Example:
    >>> import cyborgdb
    >>> dataset = cyborgdb.load_sample_dataset()
    >>> index = client.create_index(index_name="demo", index_key=index_key)
    >>> index.upsert(dataset.items)
"""

import gzip
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Base URL for hosted sample datasets (public-read S3 bucket). Datasets live at
# versioned per-dataset paths (``<name>/v<n>/dataset.json.gz``), so the dataset
# can be iterated without an SDK release: re-upload under a new version path and
# bump the entry in ``_DATASETS``.
SAMPLE_DATASETS_BASE_URL = "https://cyborgdb-sample-datasets.s3.amazonaws.com"

# Default dataset returned by ``load_sample_dataset()`` with no arguments.
DEFAULT_SAMPLE_DATASET = "quickstart-75k"

# Catalog of available datasets -> their object path within the bucket.
_DATASETS: Dict[str, str] = {
    "quickstart-75k": "quickstart-75k/v1/dataset.json.gz",
}

# Number of leading ``queries`` exposed as ``sample_queries`` for quick demos.
_NUM_SAMPLE_QUERIES = 10


@dataclass
class SampleDataset:
    """A fully-loaded sample dataset, ready to upsert and query.

    Combines dataset metadata, loader-derived convenience fields
    (``items``, ``sample_queries``, ``example_filters``), the raw parallel
    arrays (``ids`` / ``vectors`` / ``metadata``), and the ground-truth fixture
    data (``queries``, ``*_neighbors``, ``*_recall``, ...) used to validate
    recall/accuracy. Arrays are aligned by index.
    """

    # ---- dataset metadata ----
    name: str
    version: int
    description: str
    dimension: int
    metric: str
    count: int

    # ---- convenience (built by the loader) ----
    items: List[Dict[str, Any]]
    sample_queries: List[List[float]]
    example_filters: List[Dict[str, Any]]

    # ---- raw parallel arrays (aligned by index) ----
    ids: List[str]
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]

    # ---- ground-truth fixture data (for recall / accuracy validation) ----
    queries: List[List[float]]
    metadata_queries: List[Dict[str, Any]]
    metadata_query_names: List[str]
    untrained_neighbors: List[List[int]]
    trained_neighbors: List[List[int]]
    untrained_metadata_matches: List[List[int]]
    trained_metadata_matches: List[List[int]]
    untrained_metadata_neighbors: List[List[List[int]]]
    trained_metadata_neighbors: List[List[List[int]]]
    untrained_recall: float
    trained_recall: float
    num_untrained_vectors: int
    num_trained_vectors: int


def _default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return Path(base) / "cyborgdb"


def _hydrate(raw: Dict[str, Any]) -> SampleDataset:
    """Build the loader-derived convenience fields from the raw arrays.

    The hosted artifact and the local cache store only the raw arrays (no
    duplicated vectors), so ``items`` and ``sample_queries`` are reconstructed
    on every load.
    """
    ids = raw["ids"]
    vectors = raw["vectors"]
    metadata = raw["metadata"]
    items = [
        {"id": ids[i], "vector": vectors[i], "metadata": metadata[i]}
        for i in range(len(ids))
    ]
    return SampleDataset(
        name=raw["name"],
        version=raw["version"],
        description=raw["description"],
        dimension=raw["dimension"],
        metric=raw["metric"],
        count=raw["count"],
        items=items,
        sample_queries=raw["queries"][:_NUM_SAMPLE_QUERIES],
        example_filters=raw["exampleFilters"],
        ids=ids,
        vectors=vectors,
        metadata=metadata,
        queries=raw["queries"],
        metadata_queries=raw["metadata_queries"],
        metadata_query_names=raw["metadata_query_names"],
        untrained_neighbors=raw["untrained_neighbors"],
        trained_neighbors=raw["trained_neighbors"],
        untrained_metadata_matches=raw["untrained_metadata_matches"],
        trained_metadata_matches=raw["trained_metadata_matches"],
        untrained_metadata_neighbors=raw["untrained_metadata_neighbors"],
        trained_metadata_neighbors=raw["trained_metadata_neighbors"],
        untrained_recall=raw["untrained_recall"],
        trained_recall=raw["trained_recall"],
        num_untrained_vectors=raw["num_untrained_vectors"],
        num_trained_vectors=raw["num_trained_vectors"],
    )


def load_sample_dataset(
    name: str = DEFAULT_SAMPLE_DATASET,
    cache_dir: Optional[str] = None,
    force_download: bool = False,
) -> SampleDataset:
    """Load a hosted sample dataset, fetching from S3 on first use and caching
    the decompressed copy locally for subsequent calls.

    Args:
        name: Dataset name (default: ``"quickstart-75k"``).
        cache_dir: Directory to cache the decompressed dataset in. Defaults to
            ``$XDG_CACHE_HOME/cyborgdb`` or ``~/.cache/cyborgdb``.
        force_download: Re-download even if a cached copy exists.

    Returns:
        SampleDataset: The parsed dataset, ready to ``upsert`` and ``query``.

    Raises:
        ValueError: If the dataset name is unknown.
        RuntimeError: If the download fails.
    """
    if name not in _DATASETS:
        known = ", ".join(_DATASETS)
        raise ValueError(
            f'Unknown sample dataset "{name}". Available datasets: {known}.'
        )

    object_path = _DATASETS[name]
    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir()
    # Cache key mirrors the versioned object path so a dataset bump never serves
    # a stale cached copy.
    cache_name = object_path.replace("/", "_")
    if cache_name.endswith(".gz"):
        cache_name = cache_name[: -len(".gz")]
    cache_file = cache_root / cache_name

    if not force_download and cache_file.exists():
        try:
            return _hydrate(json.loads(cache_file.read_text("utf-8")))
        except (ValueError, KeyError, OSError):
            # Corrupt cache -- fall through and re-download.
            pass

    url = f"{SAMPLE_DATASETS_BASE_URL}/{object_path}"
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f'Failed to download sample dataset "{name}" from {url}: {exc}'
        ) from exc

    # The object is stored as an opaque gzip blob (no Content-Encoding: gzip),
    # so requests does not auto-decompress -- we own the gunzip step.
    text = gzip.decompress(response.content).decode("utf-8")
    raw = json.loads(text)

    # Best-effort local cache of the raw payload; a failed write must not break
    # the load. items/sample_queries are rebuilt by _hydrate() on read.
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, "utf-8")
    except OSError:
        pass

    return _hydrate(raw)
