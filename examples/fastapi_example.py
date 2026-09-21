"""FastAPI example: async CyborgDB vector search.

Demonstrates ``AsyncClient`` and ``await index.query(…)`` inside FastAPI
route handlers.  Runs end-to-end against a local CyborgDB service instance.

Usage::

    pip install cyborgdb fastapi uvicorn
    CYBORGDB_API_KEY=your_key uvicorn examples.fastapi_example:app --reload

Environment variables:

    CYBORGDB_SERVICE_URL  Base URL of the CyborgDB service
                          (default: http://localhost:8000)
    CYBORGDB_API_KEY      API key (optional when auth is disabled)
    CYBORGDB_INDEX_NAME   Index to query (default: fastapi-demo)
    CYBORGDB_INDEX_KEY    Hex-encoded 32-byte index key (optional; not needed
                          for KMS-backed indexes)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cyborgdb import AsyncClient, AsyncEncryptedIndex

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SERVICE_URL: str = os.getenv("CYBORGDB_SERVICE_URL", "http://localhost:8000")
API_KEY: Optional[str] = os.getenv("CYBORGDB_API_KEY")
INDEX_NAME: str = os.getenv("CYBORGDB_INDEX_NAME", "fastapi-demo")
_INDEX_KEY_HEX: Optional[str] = os.getenv("CYBORGDB_INDEX_KEY")


def _resolve_index_key() -> Optional[bytes]:
    if _INDEX_KEY_HEX:
        import binascii

        return binascii.unhexlify(_INDEX_KEY_HEX)
    return None


# ---------------------------------------------------------------------------
# Lifespan: open / close the async client and load the index handle once
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = AsyncClient(base_url=SERVICE_URL, api_key=API_KEY)
    app.state.client = client
    app.state.index = await client.load_index(
        index_name=INDEX_NAME,
        index_key=_resolve_index_key(),
    )
    yield
    await client.close()
    app.state.client = None
    app.state.index = None


app = FastAPI(
    title="CyborgDB FastAPI Example",
    description="Async vector search with CyborgDB",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    vector: List[float]
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Proxy the CyborgDB health endpoint."""
    client: AsyncClient = app.state.client
    try:
        return await client.get_health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a vector similarity search against the configured index."""
    index: AsyncEncryptedIndex = app.state.index
    try:
        results = await index.query(
            query_vectors=request.vector,
            top_k=request.top_k,
            filters=request.filters,
            include=request.include,
        )
        if not results:
            return QueryResponse(results=[])
        return QueryResponse(
            results=results if isinstance(results[0], dict) else results[0]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/indexes")
async def list_indexes():
    """List all indexes accessible with the configured API key."""
    client: AsyncClient = app.state.client
    try:
        return {"indexes": await client.list_indexes()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
