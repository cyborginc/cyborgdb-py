"""Serve CyborgDB similarity search from a FastAPI app with AsyncClient.

One ``AsyncClient`` and one index handle are opened in the application
lifespan and shared by every request; the client is closed on shutdown.

Configure through the environment:

    CYBORGDB_SERVICE_URL   service base URL (default http://localhost:8000)
    CYBORGDB_API_KEY       API key for the service
    CYBORGDB_INDEX_NAME    name of an existing index to load
    CYBORGDB_INDEX_KEY     hex-encoded 32-byte index key; omit for a
                           KMS-backed index

Run with:

    pip install fastapi uvicorn
    uvicorn examples.fastapi_example:app

Then query it:

    curl -X POST http://127.0.0.1:8000/query \\
        -H 'Content-Type: application/json' \\
        -d '{"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 5}'
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cyborgdb import AsyncClient, CyborgDBError, NotFoundError, ValidationError


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_key_hex = os.getenv("CYBORGDB_INDEX_KEY")
    client = AsyncClient(
        os.getenv("CYBORGDB_SERVICE_URL", "http://localhost:8000"),
        api_key=os.getenv("CYBORGDB_API_KEY"),
    )
    try:
        app.state.index = await client.load_index(
            os.environ["CYBORGDB_INDEX_NAME"],
            bytes.fromhex(index_key_hex) if index_key_hex else None,
        )
        app.state.client = client
        yield
    finally:
        await client.close()


app = FastAPI(title="CyborgDB search", lifespan=lifespan)


class QueryBody(BaseModel):
    vector: List[float] = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=1000)
    filters: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int


@app.get("/health")
async def health() -> Dict[str, Any]:
    return await app.state.client.get_health()


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryBody) -> QueryResponse:
    try:
        results = await app.state.index.query(
            query_vectors=body.vector,
            top_k=body.top_k,
            filters=body.filters,
            include=body.include,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except CyborgDBError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    # An empty index, or a filter that matches nothing, is a normal answer.
    return QueryResponse(results=results or [], count=len(results or []))
