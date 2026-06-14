"""
FastMCP RAG Search Server
=========================

Tools exposed:
  - semantic_search   dense vector similarity (OpenAI embeddings + Pinecone)
  - keyword_search    sparse BM25-style search
  - hybrid_search     dense + sparse combined
  - metadata_search   filter by document attributes
  - multi_search      all three strategies in parallel, merged

Run:
    python mcp_server/server.py
    # or via main.py:
    python main.py server
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Annotated, Any, Dict, List

from dotenv import load_dotenv
from fastmcp import FastMCP
from openai import AsyncOpenAI
from pinecone import Pinecone
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()

# ── Config from .env ──────────────────────────────────────────────────────────

PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-index")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "default")
PINECONE_DIMENSION  = int(os.getenv("PINECONE_DIMENSION", "1536"))

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

MAX_RETRIES    = int(os.getenv("MAX_RETRIES", "3"))
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Pydantic models ───────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)
    total: int = 0
    search_type: str
    query: str

# ── Pinecone + OpenAI singletons ──────────────────────────────────────────────

_pc     = Pinecone(api_key=PINECONE_API_KEY)
_index  = _pc.Index(PINECONE_INDEX_NAME)
_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

_ZERO_VECTOR: List[float] = [0.0] * PINECONE_DIMENSION

logger.info("Pinecone ready — index=%s  namespace=%s", PINECONE_INDEX_NAME, PINECONE_NAMESPACE)

# ── Retry decorator ───────────────────────────────────────────────────────────

def _retried(fn):
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )(fn)

# ── Helpers ───────────────────────────────────────────────────────────────────

@_retried
async def _embed(text: str) -> List[float]:
    resp = await _openai.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding

def _sparse(text: str) -> Dict[str, Any]:
    """BM25-style sparse vector via MD5 bucket hashing."""
    counts: Dict[int, float] = {}
    for tok in text.lower().split():
        b = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % 100_000
        counts[b] = counts.get(b, 0) + 1
    if not counts:
        return {"indices": [0], "values": [1.0]}
    total = sum(counts.values())
    return {"indices": list(counts.keys()), "values": [v / total for v in counts.values()]}

def _pinecone_filter(filters: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v if isinstance(v, dict) else {"$eq": v}) for k, v in filters.items()}

def _parse(matches: list) -> List[SearchResult]:
    out: List[SearchResult] = []
    for m in matches:
        meta = dict(m.metadata or {})
        text = meta.pop("text", meta.pop("content", ""))
        out.append(SearchResult(id=m.id, score=float(m.score), text=text, metadata=meta))
    return out

# ── Core search functions ─────────────────────────────────────────────────────

@_retried
async def _semantic(query: str, top_k: int) -> List[SearchResult]:
    vector = await _embed(query)
    resp = _index.query(vector=vector, top_k=top_k, namespace=PINECONE_NAMESPACE, include_metadata=True)
    return _parse(resp.matches)

@_retried
async def _keyword(query: str, top_k: int) -> List[SearchResult]:
    sparse = _sparse(query)
    try:
        resp = _index.query(vector=_ZERO_VECTOR, sparse_vector=sparse, top_k=top_k,
                            namespace=PINECONE_NAMESPACE, include_metadata=True)
        return _parse(resp.matches)
    except Exception as exc:
        logger.warning("Sparse unavailable (%s) — falling back to semantic", exc)
        return await _semantic(query, top_k)

@_retried
async def _hybrid(query: str, top_k: int, alpha: float = 0.5) -> List[SearchResult]:
    vector = await _embed(query)
    sparse = _sparse(query)
    try:
        resp = _index.query(
            vector=[v * alpha for v in vector],
            sparse_vector={"indices": sparse["indices"], "values": [v * (1 - alpha) for v in sparse["values"]]},
            top_k=top_k, namespace=PINECONE_NAMESPACE, include_metadata=True,
        )
        return _parse(resp.matches)
    except Exception as exc:
        logger.warning("Hybrid failed (%s) — falling back to semantic", exc)
        return await _semantic(query, top_k)

@_retried
async def _metadata(filters: Dict[str, Any], top_k: int) -> List[SearchResult]:
    resp = _index.query(vector=_ZERO_VECTOR, top_k=top_k, namespace=PINECONE_NAMESPACE,
                        include_metadata=True, filter=_pinecone_filter(filters))
    return _parse(resp.matches)

async def _multi(query: str, top_k: int) -> List[SearchResult]:
    results = await asyncio.gather(
        _semantic(query, top_k), _keyword(query, top_k), _hybrid(query, top_k),
        return_exceptions=True,
    )
    merged: Dict[str, SearchResult] = {}
    for batch in results:
        if isinstance(batch, Exception):
            logger.warning("multi_search sub-strategy failed: %s", batch)
            continue
        for r in batch:
            if r.id not in merged or r.score > merged[r.id].score:
                merged[r.id] = r
    return sorted(merged.values(), key=lambda x: x.score, reverse=True)[:top_k]

# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    name="RAG Search Server",
    instructions=(
        "Pinecone-backed RAG search server. "
        "Provides semantic, keyword, hybrid, metadata-filter, and multi-strategy search."
    ),
)

@mcp.tool()
async def semantic_search(
    query: Annotated[str, Field(description="Natural-language query for dense vector search")],
    top_k: Annotated[int, Field(description="Number of results (1-50)", ge=1, le=50)] = 5,
) -> Dict[str, Any]:
    """Semantic vector-similarity search using OpenAI dense embeddings. Best for conceptual queries."""
    logger.info("semantic_search | query='%.80s' top_k=%d", query, top_k)
    try:
        results = await _semantic(query, top_k)
        return SearchResponse(results=results, total=len(results), search_type="semantic", query=query).model_dump()
    except Exception as exc:
        logger.error("semantic_search failed: %s", exc, exc_info=True)
        raise

@mcp.tool()
async def keyword_search(
    query: Annotated[str, Field(description="Keyword / BM25 query string")],
    top_k: Annotated[int, Field(description="Number of results (1-50)", ge=1, le=50)] = 5,
) -> Dict[str, Any]:
    """Sparse BM25-style keyword search. Best for exact terms, names, acronyms, code identifiers."""
    logger.info("keyword_search | query='%.80s' top_k=%d", query, top_k)
    try:
        results = await _keyword(query, top_k)
        return SearchResponse(results=results, total=len(results), search_type="keyword", query=query).model_dump()
    except Exception as exc:
        logger.error("keyword_search failed: %s", exc, exc_info=True)
        raise

@mcp.tool()
async def hybrid_search(
    query: Annotated[str, Field(description="Query for hybrid dense + sparse search")],
    top_k: Annotated[int, Field(description="Number of results (1-50)", ge=1, le=50)] = 5,
) -> Dict[str, Any]:
    """Hybrid search combining dense embeddings and sparse BM25 (alpha=0.5). Best general-purpose retrieval."""
    logger.info("hybrid_search | query='%.80s' top_k=%d", query, top_k)
    try:
        results = await _hybrid(query, top_k)
        return SearchResponse(results=results, total=len(results), search_type="hybrid", query=query).model_dump()
    except Exception as exc:
        logger.error("hybrid_search failed: %s", exc, exc_info=True)
        raise

@mcp.tool()
async def metadata_search(
    filters: Annotated[
        Dict[str, Any],
        Field(description=(
            "Pinecone metadata filter dict. "
            "Equality: {'category': 'AI'}. "
            "Operators: {'year': {'$gte': 2023}}. "
            "IN: {'tag': {'$in': ['RAG', 'LLM']}}."
        )),
    ],
    top_k: Annotated[int, Field(description="Number of results (1-50)", ge=1, le=50)] = 5,
) -> Dict[str, Any]:
    """Metadata-filter search — retrieves documents matching structured attribute conditions."""
    logger.info("metadata_search | filters=%s top_k=%d", filters, top_k)
    try:
        results = await _metadata(filters, top_k)
        return SearchResponse(results=results, total=len(results), search_type="metadata", query=str(filters)).model_dump()
    except Exception as exc:
        logger.error("metadata_search failed: %s", exc, exc_info=True)
        raise

@mcp.tool()
async def multi_search(
    query: Annotated[str, Field(description="Query to run across all search strategies in parallel")],
    top_k: Annotated[int, Field(description="Number of results (1-50)", ge=1, le=50)] = 5,
) -> Dict[str, Any]:
    """Runs semantic, keyword, and hybrid in parallel then deduplicates. Maximum recall."""
    logger.info("multi_search | query='%.80s' top_k=%d", query, top_k)
    try:
        results = await _multi(query, top_k)
        return SearchResponse(results=results, total=len(results), search_type="multi", query=query).model_dump()
    except Exception as exc:
        logger.error("multi_search failed: %s", exc, exc_info=True)
        raise

# ── Entry-point ───────────────────────────────────────────────────────────────

def run_server() -> None:
    logger.info("Starting MCP server — %s:%d", MCP_HOST, MCP_PORT)
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)

if __name__ == "__main__":
    run_server()
