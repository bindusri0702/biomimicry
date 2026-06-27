"""Retriever interface + factory — the offline/online seam, mirroring `LLM`.

Node code depends only on `Retriever.search()` and never names a concrete class,
so a future `LiveRetriever` (httpx + AskNature/OpenAlex/RSS, writing into the same
corpus cache) drops in by extending `get_retriever()` with zero changes upstream.

`search()` accepts an optional `filters` dict ({function, environment, scale, taxon})
so the Discover stage can drive metadata-aware retrieval. Translating those filters
into backend-specific constraints (e.g. a Weaviate `Filter`) is USER-OWNED — the
default backends accept the param and leave the matching seam to the user.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .. import config

_TOKEN = re.compile(r"[a-z][a-z\-]+")


def tokenize(text: str) -> list[str]:
    """Shared tokenizer (same pattern as metrics.py)."""
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class RetrievalHit:
    doc_id: str
    score: float            # relevance normalized 0..1 within a query's result set
    doc: dict               # the raw StrategyDoc dict
    source_tier: str
    query_variant: str      # the query string that produced this hit


class Retriever(ABC):
    @abstractmethod
    def search(self, query: str, *, k: int = config.RETRIEVAL_K,
               source_tier: str | None = None,
               filters: dict | None = None) -> list[RetrievalHit]:
        """Return up to k hits for a query.

        `source_tier` optionally restricts to one credibility tier. `filters` is an
        optional metadata constraint ({function, environment, scale, taxon}); how it
        is applied is backend-specific and user-owned.
        """


def get_retriever() -> Retriever:
    """Factory mirroring `LLM()`. Keyed off RETRIEVAL_BACKEND; lexical BM25 is the
    keyless default so the app always works without a vector store."""
    backend = config.RETRIEVAL_BACKEND

    if backend == "weaviate":
        # Local e5 vectors over a Weaviate Cloud collection (only the store is remote).
        from .weaviate_store import WeaviateRetriever
        return WeaviateRetriever()

    if backend == "embedding" and config.HAS_LLM_KEY:
        from .embedding import EmbeddingRetriever
        return EmbeddingRetriever()

    from .lexical import OfflineRetriever
    return OfflineRetriever()
