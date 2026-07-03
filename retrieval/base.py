"""Retriever interface + factory, mirroring `LLM`.

Node code depends only on `Retriever.search()` and never names a concrete class,
so the backend (currently the sole `WeaviateRetriever`) stays swappable behind
`get_retriever()` with zero changes upstream.

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


# Invisible/format chars scraped from web HTML that corrupt text and break search:
# soft hyphen, zero-width space/non-joiner/joiner, word-joiner, BOM.
_DELETE = dict.fromkeys([0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF], None)
# Non-breaking spaces -> a normal space so tokenization and Ctrl+F behave.
_SPACES = {0x00A0: " ", 0x202F: " "}
_CLEAN_MAP = {**_DELETE, **_SPACES}


def clean_text(text: str) -> str:
    """Strip invisible web-scraping artifacts and normalize non-breaking spaces."""
    return text.translate(_CLEAN_MAP) if text else text


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
    """Factory mirroring `LLM()`. Weaviate is the only retrieval backend: local
    BGE-M3 vectors + hybrid search over a Weaviate Cloud collection."""
    from .weaviate_store import WeaviateRetriever
    return WeaviateRetriever()
