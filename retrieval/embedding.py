"""EmbeddingRetriever — online semantic backend behind the same interface.

Computes cosine similarity between a query embedding and precomputed doc
embeddings via LiteLLM. Active only when RETRIEVAL_BACKEND=embedding and a key is
present; otherwise the factory returns the lexical backend. Kept thin: the offline
BM25 path is the guaranteed default.
"""
from __future__ import annotations

import math

from .. import config
from .base import Retriever, RetrievalHit
from .corpus import StrategyDoc, load_corpus


class EmbeddingRetriever(Retriever):
    def __init__(self, docs: list[dict] | None = None):
        if not config.HAS_LLM_KEY:
            raise RuntimeError(
                "EmbeddingRetriever needs an API key. Set RETRIEVAL_BACKEND=lexical "
                "for keyless use, or provide an LLM provider key."
            )
        self.docs = docs if docs is not None else load_corpus()
        self._doc_vecs = [self._embed(StrategyDoc(**d).index_text()) for d in self.docs]

    @staticmethod
    def _embed(text: str) -> list[float]:
        from litellm import embedding
        resp = embedding(model=config.EMBED_MODEL, input=[text])
        return resp["data"][0]["embedding"]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, *, k: int = config.RETRIEVAL_K,
               source_tier: str | None = None,
               filters: dict | None = None) -> list[RetrievalHit]:
        # `filters` accepted for interface parity; applying it is user-owned.
        q = self._embed(query)
        scored = []
        for d, v in zip(self.docs, self._doc_vecs):
            if source_tier and d.get("source_tier") != source_tier:
                continue
            scored.append((max(0.0, self._cosine(q, v)), d))
        if not scored:
            return []
        top = max(s for s, _ in scored) or 1.0
        scored.sort(key=lambda x: (-x[0], x[1]["doc_id"]))
        return [
            RetrievalHit(doc_id=d["doc_id"], score=round(s / top, 4), doc=d,
                         source_tier=d.get("source_tier", "grey_literature"),
                         query_variant=query)
            for s, d in scored[:k]
        ]
