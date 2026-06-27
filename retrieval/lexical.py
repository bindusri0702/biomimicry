"""OfflineRetriever — dependency-free Okapi BM25 over the local corpus.

Pure stdlib (Counter + math.log). Deterministic: corpus loaded once in sorted
doc_id order, ties broken by doc_id, scores normalized to 0..1 per query.
"""
from __future__ import annotations

import math
from collections import Counter

from .. import config
from .base import Retriever, RetrievalHit, tokenize
from .corpus import load_corpus


class OfflineRetriever(Retriever):
    def __init__(self, docs: list[dict] | None = None,
                 k1: float | None = None, b: float | None = None):
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b
        self.docs = docs if docs is not None else load_corpus()
        self._build_index()

    def _build_index(self) -> None:
        from .corpus import StrategyDoc
        self._tf: list[Counter] = []
        self._len: list[int] = []
        df: Counter = Counter()
        for d in self.docs:
            text = StrategyDoc(**d).index_text()
            toks = tokenize(text)
            tf = Counter(toks)
            self._tf.append(tf)
            self._len.append(len(toks))
            df.update(tf.keys())
        self._n = len(self.docs)
        self._avgdl = (sum(self._len) / self._n) if self._n else 0.0
        self._idf = {
            term: math.log(1 + (self._n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def _bm25(self, query_terms: list[str], i: int) -> float:
        tf, dl = self._tf[i], self._len[i]
        if not dl:
            return 0.0
        score = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            idf = self._idf.get(term, 0.0)
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1))
            score += idf * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, *, k: int = config.RETRIEVAL_K,
               source_tier: str | None = None,
               filters: dict | None = None) -> list[RetrievalHit]:
        # `filters` (metadata-aware retrieval) is accepted for interface parity;
        # applying it is user-owned. Default lexical behaviour ignores it.
        terms = tokenize(query)
        raw = []
        for i, d in enumerate(self.docs):
            if source_tier and d.get("source_tier") != source_tier:
                continue
            s = self._bm25(terms, i)
            if s > 0:
                raw.append((s, d))
        if not raw:
            return []
        top = max(s for s, _ in raw)
        # Deterministic order: score desc, then doc_id asc.
        raw.sort(key=lambda x: (-x[0], x[1]["doc_id"]))
        return [
            RetrievalHit(
                doc_id=d["doc_id"],
                score=round(s / top, 4),
                doc=d,
                source_tier=d.get("source_tier", "grey_literature"),
                query_variant=query,
            )
            for s, d in raw[:k]
        ]
