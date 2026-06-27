"""Local e5-large-v2 embeddings for the Weaviate vector backend.

intfloat/e5-large-v2 (1024-dim) via sentence-transformers, run locally on CPU. The
model is asymmetric and REQUIRES input prefixes — ``"passage: "`` for stored documents
and ``"query: "`` for queries — and works best with L2-normalized output (so cosine
similarity == dot product). Ingest (`build_weaviate.py`) and the `WeaviateRetriever`
both go through here, so document- and query-side encoding can never drift.
"""
from __future__ import annotations

import re

from .. import config

_model = None  # lazily loaded singleton — importing this module must stay cheap


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.E5_MODEL)
    return _model


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def build_passage(record: dict) -> str:
    """Assemble the text embedded for one strategy (without the "passage: " prefix).

    High-signal fields first, because e5-large-v2 truncates at 512 tokens — the tail
    (end of `strategy` / `potential`) is what gets dropped. Blank fields are skipped;
    `strategy` is always present, so every record yields a non-empty passage.
    """
    title = _clean(record.get("title", ""))
    organism = _clean(record.get("organism_name", ""))
    funcs = "; ".join(f for f in (_clean(x) for x in record.get("functions_performed", [])) if f)

    head_bits = []
    if title:
        head_bits.append(f"{title}.")
    if organism:
        head_bits.append(f"Organism: {organism}.")
    if funcs:
        head_bits.append(f"Functions: {funcs}.")

    body_bits = [_clean(record.get(k, "")) for k in ("introduction", "strategy", "potential")]
    parts = [" ".join(head_bits)] + [b for b in body_bits if b]
    return "\n".join(p for p in parts if p).strip()


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed stored documents: prepend the e5 'passage:' prefix, L2-normalize."""
    model = _get_model()
    inputs = [f"passage: {t}" for t in texts]
    vecs = model.encode(
        inputs,
        normalize_embeddings=True,
        batch_size=config.EMBED_BATCH,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    """Embed a query: prepend the e5 'query:' prefix, L2-normalize."""
    model = _get_model()
    vec = model.encode(
        f"query: {text}",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.tolist()
