"""Local embeddings for the Weaviate vector backend.

BAAI/bge-m3 (1024-dim) via sentence-transformers, run locally on CPU. Unlike e5, BGE-M3
is symmetric and needs NO input prefix on either side; we still L2-normalize the output
(so cosine similarity == dot product, matching the index). Its 8192-token context fits a
whole strategy, so passages are no longer truncated the way e5's 512-token cap forced.
Ingest (`build_weaviate.py`) and the `WeaviateRetriever` both go through here, so
document- and query-side encoding can never drift.
"""
from __future__ import annotations

import contextlib
import re
import sys

from .. import config

_model = None  # lazily loaded singleton — importing this module must stay cheap

# hf_xet (huggingface_hub's Rust download backend) prints "You are sending unauthenticated
# requests to the HF Hub..." straight to sys.stderr — not via `warnings` or `logging`, so
# filters on those don't catch it. It's benign (BGE-M3 downloads fine anonymously), so we
# drop just that line while the model loads. Set config.HF_TOKEN (env HF_TOKEN) for higher
# rate limits / faster downloads and the message goes away at its source.
_HF_NOISE = "unauthenticated requests to the HF Hub"


class _DropLines:
    """sys.stderr proxy that suppresses lines containing `needle`, passing everything else."""

    def __init__(self, real, needle: str):
        self._real, self._needle = real, needle

    def write(self, s):
        if self._needle in s:
            s = "".join(ln for ln in s.splitlines(keepends=True) if self._needle not in ln)
        return self._real.write(s) if s else 0

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):  # delegate isatty/fileno/etc. to the real stream
        return getattr(self._real, name)


@contextlib.contextmanager
def _quiet_hf_stderr():
    prev = sys.stderr
    sys.stderr = _DropLines(prev, _HF_NOISE)
    try:
        yield
    finally:
        sys.stderr = prev


def _get_model():
    global _model
    if _model is None:
        # The stderr swap must wrap the import too: hf_xet's Rust extension binds to
        # sys.stderr at import time, so swapping only around construction is too late.
        # token=None (default) => anonymous; uses HF_TOKEN automatically once it's set.
        with _quiet_hf_stderr():
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(config.E5_MODEL, token=config.HF_TOKEN)
    return _model


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def build_passage(record: dict) -> str:
    """Assemble the text embedded for one strategy.

    High-signal fields (title/organism/functions) lead, then the full editorial body.
    BGE-M3's 8192-token context fits a whole strategy, so the body is no longer trimmed
    to fit a 512-token cap; ordering is kept only so the most discriminative signal comes
    first. Blank fields are skipped; `strategy` is always present, so every record yields
    a non-empty passage.
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
    """Embed stored documents (BGE-M3: no prefix), L2-normalized."""
    model = _get_model()
    vecs = model.encode(
        list(texts),
        normalize_embeddings=True,
        batch_size=config.EMBED_BATCH,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    """Embed a query (BGE-M3: no prefix), L2-normalized."""
    model = _get_model()
    vec = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vec.tolist()
