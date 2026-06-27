"""Central configuration for the biomimicry spiral assistant.

Model selection goes through LiteLLM so any provider can be swapped via the
``BIOMIMICRY_MODEL`` env var. Default is ``gemini/gemini-2.5-flash``.

The pipeline is fully LLM-driven: there is no offline stub and no baked-in
heuristic knowledge. Everything below is operational tuning, not domain knowledge.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # biomimicry/.env, any CWD
load_dotenv()

# --- LLM ---------------------------------------------------------------------
# LiteLLM model string. Swap providers freely, e.g. "anthropic/claude-opus-4-8"
# or "openai/gpt-4o". Default: Gemini 2.5 Flash.
MODEL: str = os.getenv("BIOMIMICRY_MODEL", "gemini/gemini-2.5-flash")

# An API key is REQUIRED — there is no offline mode. We check the common provider
# keys for a friendly startup error; litellm itself reads the key from the env.
HAS_LLM_KEY: bool = bool(
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
)

# Generation vs critic temperatures (breadth via high temp, judging via low temp).
GEN_TEMPERATURE: float = float(os.getenv("BIOMIMICRY_GEN_TEMP", "0.9"))
CRITIC_TEMPERATURE: float = float(os.getenv("BIOMIMICRY_CRITIC_TEMP", "0.2"))

# Resilience: retry the model with exponential backoff (litellm handles backoff),
# then auto-fall back to lighter models on persistent overload/503.
LLM_NUM_RETRIES: int = int(os.getenv("BIOMIMICRY_LLM_RETRIES", "4"))
LLM_TIMEOUT: float = float(os.getenv("BIOMIMICRY_LLM_TIMEOUT", "60"))
LLM_FALLBACKS: tuple = tuple(
    m.strip() for m in os.getenv(
        "BIOMIMICRY_LLM_FALLBACKS", "gemini/gemini-2.5-flash-lite,gemini/gemini-2.0-flash"
    ).split(",") if m.strip()
)

# --- Stage fan-out + evaluator parameters ------------------------------------
DEFINE_HMW_PER_FUNCTION: int = int(os.getenv("BIOMIMICRY_HMW_PER_FUNCTION", "3"))
BIOLOGIZE_HDN_PER_QUESTION: int = int(os.getenv("BIOMIMICRY_HDN_PER_QUESTION", "3"))
DISCOVER_K_PER_HDN: int = int(os.getenv("BIOMIMICRY_DISCOVER_K", "8"))
# Capped retry-with-feedback for every evaluator (then proceed best-effort).
EVALUATOR_MAX_RETRIES: int = int(os.getenv("BIOMIMICRY_EVAL_RETRIES", "2"))

# --- Biomimicry Taxonomy -----------------------------------------------------
# Canonical Group -> Sub-group -> Function hierarchy the Biologize step maps onto.
TAXONOMY_PATH: str = os.getenv(
    "BIOMIMICRY_TAXONOMY_PATH",
    str(Path(__file__).resolve().parent / "taxonomy_hierarchy.json"),
)

# --- Retrieval ---------------------------------------------------------------
# Retrieval backend behind the Retriever factory. "lexical" = dependency-free
# BM25 over the local corpus (no key); "embedding" = LiteLLM embeddings;
# "weaviate" = local e5 vectors over a Weaviate Cloud collection.
RETRIEVAL_BACKEND: str = os.getenv("BIOMIMICRY_RETRIEVAL", "lexical")
EMBED_MODEL: str = os.getenv("BIOMIMICRY_EMBED_MODEL", "gemini/text-embedding-004")

RETRIEVAL_K: int = int(os.getenv("BIOMIMICRY_RETRIEVAL_K", "8"))  # default hits per query
BM25_K1, BM25_B = 1.5, 0.75                   # Okapi BM25 term-saturation / length-norm

# --- Weaviate vector backend (RETRIEVAL_BACKEND=weaviate) ---------------------
# Local sentence-transformers embeddings (intfloat/e5-large-v2, 1024-dim) stored in
# a Weaviate Cloud collection (bring-your-own vectors, vectorizer=none). e5 needs the
# "passage: " / "query: " input prefixes and L2-normalized output; see retrieval/e5_embedder.py.
E5_MODEL: str = os.getenv("BIOMIMICRY_E5_MODEL", "intfloat/e5-large-v2")
EMBED_DIM: int = int(os.getenv("BIOMIMICRY_EMBED_DIM", "1024"))
EMBED_BATCH: int = int(os.getenv("BIOMIMICRY_EMBED_BATCH", "32"))

# Search mode for the Weaviate backend:
#   filtered_hybrid (default) = pre-filter on function keys, then hybrid (BM25 + vector)
#   filtered_vector           = pre-filter on function keys, then pure vector
#   hybrid / vector           = no function filter (baselines for the perf comparison)
# HYBRID_ALPHA fuses keyword vs vector (0 = pure BM25, 1 = pure vector).
# FUNCTION_FILTER_LEVEL picks the filter granularity (subgroup = robust, leaf = precise);
# a filtered query that returns fewer than FILTER_MIN_HITS (0 => max(1, k//2)) relaxes
# leaf -> sub-group -> unfiltered. See retrieval/weaviate_store.py and function_keys.py.
WEAVIATE_SEARCH_MODE: str = os.getenv("BIOMIMICRY_WEAVIATE_SEARCH_MODE", "filtered_hybrid")
HYBRID_ALPHA: float = float(os.getenv("BIOMIMICRY_HYBRID_ALPHA", "0.5"))
FUNCTION_FILTER_LEVEL: str = os.getenv("BIOMIMICRY_FUNCTION_FILTER_LEVEL", "subgroup")
FILTER_MIN_HITS: int = int(os.getenv("BIOMIMICRY_FILTER_MIN_HITS", "0"))

WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY: str = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_COLLECTION: str = os.getenv("WEAVIATE_COLLECTION", "BiologicalStrategy")
# Weaviate Cloud's first gRPC handshake can exceed the client's ~2s default init check on
# higher-latency networks; widen the init window and skip the startup ping by default.
WEAVIATE_INIT_TIMEOUT: int = int(os.getenv("WEAVIATE_INIT_TIMEOUT", "30"))
WEAVIATE_SKIP_INIT_CHECKS: bool = os.getenv(
    "WEAVIATE_SKIP_INIT_CHECKS", "true").strip().lower() in ("1", "true", "yes", "on")
