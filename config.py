"""Central configuration for the biomimicry spiral assistant.

Model selection goes through LiteLLM. By default each LLM task is routed by complexity to
one of two NVIDIA Llama-Nemotron tiers (``MODEL_SUPER`` / ``MODEL_NANO``); set
``BIOMIMICRY_MODEL`` to pin every task to one model instead (e.g. ``gemini/gemini-2.5-flash``
or ``anthropic/claude-opus-4-8``). See llm.py for the routing.

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
# Primary model is Groq's Llama-3.3-70b (fast, reliable) behind the LiteLLM `groq/` provider.
# Both task tiers (SUPER for reasoning-heavy tasks, NANO for simple ones — see llm.py /
# COMPLEX_TASKS) point at it by default; they stay separate env vars so either can be
# retargeted (e.g. a lighter NANO model) without code edits.
MODEL_SUPER: str = os.getenv(
    "BIOMIMICRY_MODEL_SUPER", "mistral/mistral-small-latest")
MODEL_NANO: str = os.getenv(
    "BIOMIMICRY_MODEL_NANO", "mistral/mistral-small-latest")

# Optional escape hatch: when BIOMIMICRY_MODEL is set, ALL tasks use that one model and the
# super/nano routing is bypassed (None when unset — do NOT default it, or routing never fires).
MODEL_OVERRIDE: str | None = os.getenv("BIOMIMICRY_MODEL") or None

# Tasks routed to the SUPER tier (reasoning-heavy generation/extraction); every other task
# goes to NANO. Override with a comma list via BIOMIMICRY_COMPLEX_TASKS.
COMPLEX_TASKS: frozenset = frozenset(
    t.strip() for t in os.getenv(
        "BIOMIMICRY_COMPLEX_TASKS", "define,biologize_map,biologize_frame,abstract"
    ).split(",") if t.strip()
)

# An API key is REQUIRED — there is no offline mode. We check the common provider
# keys for a friendly startup error; litellm itself reads the key from the env.
HAS_LLM_KEY: bool = bool(
    os.getenv("GROQ_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY")
    or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("MISTRAL_API_KEY")
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
        "BIOMIMICRY_LLM_FALLBACKS", "gemini/gemini-2.5-flash,gemini/gemini-3.1-flash-lite"
    ).split(",") if m.strip()
)

# Per-call usage logging: after each model call, print a concise "[llm] … used N tok |
# req R/L …" line to stderr. Groq exposes remaining-request/-token headers; Gemini does
# not (shown as n/a). Set BIOMIMICRY_LOG_USAGE=0 to silence.
LOG_USAGE: bool = os.getenv(
    "BIOMIMICRY_LOG_USAGE", "1").strip().lower() in ("1", "true", "yes", "on")

# --- Stage fan-out + evaluator parameters ------------------------------------
DEFINE_HMW: int = int(os.getenv("BIOMIMICRY_HMW", "3"))
BIOLOGIZE_HDN_PER_QUESTION: int = int(os.getenv("BIOMIMICRY_HDN_PER_QUESTION", "3"))
DISCOVER_K_PER_HDN: int = int(os.getenv("BIOMIMICRY_DISCOVER_K", "4"))
# Capped retry-with-feedback for every evaluator (then proceed best-effort).
EVALUATOR_MAX_RETRIES: int = int(os.getenv("BIOMIMICRY_EVAL_RETRIES", "2"))

# Max simultaneous in-flight LLM requests for stage fan-out (Biologize/Discover). Default 1
# = fully sequential (safe for any provider tier, identical to the pre-parallel behavior).
# Raise it to overlap the per-item calls — via BIOMIMICRY_MAX_CONCURRENCY here, or at runtime
# with `demo.py --parallel [N]`. Keep it <= your provider tier's requests-per-second (RPS)
# to avoid 429s; the 429 -> fallback chain in llm.py is the backstop if you set it too high.
MAX_CONCURRENCY: int = int(os.getenv("BIOMIMICRY_MAX_CONCURRENCY", "1"))

# --- Biomimicry Taxonomy -----------------------------------------------------
# Canonical Group -> Sub-group -> Function hierarchy the Biologize step maps onto.
TAXONOMY_PATH: str = os.getenv(
    "BIOMIMICRY_TAXONOMY_PATH",
    str(Path(__file__).resolve().parent / "taxonomy_hierarchy.json"),
)

# --- Retrieval ---------------------------------------------------------------
# Retrieval is served exclusively by the Weaviate backend (local BGE-M3 vectors +
# hybrid search over a Weaviate Cloud collection); see the Weaviate section below.
RETRIEVAL_K: int = int(os.getenv("BIOMIMICRY_RETRIEVAL_K", "5"))  # default hits per query

# --- Weaviate vector backend (the sole retrieval backend) --------------------
# Local sentence-transformers embeddings (BAAI/bge-m3, 1024-dim) stored in a Weaviate
# Cloud collection (bring-your-own vectors, vectorizer=none). BGE-M3 is symmetric (no
# input prefix) with an 8192-token context, and we L2-normalize the output (cosine ==
# dot product); see retrieval/e5_embedder.py. Env var name kept for back-compat.
E5_MODEL: str = os.getenv("BIOMIMICRY_E5_MODEL", "BAAI/bge-m3")
EMBED_DIM: int = int(os.getenv("BIOMIMICRY_EMBED_DIM", "1024"))
EMBED_BATCH: int = int(os.getenv("BIOMIMICRY_EMBED_BATCH", "32"))
# Optional HuggingFace token for the BGE-M3 embedder download (higher rate limits /
# faster downloads). Anonymous access works fine; this is only used if set.
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None

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
