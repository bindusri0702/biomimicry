"""WeaviateRetriever — semantic backend over a Weaviate Cloud collection.

Stores the ORIGINAL scraped AskNature records (title, organism_name, functions_performed,
introduction, strategy, potential, related_innovation, source_url) with bring-your-own
BGE-M3 vectors (`BAAI/bge-m3`, 1024-dim, vectorizer=none, cosine). At query time the original properties are
mapped into a StrategyDoc-shaped dict (reusing `build_asknature_corpus.convert_one`) so the
Discover stage nodes consume Weaviate hits exactly like local-corpus hits.

The sole retrieval backend; needs WEAVIATE_URL + WEAVIATE_API_KEY.
"""
from __future__ import annotations

import logging

from .. import config
from .base import Retriever, RetrievalHit
from .build_asknature_corpus import convert_one
from .e5_embedder import embed_query

_log = logging.getLogger(__name__)

# Original-record property names returned on each Weaviate object (the canonical
# function_keys/subgroup_keys are indexed for filtering but rebuilt from functions_performed
# by convert_one, so they need not be fetched here).
PROPERTIES = (
    "title", "organism_name", "functions_performed", "introduction", "strategy",
    "potential", "related_innovation", "source_url", "slug",
)


def connect():
    """Open a Weaviate Cloud client. Raises a clear error if creds are unset."""
    if not config.WEAVIATE_URL or not config.WEAVIATE_API_KEY:
        raise RuntimeError(
            "Weaviate backend needs WEAVIATE_URL and WEAVIATE_API_KEY (set them in "
            "biomimicry/.env)."
        )
    import weaviate
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=config.WEAVIATE_URL,
        auth_credentials=Auth.api_key(config.WEAVIATE_API_KEY),
        skip_init_checks=config.WEAVIATE_SKIP_INIT_CHECKS,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=config.WEAVIATE_INIT_TIMEOUT, query=60, insert=180),
        ),
    )


def _props_to_doc(props: dict) -> dict:
    """Map stored original properties -> StrategyDoc-shaped dict for downstream nodes."""
    raw = {
        "title": props.get("title", ""),
        "organism_name": props.get("organism_name", ""),
        "functions_performed": props.get("functions_performed") or [],
        "introduction": props.get("introduction", ""),
        "strategy": props.get("strategy", ""),
        "potential": props.get("potential", ""),
        "related_innovation": props.get("related_innovation") or [],
        "source_url": props.get("source_url", ""),
    }
    slug = props.get("slug") or ""
    _, doc = convert_one(raw, slug)
    return doc


class WeaviateRetriever(Retriever):
    def __init__(self):
        self.client = connect()
        self.collection = self.client.collections.get(config.WEAVIATE_COLLECTION)

    def search(self, query: str, *, k: int = config.RETRIEVAL_K,
               source_tier: str | None = None,
               filters: dict | None = None) -> list[RetrievalHit]:
        """Vector / hybrid search, optionally pre-filtered on canonical function keys.

        `filters` (from discover.search_query_builder) may carry `function_keys` (leaf) and
        `subgroup_keys` (sub-group). The pre-filter restricts the candidate set BEFORE ranking;
        `WEAVIATE_SEARCH_MODE` picks vector vs hybrid and whether the filter is applied. To avoid
        over-restriction, a filtered query that under-fills falls back leaf -> sub-group -> none."""
        qvec = embed_query(query)
        mode = config.WEAVIATE_SEARCH_MODE
        chain = self._filter_chain(filters) if mode.startswith("filtered") else [(None, "none")]
        floor = config.FILTER_MIN_HITS or max(1, k // 2)

        rows: list[tuple[float, dict, str]] = []
        used = "none"
        for flt, level in chain:
            res = self._query(query, qvec, k, mode, flt)
            rows = self._rows(res, source_tier)
            used = level
            if flt is None or len(rows) >= floor:
                break
        if used != chain[0][1]:
            _log.info("function filter relaxed to %r (%d hits, floor %d) for query: %.60s",
                      used, len(rows), floor, query)
        if not rows:
            return []
        top = max(s for s, _, _ in rows) or 1.0
        rows.sort(key=lambda x: (-x[0], x[1]["doc_id"]))
        return [
            RetrievalHit(doc_id=doc["doc_id"], score=round(s / top, 4), doc=doc,
                         source_tier=tier, query_variant=query)
            for s, doc, tier in rows
        ]

    def _filter_chain(self, filters: dict | None):
        """Ordered (Filter|None, level) attempts: configured level -> sub-group -> unfiltered."""
        from weaviate.classes.query import Filter

        filters = filters or {}
        leaf = list(filters.get("function_keys") or [])
        sub = list(filters.get("subgroup_keys") or [])
        chain = []
        if config.FUNCTION_FILTER_LEVEL == "leaf" and leaf:
            chain.append((Filter.by_property("function_keys").contains_any(leaf), "leaf"))
        if sub:
            chain.append((Filter.by_property("subgroup_keys").contains_any(sub), "subgroup"))
        chain.append((None, "none"))
        return chain

    def _query(self, query: str, qvec, k: int, mode: str, flt):
        from weaviate.classes.query import MetadataQuery

        if mode in ("hybrid", "filtered_hybrid"):
            # Pre-filter applies first, then BM25(query) + vector(qvec) are fused by alpha.
            return self.collection.query.hybrid(
                query=query, vector=qvec, alpha=config.HYBRID_ALPHA, limit=k, filters=flt,
                return_metadata=MetadataQuery(score=True),
                return_properties=list(PROPERTIES),
            )
        return self.collection.query.near_vector(
            near_vector=qvec, limit=k, filters=flt,
            return_metadata=MetadataQuery(distance=True),
            return_properties=list(PROPERTIES),
        )

    @staticmethod
    def _rows(res, source_tier: str | None) -> list[tuple[float, dict, str]]:
        rows = []
        for obj in res.objects:
            doc = _props_to_doc(obj.properties)
            tier = doc.get("source_tier", "science_journalism")
            if source_tier and tier != source_tier:
                continue
            md = obj.metadata
            if getattr(md, "score", None) is not None:          # hybrid fused score
                rel = max(0.0, md.score)
            else:                                                # vector distance -> relevance
                dist = md.distance if md.distance is not None else 1.0
                rel = max(0.0, 1.0 - dist)
            rows.append((rel, doc, tier))
        return rows

    def close(self) -> None:
        self.client.close()
