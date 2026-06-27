"""Stage 3 - Discover: Challenge Brief v2 -> Challenge Brief v3.

Fans out retrieval per biologize question, then filters organisms with LLM
reasoning. Fully automated, no human gate. Sub-components (one node each):
  search_query_builder  -> per accepted HDN: a query + a metadata `filters` dict
  fanout_retriever      -> retrieve per HDN (Weaviate/lexical), entity-dedup the pool
  organism_profile_builder -> canonical BiologicalModel records
  filter_with_reasoning -> per organism: function-fit? environment-fit? + written
                           justification using the organism's strategy -> keep/drop
                           (per-HDN floor: keep top-1 by relevance if none survive)
  citation_ledger_writer-> append every kept model to the Citation Ledger
  compute_metrics       -> the Discover evaluation metrics
  finalize              -> stamps Challenge Brief v3

Retrieval goes through the pluggable `get_retriever()`; metadata filtering inside
the retriever is user-owned (see retrieval/base.py).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import config
from ..llm import LLM
from ..metrics import discover_metrics
from ..retrieval import get_retriever, tokenize
from ..retrieval.function_keys import keys_for_triple
from ..state import BiologicalModel, SpiralState, log_entry

STAGE = "discover"
_llm = LLM()           # injectable for tests
_retriever = get_retriever()


# --- nodes -------------------------------------------------------------------
def search_query_builder(state: SpiralState) -> dict:
    """One query + metadata filters per accepted biologize question.

    The HDN text is itself a natural-language search query. The filter is the question's
    verbatim taxonomy path (group/sub_group/function), re-written to the same canonical keys
    the corpus is indexed under (see retrieval/function_keys.py) so the retriever can
    pre-filter on an exact match."""
    dq_by_id = {q["id"]: q for q in state["defined_questions"]}
    ctx = state.get("context", {}) or {}
    environment = ctx.get("operating_environment", "")
    queries = []
    for h in state["hdn_questions"]:
        if not h.get("accepted"):
            continue
        define_q = (dq_by_id.get(h["define_question_id"]) or {}).get("text", "")
        leaf_key, sub_key = keys_for_triple(
            h.get("group", ""), h.get("sub_group", ""), h.get("function", ""))
        queries.append({
            "hdn_id": h["id"],
            "query": h["text"],
            "filters": {
                "function_keys": [leaf_key] if leaf_key else [],
                "subgroup_keys": [sub_key] if sub_key else [],
                "function": define_q,        # retained for downstream LLM fit-judge context
                "environment": environment,
                "scale": None,
                "taxon": None,
            },
        })
    return {"search_queries": queries,
            "spiral_log": [log_entry(STAGE, "queries_built", f"{len(queries)} queries")]}


def fanout_retriever(state: SpiralState) -> dict:
    """Retrieve per biologize question, then entity-dedup across the pool."""
    pooled: dict[str, dict] = {}
    for q in state["search_queries"]:
        for hit in _retriever.search(q["query"], k=config.DISCOVER_K_PER_HDN,
                                     filters=q.get("filters")):
            key = _merge_key(hit.doc)
            cur = pooled.get(key)
            if cur is None:
                pooled[key] = {
                    "doc": hit.doc, "doc_id": hit.doc_id, "relevance": hit.score,
                    "source_tier": hit.source_tier, "hdn_ids": [q["hdn_id"]],
                }
            else:
                cur["hdn_ids"] = sorted(set(cur["hdn_ids"]) | {q["hdn_id"]})
                if hit.score > cur["relevance"]:    # higher relevance wins canonical record
                    cur.update(doc=hit.doc, doc_id=hit.doc_id, relevance=hit.score,
                               source_tier=hit.source_tier)
    hits = sorted(pooled.values(), key=lambda x: (-x["relevance"], x["doc_id"]))
    return {"raw_hits": hits,
            "spiral_log": [log_entry(STAGE, "retrieved", f"{len(hits)} deduped hits")]}


def organism_profile_builder(state: SpiralState) -> dict:
    models = []
    for i, h in enumerate(state["raw_hits"]):
        d = h["doc"]
        models.append(BiologicalModel(
            id=i,
            organism_common=d.get("organism_common", ""),
            organism_scientific=d.get("organism_scientific", ""),
            strategy_summary=d.get("strategy_summary", ""),
            mechanism=d.get("mechanism", ""),
            function_addressed=d.get("function_addressed", []),
            environment=d.get("environment", ""),
            taxon=d.get("taxon", ""),
            scale=d.get("scale"),
            source_url=d.get("source_url", ""),
            source_tier=h.get("source_tier"),
            doc_id=h["doc_id"],
            hdn_ids=h["hdn_ids"],
            relevance_score=h["relevance"],
        ).model_dump())
    return {"biological_models": models,
            "spiral_log": [log_entry(STAGE, "profiles_built", f"{len(models)} models")]}


def filter_with_reasoning(state: SpiralState) -> dict:
    """Per organism: LLM judges function-fit + environment-fit and writes the rationale."""
    hdn_by_id = {h["id"]: h for h in state["hdn_questions"]}
    dq_by_id = {q["id"]: q for q in state["defined_questions"]}
    environment = (state.get("context", {}) or {}).get("operating_environment", "")
    models = [dict(m) for m in state["biological_models"]]

    for m in models:
        hids = [i for i in m.get("hdn_ids", []) if i in hdn_by_id]
        questions = [hdn_by_id[i]["text"] for i in hids]
        functions = sorted({(dq_by_id.get(hdn_by_id[i]["define_question_id"]) or {}).get("text", "")
                            for i in hids})
        functions = [f for f in functions if f]
        verdict = _eval_fit(m, questions, functions, environment)
        m["function_fit"] = verdict.get("function_fit")
        m["environment_fit"] = verdict.get("environment_fit")
        m["filter_reasoning"] = verdict.get("reasoning", "")
        m["keep"] = bool(verdict.get("keep"))

    _apply_per_hdn_floor(models, state["hdn_questions"])
    kept = sum(1 for m in models if m["keep"])
    return {"biological_models": models,
            "spiral_log": [log_entry(STAGE, "filtered_with_reasoning", f"{kept} kept")]}


def citation_ledger_writer(state: SpiralState) -> dict:
    """Persist every kept model — the anti-hallucination ground truth."""
    prov = {h["doc_id"]: h["doc"].get("provenance", "synthetic")
            for h in state.get("raw_hits", [])}
    entries = [{
        "stage": STAGE,
        "organism_common": m["organism_common"],
        "organism_scientific": m["organism_scientific"],
        "strategy": m["strategy_summary"][:160],
        "source_url": m["source_url"],
        "source_tier": m.get("source_tier"),
        "doc_id": m["doc_id"],
        "provenance": prov.get(m["doc_id"], "synthetic"),
        "hdn_ids": m["hdn_ids"],
        "filter_reasoning": m.get("filter_reasoning", ""),
    } for m in state["biological_models"] if m.get("keep")]
    return {"citation_ledger": entries,
            "spiral_log": [log_entry(STAGE, "citations_logged", f"{len(entries)} sources")]}


def compute_metrics(state: SpiralState) -> dict:
    return {"discover_metrics": discover_metrics(state["biological_models"], state["hdn_questions"]),
            "spiral_log": [log_entry(STAGE, "metrics_computed")]}


def finalize(state: SpiralState) -> dict:
    return {"version": "v3", "current_stage": STAGE,
            "spiral_log": [log_entry(STAGE, "challenge_brief_finalized", "Challenge Brief v3 ready")]}


# --- LLM helper --------------------------------------------------------------
def _eval_fit(model: dict, questions: list, functions: list, environment: str) -> dict:
    return _llm.complete_json(
        task="filter_with_reasoning",
        system=(
            "Decide whether a biological organism/strategy fits the biologize question(s). Judge:\n"
            " - function_fit: does the organism perform a function similar to the one asked?\n"
            " - environment_fit: does it live in / operate in a similar environment/context?\n"
            "Explain WHY it fits (or not) by referring to the organism's strategy. Keep it only "
            "if it genuinely fits. Return strict JSON: {\"function_fit\":bool,"
            "\"environment_fit\":bool,\"reasoning\":str,\"keep\":bool}."
        ),
        user=(f"Biologize question(s): {questions}\nTarget function(s): {functions}\n"
              f"Target environment/context: {environment}\n\n"
              f"Organism: {model.get('organism_common')} ({model.get('organism_scientific')})\n"
              f"Functions: {model.get('function_addressed')}\n"
              f"Environment: {model.get('environment')}\n"
              f"Strategy: {model.get('strategy_summary')}\n"
              f"Mechanism: {model.get('mechanism')}"),
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"model_id": model.get("id"), "questions": questions},
    )


# --- helpers -----------------------------------------------------------------
def _merge_key(doc: dict) -> str:
    """Entity key: organism (scientific, else common) + primary strategy token.

    Under-merges rather than over-merges: two distinct strategies of one organism
    stay separate records."""
    name = (doc.get("organism_scientific") or doc.get("organism_common") or "").lower().strip()
    kws = doc.get("keywords") or tokenize(doc.get("strategy_summary", ""))
    primary = (kws[0] if kws else "").lower()
    return f"{name}|{primary}"


def _apply_per_hdn_floor(models: list[dict], hdn_questions: list[dict]) -> None:
    """If a biologize question has zero kept organisms, retain its top-1 by relevance."""
    for h in hdn_questions:
        if not h.get("accepted"):
            continue
        hid = h["id"]
        for_hdn = [m for m in models if hid in (m.get("hdn_ids") or [])]
        if for_hdn and not any(m["keep"] for m in for_hdn):
            top = max(for_hdn, key=lambda m: (m.get("relevance_score") or 0, -m["id"]))
            top["keep"] = True
            top.setdefault("filter_reasoning", "")
            top["filter_reasoning"] = (top["filter_reasoning"]
                                       + " [floor: best available for this question]").strip()


# --- subgraph ----------------------------------------------------------------
def build_discover_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("search_query_builder", search_query_builder)
    g.add_node("fanout_retriever", fanout_retriever)
    g.add_node("organism_profile_builder", organism_profile_builder)
    g.add_node("filter_with_reasoning", filter_with_reasoning)
    g.add_node("citation_ledger_writer", citation_ledger_writer)
    g.add_node("compute_metrics", compute_metrics)
    g.add_node("finalize", finalize)

    g.add_edge(START, "search_query_builder")
    g.add_edge("search_query_builder", "fanout_retriever")
    g.add_edge("fanout_retriever", "organism_profile_builder")
    g.add_edge("organism_profile_builder", "filter_with_reasoning")
    g.add_edge("filter_with_reasoning", "citation_ledger_writer")
    g.add_edge("citation_ledger_writer", "compute_metrics")
    g.add_edge("compute_metrics", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
