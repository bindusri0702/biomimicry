"""Stage 3 - Discover: Challenge Brief v2 -> Challenge Brief v3.

Fans out retrieval per biologize question, then filters organisms with LLM
reasoning. Fully automated, no human gate. Sub-components (one node each):
  search_query_builder  -> per accepted HDN: a query + a metadata `filters` dict
  fanout_retriever      -> retrieve per HDN (Weaviate), entity-dedup the pool
  organism_profile_builder -> canonical BiologicalModel records
  filter_with_reasoning -> per HDN batch (all organisms retrieved for one HDN graded in a
                           single call): functional-relevance (yes|partial|no) and
                           mechanistic-adequacy (sufficient|thin|unusable) + reasoning;
                           keep if any HDN passes rel∈{yes,partial} AND adeq∈{sufficient,thin}
                           (per-HDN floor: keep top-1 by relevance if none survive)
  citation_ledger_writer-> append every kept model to the Citation Ledger

Retrieval goes through the pluggable `get_retriever()`; metadata filtering inside
the retriever is user-owned (see retrieval/base.py). The Discover metrics are derived
on demand from the final state in demo.py (see metrics.discover_metrics).
"""
from __future__ import annotations

import atexit
import json

from langgraph.graph import END, START, StateGraph

from .. import config
from ..llm import LLM
from ..parallel import bounded_map
from ..retrieval import get_retriever, tokenize
from ..retrieval.function_keys import keys_for_triple
from ..schemas import DiscoverEvalResponse
from ..state import BiologicalModel, SpiralState, log_entry

STAGE = "discover"
_llm = LLM()           # injectable for tests
_retriever = get_retriever()

# The retriever holds a live Weaviate connection for the process lifetime; close it
# at interpreter exit so the socket isn't leaked (weaviate emits a ResourceWarning
# otherwise). Guarded + best-effort so a backend without close() or a shutdown-time
# error never crashes exit.
if hasattr(_retriever, "close"):
    @atexit.register
    def _close_retriever() -> None:
        try:
            _retriever.close()
        except Exception:  # noqa: BLE001 - shutdown best-effort
            pass


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
    """Per HDN batch: LLM grades functional-relevance + mechanistic-adequacy for all
    organisms retrieved for one HDN in a single call.

    Each organism is still evaluated against every HDN question that retrieved it (the
    lens `approach` matters), and kept if at least one HDN pair passes the filter rule.
    Batching by HDN cuts the call count from one-per-pair to one-per-HDN. The deciding
    pair's verdict is surfaced on the flat fields; all pair verdicts are kept in
    `hdn_verdicts` for traceability."""
    hdn_by_id = {h["id"]: h for h in state["hdn_questions"]}
    models = [dict(m) for m in state["biological_models"]]
    for m in models:
        m["hdn_verdicts"] = []

    # Group organisms by the HDN that retrieved them — one grader batch per HDN.
    models_for_hdn: dict[int, list[dict]] = {}
    for m in models:
        for hid in m.get("hdn_ids", []):
            if hid in hdn_by_id:
                models_for_hdn.setdefault(hid, []).append(m)

    # Parallelize only the grader call across HDNs. The append below MUST stay single-threaded:
    # one organism can belong to several HDN batches (its hdn_ids), so concurrent appends to the
    # same model's hdn_verdicts list would race.
    def _eval_one(hid: int) -> tuple[int, dict]:
        h = hdn_by_id[hid]
        return hid, _eval_discover_batch(h["text"], h.get("approach", ""), models_for_hdn[hid])

    verdicts_by_hid = dict(bounded_map(_eval_one, list(models_for_hdn.keys())))

    for hid, batch in models_for_hdn.items():
        h = hdn_by_id[hid]
        verdicts = verdicts_by_hid.get(hid, {})
        for m in batch:
            v = verdicts.get(m["id"], {})
            rel = v.get("functional_relevance")
            adeq = v.get("mechanistic_adequacy")
            m["hdn_verdicts"].append({
                "hdn_id": hid,
                "approach": h.get("approach", ""),
                "functional_relevance": rel,
                "mechanistic_adequacy": adeq,
                "relevance_reasoning": v.get("relevance_reasoning", ""),
                "adequacy_reasoning": v.get("adequacy_reasoning", ""),
                "passes": _passes(rel, adeq),
            })

    for m in models:
        m["hdn_verdicts"].sort(key=lambda v: v["hdn_id"])   # match hdn_ids (sorted) ordering
        m["keep"] = any(v["passes"] for v in m["hdn_verdicts"])
        deciding = _deciding_verdict(m["hdn_verdicts"])
        m["functional_relevance"] = deciding.get("functional_relevance") if deciding else None
        m["mechanistic_adequacy"] = deciding.get("mechanistic_adequacy") if deciding else None
        m["relevance_reasoning"] = deciding.get("relevance_reasoning", "") if deciding else ""
        m["adequacy_reasoning"] = deciding.get("adequacy_reasoning", "") if deciding else ""

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
        "functional_relevance": m.get("functional_relevance"),
        "mechanistic_adequacy": m.get("mechanistic_adequacy"),
        "relevance_reasoning": m.get("relevance_reasoning", ""),
        "adequacy_reasoning": m.get("adequacy_reasoning", ""),
    } for m in state["biological_models"] if m.get("keep")]
    return {"citation_ledger": entries,
            "spiral_log": [log_entry(STAGE, "citations_logged", f"{len(entries)} sources")]}


# --- LLM helper --------------------------------------------------------------
_EVAL_DISCOVER_SYSTEM = (
    "You are a calibrated grader for the Discover step of the Biomimicry Design Spiral. You "
    "evaluate a BATCH of retrieved biological strategies that were ALL retrieved for the SAME "
    "\"How does nature...?\" (HDN) question. For each, judge whether the RIGHT content was "
    "retrieved and whether it is USABLE downstream. The strategy text is verbatim from source "
    "(extractive, not generated) - do NOT assess factual accuracy or citations; assume the "
    "content is true to source.\n\n"

    "INDEPENDENCE - READ FIRST (this is a batch of retrieval candidates; grade as if each "
    "were alone):\n"
    "- These strategies were retrieved for the same question, so they will resemble one "
    "another. Judge each ONLY against the HDN question and the rubric - NEVER against the "
    "other strategies. This is absolute grading, not ranking.\n"
    "- Do NOT pick a 'best' one, do NOT rank, do NOT grade on a curve. Multiple strategies "
    "can all be 'yes', or all be 'no'. If every candidate genuinely performs the function, "
    "every one gets 'yes' - retrieval returning several good hits is a success, not a "
    "quota to distribute.\n"
    "- A strategy being the strongest IN THIS BATCH does not make it 'yes'; a weak batch "
    "does not promote its least-bad member. Each is measured against the question, not its "
    "neighbors.\n"
    "- Verdicts on earlier items have ZERO bearing on later ones. Item order is "
    "meaningless; grade the last with the same care as the first.\n\n"

    "Within each item you score TWO INDEPENDENT dimensions. They are orthogonal: a strategy "
    "can be relevant but too vague to use, or richly detailed but about the wrong function. "
    "Score each on its own axis; do not let one pull the other.\n\n"

    "INPUTS\n"
    "- hdn_question: the question all strategies were retrieved for (applies to ALL items).\n"
    "- approach: direct | analogous | inverted.\n"
    "- items: an array; each element is { id, organism, content } (content is the verbatim "
    "retrieved record). Echo each item's id back verbatim in your output.\n\n"

    "DIMENSION 1 - FUNCTIONAL RELEVANCE  (axis: right content vs keyword collision)\n"
    "Does the organism's mechanism actually perform the FUNCTION the question asks about, or "
    "was it retrieved on semantic/keyword proximity while serving a DIFFERENT function? This "
    "is the classic vector-search miss: shared vocabulary, wrong function (e.g. a "
    "fire-adapted seed retrieved for 'how does nature interrupt combustion?' - all the fire "
    "words match, but germinating after a burn does not interrupt combustion).\n"
    "For approach = analogous or inverted, the organism's CONTEXT is expected to differ from "
    "the challenge - do not penalize a different context; judge only whether the mechanism "
    "serves the asked function.\n"
    "Verdict: yes | partial | no.\n\n"

    "DIMENSION 2 - MECHANISTIC ADEQUACY  (axis: usable how-it-works detail)\n"
    "Does the retrieved content carry enough how-it-works detail for the Abstract step to "
    "extract a transferable principle, or only a vague outcome? ('the bark resists heat' is "
    "thin; 'an insulating low-conductivity outer layer delays heat transfer inward' is "
    "sufficient.) This judges the retrieved CHUNK, not the organism's true biology - if the "
    "mechanism is absent from the content, it is thin/unusable even if the organism is "
    "relevant.\n"
    "Verdict: sufficient | thin | unusable.\n\n"

    "DISSOCIATING EXAMPLE - AXES COME APART (within one item):\n"
    "- A strategy that clearly performs the function but only states the outcome -> "
    "functional_relevance: yes; mechanistic_adequacy: thin.\n"
    "- A detailed, mechanism-rich strategy that actually serves a different function -> "
    "functional_relevance: no; mechanistic_adequacy: sufficient.\n\n"

    "DISSOCIATING EXAMPLE - ITEMS COME APART (verdicts must not drift across a batch):\n"
    "- If four strong on-function strategies are followed by a fifth that only shares "
    "vocabulary, the fifth is STILL functional_relevance: no. The four good hits before it "
    "change nothing. And if all five genuinely serve the function, all five are 'yes' - do "
    "not demote any to force spread.\n\n"

    "OUTPUT\n"
    "A single JSON object and nothing else - no markdown fences, no commentary, no trailing "
    "text. It has exactly one field:\n"
    "  verdicts : an array with ONE object per input item, in the SAME order as the input, "
    "and with the SAME COUNT as the input. Do not merge, split, skip, dedupe, or add items - "
    "even if two strategies look near-identical, grade and return BOTH. Each object has "
    "exactly these fields, in this order:\n"
    "    id                    echo the input item's id verbatim\n"
    "    relevance_reasoning   a sentence or two\n"
    "    functional_relevance  yes | partial | no\n"
    "    adequacy_reasoning    a sentence or two\n"
    "    mechanistic_adequacy  sufficient | thin | unusable\n\n"

    "Before returning, confirm len(verdicts) == number of input items and every input id "
    "appears exactly once."
)


def _eval_discover_batch(hdn_question: str, approach: str,
                         models: list[dict]) -> dict[int, dict]:
    """Grade a batch of retrieved strategies (same HDN) on relevance + adequacy.

    Returns {model_id: verdict}. Content is built exactly as before (strategy_summary +
    mechanism) so verdicts are unchanged from the per-pair grader — only batched."""
    items = []
    for m in models:
        organism = f"{m.get('organism_common', '')} ({m.get('organism_scientific', '')})".strip()
        content = "\n".join(p for p in (m.get("strategy_summary", ""),
                                        m.get("mechanism", "")) if p)
        items.append({"id": m["id"], "organism": organism, "content": content})
    raw = _llm.complete(
        task="evaluate_discover",
        system=_EVAL_DISCOVER_SYSTEM,
        user=(f'hdn_question:\n"""\n{hdn_question}\n"""\n\n'
              f'approach:\n"""\n{approach}\n"""\n\n'
              f'items:\n"""\n{json.dumps(items, ensure_ascii=False)}\n"""'),
        schema=DiscoverEvalResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"hdn_question": hdn_question, "n_items": len(items)},
    )
    # .model_dump() each verdict so the dict-based verdict-rule helpers stay unchanged.
    return {v.id: v.model_dump() for v in raw.verdicts}


# --- verdict rules -----------------------------------------------------------
_REL_RANK = {"yes": 2, "partial": 1, "no": 0}
_ADEQ_RANK = {"sufficient": 2, "thin": 1, "unusable": 0}


def _passes(functional_relevance: str | None, mechanistic_adequacy: str | None) -> bool:
    """The keep rule: functional_relevance ∈ {yes, partial} AND adequacy ∈ {sufficient, thin}."""
    return (functional_relevance in {"yes", "partial"}
            and mechanistic_adequacy in {"sufficient", "thin"})


def _deciding_verdict(verdicts: list[dict]) -> dict | None:
    """The verdict surfaced on the flat fields: the strongest passing pair, else strongest overall."""
    if not verdicts:
        return None
    return max(verdicts, key=lambda v: (
        v["passes"],
        _REL_RANK.get(v.get("functional_relevance"), -1),
        _ADEQ_RANK.get(v.get("mechanistic_adequacy"), -1),
    ))


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
            top["relevance_reasoning"] = ((top.get("relevance_reasoning", "") or "")
                                          + " [floor: best available for this question]").strip()


# --- subgraph ----------------------------------------------------------------
def build_discover_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("search_query_builder", search_query_builder)
    g.add_node("fanout_retriever", fanout_retriever)
    g.add_node("organism_profile_builder", organism_profile_builder)
    g.add_node("filter_with_reasoning", filter_with_reasoning)
    g.add_node("citation_ledger_writer", citation_ledger_writer)

    g.add_edge(START, "search_query_builder")
    g.add_edge("search_query_builder", "fanout_retriever")
    g.add_edge("fanout_retriever", "organism_profile_builder")
    g.add_edge("organism_profile_builder", "filter_with_reasoning")
    g.add_edge("filter_with_reasoning", "citation_ledger_writer")
    g.add_edge("citation_ledger_writer", END)
    return g.compile()
