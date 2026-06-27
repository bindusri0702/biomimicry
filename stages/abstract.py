"""Stage 4 - Abstract: Challenge Brief v3 -> biological abstractions (v4).

Produces a plain-English account of each kept organism's features/mechanism —
scientific/biological terms removed but the science kept — WITHOUT concluding a
design. Fully automated, LLM-driven, no human gate. Sub-components (one node each):
  mechanism_summarizer -> mechanism-first paragraph per kept organism
  term_neutralizer     -> LLM plain-English rewrite (remove sci terms, keep science)
  mechanism_abstraction-> the canonical faithful features/mechanism statement
  fidelity_evaluator   -> true to biology AND does not conclude a design; regenerate
                          with feedback when rejected (capped), then best-effort
  compute_metrics      -> the Abstract evaluation metrics
  finalize             -> emits the biological abstractions (v4)
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import config
from ..llm import LLM
from ..metrics import abstract_metrics
from ..state import BiologicalAbstraction, SpiralState, log_entry

STAGE = "abstract"
_llm = LLM()   # injectable for tests


# --- nodes -------------------------------------------------------------------
def mechanism_summarizer(state: SpiralState) -> dict:
    """Mechanism-first paragraph per kept organism (HOW it works, not a species story)."""
    models = [m for m in state["biological_models"] if m.get("keep")]
    raw = _llm.complete_json(
        task="mechanism_summary",
        system=(
            "Distill each organism profile into a MECHANISM-FIRST paragraph: a functional account "
            "of HOW the mechanism works, not a species description and not a design. Return strict "
            'JSON: {"summaries":[{"model_id":int,"summary":str}]}.'
        ),
        user="\n\n".join(f'model {m["id"]} ({m["organism_common"]}): {m["mechanism"]}'
                         for m in models),
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"models": models},
    )
    by_model = {x["model_id"]: x["summary"] for x in raw.get("summaries", [])}
    abstractions = [
        BiologicalAbstraction(
            id=i, model_id=m["id"], organism_common=m["organism_common"],
            source_doc_id=m.get("doc_id", ""), source_taxon=m.get("taxon", ""),
            source_scale=m.get("scale"),
            mechanism_summary=(by_model.get(m["id"], "") or "").strip(),
            functions_addressed=m.get("function_addressed", []),
        ).model_dump()
        for i, m in enumerate(models)
    ]
    return {"abstractions": abstractions,
            "spiral_log": [log_entry(STAGE, "mechanisms_summarized", f"{len(abstractions)} models")]}


def term_neutralizer(state: SpiralState) -> dict:
    """LLM rewrite: remove scientific/biological terminology, keep the underlying science."""
    items = state["abstractions"]
    raw = _llm.complete_json(
        task="neutralize",
        system=(
            "Rewrite each mechanism summary in plain, discipline-neutral English: remove "
            "scientific/biological terminology and species names, but KEEP the underlying science "
            "(the causal mechanism). Do not introduce any design, material, or technology. List the "
            "scientific/biological terms you removed. Return strict JSON: "
            '{"items":[{"id":int,"neutral_summary":str,"jargon_terms":[str]}]}.'
        ),
        user="\n\n".join(f'{s["id"]}: {s["mechanism_summary"]}' for s in items),
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"abstractions": items},
    )
    by_id = {x["id"]: x for x in raw.get("items", [])}
    out = [dict(s) for s in items]
    for s in out:
        r = by_id.get(s["id"], {})
        s["neutral_summary"] = (r.get("neutral_summary", "") or "").strip()
        s["jargon_terms"] = r.get("jargon_terms", []) or []
    return {"abstractions": out,
            "spiral_log": [log_entry(STAGE, "terms_neutralized")]}


def mechanism_abstraction(state: SpiralState) -> dict:
    """The canonical faithful features/mechanism account — NOT a design conclusion."""
    items = state["abstractions"]
    raw = _llm.complete_json(
        task="mechanism_abstraction",
        system=(
            "Write the canonical ABSTRACTION: a faithful plain-English account of the FEATURES and "
            "MECHANISM the organism uses to achieve its function. Capture the strategy alone — do "
            "NOT prescribe or conclude any design, product, material, or technology. Return strict "
            'JSON: {"accounts":[{"id":int,"statement":str}]}.'
        ),
        user="\n".join(f'{s["id"]}: functions={s["functions_addressed"]} '
                       f'mechanism={s["neutral_summary"]!r}' for s in items),
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"abstractions": items},
    )
    by_id = {x["id"]: x["statement"] for x in raw.get("accounts", [])}
    out = [dict(s) for s in items]
    for s in out:
        s["statement"] = (by_id.get(s["id"], "") or "").strip()
    return {"abstractions": out,
            "spiral_log": [log_entry(STAGE, "abstractions_written")]}


def fidelity_evaluator(state: SpiralState) -> dict:
    """Per abstraction: true to biology AND no design conclusion; regen (capped), then best-effort."""
    out = [dict(s) for s in state["abstractions"]]
    for s in out:
        attempts, verdict = 0, None
        while True:
            verdict = _eval_fidelity(s)
            attempts += 1
            ok = bool(verdict.get("true_to_biology")) and not bool(verdict.get("concludes_design"))
            if ok or attempts > config.EVALUATOR_MAX_RETRIES:
                break
            regen = _regen_statement(s, verdict.get("feedback", ""))
            if regen:
                s["statement"] = regen
        s["true_to_biology"] = verdict.get("true_to_biology")
        s["concludes_design"] = verdict.get("concludes_design")
        s["jargon_terms"] = verdict.get("jargon_terms", s.get("jargon_terms", [])) or []
        s["evaluator_feedback"] = verdict.get("feedback", "")
        passed = bool(verdict.get("true_to_biology")) and not bool(verdict.get("concludes_design"))
        s["accepted"] = True                       # all carry forward (best-effort if needed)
        s["eval_status"] = "accepted" if passed else "best_effort"
        s["eval_attempts"] = attempts
    return {"abstractions": out,
            "spiral_log": [log_entry(STAGE, "fidelity_evaluated",
                                     f"{sum(1 for s in out if s['eval_status']=='accepted')} faithful")]}


def compute_metrics(state: SpiralState) -> dict:
    return {"abstract_metrics": abstract_metrics(state["abstractions"]),
            "spiral_log": [log_entry(STAGE, "metrics_computed")]}


def finalize(state: SpiralState) -> dict:
    return {"version": "v4", "current_stage": STAGE,
            "spiral_log": [log_entry(STAGE, "abstractions_finalized",
                                     "Biological abstractions ready (Challenge Brief v4)")]}


# --- LLM helpers -------------------------------------------------------------
def _eval_fidelity(s: dict) -> dict:
    return _llm.complete_json(
        task="fidelity_eval",
        system=(
            "Evaluate a biological abstraction against its source mechanism. Judge:\n"
            " - true_to_biology: does it faithfully capture the actual strategy/mechanism, with no "
            "distortion or invented detail?\n"
            " - concludes_design: does it prescribe or imply a specific design, product, material, or "
            "technology? (it must NOT)\n"
            "Also list any residual scientific/biological jargon. Return strict JSON: "
            '{"true_to_biology":bool,"concludes_design":bool,"jargon_terms":[str],"feedback":str}.'
        ),
        user=f"Source mechanism: {s.get('mechanism_summary')}\n\nAbstraction: {s.get('statement')}",
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"id": s.get("id")},
    )


def _regen_statement(s: dict, feedback: str) -> str:
    raw = _llm.complete_json(
        task="mechanism_abstraction",
        system=(
            "Rewrite the abstraction to fix the issue: it must faithfully capture the biological "
            f"mechanism and NOT conclude any design. Issue: {feedback}. Return strict JSON: "
            '{"accounts":[{"id":int,"statement":str}]}.'
        ),
        user=f'{s["id"]}: functions={s["functions_addressed"]} mechanism={s.get("neutral_summary")!r}',
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"id": s.get("id"), "feedback": feedback},
    )
    accounts = raw.get("accounts", [])
    return (accounts[0].get("statement", "").strip() if accounts else "")


# --- subgraph ----------------------------------------------------------------
def build_abstract_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("mechanism_summarizer", mechanism_summarizer)
    g.add_node("term_neutralizer", term_neutralizer)
    g.add_node("mechanism_abstraction", mechanism_abstraction)
    g.add_node("fidelity_evaluator", fidelity_evaluator)
    g.add_node("compute_metrics", compute_metrics)
    g.add_node("finalize", finalize)

    g.add_edge(START, "mechanism_summarizer")
    g.add_edge("mechanism_summarizer", "term_neutralizer")
    g.add_edge("term_neutralizer", "mechanism_abstraction")
    g.add_edge("mechanism_abstraction", "fidelity_evaluator")
    g.add_edge("fidelity_evaluator", "compute_metrics")
    g.add_edge("compute_metrics", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
