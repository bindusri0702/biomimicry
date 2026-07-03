"""Stage 4 - Abstract: Challenge Brief v3 -> bio-inspired design strategies (v4).

Translates ONE biological strategy at a time into a discipline-neutral design
strategy: same function and mechanism, all biological terms removed, strictly true
to the source biology. Fully automated, LLM-driven, no human gate. Nodes:
  abstract_strategy  -> run the Abstract prompt once per kept organism: summarize the
                        mechanism, translate biological terms (paired), and write the
                        neutral function+mechanism design strategy (a launching pad,
                        not a solution); self-flag `abstainable` when the source is thin
  fidelity_evaluator -> true to biology AND does not conclude a design; regenerate
                        with feedback when rejected (capped), then best-effort

The Abstract metrics are derived on demand from the final state in demo.py
(see metrics.abstract_metrics).
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .. import config
from ..biology_denylist import check_biology_residue
from ..llm import LLM
from ..schemas import AbstractEvalResponse, AbstractResponse
from ..state import BiologicalAbstraction, SpiralState, log_entry

STAGE = "abstract"
_llm = LLM()   # injectable for tests


# --- prompts -----------------------------------------------------------------
_ABSTRACT_SYSTEM = (
    "You are a biomimicry practitioner working the Abstract step of the Biomimicry Design "
    "Spiral. You translate ONE biological strategy into a bio-inspired DESIGN strategy: a "
    "statement of the same function and mechanism, with all biological terms removed, staying "
    "strictly true to the source biology.\n\n"
    "A design strategy describes HOW the biological strategy works in discipline-neutral "
    "language so a non-biologist can build on it. It is NOT a description of a solution or "
    "product — it is a launching pad for brainstorming. You do not design anything.\n\n"
    "INPUTS\n"
    "- challenge: the design challenge from the Define step (gives the context the strategy will "
    "be used in).\n"
    "- function: the function / HDN question this strategy was retrieved to answer.\n"
    "- biological_strategy: { organism, content } — verbatim source text from Discover. This is "
    "your ONLY source of truth about the biology.\n\n"
    "PROCESS\n"
    "1. SUMMARIZE: in plain language, state how the strategy works to meet the function — the "
    "essential features/mechanism, nothing decorative.\n"
    "2. TRANSLATE TERMS: list each biological term in that mechanism and give a discipline-neutral "
    "synonym (e.g. fur -> fibers, skin -> membrane, bone -> rigid strut). Translate the term, "
    "never delete the mechanism it carries.\n"
    "3. WRITE THE DESIGN STRATEGY: restate the strategy in neutral terms. Name the FUNCTION and "
    "the CONTEXT it serves; describe the MECHANISM in neutral terms. Think like an engineer "
    "describing a mechanical system or process — not an organism.\n\n"
    "HARD RULES\n"
    "- NO biological terms in design_strategy: no species, body parts, taxa, or biology-specific "
    "vocabulary. If a term names a mechanism, replace it with a functional equivalent; do not just "
    "drop it.\n"
    "- STAY TRUE TO THE SOURCE: every mechanism or effect in design_strategy must be supported by "
    "biological_strategy.content. Do NOT add mechanisms, numbers, causes, or generalizations the "
    "source doesn't state. If the source is too thin to support a faithful mechanism, say so in "
    "reasoning and set abstainable = true.\n"
    "- KEEP THE MECHANISM: do not flatten to a bare function (\"a system that manages heat\"). The "
    "how must survive the translation, or the abstraction is useless to the next step.\n"
    "- NOT A SOLUTION: do not propose a design, product, material choice, or application. No \"the "
    "building could...\", no \"use a coating that...\". Describe the strategy only.\n"
    "- The cited organism/source identifier MUST be carried through verbatim for traceability.\n\n"
    "OUTPUT\n"
    "A single JSON object, nothing else — no markdown fences, no commentary. Exactly these "
    "fields, in this order:\n"
    "  reasoning           str  — how you preserved mechanism + function while removing biology, "
    "and a one-line check that nothing unsupported was added.\n"
    "  summary             str  — plain-language mechanism (step 1; may keep biology terms).\n"
    "  term_translations   array of { biological_term: str, neutral_term: str }\n"
    "  design_strategy     str  — the neutral function+mechanism statement (step 3).\n"
    "  function            str  — verbatim from input.\n"
    "  organism            str  — verbatim from biological_strategy.\n"
    "  source_id           str  — verbatim cited identifier/URL from biological_strategy.\n"
    "  abstainable         bool — true if the source can't support a faithful abstraction."
)


def _challenge_text(state: SpiralState) -> str:
    """The Define-step context the strategy will be used in: the user challenge plus any
    solution-neutral defined questions framed from it."""
    parts = [(state.get("raw_idea") or "").strip()]
    for q in state.get("defined_questions", []) or []:
        text = (q.get("text") or "").strip()
        if text:
            parts.append(f"- {text}")
    return "\n".join(p for p in parts if p)


def _abstract_user(challenge: str, function: str, organism: str, content: str,
                   source_id: str) -> str:
    return (
        f'challenge:\n"""\n{challenge}\n"""\n\n'
        f'function:\n"""\n{function}\n"""\n\n'
        f'biological_strategy:\n"""\n'
        f"organism: {organism}\n"
        f"source_id: {source_id}\n"
        f"content: {content}\n"
        f'"""'
    )


# --- nodes -------------------------------------------------------------------
def abstract_strategy(state: SpiralState) -> dict:
    """Run the Abstract prompt once per kept organism: summarize, translate terms, and write the
    discipline-neutral design strategy (function + mechanism), or flag it abstainable."""
    models = [m for m in state["biological_models"] if m.get("keep")]
    challenge = _challenge_text(state)
    abstractions = []
    for i, m in enumerate(models):
        function = "; ".join(m.get("function_addressed", []) or [])
        organism = m.get("organism_common", "")
        if m.get("organism_scientific"):
            organism = f'{organism} ({m["organism_scientific"]})'
        content = m.get("mechanism") or m.get("strategy_summary", "")
        source_id = m.get("doc_id") or m.get("source_url", "")
        raw = _llm.complete(
            task="abstract",
            system=_ABSTRACT_SYSTEM,
            user=_abstract_user(challenge, function, organism, content, source_id),
            schema=AbstractResponse,
            temperature=config.CRITIC_TEMPERATURE,
            ctx={"model_id": m["id"]},
        )
        translations = [t.model_dump() for t in raw.term_translations]
        summary = raw.summary.strip()
        design = raw.design_strategy.strip()
        abstractions.append(
            BiologicalAbstraction(
                id=i, model_id=m["id"], organism_common=m["organism_common"],
                source_doc_id=(raw.source_id or source_id or "").strip(),
                source_taxon=m.get("taxon", ""), source_scale=m.get("scale"),
                function=(raw.function or function or "").strip(),
                mechanism_summary=summary, neutral_summary=design, summary=summary,
                design_strategy=design, statement=design,
                term_translations=translations,
                jargon_terms=[t.get("biological_term", "") for t in translations
                              if t.get("biological_term")],
                functions_addressed=m.get("function_addressed", []),
                abstract_reasoning=raw.reasoning.strip(),
                abstainable=bool(raw.abstainable),
                source_content=content,    # verbatim source the grader judges against (mechanism)
                source_organism_terms=[t for t in (m.get("organism_common", ""),
                                                   m.get("organism_scientific", "")) if t],
            ).model_dump()
        )
    return {"abstractions": abstractions,
            "spiral_log": [log_entry(STAGE, "strategies_abstracted", f"{len(abstractions)} models")]}


def fidelity_evaluator(state: SpiralState) -> dict:
    """Per abstraction: grade completeness + faithfulness vs the verbatim source; regen (capped),
    then best-effort. Strict pass = completeness 'complete' AND faithfulness 'faithful'.

    An abstraction the prompt itself flagged `abstainable` is carried through as best-effort
    without regeneration — the source is too thin to support a faithful mechanism."""
    out = [dict(s) for s in state["abstractions"]]
    for s in out:
        # Layer 0 (deterministic, notify-only): flag biological residue in the design strategy.
        # Records hits for visibility; does NOT regenerate or change the accept decision.
        residue = check_biology_residue(
            s.get("design_strategy") or s.get("statement") or "",
            source_organism_terms=s.get("source_organism_terms") or [],
        )
        s["biology_residue_flag"] = residue.flag
        s["biology_hard_hits"] = residue.hard_hits
        s["biology_ambiguous_hits"] = residue.ambiguous_hits
        s["biology_escalate"] = residue.escalate

        if s.get("abstainable"):
            s["evaluator_feedback"] = "source too thin for a faithful abstraction (abstained)"
            s["accepted"] = True
            s["eval_status"] = "best_effort"
            s["eval_attempts"] = 0
            continue
        attempts, verdict = 0, {}
        while True:
            verdict = _eval_abstraction(s)
            attempts += 1
            passed = (verdict.get("completeness") == "complete"
                      and verdict.get("faithfulness") == "faithful")
            if passed or attempts > config.EVALUATOR_MAX_RETRIES:
                break
            regen = _regen_statement(s, _eval_feedback(verdict))
            if regen:
                s["design_strategy"] = regen
                s["statement"] = regen
        s["source_steps"] = verdict.get("source_steps", []) or []
        s["completeness_reasoning"] = verdict.get("completeness_reasoning", "") or ""
        s["step_coverage"] = verdict.get("step_coverage", []) or []
        s["completeness"] = verdict.get("completeness")
        s["faithfulness_reasoning"] = verdict.get("faithfulness_reasoning", "") or ""
        s["added_claims"] = verdict.get("added_claims", []) or []
        s["faithfulness"] = verdict.get("faithfulness")
        s["evaluator_feedback"] = _eval_feedback(verdict)
        passed = (verdict.get("completeness") == "complete"
                  and verdict.get("faithfulness") == "faithful")
        s["accepted"] = True                       # all carry forward (best-effort if needed)
        s["eval_status"] = "accepted" if passed else "best_effort"
        s["eval_attempts"] = attempts
    residue_n = sum(1 for s in out if s.get("biology_residue_flag"))
    return {"abstractions": out,
            "spiral_log": [
                log_entry(STAGE, "biology_residue_checked", f"{residue_n} with residual biology"),
                log_entry(STAGE, "fidelity_evaluated",
                          f"{sum(1 for s in out if s['eval_status']=='accepted')} faithful")]}


# --- LLM helpers -------------------------------------------------------------
_EVAL_SYSTEM = (
    "You are a calibrated grader for the Abstract step of the Biomimicry Design Spiral. You check a "
    "bio-inspired DESIGN STRATEGY against the BIOLOGICAL STRATEGY it was abstracted from, on two "
    "axes:\n"
    "  COMPLETENESS  — every causal step in the source survived the abstraction.\n"
    "  FAITHFULNESS  — nothing was introduced that the source doesn't support.\n"
    "You are not a generator and you do not rewrite. The BIOLOGICAL STRATEGY is the ONLY source of "
    "truth — judge against it, never against your own knowledge of biology. If a claim's correctness "
    "depends on what you happen to know rather than on what the source says, that is OUT OF SCOPE "
    "here.\n\n"
    "INPUTS\n"
    "- biological_strategy: verbatim source text (the ground truth).\n"
    "- design_strategy:     the abstracted statement under evaluation.\n\n"
    "METHOD (follow in order)\n"
    "Step A — Decompose the SOURCE. List the distinct CAUSAL STEPS in biological_strategy: each "
    "feature/structure and the effect it produces that the mechanism depends on. A causal step = "
    '"X does/enables Y." Number them S1, S2, … Decompose only what the source states; do not add '
    "steps from your own knowledge.\n\n"
    "Step B — COMPLETENESS. For each source step Sn, is its mechanistic content present in "
    "design_strategy (in neutral wording)? present | missing | weakened (weakened = the step is "
    "gestured at but its causal content is lost, e.g. flattened to a bare function). Missing or "
    "weakened steps are completeness failures.\n\n"
    "Step C — FAITHFULNESS. Examine each mechanistic claim in design_strategy. Does it trace to a "
    "source step? supported | unsupported | contradicted. Flag any mechanism, cause, quantity, or "
    "generalization NOT present in the source — including plausible-sounding additions (those are "
    "the dangerous ones). Re-wording and neutral synonyms are fine; new mechanistic content is "
    "not.\n\n"
    "Do not conflate the axes: a strategy can include every source step (complete) yet also add an "
    "invented one (unfaithful); or add nothing (faithful) yet drop a step (incomplete). Score them "
    "separately.\n\n"
    "OUTPUT — single JSON object, nothing else, no markdown fences. Exactly these fields, in order:\n"
    "  source_steps          array of { id: str, step: str }      // Step A\n"
    "  completeness_reasoning str\n"
    '  step_coverage         array of { id: str, status: "present"|"missing"|"weakened" }\n'
    "  completeness          complete | partial | incomplete\n"
    "  faithfulness_reasoning str\n"
    '  added_claims          array of { claim: str, status: "unsupported"|"contradicted" }  // [] if none\n'
    "  faithfulness          faithful | minor_additions | unfaithful"
)


def _eval_user(biological_strategy: str, design_strategy: str) -> str:
    return (
        f'biological_strategy:\n"""\n{biological_strategy}\n"""\n\n'
        f'design_strategy:\n"""\n{design_strategy}\n"""'
    )


def _eval_abstraction(s: dict) -> dict:
    """Grade a design strategy for completeness + faithfulness against its verbatim source."""
    # .model_dump() so the dict-based verdict readers (_eval_feedback, fidelity_evaluator)
    # stay unchanged.
    return _llm.complete(
        task="abstract_eval",
        system=_EVAL_SYSTEM,
        user=_eval_user(
            biological_strategy=(s.get("source_content") or "").strip(),
            design_strategy=(s.get("design_strategy") or s.get("statement") or "").strip(),
        ),
        schema=AbstractEvalResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"id": s.get("id")},
    ).model_dump()


def _eval_feedback(verdict: dict) -> str:
    """Synthesize regeneration feedback from the grader verdict (no single feedback field)."""
    bits = []
    if verdict.get("completeness_reasoning"):
        bits.append(f"Completeness: {verdict['completeness_reasoning']}")
    lost = [c.get("id", "") for c in (verdict.get("step_coverage") or [])
            if c.get("status") in ("missing", "weakened")]
    if lost:
        bits.append(f"Restore the missing/weakened source steps: {', '.join(s for s in lost if s)}.")
    if verdict.get("faithfulness_reasoning"):
        bits.append(f"Faithfulness: {verdict['faithfulness_reasoning']}")
    added = [c.get("claim", "") for c in (verdict.get("added_claims") or []) if c.get("claim")]
    if added:
        bits.append(f"Remove unsupported claims: {'; '.join(added)}.")
    return " ".join(bits)


def _regen_statement(s: dict, feedback: str) -> str:
    """Rewrite the design strategy via the Abstract prompt to fix an evaluator issue."""
    function = s.get("function") or "; ".join(s.get("functions_addressed", []) or [])
    raw = _llm.complete(
        task="abstract",
        system=(
            _ABSTRACT_SYSTEM
            + f"\n\nThe previous design_strategy was rejected. Fix this issue and try again: "
              f"{feedback}"
        ),
        user=_abstract_user(
            challenge=function, function=function,
            organism=s.get("organism_common", ""),
            content=s.get("summary") or s.get("mechanism_summary", ""),
            source_id=s.get("source_doc_id", ""),
        ),
        schema=AbstractResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"id": s.get("id"), "feedback": feedback},
    )
    return raw.design_strategy.strip()


# --- subgraph ----------------------------------------------------------------
def build_abstract_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("abstract_strategy", abstract_strategy)
    g.add_node("fidelity_evaluator", fidelity_evaluator)

    g.add_edge(START, "abstract_strategy")
    g.add_edge("abstract_strategy", "fidelity_evaluator")
    g.add_edge("fidelity_evaluator", END)
    return g.compile()
