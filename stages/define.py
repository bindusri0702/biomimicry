"""Stage 1 - Define: user challenge -> Challenge Brief v1.

Fully automated, LLM-driven, no human gate. Two nodes do the work:
  define               -> ONE LLM call (the DEFINE prompt) returning parsed context,
                          system_context, one "How might we...?" defined question per
                          function, and assumptions
  goldilocks_evaluator -> judge each defined question on two independent axes (breadth +
                          solution-neutrality); revise in place via the prompt's
                          suggested_revision, capped at EVALUATOR_MAX_RETRIES, then best-effort

Exported as a compiled subgraph; the orchestrator wires it into the spiral. The Define
metrics are derived on demand from the final state in demo.py (see metrics.define_metrics).
"""
from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from .. import config
from ..llm import LLM
from ..schemas import DefineResponse, GoldilocksResponse, GoldilocksVerdict
from ..state import ContextProfile, DefinedQuestion, SpiralState, log_entry

STAGE = "define"
_llm = LLM()  # module-level so tests can inject a fake: `define._llm = FakeLLM()`

# --- prompts (verbatim from the practitioner spec) ---------------------------
_DEFINE_SYSTEM = (
    "You are a biomimicry practitioner trained in the Biomimicry Institute's Design Spiral. "
    "You are guiding the user through the DEFINE step. The aim is not to decide what to build, "
    "but to articulate what the design must DO, for WHOM, and in WHAT CONTEXT — in "
    "solution-neutral, functional terms.\n\n"
    "1. Frame the challenge - Explain the impact needed. Name the function to achieve, never a "
    "mechanism or product.\n"
    "2. Consider context - Describe some of the contextual factors (stakeholders, location "
    "conditions, resource availability, etc.)\n"
    f"3. Design question - Using the information above, frame {config.DEFINE_HMW} \"How might we…?\" "
    "questions, based on the functions stated in the user challenge.\n\n"
    "Note: If the input lacks the information for a field, do not invent it. Use an empty list/null "
    "and add an item to \"assumptions\" instead.\n\n"
    "Output: respond with ONLY a valid JSON object — no markdown fences, no commentary — matching "
    "exactly this shape:\n"
    "{\n"
    '  "stakeholders": [str],\n'
    '  "operating_environment": str,\n'
    '  "hard_constraints": [str],\n'
    '  "system_context": {\n'
    '    "interactions": [str],\n'
    '    "boundaries": [str],\n'
    '    "adjacent_systems": [str],\n'
    '    "leverage_points": [str]\n'
    "  },\n"
    '  "defined_questions": [str],\n'
    '  "assumptions": [str]\n'
    "}"
)

_GOLDILOCKS_SYSTEM = (
    "You are a Goldilocks critic for the Biomimicry Design Spiral. You judge whether a DEFINE-step "
    "\"How might we...?\" question is pitched at the right ALTITUDE to be biologized — abstracted to "
    "a function, but specific enough to discipline the search.\n\n"
    "You are given the challenge context (stakeholders, environment, constraints) and a list of "
    "QUESTIONS.\n\n"
    "Judge each question on TWO INDEPENDENT axes:\n\n"
    "1. breadth_label — altitude:\n"
    "   - too_broad: so generic it would return nearly any biological strategy; the functional "
    "stressors are missing (e.g. \"How does nature manage gases?\").\n"
    "   - just_right: names the core function and its essential stressors, nothing more.\n"
    "   - too_narrow: over-constrained — piles on incidental constraints that exclude valid "
    "strategies without adding function.\n\n"
    "2. solution_neutral — does the question presuppose a mechanism, material, or design? This is "
    "SEPARATE from breadth. true = purely functional; false = a solution is baked in (e.g. "
    "\"...using a membrane filter\").\n\n"
    "For each question, reason first, then label. Output ONLY a valid JSON object — no markdown "
    "fences, no commentary — matching exactly this shape:\n"
    "{\n"
    '  "verdicts": [\n'
    '    {\n'
    '      "index": int,\n'
    '      "question": str,\n'
    '      "reasoning": str,\n'
    '      "breadth_label": "too_broad" | "just_right" | "too_narrow",\n'
    '      "solution_neutral": true | false,\n'
    '      "suggested_revision": str | null\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    "Calibration examples:\n"
    "- \"How does nature manage gases?\" -> too_broad, solution_neutral true (no stressors).\n"
    "- \"How might we keep occupied air survivable for the minutes people need?\" -> just_right, "
    "solution_neutral true.\n"
    "- \"How might we use a CO2-scrubbing membrane to clean trapped air?\" -> too_narrow, "
    "solution_neutral FALSE (membrane is a presupposed mechanism, doesn't leave room for innovation.)."
)


# --- nodes -------------------------------------------------------------------
def define(state: SpiralState) -> dict:
    """One LLM call: parse context + system_context + one defined question per function."""
    idea = state["raw_idea"]
    raw = _llm.complete(
        task="define",
        system=_DEFINE_SYSTEM,
        user=f'USER CHALLENGE:\n"""\n{idea}\n"""',
        schema=DefineResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"idea": idea},
    )
    context = ContextProfile(
        stakeholders=raw.stakeholders,
        operating_environment=raw.operating_environment,
        hard_constraints=raw.hard_constraints,
    ).model_dump()
    system_context = raw.system_context.model_dump()
    dqs = [DefinedQuestion(id=i, text=q.strip()).model_dump()
           for i, q in enumerate(raw.defined_questions) if q and q.strip()]
    return {
        "context": context,
        "system_context": system_context,
        "assumptions": raw.assumptions,
        "defined_questions": dqs,
        "spiral_log": [log_entry(STAGE, "defined", f"{len(dqs)} questions")],
    }


def goldilocks_evaluator(state: SpiralState) -> dict:
    """Judge each defined question (breadth + solution-neutral); revise in place (capped)."""
    context = state.get("context", {})
    dqs = [dict(q) for q in state["defined_questions"]]
    pending = list(dqs)
    rnd = 0
    while pending and rnd <= config.EVALUATOR_MAX_RETRIES:
        verdict = _eval_goldilocks(context, [q["text"] for q in pending])
        by_index = {v.index: v for v in verdict.verdicts}
        rnd += 1
        still = []
        for idx, q in enumerate(pending):
            v = by_index.get(idx) or GoldilocksVerdict()
            q["breadth_label"] = v.breadth_label
            q["solution_neutral"] = v.solution_neutral
            q["reasoning"] = v.reasoning
            q["suggested_revision"] = v.suggested_revision
            q["eval_attempts"] += 1
            passed = q["breadth_label"] == "just_right" and bool(q["solution_neutral"])
            revision = (v.suggested_revision or "").strip()
            if passed:
                q["eval_status"] = "accepted"
            elif rnd <= config.EVALUATOR_MAX_RETRIES and revision:
                q["text"] = revision            # apply revision and retry
                still.append(q)
            else:
                if revision:
                    q["text"] = revision        # apply the final suggested revision
                q["eval_status"] = "best_effort"
        pending = still
    for q in pending:                            # safety: no revision offered at the cap
        q["eval_status"] = q.get("eval_status") or "best_effort"
    return {"defined_questions": dqs,
            "spiral_log": [log_entry(STAGE, "goldilocks_evaluated",
                                     f"{sum(1 for q in dqs if q['eval_status']=='accepted')} just-right")]}


# --- LLM helper --------------------------------------------------------------
def _eval_goldilocks(context: dict, questions: list[str]) -> GoldilocksResponse:
    numbered = "\n".join(f"{i}: {q}" for i, q in enumerate(questions))
    return _llm.complete(
        task="goldilocks",
        system=_GOLDILOCKS_SYSTEM,
        user=f'CHALLENGE CONTEXT:\n"""\n{json.dumps(context, ensure_ascii=False)}\n"""\n\n'
             f'QUESTIONS:\n"""\n{numbered}\n"""',
        schema=GoldilocksResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"context": context, "questions": questions},
    )


# --- subgraph ----------------------------------------------------------------
def build_define_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("define", define)
    g.add_node("goldilocks_evaluator", goldilocks_evaluator)

    g.add_edge(START, "define")
    g.add_edge("define", "goldilocks_evaluator")
    g.add_edge("goldilocks_evaluator", END)
    return g.compile()
