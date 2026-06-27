"""Stage 2 - Biologize: Challenge Brief v1 -> Challenge Brief v2.

Maps each defined "How might we...?" (HMW) question onto the Biomimicry Taxonomy and
turns the selected functions into "How does nature...?" (HDN) questions for AskNature.
Fully automated, LLM-driven, no human gate. Each defined question is the `challenge`
fed to every prompt. Sub-components (one node each):
  function_mapper      -> per HMW: map onto 4-8 taxonomy functions (group/sub_group/
                          function paths) via three lenses (direct / analogous / inverted)
  hdn_framer           -> per mapped function: frame one (occasionally two) HDN question(s)
                          at the right altitude and solution-neutral
  biologize_evaluator  -> per HDN: grade neutrality / altitude / fidelity; on failure rework
                          with the regeneration prompt, capped at EVALUATOR_MAX_RETRIES,
                          then best-effort
  compute_metrics      -> the Biologize evaluation metrics
  finalize             -> stamps Challenge Brief v2
"""
from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from .. import config, taxonomy
from ..llm import LLM
from ..metrics import biologize_metrics
from ..state import HDNQuestion, MappedFunction, SpiralState, log_entry

STAGE = "biologize"
_llm = LLM()  # injectable for tests: `biologize._llm = FakeLLM()`


# --- prompts (verbatim from the practitioner spec) ---------------------------
# Output-shape note: LLM.complete_json forces a JSON *object* response, so the two
# array-producing prompts wrap their array in a single key ("functions"/"questions"),
# exactly as define._GOLDILOCKS_SYSTEM wraps "verdicts". All other content is verbatim.
_MAP_FUNCTIONS_SYSTEM = (
    "You are a biomimicry practitioner trained in the Biomimicry Institute's Design Spiral, "
    "working the Biologize step. Given a design Challenge and the full Biomimicry Taxonomy, map "
    "the challenge to the biological functions whose scope best matches its underlying purpose.\n\n"
    "Two objectives, both required: precisely match the challenge's purpose AND deliberately "
    "broaden beyond it. Produce candidates through three lenses:\n"
    "- direct: verbs naming the challenge's purpose or required outcomes.\n"
    "- analogous: adjacent purposes, related nouns/synonyms, or organisms facing a similar problem "
    "in another context. Go beyond the literal challenge.\n"
    "- inverted: the opposite framing (e.g. instead of \"store water\" -> \"keep water out\"). "
    "Opposites often surface new strategies.\n\n"
    "Rules:\n"
    "- Return 4-8 functions total, with at least one from each lens.\n"
    "- Span multiple groups when the challenge is multi-faceted; do not anchor on one.\n"
    "- group, sub_group, and function MUST be copied verbatim from the provided taxonomy and form "
    "a valid path (function nested under that sub_group under that group). Never invent, rename, "
    "or paraphrase a label.\n"
    "- If a facet has no fitting function, omit it rather than force a weak match.\n"
    "- Keep reasoning at the level of purpose/function - no mechanisms or solutions.\n"
    "- Order items most- to least-relevant.\n\n"
    "Output a single JSON object and nothing else - no markdown fences, no commentary - with one "
    "key \"functions\" whose value is an array. Each array element has exactly these keys, in this "
    "order:\n"
    "  reasoning   (string, written before you commit to the path)\n"
    "  approach    (one of: \"direct\", \"analogous\", \"inverted\")\n"
    "  group       (string, verbatim from taxonomy)\n"
    "  sub_group   (string, verbatim from taxonomy)\n"
    "  function    (string, verbatim from taxonomy)"
)

_NEUTRALITY_BLOCK = (
    "SOLUTION-NEUTRALITY - strictly enforced:\n"
    "- Describe the function or outcome, not the means. Strip any human technology, material, "
    "mechanism, or implementation (\"sprinkler\", \"foam\", \"sensor\", \"pump\", \"alarm\") and "
    "any verb that implies one (\"spray\", \"vent via ducts\").\n"
    "- The challenge's domain nouns may stay only if they name the *outcome* (heat, smoke, crowd, "
    "evacuation), not a chosen solution."
)

_FRAME_HDN_SYSTEM = (
    "You are a biomimicry practitioner trained in the Biomimicry Institute's Design Spiral, "
    "working the Biologize step. The functions that fit this challenge have already been selected. "
    "Your job is to translate each into a \"How does nature...?\" (HDN) question that a designer "
    "can take to AskNature.\n\n"
    "INPUTS\n"
    "- challenge: the design challenge from the Define step.\n"
    "- functions: a JSON array of selected taxonomy paths, each with an approach lens:\n"
    "  [{ \"approach\": \"direct\"|\"analogous\"|\"inverted\", \"group\": str, \"sub_group\": str, "
    "\"function\": str }]\n\n"
    "TASK\n"
    "For each selected function, write one HDN question (occasionally two if the function genuinely "
    "carries two distinct purposes). Phrase it as the biological purpose the challenge needs solved "
    "- never as the challenge's own wording, and never as a solution.\n\n"
    "ALTITUDE - the single most important constraint (aim for the middle):\n"
    "- TOO HIGH (reject): restates a whole domain; maps to hundreds of strategies.\n"
    "    e.g. \"How does nature survive?\"  \"How does nature deal with fire?\"\n"
    "- TOO LOW (reject): names a specific mechanism, material, device, or the challenge's literal "
    "context; presupposes the answer.\n"
    "    e.g. \"How does nature make a foam retardant?\"  \"How does nature build an exit?\"\n"
    "- JUST RIGHT (target): a single biological function/outcome, context-free, answerable by a "
    "tractable set of organisms.\n"
    "    e.g. \"How does nature protect tissue from extreme heat?\"\n"
    "         \"How does nature move many individuals through a narrow opening?\"\n\n"
    + _NEUTRALITY_BLOCK + "\n\n"
    "APPROACH LENS - preserve and express:\n"
    "- direct:    the question states the challenge's literal purpose.\n"
    "- analogous: shift to an adjacent purpose, synonym, or another context where the same function "
    "appears (e.g. heat in a desert organism, not a building).\n"
    "- inverted:  ask the opposite (instead of \"remove heat\" -> \"retain heat\"; instead of "
    "\"let crowds out\" -> \"control how a group concentrates\").\n\n"
    "RULES\n"
    "- Every question begins with \"How does nature\" and ends with \"?\".\n"
    "- group/sub_group/function in each output item MUST be copied verbatim from the corresponding "
    "input item - this is the traceability link; do not alter them.\n"
    "- Preserve the input's approach label on each question.\n"
    "- Keep all three lenses represented in the output if they were in the input.\n"
    "- If a selected function cannot be framed at the right altitude AND neutrally, omit it rather "
    "than emit a weak question (do not output a placeholder).\n"
    "- Order items: direct first, then analogous, then inverted.\n\n"
    "OUTPUT\n"
    "A single JSON object and nothing else - no markdown fences, no commentary - with one key "
    "\"questions\" whose value is an array. Each array element has exactly these keys, in this "
    "order:\n"
    "  reasoning    string - written BEFORE the question; must state why the chosen altitude is "
    "neither too broad nor too narrow, and confirm no mechanism/solution leaked in.\n"
    "  hdn_question    string - the \"How does nature...?\" question.\n"
    "  approach    string - one of: direct, analogous, inverted.\n"
    "  group    string - verbatim from the matching input item.\n"
    "  sub_group    string - verbatim from the matching input item.\n"
    "  function    string - verbatim from the matching input item."
)

_EVAL_SYSTEM = (
    "You are a calibrated rubric grader for the Biologize step of the Biomimicry Design Spiral. "
    "You evaluate ONE \"How does nature...?\" (HDN) question as written. You are not a generator. "
    "If the author attached any rationale, ignore it and judge the question text itself.\n\n"
    "You score THREE independent dimensions. They are orthogonal: a question can pass one and fail "
    "another. Reason about each on its own axis, in the order given, and do not let one verdict "
    "pull another (the dissociating examples below prove they come apart).\n\n"
    "INPUTS\n"
    "- challenge: the design challenge from the Define step.\n"
    "- item: { hdn_question, approach, group, sub_group, function }, approach is one of: direct, "
    "analogous, inverted.\n\n"
    "DIMENSION 1 - SOLUTION-NEUTRALITY  (axis: WHAT vs HOW)\n"
    "Neutral = names a function or outcome, never a means. Fails if it contains OR implies a "
    "specific mechanism, material, device, or implementation - human or biological alike "
    "(\"evaporation\", \"capillary action\", \"foam\", \"valve\").\n"
    "The test: does it presuppose HOW, or only state WHAT must be achieved?\n"
    "Verdicts: clean | borderline | leaks. If not clean, name the offending term.\n\n"
    "DIMENSION 2 - ALTITUDE  (axis: SCOPE / granularity - independent of Dimension 1)\n"
    "- too_high : restates a whole domain or survival problem; maps to hundreds of unrelated "
    "strategies.\n"
    "- just_right: a single biological function/outcome, context-free, answerable by a tractable, "
    "related set of organisms.\n"
    "- too_low  : collapses to one mechanism, OR names the challenge's literal artifact/context; "
    "presupposes the answer.\n"
    "Verdicts: too_high | just_right | too_low.\n\n"
    "DIMENSION 3 - FIDELITY / LENS APPROPRIATENESS  (axis: relationship to challenge, conditioned "
    "on the declared lens - divergence is CORRECT for two of three)\n"
    "- direct   : must stay tethered to the challenge's literal purpose/context.\n"
    "- analogous: must shift to an ADJACENT purpose, a synonym, or another context where the same "
    "function appears - clearly related, but neither a literal restatement nor unrelated drift.\n"
    "- inverted : must be a genuine OPPOSITE framing of the challenge's purpose, not a reworded "
    "direct question.\n"
    "In all cases the question must still be RELEVANT to the challenge's underlying need. Do NOT "
    "penalize analogous/inverted questions for failing to match the literal challenge - that is "
    "intended. Also flag if the lens label looks wrong (e.g. labeled inverted but reads direct).\n"
    "Verdicts: on_lens_and_relevant | weak | off_lens_or_irrelevant.\n\n"
    "DISSOCIATING EXAMPLES (these show the three axes are independent - internalize)\n"
    "- \"How does nature survive a wildfire?\"\n"
    "    neutrality: clean (no mechanism); altitude: too_high -> clean but unusable\n"
    "- \"How does nature spray to put out flames?\"\n"
    "    neutrality: leaks (\"spray\"); altitude: too_low -> both fail, separately\n"
    "- \"How does nature build a fire exit?\"\n"
    "    neutrality: clean (names no mechanism); altitude: too_low (names the literal artifact "
    "\"exit\") -> proves altitude != neutrality\n"
    "- \"How does nature interrupt combustion?\"\n"
    "    neutrality: clean; altitude: just_right\n\n"
    "OUTPUT\n"
    "A single JSON object and nothing else - no markdown fences, no commentary, no trailing text. "
    "Write the reasoning field BEFORE its verdict for each dimension. The object has exactly these "
    "fields, in this order:\n"
    "  neutrality_reasoning       a sentence or two\n"
    "  neutrality                 clean | borderline | leaks\n"
    "  neutrality_offending_term  the term, or null\n"
    "  altitude_reasoning         a sentence or two\n"
    "  altitude                   too_high | just_right | too_low\n"
    "  fidelity_reasoning         a sentence or two\n"
    "  fidelity                   on_lens_and_relevant | weak | off_lens_or_irrelevant\n"
    "  lens_label_looks_correct   true | false"
)

_REGEN_SYSTEM = (
    "You are a biomimicry practitioner reworking ONE rejected \"How does nature...?\" question for "
    "the Biologize step. A separate evaluator rejected it; your job is to fix the specific "
    "failure(s) and return a corrected question. You do NOT score it.\n\n"
    "INPUTS\n"
    "- challenge: the Define-stage challenge.\n"
    "- approach:  direct | analogous | inverted.\n"
    "- rejected:  the prior question text.\n"
    "- failures:  array of { dimension, verdict, reason } from the evaluator (dimension in "
    "neutrality | altitude | fidelity).\n"
    "- history:   prior rejected attempts for this function, if any (avoid repeating).\n"
    "- group / sub_group / function: the taxonomy path; echo them back verbatim.\n\n"
    "FIX BY DIMENSION\n"
    "- neutrality leak -> remove the named mechanism/material/device and restate as the pure "
    "outcome.\n"
    + _NEUTRALITY_BLOCK + "\n"
    "- altitude too_high -> narrow to one biological function. too_low -> lift off the specific "
    "mechanism/literal artifact to the function it serves.\n"
    "- fidelity off-lens -> re-anchor to the challenge's purpose AS REQUIRED BY 'approach' "
    "(direct = literal; analogous = adjacent/another context; inverted = true opposite).\n"
    "Fix only what failed; do not regress a dimension that passed.\n\n"
    "OUTPUT: a single JSON object, no fences:\n"
    "  { \"reasoning\": str, \"hdn_question\": str, \"approach\": str,\n"
    "    \"group\": str, \"sub_group\": str, \"function\": str }"
)


# --- LLM helpers -------------------------------------------------------------
def _map_functions(taxonomy_text: str, challenge: str) -> list[dict]:
    """Map one challenge (HMW question) onto taxonomy functions through three lenses."""
    raw = _llm.complete_json(
        task="biologize_map",
        system=_MAP_FUNCTIONS_SYSTEM,
        user=f'TAXONOMY:\n"""\n{taxonomy_text}\n"""\n\nCHALLENGE:\n"""\n{challenge}\n"""',
        temperature=config.GEN_TEMPERATURE,
        ctx={"challenge": challenge},
    )
    return raw.get("functions", []) or []


def _frame_hdns(challenge: str, functions: list[dict]) -> list[dict]:
    """Frame HDN questions from the selected taxonomy functions for one challenge."""
    raw = _llm.complete_json(
        task="biologize_frame",
        system=_FRAME_HDN_SYSTEM,
        user=f'FUNCTIONS:\n"""\n{json.dumps(functions, ensure_ascii=False)}\n"""\n\n'
             f'CHALLENGE:\n"""\n{challenge}\n"""',
        temperature=config.GEN_TEMPERATURE,
        ctx={"challenge": challenge, "n_functions": len(functions)},
    )
    return raw.get("questions", []) or []


def _eval_hdn(item: dict, challenge: str) -> dict:
    """Grade one HDN on the three orthogonal axes (neutrality / altitude / fidelity)."""
    return _llm.complete_json(
        task="biologize_eval",
        system=_EVAL_SYSTEM,
        user=f'ITEM:\n"""\n{json.dumps(item, ensure_ascii=False)}\n"""\n\n'
             f'CHALLENGE:\n"""\n{challenge}\n"""',
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"hdn": item.get("hdn_question")},
    )


def _regen_hdn(challenge: str, h: dict, failures: list[dict], history: list[str]) -> dict:
    """Rework one rejected HDN, fixing only the failed dimensions."""
    inputs = {
        "challenge": challenge,
        "approach": h["approach"],
        "rejected": h["text"],
        "failures": failures,
        "history": history,
        "group": h["group"],
        "sub_group": h["sub_group"],
        "function": h["function"],
    }
    return _llm.complete_json(
        task="biologize_regen",
        system=_REGEN_SYSTEM,
        user=f'INPUTS:\n"""\n{json.dumps(inputs, ensure_ascii=False)}\n"""',
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"approach": h["approach"]},
    )


# --- evaluator verdict helpers -----------------------------------------------
def _passed(v: dict) -> bool:
    """An HDN is clean only when every axis is at its best verdict."""
    return (v.get("neutrality") == "clean"
            and v.get("altitude") == "just_right"
            and v.get("fidelity") == "on_lens_and_relevant"
            and bool(v.get("lens_label_looks_correct")))


def _failures_from_verdict(v: dict) -> list[dict]:
    """One {dimension, verdict, reason} per axis not at its best verdict (feeds regen)."""
    fails: list[dict] = []
    if v.get("neutrality") != "clean":
        fails.append({"dimension": "neutrality", "verdict": v.get("neutrality"),
                      "reason": v.get("neutrality_reasoning", "")})
    if v.get("altitude") != "just_right":
        fails.append({"dimension": "altitude", "verdict": v.get("altitude"),
                      "reason": v.get("altitude_reasoning", "")})
    if v.get("fidelity") != "on_lens_and_relevant" or not v.get("lens_label_looks_correct"):
        reason = v.get("fidelity_reasoning", "")
        if not v.get("lens_label_looks_correct"):
            reason = (reason + " (declared lens label appears incorrect.)").strip()
        fails.append({"dimension": "fidelity", "verdict": v.get("fidelity"), "reason": reason})
    return fails


def _item(h: dict) -> dict:
    """The grader's ITEM view of an HDN record."""
    return {"hdn_question": h["text"], "approach": h["approach"], "group": h["group"],
            "sub_group": h["sub_group"], "function": h["function"]}


# --- nodes -------------------------------------------------------------------
def function_mapper(state: SpiralState) -> dict:
    """Per defined question: map onto 4-8 taxonomy functions; drop any invalid path."""
    taxonomy_text = taxonomy.render_for_prompt()
    mapped: list[dict] = []
    dropped = 0
    for dq in state["defined_questions"]:
        challenge = dq.get("text", "")
        if not challenge:
            continue
        for item in _map_functions(taxonomy_text, challenge):
            group = (item.get("group") or "").strip()
            sub_group = (item.get("sub_group") or "").strip()
            function = (item.get("function") or "").strip()
            if not taxonomy.is_valid_path(group, sub_group, function):
                dropped += 1
                continue
            mapped.append(MappedFunction(
                define_question_id=dq["id"],
                approach=(item.get("approach") or "").strip().lower(),
                group=group, sub_group=sub_group, function=function,
                reasoning=(item.get("reasoning") or "").strip(),
            ).model_dump())
    detail = f"{len(mapped)} functions" + (f" ({dropped} invalid dropped)" if dropped else "")
    return {"mapped_functions": mapped,
            "spiral_log": [log_entry(STAGE, "functions_mapped", detail)]}


def hdn_framer(state: SpiralState) -> dict:
    """Per mapped function: frame one (occasionally two) HDN question(s)."""
    funcs_by_dq: dict[int, list[dict]] = {}
    for mf in state.get("mapped_functions", []):
        funcs_by_dq.setdefault(mf["define_question_id"], []).append(mf)

    hdns: list[dict] = []
    next_id = 0
    for dq in state["defined_questions"]:
        funcs = funcs_by_dq.get(dq["id"], [])
        if not funcs:
            continue
        challenge = dq.get("text", "")
        func_inputs = [{"approach": f["approach"], "group": f["group"],
                        "sub_group": f["sub_group"], "function": f["function"]} for f in funcs]
        for item in _frame_hdns(challenge, func_inputs):
            text = (item.get("hdn_question") or "").strip()
            if not text:
                continue
            hdns.append(HDNQuestion(
                id=next_id, define_question_id=dq["id"],
                approach=(item.get("approach") or "").strip().lower(),
                group=(item.get("group") or "").strip(),
                sub_group=(item.get("sub_group") or "").strip(),
                function=(item.get("function") or "").strip(),
                reasoning=(item.get("reasoning") or "").strip(),
                text=text,
            ).model_dump())
            next_id += 1
    return {"hdn_questions": hdns,
            "spiral_log": [log_entry(STAGE, "hdn_framed", f"{len(hdns)} questions")]}


def biologize_evaluator(state: SpiralState) -> dict:
    """Per HDN: grade the 3 axes; rework failures (capped), then carry best-effort."""
    challenge_by_dq = {q["id"]: q.get("text", "") for q in state["defined_questions"]}
    hdns = [dict(h) for h in state["hdn_questions"]]

    for h in hdns:
        challenge = challenge_by_dq.get(h["define_question_id"], "")
        history: list[str] = []
        attempts = 0
        verdict = _eval_hdn(_item(h), challenge)
        while not _passed(verdict) and attempts < config.EVALUATOR_MAX_RETRIES:
            regen = _regen_hdn(challenge, h, _failures_from_verdict(verdict), history)
            history.append(h["text"])
            attempts += 1
            new_text = (regen.get("hdn_question") or "").strip()
            if not new_text:
                break
            h["text"] = new_text                      # taxonomy path/approach stay fixed
            h["reasoning"] = (regen.get("reasoning") or "").strip() or h["reasoning"]
            verdict = _eval_hdn(_item(h), challenge)

        h["neutrality"] = verdict.get("neutrality")
        h["neutrality_offending_term"] = verdict.get("neutrality_offending_term")
        h["altitude"] = verdict.get("altitude")
        h["fidelity"] = verdict.get("fidelity")
        h["lens_label_looks_correct"] = verdict.get("lens_label_looks_correct")
        h["evaluator_reasoning"] = {
            "neutrality": verdict.get("neutrality_reasoning", ""),
            "altitude": verdict.get("altitude_reasoning", ""),
            "fidelity": verdict.get("fidelity_reasoning", ""),
        }
        h["accepted"] = True                           # all carry forward (best-effort if needed)
        h["eval_status"] = "accepted" if _passed(verdict) else "best_effort"
        h["eval_attempts"] = attempts

    clean = sum(1 for h in hdns if h["eval_status"] == "accepted")
    return {"hdn_questions": hdns,
            "spiral_log": [log_entry(STAGE, "biologize_evaluated", f"{clean} clean")]}


def compute_metrics(state: SpiralState) -> dict:
    return {"biologize_metrics": biologize_metrics(state["hdn_questions"],
                                                   state.get("mapped_functions", [])),
            "spiral_log": [log_entry(STAGE, "metrics_computed")]}


def finalize(state: SpiralState) -> dict:
    return {"version": "v2", "current_stage": STAGE,
            "spiral_log": [log_entry(STAGE, "challenge_brief_finalized", "Challenge Brief v2 ready")]}


# --- subgraph ----------------------------------------------------------------
def build_biologize_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("function_mapper", function_mapper)
    g.add_node("hdn_framer", hdn_framer)
    g.add_node("biologize_evaluator", biologize_evaluator)
    g.add_node("compute_metrics", compute_metrics)
    g.add_node("finalize", finalize)

    g.add_edge(START, "function_mapper")
    g.add_edge("function_mapper", "hdn_framer")
    g.add_edge("hdn_framer", "biologize_evaluator")
    g.add_edge("biologize_evaluator", "compute_metrics")
    g.add_edge("compute_metrics", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
