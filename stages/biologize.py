"""Stage 2 - Biologize: Challenge Brief v1 -> Challenge Brief v2.

Maps each defined "How might we...?" (HMW) question onto the Biomimicry Taxonomy and
turns the selected functions into "How does nature...?" (HDN) questions for AskNature.
Fully automated, LLM-driven, no human gate. Each defined question is the `challenge`
fed to every prompt. Sub-components (one node each):
  function_mapper      -> per HMW: map onto 3-5 taxonomy functions (group/sub_group/
                          function paths) via three lenses (direct / analogous / inverted)
  hdn_framer           -> per mapped function: frame one (occasionally two) HDN question(s)
                          at the right altitude and solution-neutral
  biologize_evaluator  -> per challenge batch: grade neutrality / altitude / fidelity for all
                          HDNs of a defined question in one call; on failure rework per-item
                          with the regeneration prompt, capped at EVALUATOR_MAX_RETRIES,
                          then best-effort

The Biologize metrics are derived on demand from the final state in demo.py
(see metrics.biologize_metrics).
"""
from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from .. import config, taxonomy
from ..llm import LLM
from ..parallel import bounded_map
from ..schemas import (BiologizeEvalResponse, FrameHDNResponse, MapFunctionsResponse,
                       RegenHDNResponse)
from ..state import HDNQuestion, MappedFunction, SpiralState, log_entry

STAGE = "biologize"
_llm = LLM()  # injectable for tests: `biologize._llm = FakeLLM()`


# --- prompts (verbatim from the practitioner spec) ---------------------------
# Output-shape note: LLM.complete restricts output to a JSON *object* schema, so the two
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
    "- Return 3-5 functions, with at least one from each lens.\n"
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
    "You are a calibrated rubric grader for the Biologize step of the Biomimicry Design "
    "Spiral. You evaluate a BATCH of \"How does nature...?\" (HDN) questions, all framed "
    "for the SAME design challenge. You are not a generator. If any item carries author "
    "rationale, ignore it and judge the question text itself.\n\n"

    "INDEPENDENCE - READ FIRST (this is a batch; grade as if each were alone):\n"
    "- Judge every item ONLY against the challenge and the rubric below - NEVER against "
    "the other items in the batch. This is absolute grading, not ranking.\n"
    "- Do not let the verdicts you gave earlier items influence later ones. If six items "
    "were 'clean', that has ZERO bearing on the seventh - reset each time.\n"
    "- Do not normalize, spread, or balance verdicts across the batch. Identical "
    "questions must get identical verdicts regardless of position. If every item "
    "deserves the same verdict, give it; if each differs, so be it.\n"
    "- Item order carries no meaning. Grade the last item with the same care as the first.\n\n"

    "Within each item you score THREE INDEPENDENT dimensions. They are orthogonal: an "
    "item can pass one and fail another. Reason about each on its own axis, in order, and "
    "do not let one verdict pull another (the dissociating examples prove they come "
    "apart).\n\n"

    "INPUTS\n"
    "- challenge: the design challenge from the Define step (applies to ALL items).\n"
    "- items: an array; each element is { id, hdn_question, approach, group, sub_group, "
    "function }, where approach is one of: direct, analogous, inverted. Echo each item's "
    "id back verbatim in your output.\n\n"

    "DIMENSION 1 - SOLUTION-NEUTRALITY  (axis: WHAT vs HOW)\n"
    "Neutral = names a function or outcome, never a means. Fails if it contains OR "
    "implies a specific mechanism, material, device, or implementation - human or "
    "biological alike (\"evaporation\", \"capillary action\", \"foam\", \"valve\").\n"
    "The test: does it presuppose HOW, or only state WHAT must be achieved?\n"
    "Verdicts: clean | borderline | leaks. If not clean, name the offending term.\n\n"

    "DIMENSION 2 - ALTITUDE  (axis: SCOPE / granularity - independent of Dimension 1)\n"
    "- too_high : restates a whole domain or survival problem; maps to hundreds of "
    "unrelated strategies.\n"
    "- just_right: a single biological function/outcome, context-free, answerable by a "
    "tractable, related set of organisms.\n"
    "- too_low  : collapses to one mechanism, OR names the challenge's literal "
    "artifact/context; presupposes the answer.\n"
    "Verdicts: too_high | just_right | too_low.\n\n"

    "DIMENSION 3 - FIDELITY / LENS APPROPRIATENESS  (axis: relationship to challenge, "
    "conditioned on the declared lens - divergence is CORRECT for two of three)\n"
    "- direct   : must stay tethered to the challenge's literal purpose/context.\n"
    "- analogous: must shift to an ADJACENT purpose, a synonym, or another context where "
    "the same function appears - clearly related, but neither a literal restatement nor "
    "unrelated drift.\n"
    "- inverted : must be a genuine OPPOSITE framing of the challenge's purpose, not a "
    "reworded direct question.\n"
    "In all cases the question must still be RELEVANT to the challenge's underlying need. "
    "Do NOT penalize analogous/inverted questions for failing to match the literal "
    "challenge - that is intended. Also flag if the lens label looks wrong (e.g. labeled "
    "inverted but reads direct).\n"
    "Verdicts: on_lens_and_relevant | weak | off_lens_or_irrelevant.\n\n"

    "DISSOCIATING EXAMPLES - AXIS INDEPENDENCE (these come apart within one item):\n"
    "- \"How does nature survive a wildfire?\"\n"
    "    neutrality: clean (no mechanism); altitude: too_high -> clean but unusable\n"
    "- \"How does nature spray to put out flames?\"\n"
    "    neutrality: leaks (\"spray\"); altitude: too_low -> both fail, separately\n"
    "- \"How does nature build a fire exit?\"\n"
    "    neutrality: clean (names no mechanism); altitude: too_low (names the literal "
    "artifact \"exit\") -> proves altitude != neutrality\n"
    "- \"How does nature interrupt combustion?\"\n"
    "    neutrality: clean; altitude: just_right\n\n"

    "DISSOCIATING EXAMPLE - ITEM INDEPENDENCE (verdicts must not drift across a batch):\n"
    "- If a batch contains five strong 'just_right / clean / on_lens' items followed by "
    "\"How does nature survive a wildfire?\", the sixth is STILL altitude: too_high. The "
    "five good items before it change nothing. Grade it exactly as if it arrived alone.\n\n"

    "OUTPUT\n"
    "A single JSON object and nothing else - no markdown fences, no commentary, no "
    "trailing text. It has exactly one field:\n"
    "  verdicts : an array with ONE object per input item, in the SAME order as the "
    "input, and with the SAME COUNT as the input. Do not merge, split, skip, or add "
    "items. Each object has exactly these fields, in this order:\n"
    "    id                         echo the input item's id verbatim\n"
    "    neutrality_reasoning       a sentence or two\n"
    "    neutrality                 clean | borderline | leaks\n"
    "    neutrality_offending_term  the term, or null\n"
    "    altitude_reasoning         a sentence or two\n"
    "    altitude                   too_high | just_right | too_low\n"
    "    fidelity_reasoning         a sentence or two\n"
    "    fidelity                   on_lens_and_relevant | weak | off_lens_or_irrelevant\n"
    "    lens_label_looks_correct   true | false\n\n"

    "Before returning, confirm len(verdicts) == number of input items and every input id "
    "appears exactly once."
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
    raw = _llm.complete(
        task="biologize_map",
        system=_MAP_FUNCTIONS_SYSTEM,
        user=f'TAXONOMY:\n"""\n{taxonomy_text}\n"""\n\nCHALLENGE:\n"""\n{challenge}\n"""',
        schema=MapFunctionsResponse,
        temperature=config.GEN_TEMPERATURE,
        ctx={"challenge": challenge},
    )
    return [f.model_dump() for f in raw.functions]


def _frame_hdns(challenge: str, functions: list[dict]) -> list[dict]:
    """Frame HDN questions from the selected taxonomy functions for one challenge."""
    raw = _llm.complete(
        task="biologize_frame",
        system=_FRAME_HDN_SYSTEM,
        user=f'FUNCTIONS:\n"""\n{json.dumps(functions, ensure_ascii=False)}\n"""\n\n'
             f'CHALLENGE:\n"""\n{challenge}\n"""',
        schema=FrameHDNResponse,
        temperature=config.GEN_TEMPERATURE,
        ctx={"challenge": challenge, "n_functions": len(functions)},
    )
    return [q.model_dump() for q in raw.questions]


def _eval_hdns(challenge: str, hdns: list[dict]) -> dict[int, dict]:
    """Grade a batch of HDNs (same challenge) on the 3 axes; return {id: verdict}.

    One LLM call for the whole batch (vs one per HDN before). All HDNs in a batch share
    the same Define-stage challenge — the grader prompt requires it — so callers batch
    per defined question."""
    items = [{"id": h["id"], **_item(h)} for h in hdns]
    raw = _llm.complete(
        task="biologize_eval",
        system=_EVAL_SYSTEM,
        user=f'challenge:\n"""\n{challenge}\n"""\n\n'
             f'items:\n"""\n{json.dumps(items, ensure_ascii=False)}\n"""',
        schema=BiologizeEvalResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"n_items": len(items)},
    )
    # .model_dump() each verdict so the dict-based verdict-rule helpers stay unchanged.
    return {v.id: v.model_dump() for v in raw.verdicts}


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
    return _llm.complete(
        task="biologize_regen",
        system=_REGEN_SYSTEM,
        user=f'INPUTS:\n"""\n{json.dumps(inputs, ensure_ascii=False)}\n"""',
        schema=RegenHDNResponse,
        temperature=config.CRITIC_TEMPERATURE,
        ctx={"approach": h["approach"]},
    ).model_dump()


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
    """Per defined question: map onto 3-5 taxonomy functions; drop any invalid path."""
    taxonomy_text = taxonomy.render_for_prompt()
    dqs = [dq for dq in state["defined_questions"] if dq.get("text", "")]
    # Parallelize only the LLM call across defined questions; validate/append below (main
    # thread), zipping each DQ's raw functions back to it so ordering stays deterministic.
    per_dq = bounded_map(lambda dq: _map_functions(taxonomy_text, dq["text"]), dqs)
    mapped: list[dict] = []
    dropped = 0
    for dq, items in zip(dqs, per_dq):
        for item in items:
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

    # Dedup by taxonomy triple: several defined questions frequently map the same
    # (group, sub_group, function). Collapse them into one row so the function is
    # framed/retrieved/graded once. First-seen wins the primary id/approach (order is
    # deterministic: DQ order, then item order); every contributing question and lens
    # is retained in the *_ids / approaches lists for traceability.
    pre_dedup = len(mapped)
    deduped: dict[tuple, dict] = {}
    for mf in mapped:
        key = (mf["group"], mf["sub_group"], mf["function"])
        cur = deduped.get(key)
        if cur is None:
            mf["define_question_ids"] = [mf["define_question_id"]]
            mf["approaches"] = [mf["approach"]] if mf["approach"] else []
            deduped[key] = mf
        else:
            cur["define_question_ids"] = sorted(
                set(cur["define_question_ids"]) | {mf["define_question_id"]})
            if mf["approach"] and mf["approach"] not in cur["approaches"]:
                cur["approaches"].append(mf["approach"])
    mapped = list(deduped.values())
    merged = pre_dedup - len(mapped)

    detail = f"{len(mapped)} functions" + (
        f" ({dropped} invalid dropped)" if dropped else "") + (
        f" ({merged} duplicates merged)" if merged else "")
    return {"mapped_functions": mapped,
            "spiral_log": [log_entry(STAGE, "functions_mapped", detail)]}


def hdn_framer(state: SpiralState) -> dict:
    """Per mapped function: frame one (occasionally two) HDN question(s)."""
    funcs_by_dq: dict[int, list[dict]] = {}
    for mf in state.get("mapped_functions", []):
        funcs_by_dq.setdefault(mf["define_question_id"], []).append(mf)

    dqs = [dq for dq in state["defined_questions"] if funcs_by_dq.get(dq["id"])]

    def _frame_one(dq: dict) -> list[dict]:
        funcs = funcs_by_dq[dq["id"]]
        func_inputs = [{"approach": f["approach"], "group": f["group"],
                        "sub_group": f["sub_group"], "function": f["function"]} for f in funcs]
        return _frame_hdns(dq.get("text", ""), func_inputs)

    # Each unique function is framed once (funcs_by_dq keys on the primary id); look up
    # its merged define_question_ids / approaches by triple to carry onto the HDN.
    mf_by_triple = {(mf["group"], mf["sub_group"], mf["function"]): mf
                    for mf in state.get("mapped_functions", [])}

    # Parallelize only the framing LLM call; assign ids single-threaded in the merge below.
    per_dq = bounded_map(_frame_one, dqs)
    hdns: list[dict] = []
    next_id = 0
    for dq, items in zip(dqs, per_dq):
        for item in items:
            text = (item.get("hdn_question") or "").strip()
            if not text:
                continue
            approach = (item.get("approach") or "").strip().lower()
            group = (item.get("group") or "").strip()
            sub_group = (item.get("sub_group") or "").strip()
            function = (item.get("function") or "").strip()
            mf = mf_by_triple.get((group, sub_group, function))
            hdns.append(HDNQuestion(
                id=next_id, define_question_id=dq["id"],
                define_question_ids=(mf or {}).get("define_question_ids", [dq["id"]]),
                approach=approach,
                approaches=(mf or {}).get("approaches", [approach] if approach else []),
                group=group, sub_group=sub_group, function=function,
                reasoning=(item.get("reasoning") or "").strip(),
                text=text,
            ).model_dump())
            next_id += 1
    return {"hdn_questions": hdns,
            "spiral_log": [log_entry(STAGE, "hdn_framed", f"{len(hdns)} questions")]}


def biologize_evaluator(state: SpiralState) -> dict:
    """Per challenge batch: grade all HDNs of a defined question in one call; rework
    failures per-item (capped), then carry best-effort.

    Batching by challenge cuts the best-case call count from one-per-HDN to one-per-
    defined-question; regeneration stays per-item and each retry round re-grades only
    the still-failing HDNs in a single batched call."""
    challenge_by_dq = {q["id"]: q.get("text", "") for q in state["defined_questions"]}
    hdns = [dict(h) for h in state["hdn_questions"]]

    # Group by defined question — one shared challenge per batch (grader requirement).
    groups: dict[int, list[dict]] = {}
    for h in hdns:
        groups.setdefault(h["define_question_id"], []).append(h)

    # One defined question per group, and each HDN belongs to exactly one group, so a worker
    # mutating its group's `h` dicts (during regen) touches objects disjoint from every other
    # worker — safe to run groups in parallel. Each returns its own verdicts/attempts, merged
    # single-threaded below. Regen stays per-item within a group.
    def _eval_group(item: tuple[int, list[dict]]) -> tuple[dict, dict]:
        dq_id, group = item
        challenge = challenge_by_dq.get(dq_id, "")
        local_verdicts: dict[int, dict] = {}
        local_attempts: dict[int, int] = {h["id"]: 0 for h in group}
        histories: dict[int, list[str]] = {h["id"]: [] for h in group}

        local_verdicts.update(_eval_hdns(challenge, group))
        pending = [h for h in group if not _passed(local_verdicts.get(h["id"], {}))]
        rnd = 0
        while pending and rnd < config.EVALUATOR_MAX_RETRIES:
            rnd += 1
            regenerated = []
            for h in pending:
                regen = _regen_hdn(challenge, h,
                                   _failures_from_verdict(local_verdicts.get(h["id"], {})),
                                   histories[h["id"]])
                histories[h["id"]].append(h["text"])
                local_attempts[h["id"]] += 1
                new_text = (regen.get("hdn_question") or "").strip()
                if not new_text:                       # no revision offered -> stop retrying it
                    continue
                h["text"] = new_text                   # taxonomy path/approach stay fixed
                h["reasoning"] = (regen.get("reasoning") or "").strip() or h["reasoning"]
                regenerated.append(h)
            if not regenerated:
                break
            local_verdicts.update(_eval_hdns(challenge, regenerated))
            pending = [h for h in regenerated if not _passed(local_verdicts.get(h["id"], {}))]
        return local_verdicts, local_attempts

    verdicts: dict[int, dict] = {}
    attempts: dict[int, int] = {}
    for local_verdicts, local_attempts in bounded_map(_eval_group, list(groups.items())):
        verdicts.update(local_verdicts)
        attempts.update(local_attempts)

    for h in hdns:
        verdict = verdicts.get(h["id"], {})
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
        h["eval_attempts"] = attempts.get(h["id"], 0)

    clean = sum(1 for h in hdns if h["eval_status"] == "accepted")
    return {"hdn_questions": hdns,
            "spiral_log": [log_entry(STAGE, "biologize_evaluated", f"{clean} clean")]}


# --- subgraph ----------------------------------------------------------------
def build_biologize_subgraph():
    g = StateGraph(SpiralState)
    g.add_node("function_mapper", function_mapper)
    g.add_node("hdn_framer", hdn_framer)
    g.add_node("biologize_evaluator", biologize_evaluator)

    g.add_edge(START, "function_mapper")
    g.add_edge("function_mapper", "hdn_framer")
    g.add_edge("hdn_framer", "biologize_evaluator")
    g.add_edge("biologize_evaluator", END)
    return g.compile()
