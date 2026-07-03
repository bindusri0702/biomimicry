"""Per-stage evaluation metrics.

All baked-in heuristics (solution-term lists, breadth/neutrality scoring formulas,
mechanism-token completeness, tier weights) were removed — judgement now lives in
the LLM evaluators. What remains here is pure-math, knowledge-free aggregation over
the evaluator verdicts already attached to each payload: ratios, counts, and a
token-overlap diversity measure.
"""
from __future__ import annotations

import itertools
import re

_TOKEN = re.compile(r"[a-z][a-z\-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


# --- generic helpers ----------------------------------------------------------
def _avg(xs: list) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def _proportion(items: list, pred) -> float:
    return round(sum(1 for x in items if pred(x)) / len(items), 3) if items else 0.0


def _jaccard_dissimilarity(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa or not sb:
        return 1.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def candidate_uniqueness(texts: list[str]) -> float:
    """Mean pairwise token dissimilarity across texts (1 = all distinct)."""
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return 1.0
    return round(sum(_jaccard_dissimilarity(a, b) for a, b in pairs) / len(pairs), 3)


def _simpson(values: list) -> float:
    """Simpson diversity 1 - Σ p² over category labels; 0 (uniform) .. ~1 (diverse)."""
    items = [v for v in values if v]
    if not items:
        return 0.0
    counts: dict = {}
    for v in items:
        counts[v] = counts.get(v, 0) + 1
    n = len(items)
    return round(1 - sum((c / n) ** 2 for c in counts.values()), 3)


def _coarse_taxon(taxon: str) -> str:
    """Top two lineage ranks, e.g. 'Animalia/Arthropoda/Insecta' -> 'Animalia/Arthropoda'."""
    parts = [p for p in (taxon or "").split("/") if p]
    return "/".join(parts[:2]) if parts else ""


# --- Define -------------------------------------------------------------------
def define_metrics(defined_questions: list[dict], context: dict) -> dict:
    """Define-stage metrics over the defined questions (Goldilocks two-axis verdicts)."""
    qs = defined_questions
    labels = [q.get("breadth_label") for q in qs]
    label_dist = {lab: labels.count(lab) for lab in set(filter(None, labels))}

    context = context or {}
    completeness = round(
        (bool(context.get("stakeholders")) + bool(context.get("operating_environment"))
         + bool(context.get("hard_constraints"))) / 3, 3)

    return {
        "defined_question_count": len(qs),
        "just_right_ratio": _proportion(qs, lambda q: q.get("breadth_label") == "just_right"),
        "solution_neutral_ratio": _proportion(qs, lambda q: bool(q.get("solution_neutral"))),
        "clean_count": sum(1 for q in qs if q.get("eval_status") == "accepted"),
        "breadth_label_distribution": label_dist,
        "question_uniqueness": candidate_uniqueness([q["text"] for q in qs]),
        "context_completeness": completeness,
    }


# --- Biologize ----------------------------------------------------------------
def biologize_metrics(hdn_questions: list[dict], mapped_functions: list[dict] | None = None) -> dict:
    """Biologize-stage metrics over the taxonomy-anchored HDN fan-out (3-axis verdicts)."""
    accepted = [h for h in hdn_questions if h.get("accepted")]

    # HDNs are deduped by taxonomy triple, so one HDN can serve several defined
    # questions — flatten over define_question_ids (fall back to the singular id) so
    # coverage counts every question a kept HDN serves, not just the primary.
    def _dq_ids(h: dict) -> list:
        return h.get("define_question_ids") or [h.get("define_question_id")]

    define_qids = {qid for h in hdn_questions for qid in _dq_ids(h)}
    covered = {qid for h in accepted for qid in _dq_ids(h)}
    approaches = [h.get("approach") for h in hdn_questions]

    return {
        "define_question_coverage": round(len(covered) / max(len(define_qids), 1), 3),
        "hdn_per_define_question": round(
            len(hdn_questions) / max(len(define_qids), 1), 3),
        "framing_diversity_index": candidate_uniqueness([h["text"] for h in accepted]),
        "neutrality_clean_ratio": _proportion(
            hdn_questions, lambda h: h.get("neutrality") == "clean"),
        "altitude_just_right_ratio": _proportion(
            hdn_questions, lambda h: h.get("altitude") == "just_right"),
        "fidelity_on_lens_ratio": _proportion(
            hdn_questions, lambda h: h.get("fidelity") == "on_lens_and_relevant"),
        "lens_label_correct_ratio": _proportion(
            hdn_questions, lambda h: bool(h.get("lens_label_looks_correct"))),
        "approach_distribution": {a: approaches.count(a) for a in set(filter(None, approaches))},
        "clean_count": sum(1 for h in hdn_questions if h.get("eval_status") == "accepted"),
        "accepted_count": len(accepted),
        "mapped_function_count": len(mapped_functions or []),
    }


# --- Discover -----------------------------------------------------------------
def discover_metrics(models: list[dict], hdn_questions: list[dict]) -> dict:
    """Discover-stage metrics over the reasoning-filter verdicts."""
    kept = [m for m in models if m.get("keep")]

    accepted_hdn_ids = [h["id"] for h in hdn_questions if h.get("accepted")]
    per_hdn = {}
    for hid in accepted_hdn_ids:
        rels = [m.get("relevance_score") or 0 for m in kept if hid in (m.get("hdn_ids") or [])]
        per_hdn[hid] = round(max(rels), 3) if rels else 0.0

    rels = [m.get("functional_relevance") for m in models]
    adeqs = [m.get("mechanistic_adequacy") for m in models]
    return {
        "kept_count": len(kept),
        "retrieved_count": len(models),
        "keep_rate": _proportion(models, lambda m: m.get("keep")),
        "functional_relevance_distribution": {
            v: rels.count(v) for v in ("yes", "partial", "no")},
        "mechanistic_adequacy_distribution": {
            v: adeqs.count(v) for v in ("sufficient", "thin", "unusable")},
        "hdn_relevance": {
            "overall": _avg([m.get("relevance_score") for m in kept]),
            "per_hdn": per_hdn,
        },
        "taxonomic_diversity_index": _simpson([_coarse_taxon(m.get("taxon", "")) for m in kept]),
    }


# --- Abstract -----------------------------------------------------------------
def abstract_metrics(abstractions: list[dict]) -> dict:
    """Abstract-stage metrics over the two-axis grader verdicts (completeness + faithfulness)."""
    sel = [a for a in abstractions if a.get("accepted")] or abstractions
    comp = [a.get("completeness") for a in sel]
    faith = [a.get("faithfulness") for a in sel]
    return {
        "complete_ratio": _proportion(sel, lambda a: a.get("completeness") == "complete"),
        "faithful_ratio": _proportion(sel, lambda a: a.get("faithfulness") == "faithful"),
        "completeness_distribution": {
            v: comp.count(v) for v in ("complete", "partial", "incomplete")},
        "faithfulness_distribution": {
            v: faith.count(v) for v in ("faithful", "minor_additions", "unfaithful")},
        "biology_free_ratio": _proportion(sel, lambda a: not a.get("biology_residue_flag")),
        "residue_escalation_count": sum(1 for a in sel if a.get("biology_escalate")),
        "abstain_rate": _proportion(abstractions, lambda a: a.get("abstainable")),
        "accepted_count": len([a for a in abstractions if a.get("accepted")]),
        "total_count": len(abstractions),
    }
