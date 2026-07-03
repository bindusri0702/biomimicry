"""Shared spiral state (the Challenge Brief) and structured payload models.

`SpiralState` is the single LangGraph state object read/written by every stage.
The pipeline is fully automated and LLM-driven: there are no human gates, so the
payloads carry evaluator verdicts (accepted / best_effort + feedback) instead of
human-selection flags. Append-only logs use a merge reducer.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from pydantic import BaseModel, Field


# --- append-only reducer ------------------------------------------------------
def merge_lists(left: list | None, right: list | None) -> list:
    left = left or []
    if right is None:
        return left
    if not isinstance(right, list):
        right = [right]
    return left + right


# --- Define (stage 1) payloads -----------------------------------------------
class ContextProfile(BaseModel):
    """Parsed from the user challenge (no elicitation gate)."""
    stakeholders: list[str] = Field(default_factory=list)
    operating_environment: str = ""
    hard_constraints: list[str] = Field(default_factory=list)


class SystemContext(BaseModel):
    interactions: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    adjacent_systems: list[str] = Field(default_factory=list)
    leverage_points: list[str] = Field(default_factory=list)


class DefinedQuestion(BaseModel):
    """A solution-neutral "How might we...?" question — one per function in the challenge."""
    id: int
    text: str
    # Goldilocks evaluator (LLM) — two independent axes
    breadth_label: Optional[str] = None          # too_broad | just_right | too_narrow
    solution_neutral: Optional[bool] = None      # true = purely functional, false = solution baked in
    reasoning: str = ""
    suggested_revision: Optional[str] = None
    accepted: bool = True                         # all carry forward (best-effort if not clean)
    eval_status: Optional[str] = None             # accepted | best_effort
    eval_attempts: int = 0


# --- Biologize (stage 2) payloads --------------------------------------------
class MappedFunction(BaseModel):
    """A taxonomy function the challenge maps onto, via one of three lenses.

    Deduped by (group, sub_group, function): when several defined questions map the
    same taxonomy triple, they collapse into one row. `define_question_id`/`approach`
    are the first-seen primaries (used downstream); `define_question_ids`/`approaches`
    record every question and lens that mapped this function, for traceability.
    """
    define_question_id: int                       # the defined question this maps (primary)
    define_question_ids: list[int] = Field(default_factory=list)  # all questions that mapped this triple
    approach: str = ""                            # direct | analogous | inverted (primary)
    approaches: list[str] = Field(default_factory=list)  # all lenses that mapped this triple
    group: str = ""                               # verbatim taxonomy path...
    sub_group: str = ""
    function: str = ""
    reasoning: str = ""                           # written before committing to the path


class HDNQuestion(BaseModel):
    """A "How does nature...?" question framed from one mapped taxonomy function."""
    id: int
    define_question_id: int                       # the defined question this reframes (primary)
    define_question_ids: list[int] = Field(default_factory=list)  # all questions this function served
    approach: str = ""                            # direct | analogous | inverted (primary)
    approaches: list[str] = Field(default_factory=list)  # all lenses that mapped this function
    group: str = ""                               # verbatim taxonomy traceability path...
    sub_group: str = ""
    function: str = ""
    reasoning: str = ""                           # framing rationale
    text: str                                     # the HDN question (hdn_question)
    # Biologize evaluator (LLM) — three orthogonal axes
    neutrality: Optional[str] = None              # clean | borderline | leaks
    neutrality_offending_term: Optional[str] = None
    altitude: Optional[str] = None                # too_high | just_right | too_low
    fidelity: Optional[str] = None                # on_lens_and_relevant | weak | off_lens_or_irrelevant
    lens_label_looks_correct: Optional[bool] = None
    evaluator_reasoning: dict = Field(default_factory=dict)   # the three reasoning strings
    accepted: bool = False                        # passed evaluator (or best-effort retained)
    eval_status: Optional[str] = None             # accepted | best_effort
    eval_attempts: int = 0


# --- Discover (stage 3) payloads ---------------------------------------------
class BiologicalModel(BaseModel):
    id: int
    organism_common: str
    organism_scientific: str = ""
    strategy_summary: str = ""
    mechanism: str = ""
    function_addressed: list[str] = Field(default_factory=list)
    environment: str = ""                        # optional metadata (often empty until enriched)
    taxon: str = ""
    scale: Optional[str] = None
    source_url: str = ""
    source_tier: Optional[str] = None
    doc_id: str = ""                             # provenance back to the corpus
    hdn_ids: list[int] = Field(default_factory=list)   # which biologize questions retrieved it
    relevance_score: Optional[float] = None
    # discover evaluation (LLM) — graded retrieval-QA verdicts (deciding HDN pair)
    functional_relevance: Optional[str] = None    # yes | partial | no
    mechanistic_adequacy: Optional[str] = None     # sufficient | thin | unusable
    relevance_reasoning: str = ""                   # why the mechanism (mis)matches the asked function
    adequacy_reasoning: str = ""                    # whether content has transferable how-it-works detail
    hdn_verdicts: list = Field(default_factory=list)  # per-(organism×HDN) verdicts for traceability
    keep: bool = False                            # derived: rel∈{yes,partial} AND adeq∈{sufficient,thin} for ≥1 HDN


# --- Abstract (stage 4) payloads ---------------------------------------------
class BiologicalAbstraction(BaseModel):
    """Plain-English account of an organism's features/mechanism — NOT a design conclusion."""
    id: int
    model_id: int                                 # source BiologicalModel id
    organism_common: str = ""
    source_doc_id: str = ""                        # citation traceability (carried verbatim)
    source_taxon: str = ""
    source_scale: Optional[str] = None
    function: str = ""                             # the function this strategy was abstracted to answer
    mechanism_summary: str = ""                    # mechanism-first paragraph (= summary, biology kept)
    neutral_summary: str = ""                      # plain-English, scientific terms removed
    summary: str = ""                              # plain-language mechanism (prompt step 1)
    design_strategy: str = ""                      # neutral function+mechanism statement (headline output)
    statement: str = ""                            # canonical faithful account (mirrors design_strategy)
    term_translations: list[dict] = Field(default_factory=list)  # {biological_term, neutral_term} pairs
    functions_addressed: list[str] = Field(default_factory=list)
    jargon_terms: list[str] = Field(default_factory=list)        # biological_term values, for metrics
    abstract_reasoning: str = ""                   # the prompt's own reasoning field
    abstainable: bool = False                      # source too thin to support a faithful abstraction
    source_content: str = ""                       # verbatim biological source the grader judges against
    source_organism_terms: list[str] = Field(default_factory=list)  # organism names for residue injection
    # Layer 0: deterministic biological-residue check (notify-only, see biology_denylist.py)
    biology_residue_flag: bool = False             # HARD biological term present (definite residue)
    biology_hard_hits: list[str] = Field(default_factory=list)
    biology_ambiguous_hits: list[str] = Field(default_factory=list)
    biology_escalate: bool = False                 # AMBIGUOUS term present (for a future contextual judge)
    # abstract evaluator (LLM, two-axis grader vs the verbatim source)
    source_steps: list[dict] = Field(default_factory=list)       # decomposed source causal steps {id, step}
    completeness_reasoning: str = ""
    step_coverage: list[dict] = Field(default_factory=list)      # per source step {id, status}
    completeness: Optional[str] = None             # complete | partial | incomplete
    faithfulness_reasoning: str = ""
    added_claims: list[dict] = Field(default_factory=list)       # unsupported/contradicted {claim, status}
    faithfulness: Optional[str] = None             # faithful | minor_additions | unfaithful
    evaluator_feedback: str = ""
    accepted: bool = False
    eval_status: Optional[str] = None              # accepted | best_effort
    eval_attempts: int = 0


# --- the shared graph state ---------------------------------------------------
class SpiralState(TypedDict, total=False):
    # input
    raw_idea: str                  # the user challenge (sole input)

    # Define outputs
    context: dict                  # ContextProfile
    defined_questions: list        # list[DefinedQuestion dict] — one per function
    system_context: dict           # SystemContext
    assumptions: list              # gaps the DEFINE step refused to invent

    # Biologize (stage 2) outputs
    mapped_functions: list         # list[MappedFunction dict] — taxonomy paths per define question
    hdn_questions: list            # list[HDNQuestion dict] — framed per mapped function

    # Discover (stage 3) outputs
    search_queries: list           # list[{hdn_id, query, filters}]
    raw_hits: list                 # deduped retrieval hits
    biological_models: list        # list[BiologicalModel dict]

    # Abstract (stage 4) outputs
    abstractions: list             # list[BiologicalAbstraction dict]

    # control / bookkeeping
    spiral_log: Annotated[list, merge_lists]
    citation_ledger: Annotated[list, merge_lists]


def log_entry(stage: str, event: str, detail: str = "", **extra: Any) -> dict:
    """One spiral-log record. Timestamps are added by the caller layer if needed."""
    return {"stage": stage, "event": event, "detail": detail, **extra}
