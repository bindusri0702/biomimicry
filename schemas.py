"""Pydantic response schemas for every LLM task — one model per `LLM.complete` call.

These restrict and validate the model's output. `LLM.complete` passes the schema to the
provider as native structured output (Gemini/OpenAI) or as a forced tool call (Groq, which
lacks json-schema but supports function calling), then validates the reply through the
model here (see llm.py).

Every field is defaulted so a provider omitting one validates to the default instead of
raising — this mirrors the old `raw.get(x) or default` defensiveness. Nullable verdict
fields are `Optional[...] = None`. Field *names* are verbatim from each prompt's OUTPUT
spec; do not rename them (they are the contract with the prompt).
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .state import SystemContext


# --- Define (stage 1) --------------------------------------------------------
class DefineResponse(BaseModel):
    """Parsed challenge context + one defined question per function + assumptions."""
    stakeholders: list[str] = Field(default_factory=list)
    operating_environment: str = ""
    hard_constraints: list[str] = Field(default_factory=list)
    system_context: SystemContext = Field(default_factory=SystemContext)
    defined_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class GoldilocksVerdict(BaseModel):
    index: int = -1
    question: str = ""
    reasoning: str = ""
    breadth_label: Optional[str] = None          # too_broad | just_right | too_narrow
    solution_neutral: Optional[bool] = None
    suggested_revision: Optional[str] = None


class GoldilocksResponse(BaseModel):
    """Altitude + solution-neutrality verdicts for a batch of defined questions."""
    verdicts: list[GoldilocksVerdict] = Field(default_factory=list)


# --- Biologize (stage 2) -----------------------------------------------------
class MappedFunctionItem(BaseModel):
    reasoning: str = ""
    approach: str = ""                           # direct | analogous | inverted
    group: str = ""
    sub_group: str = ""
    function: str = ""


class MapFunctionsResponse(BaseModel):
    """Taxonomy functions a challenge maps onto, through three lenses."""
    functions: list[MappedFunctionItem] = Field(default_factory=list)


class HDNItem(BaseModel):
    reasoning: str = ""
    hdn_question: str = ""
    approach: str = ""                           # direct | analogous | inverted
    group: str = ""
    sub_group: str = ""
    function: str = ""


class FrameHDNResponse(BaseModel):
    """"How does nature...?" questions framed from the selected functions."""
    questions: list[HDNItem] = Field(default_factory=list)


class BiologizeVerdict(BaseModel):
    id: int = -1
    neutrality_reasoning: str = ""
    neutrality: Optional[str] = None             # clean | borderline | leaks
    neutrality_offending_term: Optional[str] = None
    altitude_reasoning: str = ""
    altitude: Optional[str] = None               # too_high | just_right | too_low
    fidelity_reasoning: str = ""
    fidelity: Optional[str] = None               # on_lens_and_relevant | weak | off_lens_or_irrelevant
    lens_label_looks_correct: Optional[bool] = None


class BiologizeEvalResponse(BaseModel):
    """Three-axis verdicts for a batch of HDN questions (one challenge)."""
    verdicts: list[BiologizeVerdict] = Field(default_factory=list)


class RegenHDNResponse(BaseModel):
    """A reworked HDN question fixing the evaluator's flagged failures."""
    reasoning: str = ""
    hdn_question: str = ""
    approach: str = ""
    group: str = ""
    sub_group: str = ""
    function: str = ""


# --- Discover (stage 3) ------------------------------------------------------
class DiscoverVerdict(BaseModel):
    id: int = -1
    relevance_reasoning: str = ""
    functional_relevance: Optional[str] = None   # yes | partial | no
    adequacy_reasoning: str = ""
    mechanistic_adequacy: Optional[str] = None    # sufficient | thin | unusable


class DiscoverEvalResponse(BaseModel):
    """Relevance + adequacy verdicts for a batch of strategies retrieved for one HDN."""
    verdicts: list[DiscoverVerdict] = Field(default_factory=list)


# --- Abstract (stage 4) ------------------------------------------------------
class TermTranslation(BaseModel):
    biological_term: str = ""
    neutral_term: str = ""


class AbstractResponse(BaseModel):
    """A biological strategy translated into a discipline-neutral design strategy."""
    reasoning: str = ""
    summary: str = ""
    term_translations: list[TermTranslation] = Field(default_factory=list)
    design_strategy: str = ""
    function: str = ""
    organism: str = ""
    source_id: str = ""
    abstainable: bool = False


class SourceStep(BaseModel):
    id: str = ""
    step: str = ""


class StepCoverage(BaseModel):
    id: str = ""
    status: Optional[str] = None                 # present | missing | weakened


class AddedClaim(BaseModel):
    claim: str = ""
    status: Optional[str] = None                 # unsupported | contradicted


class AbstractEvalResponse(BaseModel):
    """Completeness + faithfulness grade of a design strategy vs its verbatim source."""
    source_steps: list[SourceStep] = Field(default_factory=list)
    completeness_reasoning: str = ""
    step_coverage: list[StepCoverage] = Field(default_factory=list)
    completeness: Optional[str] = None           # complete | partial | incomplete
    faithfulness_reasoning: str = ""
    added_claims: list[AddedClaim] = Field(default_factory=list)
    faithfulness: Optional[str] = None           # faithful | minor_additions | unfaithful
