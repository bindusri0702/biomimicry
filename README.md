# Biomimicry Spiral Assistant

LangGraph implementation of the full four-stage biomimicry spiral —
**Define → Biologize → Discover → Abstract** — orchestrated by a **Spiral Controller**
with a human gate after every stage and backward-transition support. All four stages
are built; the spiral runs end-to-end from a raw idea to discipline-neutral design
strategies (`raw idea → Challenge Brief v1 → v2 → v3 → Design strategies / v4`).

LLM calls go through **LiteLLM** (`gemini/gemini-2.5-flash` by default, swappable).
Discover uses **offline RAG**: retrieval reads a local corpus, no network at runtime.

## Run

```bash
pip install -r requirements.txt
python -m biomimicry.demo                 # auto-resumes human gates
python -m biomimicry.demo --interactive   # answer gates at the prompt
```

No API key → deterministic **offline stub** runs automatically. For real calls:

```bash
export GEMINI_API_KEY=...           # unset BIOMIMICRY_OFFLINE if previously set
export BIOMIMICRY_MODEL=gemini/gemini-2.5-flash   # or anthropic/claude-opus-4-8, etc.
```

## Layout

| File | Role |
|------|------|
| `state.py` | `SpiralState` — the shared Challenge Brief read/written by all stages; pydantic payload models; append-only log reducer |
| `llm.py` | LiteLLM wrapper (`complete_json`) + offline stub |
| `taxonomy.py` | Biomimicry Taxonomy reference + `align()` (token-overlap; swap for embeddings) |
| `lexicon.py` | Bio-to-neutral substitution table + jargon detection (Term Neutralizer) |
| `metrics.py` | Heuristic scorers + Define / Biologize / Discover / Abstract evaluation metrics |
| `retrieval/` | Pluggable `Retriever` factory, BM25 offline backend, embedding backend, corpus loader + synthetic seed docs |
| `stages/define.py` | Define subgraph: 5 sub-components, metrics, human gate, finalize |
| `stages/biologize.py` | Biologize subgraph: 6 sub-components, metrics, human gate, finalize |
| `stages/discover.py` | Discover subgraph: 6 sub-components (RAG), metrics, human gate, finalize |
| `stages/abstract.py` | Abstract subgraph: 7 sub-components, metrics, human gate, finalize |
| `orchestrator.py` | Spiral Controller — state-machine router + backward-transition handler + checkpointer |
| `demo.py` | End-to-end driver with human-gate resolution |

## Define stage → Challenge Brief v1

```
context_elicitor → hmw_generator → goldilocks_critic → scope_scorer
   → system_mapper → compute_metrics → human_gate → finalize
```

- **Human gates** use LangGraph `interrupt()`. `context_elicitor` only pauses when a
  required slot is missing; `human_gate` pauses for HMW select / edit / merge. Set
  `auto_select_id` in state to bypass the HMW gate non-interactively.
- **Metrics:** solution-neutrality, stakeholder specificity, breadth index
  (Goldilocks zone `0.3–0.7`), candidate uniqueness, context completeness.

## Stage 2 — Biologize → Challenge Brief v2

```
function_decomposer → context_mapper → taxonomy_aligner → hdn_generator
   → flip_engine → framing_ranker → compute_metrics → human_gate → finalize
```

- **Taxonomy Aligner** = heuristic recall (`taxonomy.align`, top-3 shortlist) + LLM
  pick, grounded to the shortlist so it can't hallucinate a node. Confidence is the
  overlap score (swap in embedding cosine without touching call sites).
- **Flip Engine** adds shared-mechanism inversions; **Framing Ranker** guarantees
  function coverage, keeps coherent flips, and greedily drops near-duplicate framings.
- **Metrics:** function coverage ratio, taxonomy alignment confidence, HDN biological
  sensibility, framing diversity index, flip-pair coherence.

## Stage 3 — Discover → Challenge Brief v3 (offline RAG)

```
search_query_builder → multi_source_retriever → organism_profile_builder
   → credibility_screener → diversity_ranker → citation_ledger_writer
   → compute_metrics → human_gate → finalize
```

- **Retrieval** goes through `retrieval.get_retriever()` (mirrors the `LLM` factory):
  default `OfflineRetriever` = dependency-free **BM25** over `retrieval/corpus/`;
  `EmbeddingRetriever` (LiteLLM) activates only on `BIOMIMICRY_RETRIEVAL=embedding` + a
  key. A future `LiveRetriever` (httpx + AskNature/OpenAlex/RSS) drops in via the factory
  with **zero node-code change**.
- **Corpus** = JSON `StrategyDoc`s. Seeded with 4 synthetic docs modeled on real AskNature
  strategies (`provenance:"synthetic"` + real URL). Drop a large real corpus into
  `retrieval/corpus/` and it loads unchanged (`load_corpus` validates each doc).
- **Guardrail:** `citation_ledger_writer` logs every screened model (URL, organism,
  `doc_id`, provenance) — biological claims trace to a real source, never hallucinated.
- **Metrics:** HDN relevance, taxonomic diversity, scale diversity, source credibility,
  mechanism completeness, novelty index (usual-suspects penalized unless on-target).

To extend the corpus, add `*.json` files; to go online, set `GEMINI_API_KEY` and
`BIOMIMICRY_RETRIEVAL=embedding`. BM25 params, tiers, and ranker weights live in `config.py`.

## Stage 4 — Abstract → Design strategies (Challenge Brief v4)

```
mechanism_summarizer → keyword_extractor → term_neutralizer → design_strategy_writer
   → strategy_validator → strategy_visualizer → cross_model_pattern_finder
   → compute_metrics → human_gate → finalize
```

- Climbs the abstraction ladder from each selected biological model to a transferable,
  discipline-neutral **design strategy statement** (function + mechanism, no jargon, no
  prescribed artefact).
- **Term Neutralizer** is rule-based (`lexicon.py`): a curated bio→neutral table
  (`fur→fibers`, `vascular network→channel network`, …), deterministic.
- **Strategy Validator** is the guardrail critic: (a) zero biological jargon
  (`detect_jargon`, incl. organism-name tokens minus generic descriptors), (b) original
  Define-stage function preserved, (c) no specific artefact/technology (`config.ARTEFACT_TERMS`
  — narrower than Define's solution-term list, so neutral mechanism nouns aren't penalized).
- **Cross-model Pattern Finder** clusters strategies by mechanism similarity and flags
  convergence when a cluster spans ≥2 taxa or scales.
- **Metrics:** jargon purge, function preservation, solution-neutrality, cross-disciplinary
  accessibility, abstraction appropriateness, cross-model convergence rate.

## Spiral control & backward transitions

The router (`spiral_router`) advances `define → biologize → discover → abstract → END`.
A backward revision is any node returning
`{"revision_request": {"target_stage": "define", "reason": "..."}}`; the target stage
should clear it. Every stage writes to the shared `SpiralState`, and biological claims
trace through the append-only `citation_ledger` written in Discover.

### Human-gate convention (important)

Gates use LangGraph `interrupt()`. **Always resume with a truthy, non-empty value** —
`Command(resume={})` (empty/falsy) is treated as "no resume" and the gate re-fires in
a loop. The resolvers return e.g. `{"ids": [...]}` rather than `{}`.
