# Biomimicry Spiral Assistant

LangGraph implementation of the full four-stage biomimicry spiral —
**Define → Biologize → Discover → Abstract**. The spiral is **fully automated**: a plain
linear forward chain (`orchestrator.py`) runs end-to-end from a raw idea to discipline-neutral
design strategies (`raw idea → Challenge Brief v1 → v2 → v3 → Design strategies / v4`).
There are no human gates, interrupts, or backward transitions.

LLM calls go through **LiteLLM**. By default each task is complexity-routed to a Mistral
tier (`mistral/mistral-small-latest`); set `BIOMIMICRY_MODEL` to pin every task to one model.
Discover retrieves from a **Weaviate Cloud** collection (local BGE-M3 embeddings). An LLM API key is **required**.

## Run

```bash
pip install -r requirements.txt
python -m biomimicry.demo "How might we protect people from fire accidents"
python -m biomimicry.demo "<challenge>" --quiet          # skip the terminal summary
python -m biomimicry.demo "<challenge>" --parallel 4     # bounded-parallel per-item LLM calls
```

The run writes a timestamped `brief-<slug>-<ts>.json` (full state + computed metrics) to the
cwd. A key is required — set one of:

```bash
export MISTRAL_API_KEY=...          # default provider
# or GROQ_API_KEY / NVIDIA_NIM_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY /
#    OPENAI_API_KEY / ANTHROPIC_API_KEY
export BIOMIMICRY_MODEL=gemini/gemini-2.5-flash   # optional: pin all tasks to one model
```

Retrieval also needs `WEAVIATE_URL` + `WEAVIATE_API_KEY`. Put these in `biomimicry/.env`.

## Layout

| File | Role |
|------|------|
| `state.py` | `SpiralState` — the shared Challenge Brief read/written by all stages; pydantic payload models; append-only log reducer |
| `schemas.py` | Pydantic response schemas — one per LLM task, used to restrict + validate output |
| `llm.py` | LiteLLM wrapper (`LLM.complete`, restricts output to a pydantic schema); requires an API key |
| `config.py` | Central config — model tiers/routing, temperatures, retrieval + Weaviate tunables (all env-overridable) |
| `taxonomy.py` | Biomimicry Taxonomy loader over `taxonomy_hierarchy.json` — `valid_paths`, `is_valid_path`, `render_for_prompt` |
| `biology_denylist.py` | Deterministic Layer-0 biological-residue check (`check_biology_residue`) used by Abstract |
| `metrics.py` | Pure-math per-stage metrics over evaluator verdicts (no domain heuristics) |
| `parallel.py` | `bounded_map` — thread-pool fan-out honoring `config.MAX_CONCURRENCY` |
| `retrieval/` | `get_retriever()` factory + `WeaviateRetriever` (BGE-M3 vectors, function-filtered hybrid search), corpus loader + ingest/build scripts |
| `stages/define.py` | Define subgraph (2 nodes): `define`, `goldilocks_evaluator` |
| `stages/biologize.py` | Biologize subgraph (3 nodes): `function_mapper`, `hdn_framer`, `biologize_evaluator` |
| `stages/discover.py` | Discover subgraph (5 nodes, RAG): `search_query_builder`, `fanout_retriever`, `organism_profile_builder`, `filter_with_reasoning`, `citation_ledger_writer` |
| `stages/abstract.py` | Abstract subgraph (2 nodes): `abstract_strategy`, `fidelity_evaluator` |
| `orchestrator.py` | Spiral Controller — wires the four subgraphs into a linear `define → … → abstract → END` chain; optional checkpointer |
| `demo.py` | End-to-end driver; runs the spiral and writes the timestamped brief JSON. Metrics are computed here (not graph nodes). |

## Define stage → Challenge Brief v1

```
define → goldilocks_evaluator
```

- `define` makes a single LLM call producing the context profile, system context, defined
  questions (HMWs), and assumptions.
- `goldilocks_evaluator` grades each question on two axes (breadth + solution-neutrality) and
  revises in place, capped by `config.EVALUATOR_MAX_RETRIES`.
- **Metrics** (computed at the end in `demo`): solution-neutrality, stakeholder specificity,
  breadth index (Goldilocks zone `0.3–0.7`), candidate uniqueness, context completeness.

## Stage 2 — Biologize → Challenge Brief v2

```
function_mapper → hdn_framer → biologize_evaluator
```

- `function_mapper` maps each HMW to Biomimicry-Taxonomy triples via three lenses, then
  validates every triple against the taxonomy (`taxonomy.is_valid_path`) and dedups.
- `hdn_framer` frames "How does nature…" (HDN) questions from the mapped functions.
- `biologize_evaluator` batch-grades HDNs on three axes (neutrality / altitude / fidelity) and
  regenerates the ones that fail, capped.
- **Metrics:** function coverage, taxonomy alignment, HDN sensibility, framing diversity.

## Stage 3 — Discover → Challenge Brief v3 (Weaviate RAG)

```
search_query_builder → fanout_retriever → organism_profile_builder
   → filter_with_reasoning → citation_ledger_writer
```

- **Retrieval** goes through `retrieval.get_retriever()` (mirrors the `LLM` factory) and is
  served by `WeaviateRetriever`: local **BGE-M3** dense vectors (`BAAI/bge-m3`, 1024-dim,
  L2-normalized) over a Weaviate Cloud collection, with **function-filtered hybrid search**
  (pre-filter on canonical function/sub-group keys, then BM25 + vector fusion). Filter
  relaxation (leaf → sub-group → unfiltered) fires when a filtered query underflows.
- `filter_with_reasoning` batch-grades each retrieved model per HDN for relevance + adequacy;
  `citation_ledger_writer` logs every kept model (URL, organism, `doc_id`, provenance) so
  biological claims trace to a real source and are never hallucinated.
- **Corpus** = JSON `StrategyDoc`s under `retrieval/corpus/` — the *source data* ingested into
  the Weaviate collection by `retrieval/build_weaviate.py` (which embeds each doc with BGE-M3
  and resolves function keys). Add `*.json` docs and re-run the build to grow the index.
- **Metrics:** HDN relevance, taxonomic + scale diversity, source credibility, novelty index.

Search mode, hybrid alpha, filter granularity, and the embedding model live in `config.py`
(env-overridable: `BIOMIMICRY_WEAVIATE_SEARCH_MODE`, `BIOMIMICRY_HYBRID_ALPHA`,
`BIOMIMICRY_FUNCTION_FILTER_LEVEL`, `BIOMIMICRY_E5_MODEL` — the `E5_MODEL` constant name is kept
for back-compat but the model is BGE-M3).

## Stage 4 — Abstract → Design strategies (Challenge Brief v4)

```
abstract_strategy → fidelity_evaluator
```

- `abstract_strategy` climbs the abstraction ladder from each kept biological model to a
  transferable, discipline-neutral **design strategy statement** — summarizing the mechanism,
  translating biological terms to neutral ones (LLM term-translate), and writing the strategy
  (function + mechanism, no jargon, no prescribed artefact).
- `fidelity_evaluator` first runs the deterministic Layer-0 residue check
  (`biology_denylist.check_biology_residue`, notify-only), then grades each abstraction for
  completeness + faithfulness and regenerates failures, capped.
- **Metrics:** jargon purge, function preservation, solution-neutrality, abstraction
  appropriateness.

## Spiral control

`orchestrator.build_spiral()` compiles the linear forward chain
`define → biologize → discover → abstract → END` over the shared `SpiralState`. A checkpointer
is optional (only useful for online crash-resume); none is required because nothing interrupts.
Every stage writes to `SpiralState`, and biological claims trace through the append-only
`citation_ledger` written in Discover.
