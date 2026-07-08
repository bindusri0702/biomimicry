# Biomimicry Spiral Assistant — Handoff

Status, remaining work, and how to validate it. Pairs with [README.md](README.md) (architecture)

---

## 1. Current status (done)

The full four-stage spiral runs end-to-end, fully automated and LLM-driven:
`raw idea → Challenge Brief v1 (Define) → v2 (Biologize) → v3 (Discover) → Design strategies / v4 (Abstract)`.

- **All 4 stages** built as LangGraph subgraphs over one shared `SpiralState`
  ([state.py](state.py)): `stages/define.py` (2 nodes), `biologize.py` (3), `discover.py` (5),
  `abstract.py` (2) — 12 nodes total.
- **Spiral Controller** ([orchestrator.py](orchestrator.py)) — a plain linear forward chain
  `define → biologize → discover → abstract → END`. No human gates, interrupts, or backward
  transitions; a checkpointer is optional and none is required.
- **LLM** via LiteLLM ([llm.py](llm.py)) — complexity-routed Mistral tiers
  (`mistral/mistral-small-latest` by default), Gemini fallbacks; an API key is required
  (there is no offline stub).
- **Weaviate RAG** ([retrieval/](retrieval/)) — `get_retriever()` factory + `WeaviateRetriever`
  (BGE-M3 vectors, function-filtered hybrid search), corpus loader + ingest/build scripts.
- **Full AskNature corpus ingested** — the scraped strategies were converted to `StrategyDoc`
  JSON ([retrieval/build_asknature_corpus.py](retrieval/build_asknature_corpus.py)) and embedded
  into the Weaviate collection ([retrieval/build_weaviate.py](retrieval/build_weaviate.py)).
- **All evaluation metrics** per stage ([metrics.py](metrics.py), pure math over evaluator
  verdicts); deterministic Layer-0 residue check ([biology_denylist.py](biology_denylist.py));
  citation ledger + spiral log.
- Demo driver ([demo.py](demo.py)) runs the spiral and writes a timestamped brief JSON
  (full state + computed metrics).

---

## 2. Remaining tasks (prioritized)

### P0 — correctness / coverage

| # | Task | Where | Acceptance criteria |
|---|------|-------|---------------------|
| P0-1 | **Test suite.** No automated tests exist. Add `pytest` covering nodes, metrics, retriever, the residue check, and edge cases. | new `tests/` | `pytest` green; covers items in §3 L3. |
| P0-2 | **LLM output-quality validation.** Verify every `LLM.complete` task returns parseable, sane content across providers; fix prompts as needed. | [llm.py](llm.py), all `stages/*.py` | Full spiral completes to v4 with no schema-validation errors; spot-check output quality per stage. |

### P1 — production-readiness

| # | Task | Where | Acceptance criteria |
|---|------|-------|---------------------|
| P1-1 | **Backward transitions / Revision Reasoner.** The controller is currently a strict forward chain. If desired, add logic to raise a revision (e.g. Discover finds nothing relevant → revise Biologize), route backward once, and have the target stage clear the request. | [orchestrator.py](orchestrator.py); each `stages/*.py` | A seeded revision routes backward exactly once, the target clears it, the spiral proceeds forward; logged in `spiral_log`. |
| P1-2 | **Live/incremental retrieval.** Extend beyond the ingested Weaviate corpus — fetch new AskNature pages and scholarly sources (OpenAlex/Semantic Scholar/Crossref), convert to `StrategyDoc`, embed, and upsert into the collection. | [retrieval/build_weaviate.py](retrieval/build_weaviate.py) + new fetcher | New sources ingest and become retrievable without node changes. |

### P2 — fidelity / polish

| # | Task | Where | Notes |
|---|------|-------|-------|
| P2-1 | **Embedding-based taxonomy alignment.** `taxonomy.py` validates triples against `taxonomy_hierarchy.json`; alignment fidelity could improve with embedding cosine over the canonical export. | [taxonomy.py](taxonomy.py) | Improves taxonomy alignment. |
| P2-2 | **Richer Define elicitation + LLM-judge metrics.** Make elicitation a fuller dialogue; validate metrics against a real LLM judge. | [stages/define.py](stages/define.py), [stages/biologize.py](stages/biologize.py) | |
| P2-3 | **Packaging & config.** Add `pyproject.toml`, logging config, pinned deps. | repo root | |

---

## 3. Validation process

Layered: each layer must pass before the next is meaningful. L1 is the regression gate for
**every** change; L2+ as features land.

### L1 — Static
```bash
python -m compileall -q biomimicry            # must print nothing + exit 0
python -c "import biomimicry; from biomimicry.orchestrator import build_spiral; build_spiral()"
```
Pass: compiles; graph builds (all subgraphs wire, no schema/channel errors).

### L2 — End-to-end smoke test (needs an LLM key + Weaviate creds)
```bash
export MISTRAL_API_KEY=...                    # or any supported provider key
python -m biomimicry.demo "How might we protect people from fire accidents"
```
Pass: reaches `CHALLENGE BRIEF v4` with no schema-validation errors and writes a brief JSON.
Manually spot-check that HMWs/HDNs/strategies read sensibly and are on-topic.

### L3 — Unit tests (to be written, P0-1)
Target coverage:
- **Retriever** ([retrieval/weaviate_store.py](retrieval/weaviate_store.py)): function-filtered hybrid search returns relevant strategies; filter relaxation (leaf → sub-group → unfiltered) fires when a filtered query underflows; empty query handled.
- **Corpus** ([retrieval/corpus.py](retrieval/corpus.py)): valid docs load; a malformed doc raises with its path; bad `scale`/`source_tier`/`provenance` rejected; duplicate `doc_id` rejected.
- **Residue check** ([biology_denylist.py](biology_denylist.py)): `check_biology_residue` flags HARD/AMBIGUOUS organism terms, honors the ALLOWLIST, and injects source-organism terms dynamically.
- **Metrics** ([metrics.py](metrics.py)): each `*_metrics` handles empty input (no divide-by-zero); `_simpson`, dissimilarity, novelty penalty correct.
- **Dedup** ([stages/discover.py](stages/discover.py) `_merge_key`): same organism+strategy merges, keeps higher tier, unions `hdn_ids`.

### L4 — Per-stage assertions (programmatic)
Invoke a single subgraph with fixture state and assert outputs. Example (Discover):
```python
from biomimicry.stages.discover import build_discover_subgraph
g = build_discover_subgraph()
out = g.invoke({"hdn_questions": [{"id": 0, "text": "How does nature regulate temperature?"}],
                "mapped_functions": [{"verb": "regulate", "phrase": "regulate temperature"}],
                "spiral_log": [], "citation_ledger": []})
assert out["biological_models"]
assert all(m["doc_id"] for m in out["biological_models"])     # provenance present
```
Repeat per stage with representative fixtures.

### L5 — Guardrail validation (anti-hallucination, the project's core promise)
- **Citation traceability:** every `citation_ledger` entry has a non-empty `doc_id` that exists
  in the corpus and a `source_url`; every kept `biological_model` traces to a ledger entry. No
  organism appears that isn't in the corpus.
- **Jargon purge:** delivered design strategies pass the Layer-0 residue check
  (`check_biology_residue` returns no HARD hits) and the evaluator's faithfulness grade.
- **Function preservation:** each accepted strategy's tokens intersect the Define-stage function
  vocabulary.

---

## 4. Known issues & risks

- **No automated tests yet** (→ P0-1) — L1/L2 are the only regression gates today.
- **Forward-only spiral** — there is no revision/backward-transition path; a stage that produces
  weak output cannot ask an earlier stage to redo its work (→ P1-1).
- **Retrieval quality depends on the ingested corpus** — coverage/diversity of results is bounded
  by what was scraped and embedded into Weaviate.
- **Provider variance** — structured output uses native response-schema where supported
  (Gemini/OpenAI) and `json_object` otherwise (Groq/NIM); prompt/output quality can differ by
  provider (→ P0-2).

---

## 5. Quick start for the next engineer

```bash
pip install -r biomimicry/requirements.txt
export MISTRAL_API_KEY=...                    # or any supported provider key
python -m biomimicry.demo "How might we protect people from fire accidents"
```
- Add corpus docs: drop `*.json` into `retrieval/corpus/` (schema = `StrategyDoc`), then re-run
  `retrieval/build_weaviate.py` to ingest + embed them into the Weaviate collection.
- Retrieval needs `WEAVIATE_URL` + `WEAVIATE_API_KEY` (and an LLM key) in `biomimicry/.env`.
- Tunables (models, tiers, Weaviate search mode / hybrid alpha / filter level, thresholds,
  concurrency) live in [config.py](config.py) and are all env-overridable.
- To add/extend a stage, mirror an existing `stages/*.py`: nodes → `build_<stage>_subgraph()`
  → wire in [orchestrator.py](orchestrator.py).
