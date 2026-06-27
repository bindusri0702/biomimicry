# Biomimicry Spiral Assistant — Handoff

Status, remaining work, and how to validate it. Pairs with [README.md](README.md)
(architecture) and the approved design at `~/.claude/plans/starry-stirring-sonnet.md`.

---

## 1. Current status (done)

The full four-stage spiral runs end-to-end, offline-deterministic by default:
`raw idea → Challenge Brief v1 (Define) → v2 (Biologize) → v3 (Discover) → Design strategies / v4 (Abstract)`.

- **All 4 stages** built as LangGraph subgraphs over one shared `SpiralState`
  ([state.py](state.py)): `stages/define.py`, `biologize.py`, `discover.py`, `abstract.py`
  (24 sub-components total, each a node).
- **Spiral Controller** ([orchestrator.py](orchestrator.py)) — state-machine router,
  human gate after every stage (`interrupt()`), `MemorySaver` checkpointer.
- **LLM** via LiteLLM ([llm.py](llm.py)) — `gemini/gemini-2.5-flash` default; deterministic
  offline stub auto-engages with no API key.
- **Offline RAG** ([retrieval/](retrieval/)) — pluggable `Retriever` factory, pure-python
  BM25 backend, embedding backend (untested), corpus loader + **4 synthetic seed docs**.
- **All evaluation metrics** per stage ([metrics.py](metrics.py)); bio→neutral
  [lexicon.py](lexicon.py); citation ledger + spiral log.
- Demo driver ([demo.py](demo.py)) resolves all gates; programmatic override runs with
  zero interrupts.

**Baseline that must keep passing** (see §3, L2): `python -m biomimicry.demo` reaches
`CHALLENGE BRIEF v4` and is byte-identical across runs.

---

## 2. Remaining tasks (prioritized)

### P0 — required before any real (non-stub) use

| # | Task | Where | Acceptance criteria |
|---|------|-------|---------------------|
| P0-1 | **Validate against real Gemini.** Never run with a key. Verify every `complete_json` task returns parseable JSON and sane content; fix prompts as needed. Tasks: `context`, `hmw`, `goldilocks`, `system_map`, `functions`, `context_conditions`, `taxonomy_pick`, `hdn`, `flip`, `search_queries`, `mechanism_summary`, `design_strategy`. | [llm.py](llm.py), all `stages/*.py` | With `GEMINI_API_KEY` set + `BIOMIMICRY_OFFLINE=0`, full spiral completes to v4 with no JSON parse errors; spot-check output quality per stage. |
| P0-2 | **Full corpus ingestion.** Replace the 4 seed docs with the real corpus. Build a parser that converts downloaded AskNature (and other) pages → `StrategyDoc` JSON, setting `provenance:"fetched"`. | new `retrieval/ingest.py`; `retrieval/corpus/` | ≥50 real docs load via `load_corpus()`; spread across all 4 scales, 3 tiers, many taxa; diversity/novelty/credibility metrics become differentiated (not all 1.0). |
| P0-3 | **Durable checkpointer.** `MemorySaver` loses paused state on process exit — real human gates may span hours. Swap to `SqliteSaver`/`PostgresSaver`. | [orchestrator.py](orchestrator.py) `build_spiral()` | Start spiral → hit a gate → kill process → new process resumes the same `thread_id` from the gate. |
| P0-4 | **Test suite.** No automated tests exist. Add `pytest` covering nodes, metrics, retriever, lexicon, determinism, gate resume, override, edge cases. | new `tests/` | `pytest` green; covers items in §3 L3. |

### P1 — production-readiness

| # | Task | Where | Acceptance criteria |
|---|------|-------|---------------------|
| P1-1 | **LiveRetriever.** Online retrieval behind the existing factory: httpx + AskNature fetch/parse, OpenAlex/Semantic Scholar/Crossref for scholarly (NOT Scholar scraping), EurekAlert/ScienceDaily RSS; write results into the corpus cache. Domain→tier rubric for `credibility_screener`. | new `retrieval/live.py`; `retrieval/base.py` `get_retriever()` | `BIOMIMICRY_RETRIEVAL=live` retrieves real strategies; node code unchanged; results cached and re-used. |
| P1-2 | **Backward transitions / Revision Reasoner.** Router honors `revision_request` but **no node sets it and no stage clears it → infinite-loop risk**. Add logic to raise a revision (e.g., Discover finds nothing relevant → revise Biologize) and have the target stage clear `revision_request`. | [orchestrator.py](orchestrator.py); each `stages/*.py` finalize | A seeded `revision_request` routes backward exactly once, the target clears it, the spiral then proceeds forward; logged in `spiral_log`. |
| P1-3 | **EmbeddingRetriever: validate + persist.** Implemented but never run; recomputes doc vectors every init. Add a persisted vector cache and test cosine ranking parity with BM25. | [retrieval/embedding.py](retrieval/embedding.py) | With a key + `BIOMIMICRY_RETRIEVAL=embedding`, retrieval ranks sensibly; vectors cached to disk, not recomputed. |
| P1-4 | **Challenge Brief persistence.** No save/load. Persist the final brief + `citation_ledger` + `spiral_log` to JSON; support resuming/inspecting a run. | new `persistence.py` or in `demo.py` | A completed run writes a single brief JSON; reloading reproduces the delivered strategies + full provenance. |
| P1-5 | **LLM robustness.** `complete_json` has a parse fallback but no retry on malformed JSON / rate limits / transient errors. Add bounded retry + backoff (LiteLLM supports it) and a schema re-ask. | [llm.py](llm.py) | Inject a malformed response → one retry recovers; rate-limit error backs off rather than crashing the spiral. |

### P2 — fidelity / polish

| # | Task | Where | Notes |
|---|------|-------|-------|
| P2-1 | **Strategy Visualizer → real diagram.** Currently emits a text/ASCII schematic spec. Render an actual mechanical schematic (graphviz/SVG) from `schematic{}`. | [stages/abstract.py](stages/abstract.py) `strategy_visualizer` | Architecture wants a diagram an engineer can read; the spec is render-ready. |
| P2-2 | **Full Biomimicry Taxonomy + embedding alignment.** `taxonomy.py` is a curated subset with token-overlap `align()`. Replace with the canonical AskNature export and switch `align()` to embedding cosine. | [taxonomy.py](taxonomy.py) | Improves `taxonomy_alignment_confidence` fidelity. |
| P2-3 | **Vector DB semantic cache.** Architecture lists one to dedupe fetches; offline uses an in-memory index. Add a persistent store once live retrieval lands. | `retrieval/` | Depends on P1-1. |
| P2-4 | **Richer Define context elicitation + LLM-judge metrics.** Make the elicitation a fuller dialogue; validate heuristic-vs-judge metrics (e.g., HDN biological sensibility) against the real LLM judge. | [stages/define.py](stages/define.py), [stages/biologize.py](stages/biologize.py) | |
| P2-5 | **Packaging & config.** Add `pyproject.toml`, `.env` handling, logging config, pinned deps. | repo root | |

---

## 3. Validation process

Layered: each layer must pass before the next is meaningful. L1–L2 are the regression
gate for **every** change; L3+ as features land.

### L1 — Static
```bash
python -m compileall -q biomimicry            # must print nothing + exit 0
python -c "import biomimicry; from biomimicry.orchestrator import build_spiral; build_spiral()"
```
Pass: compiles; graph builds (all subgraphs wire, no schema/channel errors).

### L2 — Offline determinism (regression baseline)
```bash
python -m biomimicry.demo > r1.txt 2>&1
python -m biomimicry.demo > r2.txt 2>&1
diff -q r1.txt r2.txt && echo OK         # must be identical
```
Pass: reaches `CHALLENGE BRIEF v4`; both runs byte-identical (BM25 + offline stubs are
deterministic). Any non-determinism is a bug (unsorted set emitted, `Date.now`-style call, etc.).

### L3 — Unit tests (to be written, P0-4)
Target coverage:
- **Retriever** ([retrieval/lexical.py](retrieval/lexical.py)): cooling query ranks termite/silver-ant > lotus; water query ranks lotus top; empty query → `[]`.
- **Corpus** ([retrieval/corpus.py](retrieval/corpus.py)): valid docs load; a malformed doc raises with its path; bad `scale`/`source_tier`/`provenance` rejected; duplicate `doc_id` rejected.
- **Lexicon** ([lexicon.py](lexicon.py)): `neutralize` replaces multi-word before single-word; `detect_jargon` flags organism tokens but not `GENERIC_DESCRIPTORS` ("water"/"silver").
- **Metrics** ([metrics.py](metrics.py)): each `*_metrics` handles empty input (no divide-by-zero); `_simpson`, `mechanism_completeness`, novelty penalty correct.
- **Dedup** ([stages/discover.py](stages/discover.py) `_merge_key`): same organism+strategy merges, keeps higher tier, unions `hdn_ids`.
- **Gate resume**: empty/falsy resume value must NOT re-fire (the documented gotcha); explicit `{"ids": [...]}` advances.
- **Override**: setting all `auto_select_*` keys runs the spiral with `__interrupt__` absent.

### L4 — Per-stage assertions (programmatic)
Invoke a single subgraph with fixture state and assert outputs. Example (Discover):
```python
from biomimicry.stages.discover import build_discover_subgraph
g = build_discover_subgraph()
out = g.invoke({"hdn_questions": [{"id":0,"text":"How does nature regulate temperature?","selected":True}],
                "functions": [{"verb":"regulate","phrase":"regulate temperature"}],
                "auto_select_model_ids": [0], "spiral_log": [], "citation_ledger": []})
assert out["biological_models"]
assert all(m["doc_id"] for m in out["biological_models"])     # provenance present
assert out["discover_metrics"]["selected_count"] >= 1
```
Repeat per stage with representative fixtures.

### L5 — Online smoke test (needs key)
```bash
export GEMINI_API_KEY=...   ;   export BIOMIMICRY_OFFLINE=0
python -m biomimicry.demo
```
Pass: completes to v4 with no JSON parse errors (validates P0-1). Manually spot-check that
HMWs/HDNs/strategies read sensibly and are on-topic.

### L6 — Guardrail validation (anti-hallucination, the project's core promise)
- **Citation traceability:** every entry in `citation_ledger` has a non-empty `doc_id` that
  exists in the corpus and a `source_url`; every selected `biological_model` traces to a
  ledger entry. No organism appears that isn't in the corpus.
- **Jargon purge:** in delivered `design_strategies`, `valid` strategies have empty
  `jargon_terms` and `jargon_purge_score == 1.0`.
- **Solution-neutrality:** no `valid` strategy statement contains a `config.ARTEFACT_TERMS` term.
- **Function preservation:** each `valid` strategy's tokens intersect the Define-stage function vocab.

Quick check:
```python
import json, subprocess
# run demo, parse the printed brief, assert the four guardrails above
```
(Fold into L3 once tests exist.)

### L7 — Human-gate validation (interactive)
```bash
python -m biomimicry.demo --interactive    # answer each of the 4 gates manually
```
Pass: each gate (`hmw_selection`, `hdn_selection`, `model_selection`, `strategy_selection`)
pauses, accepts selection / `edit` / `merge` / explicit ids, and resumes correctly. After
P0-3, verify resume survives a process restart.

### L8 — Corpus / scale validation (after P0-2)
Pass: `load_corpus()` validates the full set; with the real corpus, diversity/novelty/
credibility metrics are differentiated and the ranker visibly reorders vs raw relevance
(e.g., usual suspects penalized, scale spread enforced).

---

## 4. Known issues & risks

- **In-memory checkpoints (`MemorySaver`)** — paused gates don't survive restart (→ P0-3).
- **Backward-transition loop risk** — `spiral_router` honors `revision_request` but nothing
  clears it; a set value would loop (→ P1-2). Currently unused, so safe today.
- **Offline stub text is illustrative**, not real biology — phrasing can read awkwardly and
  the mechanism summary may leak an organism name (the validator correctly flags it, e.g.
  "ant"). Real Gemini output is clean. Don't judge content quality from offline runs.
- **4-doc corpus** makes diversity/novelty metrics under-differentiated; expected until P0-2.
- **EmbeddingRetriever untested** (no key in dev) and recomputes vectors each init (→ P1-3).
- **Gate resume must use a truthy, non-empty value** — `Command(resume={})` is treated as
  "no resume" and re-fires the interrupt (documented in [README.md](README.md)).

---

## 5. Quick start for the next engineer

```bash
pip install -r biomimicry/requirements.txt
python -m biomimicry.demo                 # offline, deterministic — see the whole spiral
python -m biomimicry.demo --interactive    # drive the human gates yourself
```
- Add corpus docs: drop `*.json` into `retrieval/corpus/` (schema = `StrategyDoc`); they load automatically.
- Go online: `export GEMINI_API_KEY=...; export BIOMIMICRY_OFFLINE=0` (and `BIOMIMICRY_RETRIEVAL=embedding` for semantic retrieval).
- Tunables (models, tiers, BM25 params, ranker weights, thresholds) live in [config.py](config.py).
- To add/extend a stage, mirror an existing `stages/*.py`: nodes → `build_<stage>_subgraph()` (compile **without** a checkpointer) → wire in [orchestrator.py](orchestrator.py).
