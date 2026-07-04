"""Run the biomimicry spiral end-to-end on a user challenge.

  python -m biomimicry.demo "How might we protect people from fire accidents"

Fully automated — no human gates. Requires an LLM API key (MISTRAL_API_KEY /
GROQ_API_KEY / NVIDIA_NIM_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY /
ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time

from . import config
from .metrics import (abstract_metrics, biologize_metrics, define_metrics,
                      discover_metrics)
from .orchestrator import build_spiral


def run(challenge: str, thread_id: str = "spiral-1") -> dict:
    """Invoke the spiral on a challenge and return the final state.

    Streams the graph (rather than a plain invoke) to time each top-level stage: the
    `updates` chunk names the stage that just finished, the `values` chunk carries the full
    merged state. Per-stage wall-clock (seconds) + a `total` are attached to the returned
    state under `stage_timings_seconds`."""
    graph = build_spiral()
    init = {"raw_idea": challenge, "spiral_log": [], "citation_ledger": []}
    cfg = {"configurable": {"thread_id": thread_id}}

    timings: dict[str, float] = {}
    final_state: dict = {}
    start = time.perf_counter()
    prev = start
    for mode, chunk in graph.stream(init, cfg, stream_mode=["updates", "values"]):
        if mode == "updates":                       # {stage_name: delta} — stage just finished
            now = time.perf_counter()
            for stage_name in chunk:
                timings[stage_name] = round(now - prev, 3)
            prev = now
        elif mode == "values":                      # full merged state after the step
            final_state = chunk
    timings["total"] = round(time.perf_counter() - start, 3)
    final_state["stage_timings_seconds"] = timings
    return final_state


def _reached(state: dict) -> tuple[str, str]:
    """Infer the (version, stage) reached from which stage payload is populated."""
    if state.get("abstractions"):
        return "v4", "abstract"
    if state.get("biological_models"):
        return "v3", "discover"
    if state.get("hdn_questions"):
        return "v2", "biologize"
    if state.get("defined_questions"):
        return "v1", "define"
    return "?", "-"


def _build_brief(state: dict) -> dict:
    """Assemble the entire spiral state plus computed metrics.

    Everything in the state is already plain dict/list/str (LangGraph state), so
    it is JSON-serializable as-is. Metrics are pure functions of the final state,
    computed here on demand rather than stored in the graph state; each metric
    block is interleaved after the stage fields it summarizes.
    """
    return {
        # per-stage wall-clock (seconds): define/biologize/discover/abstract + total
        "stage_timings_seconds": state.get("stage_timings_seconds", {}),
        # Define
        "raw_idea": state.get("raw_idea"),
        "context": state.get("context"),
        "system_context": state.get("system_context"),
        "assumptions": state.get("assumptions"),
        "defined_questions": state.get("defined_questions"),
        "define_metrics": define_metrics(state.get("defined_questions", []),
                                         state.get("context", {})),
        # Biologize
        "mapped_functions": state.get("mapped_functions"),
        "hdn_questions": state.get("hdn_questions"),
        "biologize_metrics": biologize_metrics(state.get("hdn_questions", []),
                                               state.get("mapped_functions", [])),
        # Discover
        "search_queries": state.get("search_queries"),
        "raw_hits": state.get("raw_hits"),
        "biological_models": state.get("biological_models"),
        "discover_metrics": discover_metrics(state.get("biological_models", []),
                                             state.get("hdn_questions", [])),
        "citation_ledger": state.get("citation_ledger"),
        # Abstract
        "abstractions": state.get("abstractions"),
        "abstract_metrics": abstract_metrics(state.get("abstractions", [])),
        # control / bookkeeping
        "spiral_log": state.get("spiral_log"),
    }


def _output_path(state: dict) -> str:
    """A fresh timestamped filename in the cwd, derived from the challenge."""
    slug = re.sub(r"[^a-z0-9]+", "-", (state.get("raw_idea") or "").lower()).strip("-")[:40]
    slug = slug or "brief"
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"brief-{slug}-{ts}.json"


def _write_brief(state: dict) -> str:
    """Write the full brief (entire state + metrics) to a new JSON file; return its path."""
    path = _output_path(state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_build_brief(state), f, indent=2, ensure_ascii=False)
    return path


def _print_summary(state: dict, path: str) -> None:
    version, stage = _reached(state)
    dm = define_metrics(state.get("defined_questions", []), state.get("context", {}))
    bm = biologize_metrics(state.get("hdn_questions", []), state.get("mapped_functions", []))
    disc = discover_metrics(state.get("biological_models", []), state.get("hdn_questions", []))
    am = abstract_metrics(state.get("abstractions", []))
    print("\n" + "=" * 70)
    print(f"CHALLENGE BRIEF {version}  (stage reached: {stage})")
    print("=" * 70)
    print(f"defined_questions:      {dm['defined_question_count']}")
    print(f"hdn accepted:           {bm['accepted_count']}")
    print(f"models kept:            {disc['kept_count']}/{disc['retrieved_count']}")
    print(f"abstractions accepted:  {am['accepted_count']}/{am['total_count']}")
    st = state.get("stage_timings_seconds", {})
    if st:
        parts = [f"{stg}={st[stg]}s" for stg in
                 ("define", "biologize", "discover", "abstract", "total") if stg in st]
        print(f"stage timings:          {'  '.join(parts)}")
    print(f"\n-> full brief written to: {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Biomimicry spiral — run a challenge end-to-end")
    ap.add_argument("challenge", help="the user challenge (required)")
    ap.add_argument("--quiet", action="store_true",
                    help="don't print the terminal summary (file is still written)")
    ap.add_argument("--parallel", nargs="?", type=int, const=4, default=None, metavar="N",
                    help="run the Biologize/Discover per-item LLM calls with bounded "
                         "parallelism. Bare --parallel uses 4 workers; --parallel N sets the "
                         "cap; omit for sequential. Keep N <= your provider tier's "
                         "requests-per-second (RPS) to avoid 429s.")
    args = ap.parse_args(argv)

    # CLI overrides the env/default cap; None means 'not passed' so the env value stands.
    if args.parallel is not None:
        config.MAX_CONCURRENCY = args.parallel

    if not config.HAS_LLM_KEY:
        print("ERROR: no LLM API key found (set MISTRAL_API_KEY, or GROQ_API_KEY / "
              "NVIDIA_NIM_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY / "
              "ANTHROPIC_API_KEY).", file=sys.stderr)
        return 2

    model_desc = config.MODEL_OVERRIDE or f"super={config.MODEL_SUPER}, nano={config.MODEL_NANO}"
    concurrency = (f"parallel={config.MAX_CONCURRENCY}"
                   if config.MAX_CONCURRENCY > 1 else "sequential")
    print(f"model: {model_desc} | retrieval: weaviate | {concurrency}")
    state = run(args.challenge)
    path = _write_brief(state)
    if not args.quiet:
        _print_summary(state, path)
    else:
        print(f"full brief written to: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
