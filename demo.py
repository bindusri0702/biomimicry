"""Run the biomimicry spiral end-to-end on a user challenge.

  python -m biomimicry.demo "How might we protect people from fire accidents"

Fully automated — no human gates. Requires an LLM API key (GEMINI_API_KEY /
GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY); there is no offline mode.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .orchestrator import build_spiral


def run(challenge: str, thread_id: str = "spiral-1") -> dict:
    """Invoke the spiral on a challenge and return the final state."""
    graph = build_spiral()
    init = {"raw_idea": challenge, "spiral_log": [], "citation_ledger": []}
    return graph.invoke(init, {"configurable": {"thread_id": thread_id}})


def _print_brief(state: dict) -> None:
    print("\n" + "=" * 70)
    print(f"CHALLENGE BRIEF {state.get('version', '?')}  (stage reached: "
          f"{state.get('current_stage')})")
    print("=" * 70)
    brief = {
        "raw_idea": state.get("raw_idea"),
        "context": state.get("context"),
        "system_context": state.get("system_context"),
        "assumptions": state.get("assumptions"),
        "defined_questions": state.get("defined_questions"),
        "define_metrics": state.get("define_metrics"),
        "mapped_functions": state.get("mapped_functions"),
        "hdn_questions": state.get("hdn_questions"),
        "biologize_metrics": state.get("biologize_metrics"),
        "biological_models": state.get("biological_models"),
        "discover_metrics": state.get("discover_metrics"),
        "citation_ledger": state.get("citation_ledger"),
        "abstractions": state.get("abstractions"),
        "abstract_metrics": state.get("abstract_metrics"),
        "spiral_log": state.get("spiral_log"),
    }
    print(json.dumps(brief, indent=2, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Biomimicry spiral — run a challenge end-to-end")
    ap.add_argument("challenge", help="the user challenge (required)")
    ap.add_argument("--quiet", action="store_true", help="don't print the full brief")
    args = ap.parse_args(argv)

    if not config.HAS_LLM_KEY:
        print("ERROR: no LLM API key found (set GEMINI_API_KEY / GOOGLE_API_KEY / "
              "OPENAI_API_KEY / ANTHROPIC_API_KEY). This pipeline has no offline mode.",
              file=sys.stderr)
        return 2

    print(f"model: {config.MODEL} | retrieval: {config.RETRIEVAL_BACKEND}")
    state = run(args.challenge)
    if not args.quiet:
        _print_brief(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
