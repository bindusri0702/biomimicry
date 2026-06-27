"""A/B benchmark for the Weaviate search modes — the "decide based on performance" harness.

Runs a fixed set of biologize-style (query, taxonomy-path) probes through each mode and reports,
per mode: mean hit-count, zero/under-floor rate (over-restriction), on-function precision (do the
returned strategies actually perform the asked function?), and latency. Also prints crosswalk
coverage. Pre-filter modes should show ~1.0 on-function precision; the unfiltered baselines show
how far pure ranking drifts off-function.

    python -m biomimicry.retrieval.bench_search_modes
    python -m biomimicry.retrieval.bench_search_modes --k 8 --alpha 0.5

Requires a populated Weaviate collection (run build_weaviate.py --recreate first) and creds.
The deeper "agreement with the LLM keep-filter" eval is intentionally out of scope here (it needs
the LLM and is expensive); on-function precision is the cheap mechanical proxy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import config
from .function_keys import keys_for_triple

_HERE = Path(__file__).resolve().parent
_CROSSWALK = _HERE / "function_crosswalk.json"

# Representative, well-populated taxonomy paths paired with a natural-language query.
FIXTURES = [
    {"q": "How does nature filter food particles out of water?",
     "group": "Get, store, or distribute resources",
     "sub_group": "Capture, absorb, or filter", "function": "liquids"},
    {"q": "How does nature store liquids inside the body?",
     "group": "Get, store, or distribute resources",
     "sub_group": "Store", "function": "liquids"},
    {"q": "How does nature move across solid surfaces?",
     "group": "Move or stay put", "sub_group": "Move", "function": "in/on solids"},
    {"q": "How does nature protect itself from freezing temperatures?",
     "group": "Protect from physical harm",
     "sub_group": "Protect from non-living threats", "function": "temperature"},
    {"q": "How does nature defend against predators and animals?",
     "group": "Protect from physical harm",
     "sub_group": "Protect from living threats", "function": "animals"},
    {"q": "How does nature assemble strong structures from available materials?",
     "group": "Make / modify", "sub_group": "Physically assemble", "function": "structure"},
    {"q": "How does nature manage impact and mechanical forces?",
     "group": "Maintain physical integrity",
     "sub_group": "Manage structural forces", "function": "impact"},
    {"q": "How does nature sense chemicals in its surroundings?",
     "group": "Process information",
     "sub_group": "Sense signals / environmental cues", "function": "chemicals (odor, taste, etc.)"},
]

MODES = ["vector", "hybrid", "filtered_vector", "filtered_hybrid"]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1))))
    return s[idx]


def _coverage() -> str:
    if not _CROSSWALK.exists():
        return "crosswalk.json missing"
    rows = json.loads(_CROSSWALK.read_text(encoding="utf-8")).get("rows", [])
    n = len(rows) or 1
    mapped = sum(1 for r in rows if r.get("sub_group"))
    leafed = sum(1 for r in rows if r.get("function"))
    return (f"{n} labels | sub-group-mapped {mapped} ({100*mapped//n}%) | "
            f"leaf-mapped {leafed} ({100*leafed//n}%)")


def run(k: int) -> None:
    from .weaviate_store import WeaviateRetriever

    floor = config.FILTER_MIN_HITS or max(1, k // 2)
    print(f"crosswalk coverage: {_coverage()}")
    print(f"k={k}  floor={floor}  alpha={config.HYBRID_ALPHA}  "
          f"filter_level={config.FUNCTION_FILTER_LEVEL}\n")

    retr = WeaviateRetriever()
    try:
        header = f"{'mode':<16}{'mean_hits':>10}{'under_floor':>13}{'on_function':>13}{'p50_ms':>9}{'p95_ms':>9}"
        print(header)
        print("-" * len(header))
        for mode in MODES:
            config.WEAVIATE_SEARCH_MODE = mode
            apply_filter = mode.startswith("filtered")
            hit_counts, on_fn_rates, latencies, under = [], [], [], 0
            for fx in FIXTURES:
                leaf_k, sub_k = keys_for_triple(fx["group"], fx["sub_group"], fx["function"])
                filters = ({"function_keys": [leaf_k] if leaf_k else [],
                            "subgroup_keys": [sub_k] if sub_k else []} if apply_filter else None)
                t0 = time.perf_counter()
                hits = retr.search(fx["q"], k=k, filters=filters)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                hit_counts.append(len(hits))
                if len(hits) < floor:
                    under += 1
                if hits:
                    on = sum(1 for h in hits if sub_k and sub_k in (h.doc.get("subgroup_keys") or []))
                    on_fn_rates.append(on / len(hits))
            mean_hits = sum(hit_counts) / len(hit_counts)
            on_fn = (sum(on_fn_rates) / len(on_fn_rates)) if on_fn_rates else 0.0
            print(f"{mode:<16}{mean_hits:>10.1f}{under:>9}/{len(FIXTURES):<3}"
                  f"{on_fn:>12.0%}{_percentile(latencies,50):>9.0f}{_percentile(latencies,95):>9.0f}")
    finally:
        retr.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark Weaviate search modes.")
    ap.add_argument("--k", type=int, default=config.DISCOVER_K_PER_HDN, help="hits per query")
    ap.add_argument("--alpha", type=float, default=None, help="override HYBRID_ALPHA for this run")
    args = ap.parse_args(argv)
    if args.alpha is not None:
        config.HYBRID_ALPHA = args.alpha
    run(args.k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
