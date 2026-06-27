"""Convert scraped AskNature strategies into the StrategyDoc corpus.

Reads the scraper output (``asknature_scraper/strategies/<slug>.json``) and writes one
``StrategyDoc``-valid file per strategy into ``corpus/asknature/``. ``load_corpus`` then
discovers them via its ``rglob("*.json")`` with no other code change -- the
"real downloaded corpus drops in" path documented in ``corpus.py``.

    python -m biomimicry.retrieval.build_asknature_corpus
    python -m biomimicry.retrieval.build_asknature_corpus --limit 200

Each record is validated against ``StrategyDoc`` before being written, so bad data is
reported and skipped rather than silently breaking ``load_corpus`` later.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from .base import tokenize
from .corpus import StrategyDoc
from .function_keys import keys_for_labels

_HERE = Path(__file__).resolve().parent
DEFAULT_SRC = _HERE.parents[1] / "asknature_scraper" / "strategies"
OUT_DIR = _HERE / "corpus" / "asknature"

# Short, content-free tokens we don't want as keywords / merge-key seeds.
_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "how",
    "via", "use", "uses", "using", "their", "its", "can", "help", "helps",
}
_LAST_UPDATED = re.compile(r"\s*Last Updated\b.*$", re.IGNORECASE | re.DOTALL)


def _slug_from_url(url: str, fallback: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    return slug or fallback


def _strip_boilerplate(text: str) -> str:
    text = _LAST_UPDATED.sub("", text or "")
    return text.strip()


def _first_sentences(text: str, n: int = 2, cap: int = 320) -> str:
    """First ~n sentences of `text`, capped to `cap` chars."""
    text = _strip_boilerplate(text).replace("\n", " ").strip()
    if not text:
        return ""
    # Split on sentence-ending punctuation followed by a space + capital/quote.
    parts = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(parts[:n]).strip()
    if len(summary) > cap:
        summary = summary[:cap].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return summary


def _keywords(title: str, limit: int = 6) -> list[str]:
    out: list[str] = []
    for tok in tokenize(title):
        if tok in _STOP or len(tok) < 3:
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= limit:
            break
    return out


def convert_one(raw: dict, url_fallback_slug: str) -> tuple[str, dict]:
    """Return (slug, StrategyDoc-shaped dict) for one scraped record."""
    title = (raw.get("title") or "").strip()
    organism = (raw.get("organism_name") or "").strip()
    intro = _strip_boilerplate(raw.get("introduction", ""))
    strategy = _strip_boilerplate(raw.get("strategy", ""))
    potential = _strip_boilerplate(raw.get("potential", ""))
    url = raw.get("source_url", "")
    slug = _slug_from_url(url, url_fallback_slug)

    mechanism = strategy
    if potential:
        mechanism = f"{mechanism}\n\n{potential}".strip()

    summary = _first_sentences(intro) or _first_sentences(strategy)

    raw_functions = raw.get("functions_performed", [])
    # Re-write the raw AskNature labels into canonical taxonomy keys for exact filtering.
    function_keys, subgroup_keys = keys_for_labels(raw_functions)

    # Emit only what the source actually provides. Metadata the scraper does not
    # carry (organism_scientific, environment, taxon, scale, source_tier) is left
    # to StrategyDoc's empty defaults — never fabricated. Populating it is a
    # separate, user-owned enrichment step.
    rec = {
        "doc_id": f"asn-{slug}",
        "organism_common": organism,
        "strategy_summary": summary,
        "mechanism": mechanism,
        "function_addressed": [f.strip().lower() for f in raw_functions if f.strip()],
        "function_keys": function_keys,
        "subgroup_keys": subgroup_keys,
        "source_url": url,
        "keywords": _keywords(title),
        "provenance": "fetched",
    }
    return slug, rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the AskNature StrategyDoc corpus.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="scraper output dir")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="corpus subdir to write")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N (0 = all)")
    args = ap.parse_args(argv)

    src: Path = args.src
    out: Path = args.out
    if not src.is_dir():
        ap.error(f"source dir not found: {src}")

    files = sorted(src.glob("*.json"))
    if args.limit:
        files = files[: args.limit]

    # Idempotent: recreate the output dir from scratch each run.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    written = skipped = 0
    seen: dict[str, int] = {}
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            slug, rec = convert_one(raw, path.stem)
            # Guarantee globally-unique doc_id (and output filename).
            base = rec["doc_id"]
            if base in seen:
                seen[base] += 1
                rec["doc_id"] = f"{base}-{seen[base]}"
                slug = f"{slug}-{seen[base]}"
            else:
                seen[base] = 0
            StrategyDoc(**rec)  # validate against the canonical schema before writing
            (out / f"{slug}.json").write_text(
                json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written += 1
        except Exception as exc:  # noqa: BLE001 - report and keep going
            skipped += 1
            print(f"SKIP {path.name}: {exc}")

    print(f"\nConverted {written} strategies into {out} (skipped {skipped}).")

    # Confirm the whole corpus (existing + new) still loads cleanly.
    from .corpus import load_corpus
    total = len(load_corpus())
    print(f"Total corpus size now: {total} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
