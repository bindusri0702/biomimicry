"""One-off: strip invisible web-scraping artifacts from corpus JSON in place.

Soft hyphens (U+00AD) and zero-width spaces (U+200B) scraped from AskNature's HTML
were captured verbatim into the corpus text. They render as nothing (or as ``0xad``
boxes in an editor), break Ctrl+F, and split words for the lexical tokenizer
(``ab­sorb`` -> ``ab`` + ``sorb``), so a keyword search misses the doc.

This walks the corpus tree exactly like ``load_corpus`` (``rglob("*.json")``), applies
``base.clean_text`` to every string value, and rewrites only files that actually
change -- preserving the build script's output format (``ensure_ascii=False``,
``indent=2``) so diffs stay minimal. It is idempotent: a second run rewrites nothing.

The corpus build (``build_asknature_corpus.py``) now cleans text at write time, so a
future rebuild stays clean on its own; this script repairs already-written files,
since the original scraper output is no longer available to rebuild from.

    python -m biomimicry.retrieval.clean_corpus
    python -m biomimicry.retrieval.clean_corpus --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .base import clean_text
from .corpus import load_corpus

_CORPUS_DIR = Path(__file__).parent / "corpus"


def _clean(obj):
    """Recursively apply clean_text to every string in a JSON value."""
    if isinstance(obj, str):
        return clean_text(obj)
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strip invisible artifacts from corpus JSON.")
    ap.add_argument("--corpus", type=Path, default=_CORPUS_DIR, help="corpus dir to scan")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args(argv)

    root: Path = args.corpus
    if not root.is_dir():
        ap.error(f"corpus dir not found: {root}")

    changed = 0
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cleaned_data = _clean(data)
        # Compare parsed objects: rewrite only when an artifact was actually removed,
        # never just because a file's whitespace differs from json.dumps' formatting.
        if cleaned_data == data:
            continue
        changed += 1
        print(f"{'WOULD CLEAN' if args.dry_run else 'CLEANED'} {path.relative_to(root)}")
        if not args.dry_run:
            cleaned = json.dumps(cleaned_data, ensure_ascii=False, indent=2)
            path.write_text(cleaned, encoding="utf-8")

    verb = "would be cleaned" if args.dry_run else "cleaned"
    print(f"\n{changed} file(s) {verb}.")

    if not args.dry_run:
        # Confirm the whole corpus still validates after rewriting.
        total = len(load_corpus(root))
        print(f"Corpus loads cleanly: {total} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
