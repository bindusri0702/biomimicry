"""One-off: strip AskNature boilerplate / leaked chrome from built corpus JSON.

The scraped strategy pages leaked text that is metadata, not strategy content, into
the corpus and pollutes retrieval:

  * contributor attribution -- "This summary was contributed by <name>."
  * "Last Updated <Month DD, YYYY>"
  * image-credit chrome      -- "Image: <name> / toggle icon", "right arrow right arrow"
  * parenthetical media refs -- "(see diagram)", "(see video here)"

This walks the corpus tree like ``load_corpus`` (``rglob("*.json")``), applies
``build_asknature_corpus.strip_boilerplate`` (the single source of truth, also used
by the corpus build) to the free-text fields ``strategy_summary`` and ``mechanism``,
and rewrites only files that actually change -- preserving the build's output format
(``ensure_ascii=False``, ``indent=2``). Idempotent: a second run rewrites nothing.

    python -m biomimicry.retrieval.remove_boilerplate
    python -m biomimicry.retrieval.remove_boilerplate --dry-run
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_asknature_corpus import strip_boilerplate
from .corpus import load_corpus

_CORPUS_DIR = Path(__file__).parent / "corpus"
_TEXT_FIELDS = ("strategy_summary", "mechanism")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strip boilerplate from built corpus JSON.")
    ap.add_argument("--corpus", type=Path, default=_CORPUS_DIR, help="corpus dir to scan")
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    args = ap.parse_args(argv)

    root: Path = args.corpus
    if not root.is_dir():
        ap.error(f"corpus dir not found: {root}")

    changed = 0
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        cleaned = dict(data)
        for f in _TEXT_FIELDS:
            if isinstance(cleaned.get(f), str):
                cleaned[f] = strip_boilerplate(cleaned[f])
        # Rewrite only when a field actually changed -- never just reformat.
        if cleaned == data:
            continue
        changed += 1
        print(f"{'WOULD CLEAN' if args.dry_run else 'CLEANED'} {path.relative_to(root)}")
        if not args.dry_run:
            path.write_text(
                json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    verb = "would be cleaned" if args.dry_run else "cleaned"
    print(f"\n{changed} file(s) {verb}.")

    if not args.dry_run:
        total = len(load_corpus(root))
        print(f"Corpus loads cleanly: {total} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
