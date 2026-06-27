"""Ingest the original scraped AskNature strategies into Weaviate Cloud.

Reads ``asknature_scraper/strategies/*.json`` (the original records, unchanged), embeds a
per-strategy passage with local e5-large-v2, and stores each as one object with a
bring-your-own vector (vectorizer=none, cosine). The Discover stage then retrieves over it
via ``RETRIEVAL_BACKEND=weaviate``.

    python -m biomimicry.retrieval.build_weaviate --recreate
    python -m biomimicry.retrieval.build_weaviate --limit 50 --recreate

Needs WEAVIATE_URL + WEAVIATE_API_KEY in biomimicry/.env. The deterministic per-slug UUID
makes re-ingest with --recreate idempotent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from .e5_embedder import build_passage, embed_documents
from .function_keys import keys_for_labels
from .weaviate_store import connect

_HERE = Path(__file__).resolve().parent
DEFAULT_SRC = _HERE.parents[1] / "asknature_scraper" / "strategies"


def _slug(record: dict, fallback: str) -> str:
    url = record.get("source_url", "")
    slug = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    return slug or fallback


def _create_collection(client, recreate: bool):
    from weaviate.classes.config import Configure, DataType, Property, Tokenization

    name = config.WEAVIATE_COLLECTION
    exists = client.collections.exists(name)
    if exists and recreate:
        client.collections.delete(name)
        exists = False
    if not exists:
        # Bring-your-own vectors (vectorizer=none). Don't force an index type: some
        # Weaviate Cloud tiers only permit the server-managed 'hfresh' index, so we let
        # the server apply its default (with default cosine distance, correct for the
        # L2-normalized e5 vectors we supply).
        client.collections.create(
            name=name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="organism_name", data_type=DataType.TEXT),
                Property(name="functions_performed", data_type=DataType.TEXT_ARRAY),
                # Canonical taxonomy keys for exact metadata filtering. FIELD tokenization
                # keeps each array element a single token so `contains_any` matches exactly
                # (default `word` tokenization would split "sub-group::function" apart).
                Property(name="function_keys", data_type=DataType.TEXT_ARRAY,
                         tokenization=Tokenization.FIELD),
                Property(name="subgroup_keys", data_type=DataType.TEXT_ARRAY,
                         tokenization=Tokenization.FIELD),
                Property(name="introduction", data_type=DataType.TEXT),
                Property(name="strategy", data_type=DataType.TEXT),
                Property(name="potential", data_type=DataType.TEXT),
                Property(name="related_innovation", data_type=DataType.TEXT_ARRAY),
                Property(name="source_url", data_type=DataType.TEXT),
                Property(name="slug", data_type=DataType.TEXT),
            ],
        )
    return client.collections.get(name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest scraped strategies into Weaviate.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="scraper output dir")
    ap.add_argument("--limit", type=int, default=0, help="ingest only the first N (0 = all)")
    ap.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    ap.add_argument("--batch", type=int, default=64, help="docs per embed/insert batch")
    args = ap.parse_args(argv)

    if not args.src.is_dir():
        ap.error(f"source dir not found: {args.src}")
    files = sorted(args.src.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    print(f"Ingesting {len(files)} strategies into Weaviate collection "
          f"'{config.WEAVIATE_COLLECTION}' ...", flush=True)

    from weaviate.util import generate_uuid5

    client = connect()
    try:
        collection = _create_collection(client, args.recreate)
        inserted = failed = 0
        with collection.batch.dynamic() as batch:
            for start in range(0, len(files), args.batch):
                chunk = files[start: start + args.batch]
                records, slugs, passages = [], [], []
                for path in chunk:
                    rec = json.loads(path.read_text(encoding="utf-8"))
                    slug = _slug(rec, path.stem)
                    records.append(rec)
                    slugs.append(slug)
                    passages.append(build_passage(rec))
                vectors = embed_documents(passages)
                for rec, slug, vec in zip(records, slugs, vectors):
                    raw_functions = rec.get("functions_performed") or []
                    function_keys, subgroup_keys = keys_for_labels(raw_functions)
                    props = {
                        "title": rec.get("title", ""),
                        "organism_name": rec.get("organism_name", ""),
                        "functions_performed": raw_functions,
                        "function_keys": function_keys,
                        "subgroup_keys": subgroup_keys,
                        "introduction": rec.get("introduction", ""),
                        "strategy": rec.get("strategy", ""),
                        "potential": rec.get("potential", ""),
                        "related_innovation": rec.get("related_innovation") or [],
                        "source_url": rec.get("source_url", ""),
                        "slug": slug,
                    }
                    batch.add_object(properties=props, vector=vec,
                                     uuid=generate_uuid5(slug))
                    inserted += 1
                print(f"  embedded+queued {min(start + args.batch, len(files))}/{len(files)}",
                      flush=True)

        failed = len(collection.batch.failed_objects)
        if failed:
            print(f"  WARNING: {failed} objects failed to insert. First error: "
                  f"{collection.batch.failed_objects[0].message}")
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"\nDone. queued={inserted} failed={failed} | collection now holds {total} objects.")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
