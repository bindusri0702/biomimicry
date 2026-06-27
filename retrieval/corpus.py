"""Corpus loader and document schema for offline retrieval.

`StrategyDoc` is the canonical record for one biological strategy. `load_corpus`
globs the `corpus/` tree, validates every document, and returns plain dicts (the
validate-then-store-as-dict convention used across the package). Designed so a
large, real, downloaded-and-parsed AskNature corpus drops in with no code change.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

_CORPUS_DIR = Path(__file__).parent / "corpus"


class StrategyDoc(BaseModel):
    """Canonical corpus record.

    Metadata fields (organism_scientific, environment, taxon, scale, source_tier)
    are OPTIONAL and default to empty — nothing is fabricated. They carry signal
    only when a real source or an enrichment step populates them; the enum
    validators were removed so any value (or none) is accepted.
    """
    doc_id: str
    organism_common: str
    organism_scientific: str = ""
    strategy_summary: str = ""
    mechanism: str = ""
    function_addressed: list[str] = Field(default_factory=list)
    # Canonical taxonomy keys resolved from function_addressed (see retrieval/function_keys.py),
    # used for exact metadata filtering. Empty until a crosswalk-aware build populates them.
    function_keys: list[str] = Field(default_factory=list)        # "sub-group::function" leaf keys
    subgroup_keys: list[str] = Field(default_factory=list)        # "sub-group" keys (broader filter)
    environment: str = ""                 # e.g. "extreme hot desert surface"
    taxon: str = ""                       # e.g. "Animalia/Arthropoda/Insecta"
    scale: str = ""                       # molecular | cellular | organismal | ecosystem (free-form)
    source_url: str = ""
    source_tier: str = ""                 # peer_reviewed | science_journalism | grey_literature (free-form)
    keywords: list[str] = Field(default_factory=list)
    provenance: str = "synthetic"         # synthetic | fetched

    def index_text(self) -> str:
        """Concatenated text the lexical index scores against."""
        return " ".join([
            self.organism_common, self.organism_scientific, self.strategy_summary,
            self.mechanism, " ".join(self.function_addressed),
            " ".join(self.keywords), self.environment,
        ])


def load_corpus(corpus_dir: Path | None = None) -> list[dict]:
    """Load and validate every *.json strategy doc; raise loudly on bad data.

    Returns dicts sorted by doc_id for deterministic downstream ordering.
    """
    root = corpus_dir or _CORPUS_DIR
    docs: list[dict] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            doc = StrategyDoc(**raw)
        except Exception as exc:  # noqa: BLE001 - surface the offending file
            raise ValueError(f"Invalid corpus document {path}: {exc}") from exc
        if doc.doc_id in seen:
            raise ValueError(f"Duplicate doc_id {doc.doc_id!r} at {path}")
        seen.add(doc.doc_id)
        docs.append(doc.model_dump())
    return docs
