"""Canonical function keys — the single source of truth for turning a biological function
into a stable, exact-match filter token.

Both sides of the function filter go through here so they can never drift (the same
discipline the e5 embedder applies to passage/query prefixes):

* ingest side  — ``keys_for_label("Chemically Assemble Molecular Devices")`` resolves a raw
  AskNature label via ``function_crosswalk.json`` and returns its canonical keys.
* query side   — ``keys_for_triple(group, sub_group, function)`` takes a biologize HDN's
  verbatim taxonomy path and returns the SAME keys.

Key forms (taxonomy sub-group names are unique, so the sub-group slug alone is collision-free):
    leaf_key("Chemically assemble", "molecular devices") -> "chemically-assemble::molecular-devices"
    subgroup_key("Chemically assemble")                  -> "chemically-assemble"

Stored on each Weaviate object as ``function_keys`` (leaf) and ``subgroup_keys`` (sub-group),
both with FIELD tokenization so ``Filter.by_property(...).contains_any([...])`` is exact.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_CROSSWALK_PATH = Path(__file__).resolve().parent / "function_crosswalk.json"
_WORD = re.compile(r"[a-z0-9]+")


def normalize_label(text: str) -> str:
    """Lowercase, punctuation-collapsed form used as the crosswalk lookup key."""
    return " ".join(_WORD.findall((text or "").lower()))


def slug(text: str) -> str:
    """Stable token slug: lowercase, non-alphanumeric runs -> single hyphen."""
    return "-".join(_WORD.findall((text or "").lower()))


def leaf_key(sub_group: str, function: str) -> str:
    return f"{slug(sub_group)}::{slug(function)}"


def subgroup_key(sub_group: str) -> str:
    return slug(sub_group)


@lru_cache(maxsize=1)
def _crosswalk() -> dict[str, dict]:
    """Map normalized label -> crosswalk row. Empty (not an error) if the artifact is absent
    so retrieval still works unfiltered before the crosswalk is built."""
    if not _CROSSWALK_PATH.exists():
        return {}
    data = json.loads(_CROSSWALK_PATH.read_text(encoding="utf-8"))
    return {normalize_label(r["label"]): r for r in data.get("rows", [])}


def keys_for_label(label: str) -> tuple[str | None, str | None]:
    """(leaf_key, subgroup_key) for a raw AskNature function label, via the crosswalk.

    Returns (None, None) for an unmapped label and (None, subgroup_key) for a label resolved
    only to a sub-group (no defensible leaf) — such docs stay vector-searchable, just not
    leaf-filterable."""
    row = _crosswalk().get(normalize_label(label))
    if not row or not row.get("sub_group"):
        return None, None
    sg = row["sub_group"]
    fn = row.get("function")
    return (leaf_key(sg, fn) if fn else None), subgroup_key(sg)


def keys_for_triple(group: str, sub_group: str, function: str) -> tuple[str | None, str | None]:
    """(leaf_key, subgroup_key) for a biologize taxonomy path. ``group`` is accepted for a
    symmetric call site but is not part of the key (sub-group names are globally unique)."""
    if not sub_group:
        return None, None
    return (leaf_key(sub_group, function) if function else None), subgroup_key(sub_group)


def keys_for_labels(labels: list[str]) -> tuple[list[str], list[str]]:
    """Resolve a strategy's raw function labels into sorted-unique (leaf_keys, subgroup_keys).

    Shared by the local corpus build (``convert_one``) and the Weaviate ingest so both
    representations carry identical canonical keys."""
    leaves, subs = set(), set()
    for label in labels or []:
        lk, sk = keys_for_label(label)
        if lk:
            leaves.add(lk)
        if sk:
            subs.add(sk)
    return sorted(leaves), sorted(subs)
