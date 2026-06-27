"""Biomimicry Functional Taxonomy — loader, prompt renderer, and path validator.

The taxonomy is the canonical Group -> Sub-group -> Function hierarchy used by the
Biologize step to map a challenge onto biological functions. It is read verbatim from
``taxonomy_hierarchy.json`` (path in :data:`config.TAXONOMY_PATH`); nothing here invents
or renames a label.

Empty-functions rule: a few sub-groups (Compute, Learn, [En/de]code) carry no leaf
functions in the source file. We never drop a taxonomy term — instead we synthesise a
single leaf function equal to the sub-group name, so the valid path becomes e.g.
``Process information -> Compute -> Compute``. Such terms are rendered into the prompt
and accepted by the validator like any other function.
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import config


@lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    """Parse and cache ``taxonomy_hierarchy.json``."""
    with open(config.TAXONOMY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _functions_for(sub_group: dict) -> list[str]:
    """Leaf functions of a sub-group, applying the empty-functions rule.

    When the source lists no functions, the sub-group name itself is the only leaf."""
    fns = sub_group.get("functions") or []
    return list(fns) if fns else [sub_group["sub_group"]]


@lru_cache(maxsize=1)
def valid_paths() -> frozenset[tuple[str, str, str]]:
    """All valid (group, sub_group, function) triples in the taxonomy."""
    paths: set[tuple[str, str, str]] = set()
    for grp in load_taxonomy().get("groups", []):
        group = grp["group"]
        for sg in grp.get("sub_groups", []):
            sub_group = sg["sub_group"]
            for fn in _functions_for(sg):
                paths.add((group, sub_group, fn))
    return frozenset(paths)


def is_valid_path(group: str, sub_group: str, function: str) -> bool:
    """True iff ``function`` is nested under ``sub_group`` under ``group``."""
    return (group, sub_group, function) in valid_paths()


@lru_cache(maxsize=1)
def render_for_prompt() -> str:
    """Readable indented listing of the full taxonomy for the ``{{taxonomy}}`` slot.

    Labels are reproduced verbatim so the model can copy them into a valid path."""
    lines: list[str] = []
    for grp in load_taxonomy().get("groups", []):
        lines.append(f"- {grp['group']}")
        for sg in grp.get("sub_groups", []):
            functions = ", ".join(_functions_for(sg))
            lines.append(f"    - {sg['sub_group']}: {functions}")
    return "\n".join(lines)
