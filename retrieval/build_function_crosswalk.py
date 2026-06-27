"""Generate ``function_crosswalk.json`` — the one-time resolution of raw AskNature
function labels into canonical Biomimicry-taxonomy ``(group, sub_group, function)`` triples.

AskNature's ``functions_performed`` labels are an irregular blend of ``sub_group + function``,
``function`` alone, or reworded leaves (e.g. ``"Modify Size/Shape/Mass/Volume"`` collapses the
sub-group ``"Modify physical state"`` to the verb ``"Modify"``), so they do not byte-match the
taxonomy. This script resolves each distinct label ONCE: a light-stemmed token matcher scores
every label against every taxonomy triple (see ``taxonomy.valid_paths``), an OVERRIDES table
pins the handful the matcher gets wrong, and the result is written as a reviewed artifact.

``function_keys.keys_for_label`` reads that artifact at corpus-build / ingest time so each
strategy's functions are stored as canonical keys; the query side derives the SAME keys from a
biologize HDN's taxonomy path. Re-runnable and deterministic — re-run after a corpus re-scrape
and diff the coverage report.

    python -m biomimicry.retrieval.build_function_crosswalk
    python -m biomimicry.retrieval.build_function_crosswalk --report   # print, don't write
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import taxonomy
from .function_keys import normalize_label as _norm

_HERE = Path(__file__).resolve().parent
DEFAULT_SRC = _HERE.parents[1] / "asknature_scraper" / "strategies"
OUT_PATH = _HERE / "function_crosswalk.json"

# Confidence at/above which the matcher's guess is auto-accepted without review.
AUTO_ACCEPT = 0.7

# Labels the token matcher resolves wrong (or to the wrong sub-group). Keyed by the
# normalized label -> (group, sub_group, function). Every value is a valid taxonomy triple
# (asserted at build time). This is the "reviewed" residue from the coverage report.
OVERRIDES: dict[str, tuple[str, str, str]] = {
    "catalyze chemical assembly": ("Make / modify", "Chemically assemble", "catalyze making of bonds"),
    "catalyze chemical breakdown": ("Break down", "Chemically break down", "catalyze breaking of bonds"),
    "cooperate within an ecosystem": ("Sustain ecological community", "Cooperate/compete", "within a (eco)system"),
    "cooperate between ecosystems": ("Sustain ecological community", "Cooperate/compete", "between (eco)systems"),
    "cooperate within the same species": ("Sustain ecological community", "Cooperate/compete", "within the same species"),
    "cooperate/compete between different species": ("Sustain ecological community", "Cooperate/compete", "between different species"),
    "compete between different species": ("Sustain ecological community", "Cooperate/compete", "between different species"),
    "encode/decode": ("Process information", "[En/de]code", "[En/de]code"),
    "compute": ("Process information", "Compute", "Compute"),
    "learn": ("Process information", "Learn", "Learn"),
    "coordinate by self-organization": ("Sustain ecological community", "Coordinate", "by self-organization"),
    "coordinate activities": ("Sustain ecological community", "Coordinate", "activities"),
    "coordinate systems": ("Sustain ecological community", "Coordinate", "systems"),
}

# AskNature labels with no defensible taxonomy home (the local taxonomy has the leaf under a
# DIFFERENT sub-group). Kept sub-group-filterable where reasonable, else left unkeyed.
# normalized label -> (group, sub_group) | None ; function is left null.
SUBGROUP_ONLY: dict[str, tuple[str, str] | None] = {
    # "organisms"/"chemical entities" are not leaves of "Capture, absorb, or filter" locally,
    # but the sub-group is unambiguous, so keep them filterable at sub-group level.
    "capture, absorb, or filter organisms": ("Get, store, or distribute resources", "Capture, absorb, or filter"),
    "capture, absorb, or filter chemical entities": ("Get, store, or distribute resources", "Capture, absorb, or filter"),
}

def _stem(tok: str) -> str:
    """Crude suffix stripper so 'chemically'/'chemical' and 'assembly'/'assemble' align."""
    for suf in ("ically", "ing", "ed", "ly", "es", "s", "y", "e"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def _stems(text: str) -> set[str]:
    return {_stem(t) for t in _norm(text).split()}


def _candidate_token_sets(sub_group: str, function: str) -> list[set[str]]:
    """Stemmed token sets for the label forms AskNature actually uses for a triple."""
    first = sub_group.split()[0] if sub_group.split() else sub_group
    return [
        _stems(f"{sub_group} {function}"),   # "Store" + "liquids"
        _stems(f"{first} {function}"),        # "Modify" (from "Modify physical state") + "size/..."
        _stems(function),                     # "maintain homeostasis" alone
    ]


def _best_match(label: str, triples: list[tuple[str, str, str]]) -> tuple[float, tuple[str, str, str] | None]:
    lt = _stems(label)
    if not lt:
        return 0.0, None
    best_j, best = 0.0, None
    for g, sg, fn in triples:
        for cand in _candidate_token_sets(sg, fn):
            if not cand:
                continue
            j = len(lt & cand) / len(lt | cand)
            if j > best_j:
                best_j, best = j, (g, sg, fn)
    return round(best_j, 3), best


def _distinct_labels(src: Path) -> list[str]:
    labels: set[str] = set()
    for path in sorted(src.glob("*.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for f in rec.get("functions_performed") or []:
            if f and f.strip():
                labels.add(f.strip())
    return sorted(labels)


def build(src: Path) -> dict:
    valid = taxonomy.valid_paths()
    triples = sorted(valid)
    # Curated tables are keyed by readable labels; match on the normalized form actually seen.
    overrides = {_norm(k): v for k, v in OVERRIDES.items()}
    subgroup_only = {_norm(k): v for k, v in SUBGROUP_ONLY.items()}
    # Sanity-check the hand-curated tables against the live taxonomy.
    for nlabel, triple in overrides.items():
        if triple not in valid:
            raise ValueError(f"OVERRIDES[{nlabel!r}] = {triple} is not a valid taxonomy path")

    rows = []
    for label in _distinct_labels(src):
        nlabel = _norm(label)
        if nlabel in overrides:
            g, sg, fn = overrides[nlabel]
            rows.append({"label": label, "group": g, "sub_group": sg,
                         "function": fn, "confidence": 1.0, "method": "override"})
            continue
        if nlabel in subgroup_only:
            home = subgroup_only[nlabel]
            if home is None:
                rows.append({"label": label, "group": None, "sub_group": None,
                             "function": None, "confidence": 0.0, "method": "unmapped"})
            else:
                g, sg = home
                rows.append({"label": label, "group": g, "sub_group": sg,
                             "function": None, "confidence": 1.0, "method": "subgroup_only"})
            continue
        conf, best = _best_match(label, triples)
        if best is None:
            rows.append({"label": label, "group": None, "sub_group": None,
                         "function": None, "confidence": 0.0, "method": "unmapped"})
        else:
            g, sg, fn = best
            rows.append({"label": label, "group": g, "sub_group": sg, "function": fn,
                         "confidence": conf,
                         "method": "auto" if conf >= AUTO_ACCEPT else "review"})
    return {"auto_accept": AUTO_ACCEPT, "rows": rows}


def _report(data: dict) -> None:
    rows = data["rows"]
    n = len(rows)
    by = lambda m: sum(1 for r in rows if r["method"] == m)  # noqa: E731
    mapped = sum(1 for r in rows if r["sub_group"])
    print(f"distinct labels: {n}")
    print(f"  mapped (sub-group or better): {mapped} ({100 * mapped // n}%)")
    print(f"  override : {by('override')}")
    print(f"  subgroup_only : {by('subgroup_only')}")
    print(f"  auto (conf>={data['auto_accept']}) : {by('auto')}")
    print(f"  review (conf<{data['auto_accept']}) : {by('review')}")
    print(f"  unmapped : {by('unmapped')}")
    review = [r for r in rows if r["method"] in ("review", "unmapped")]
    if review:
        print("--- needs eyes (label -> guess @conf) ---")
        for r in sorted(review, key=lambda r: r["confidence"]):
            print(f"  {r['confidence']:.2f}  {r['label']!r} -> "
                  f"[{r['sub_group']}] / [{r['function']}]")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the function-label -> taxonomy crosswalk.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="scraper output dir")
    ap.add_argument("--out", type=Path, default=OUT_PATH, help="crosswalk json to write")
    ap.add_argument("--report", action="store_true", help="print coverage only; do not write")
    args = ap.parse_args(argv)

    if not args.src.is_dir():
        ap.error(f"source dir not found: {args.src}")
    data = build(args.src)
    _report(data)
    if not args.report:
        args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote {len(data['rows'])} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
