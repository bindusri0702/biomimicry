"""Bounded-concurrency helper for stage fan-out.

`bounded_map` runs a function over many items with at most `config.MAX_CONCURRENCY`
calls in flight, so parallel fan-out (Biologize / Discover) stays within the provider's
rate limit. With a cap of 1 (the default) it runs fully sequentially — identical to the
pre-parallel behavior — and never spawns a thread.

Threads (not asyncio) because the stage nodes make synchronous `litellm.completion` calls:
they are network-I/O-bound, so the GIL is released during the request and the waits overlap.
Results are returned in input order (`ThreadPoolExecutor.map`), so callers can `zip` them
back to their inputs deterministically.

Concurrency safety is the caller's responsibility: pass a `fn` that only makes the LLM call
and returns its result; do all shared-state mutation (id assignment, list appends, dedup)
after `bounded_map` returns, on the calling thread. See the Biologize/Discover nodes.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

from . import config

T = TypeVar("T")
R = TypeVar("R")


def bounded_map(fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
    """Apply `fn` to each item, at most `config.MAX_CONCURRENCY` at a time, in input order.

    Falls back to a plain sequential comprehension (no threads) when the cap is 1 or there
    is at most one item, so the default configuration behaves exactly as the old loops did.
    """
    items = list(items)
    cap = max(1, config.MAX_CONCURRENCY)
    if cap == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=cap) as ex:
        return list(ex.map(fn, items))
