"""Generic greedy cover over a set of items with changing gains.

The cover picks the item with the largest gain, lets the language module lower
the gain of the items that share a term with the pick, and repeats. The
language module gives the gain, the heap key, the reject test, and the
callbacks. The cover keeps the heap and the reason labels.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable
from typing import Any

#: A heap entry. The first element is the negated gain and the last element is
#: the item index.
HeapKey = tuple[Any, ...]


def greedy_cover(
    order: Iterable[int],
    *,
    gain: Callable[[int], float],
    key: Callable[[int, float], HeapKey],
    reject: Callable[[int], str | None],
    on_pick: Callable[[int, float], None],
    limit: int | None = None,
    include_zero: bool = False,
    precheck: Callable[[int], str | None] | None = None,
    on_measure: Callable[[int, float], None] | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Pick items with the largest gain until the limit or the empty heap.

    :param order: Indices of the items, in the input order.
    :param gain: Item index to the gain of the item at the current state.
    :param key: Item index and gain to a heap entry. The first element is the
        negated gain and the last element is the item index.
    :param reject: Item index to a reason, or None to keep the item.
    :param on_pick: Item index and gain of each kept item.
    :param limit: Stop the loop at this count of kept items. None has no limit.
    :param include_zero: Put an item with a zero gain on the heap. A false
        value keeps the item out of the heap, so the caller must give it a
        reason.
    :param precheck: Item index to a reason that stops the item before its gain
        is measured. None skips the test.
    :param on_measure: Item index and gain of each item at each measurement.
        None skips the callback.
    :returns: The kept indices in pick order, and the reason per index of each
        item that was not kept and was removed from the heap.
    """
    indices = list(order)
    heap: list[HeapKey] = []
    for index in indices:
        value = gain(index)
        if include_zero or value > 0:
            heap.append(key(index, value))
    heapq.heapify(heap)

    kept: list[int] = []
    reasons: dict[int, str] = {}

    while heap and (limit is None or len(kept) < limit):
        stored = heapq.heappop(heap)
        index = stored[-1]
        if precheck is not None:
            reason = precheck(index)
            if reason:
                reasons[index] = reason
                continue
        value = gain(index)
        if on_measure is not None:
            on_measure(index, value)
        if value <= 0:
            reasons[index] = "no_gain"
            continue
        if abs(value + stored[0]) > 1e-12:
            heapq.heappush(heap, key(index, value))
            continue
        reason = reject(index)
        if reason:
            reasons[index] = reason
            continue
        on_pick(index, value)
        kept.append(index)

    return kept, reasons
