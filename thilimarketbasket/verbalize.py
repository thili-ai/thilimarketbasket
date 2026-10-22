"""Turn a basket into text a language model can read. (Lesson 3.)

A basket is a *set*: nothing in it comes first. A language model reads text left to right, so we have to pick an
order, and any fixed order (alphabetical, the order the till scanned) is something the model can learn and that
means nothing. So the order is shuffled, afresh every time a basket is shown.

Quantities are dropped: two packs of milk read as milk, because the exact miner ignores quantity too.
"""
from typing import Iterable, List, Optional

import numpy as np

PREFIX = "Basket: "
SEPARATOR = "; "


def verbalize_basket(items: Iterable[str], seed: Optional[int] = None, max_items: int = 30) -> str:
    """One basket as a single string, e.g. "Basket: Onion 1 kg; Gokul Fresh Milk 500 ml; Potato 1 kg".

    Each different item appears once, in a random order (`seed` makes it repeatable; None gives a fresh
    order each call). A basket with more than `max_items` different items is cut to a random `max_items`.
    """
    different = sorted(set(items))  # sorted first so that the same seed always gives the same order
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(different))[:max_items]
    return PREFIX + SEPARATOR.join(different[i] for i in order)


def parse_basket(text: str) -> List[str]:
    """The items in a verbalized basket, in the order they were written. The reverse of `verbalize_basket`."""
    if not text.startswith(PREFIX):
        raise ValueError(f"not a verbalized basket: {text[:40]!r}")
    body = text[len(PREFIX):]
    return body.split(SEPARATOR) if body else []


def verbalize_all(baskets, epochs: int = 1, seed: int = 0, max_items: int = 30) -> List[str]:
    """Every basket as text, `epochs` times over, each time in a fresh random order."""
    rng = np.random.default_rng(seed)
    return [verbalize_basket(basket, seed=int(rng.integers(2**31)), max_items=max_items)
            for _ in range(epochs) for basket in baskets]


def verbalize_prefix(items: Iterable[str], seed: Optional[int] = None, max_items: int = 30) -> str:
    """A basket written up to the point where the next item goes, e.g. "Basket: Onion 1 kg; Potato 1 kg;".

    Add a space and an item name to get a whole basket text. With no items it is just "Basket:".
    """
    text = verbalize_basket(items, seed=seed, max_items=max_items)
    return PREFIX.rstrip() if text == PREFIX else text + ";"
