"""From scores to suggestions: cross-sell lists, a counting baseline, and the choice among variants. (Lesson 11.)

The model gives, for a basket, a score for every catalogue item (`evaluate.score_candidates`). A suggestion is a
choice made from those scores. This module holds the choices that need no model, so they can be tested and
reused: the lift adjustment, the top-k list, an item-to-item counting recommender to compare against, and ranking
among the variants (brands and pack sizes) of one product.
"""
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .evaluate import rank_of


def lift_adjusted(scores: np.ndarray, blank: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """`scores - alpha * blank`, both log-probabilities: what the basket adds to an item's general popularity.

    `blank` is the model's score for each item with no basket shown. With alpha 1 the result is the log of the
    ratio P(item | basket) / P(item), a lift; with alpha 0 it is the raw score; in between is a dial.
    """
    return np.asarray(scores) - alpha * np.asarray(blank)


def top_k(scores: np.ndarray, k: int, exclude: Iterable[int] = ()) -> List[int]:
    """The positions of the `k` highest scores, best first, never an excluded one (the items already in the basket)."""
    scores = np.array(scores, dtype=float)
    scores[list(exclude)] = -np.inf
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


class ItemToItem:
    """Recommend by counting: for each item in the basket, how often did each other item share a basket with it?

    `score(visible)` is the average, over the visible items v, of P(c | v) = (baskets with v and c) / (baskets
    with v). With `lift=True` each is divided by P(c), the item's overall share, so common items do not win by
    being common. It has no training beyond a count and no notion of order, which makes it the fair baseline for
    anything fancier.
    """

    def __init__(self, train_baskets: Sequence[Sequence[str]], names: Sequence[str]):
        self.position = {name: i for i, name in enumerate(names)}
        n = len(names)
        together = np.zeros((n, n), dtype=np.float64)
        for basket in train_baskets:
            held = np.array(sorted({self.position[x] for x in basket}), dtype=int)
            together[np.ix_(held, held)] += 1
        self.alone = np.diag(together).copy()          # baskets holding each item
        self.together = together
        self.n_baskets = len(train_baskets)

    def score(self, visible: Sequence[str], lift: bool = False) -> np.ndarray:
        held = [self.position[x] for x in visible]
        confidence = (self.together[held] / np.maximum(self.alone[held, None], 1)).mean(axis=0)
        return confidence / ((self.alone + 1) / self.n_baskets) if lift else confidence


def variants_of(catalog) -> Dict[str, List[int]]:
    """Catalogue positions of every SKU of each product (same `base`: the brands and pack sizes of "Toor Dal")."""
    groups: Dict[str, List[int]] = {}
    for position, item in enumerate(catalog):
        groups.setdefault(item.base, []).append(position)
    return groups


def rank_among(scores: np.ndarray, target: int, group: Sequence[int]) -> int:
    """The rank of `target` among only the items in `group` (ties count against it, as in `rank_of`)."""
    outside = [i for i in range(len(scores)) if i not in set(group)]
    return rank_of(np.asarray(scores), target, outside)


def repeat_share(suggested: Sequence[int], visible: Sequence[int], catalog) -> float:
    """The share of suggested items that are another SKU of a product already in the basket (a second pack of the
    same Toor Dal). In our data a shopper buys one SKU of a product, so such a suggestion is never taken."""
    bases = {catalog[i].base for i in visible}
    return float(np.mean([catalog[i].base in bases for i in suggested])) if len(suggested) else 0.0
