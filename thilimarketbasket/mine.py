"""The exact answer: frequent itemsets by counting. (Lesson 2.)

Apriori, written level by level. An itemset can only be frequent if every part of it is frequent, so we
count the single items first, build pairs only from frequent items, triples only from frequent pairs, and so on.

To count fast, every item gets a *bitset*: one bit per basket, set when the basket holds the item. The number
of baskets holding both of two items is then the number of set bits in `bits_a & bits_b`, one cheap operation.
"""
import math
from itertools import combinations
from typing import Dict, List

import numpy as np

_count_bits = int.bit_count if hasattr(int, "bit_count") else (lambda value: bin(value).count("1"))


def _bitsets(baskets) -> Dict[str, int]:
    """For each item, a number whose bit i is 1 when basket i holds the item (the quantity does not matter)."""
    numbers_of = {}
    for number, basket in enumerate(baskets):
        for name in set(basket):
            numbers_of.setdefault(name, []).append(number)
    bits = {}
    for name, numbers in numbers_of.items():
        held = np.zeros(len(baskets), dtype=bool)
        held[numbers] = True
        bits[name] = int.from_bytes(np.packbits(held).tobytes(), "big")
    return bits


def frequent_itemsets(baskets, min_support: float, max_size: int = 5) -> Dict[frozenset, float]:
    """Map each itemset (a frozenset of item names) to its support, for every itemset of at most `max_size`
    items that is held by at least `min_support` of the baskets. Quantities are ignored: two packs of milk
    count as milk."""
    n = len(baskets)
    min_count = max(1, math.ceil(min_support * n - 1e-9))  # the small allowance guards against 0.0005 * 100000
    bits = _bitsets(baskets)

    level = {(name,): value for name, value in bits.items() if _count_bits(value) >= min_count}
    found = {frozenset(items): _count_bits(value) / n for items, value in level.items()}
    size = 1
    while level and size < max_size:
        by_prefix = {}  # itemsets that agree on every item but the last can be joined into a bigger one
        for items in sorted(level):
            by_prefix.setdefault(items[:-1], []).append(items[-1])
        next_level = {}
        for prefix, lasts in by_prefix.items():
            for a, b in combinations(lasts, 2):  # `lasts` is sorted, so a < b
                candidate = prefix + (a, b)
                # every part of a frequent itemset must itself be frequent
                if any(candidate[:i] + candidate[i + 1:] not in level for i in range(len(candidate) - 2)):
                    continue
                held = level[prefix + (a,)] & bits[b]
                if _count_bits(held) >= min_count:
                    next_level[candidate] = held
        level = next_level
        found.update({frozenset(items): _count_bits(value) / n for items, value in level.items()})
        size += 1
    return found


def rules(itemsets: Dict[frozenset, float], min_confidence: float = 0.3) -> List[tuple]:
    """Association rules from the itemsets: (antecedent, consequent, support, confidence, lift).

    "People who buy the antecedent also buy the consequent". Confidence is the share of baskets holding the
    antecedent that also hold the consequent; lift says how many times more often they go together than they
    would if they were unrelated (1 means unrelated).
    """
    found = []
    for items, support in itemsets.items():
        if len(items) < 2:
            continue
        for k in range(1, len(items)):
            for antecedent in map(frozenset, combinations(sorted(items), k)):
                consequent = items - antecedent
                confidence = support / itemsets[antecedent]
                if confidence >= min_confidence:
                    found.append((antecedent, consequent, support, confidence, confidence / itemsets[consequent]))
    return sorted(found, key=lambda rule: (-rule[4], -rule[2]))
