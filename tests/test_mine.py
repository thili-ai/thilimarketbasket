from itertools import combinations

import numpy as np
import pytest

from thilimarketbasket.mine import frequent_itemsets, rules

SMALL = [["a", "b", "c"], ["a", "b"], ["a", "c"], ["b", "c"], ["a", "b", "c", "d"]]


def test_supports_are_counted_exactly():
    found = frequent_itemsets(SMALL, 0.4)
    assert found[frozenset("a")] == 0.8
    assert found[frozenset("ab")] == 0.6
    assert found[frozenset("abc")] == 0.4
    assert frozenset("d") not in found  # held by one basket in five: 0.2 is below 0.4


def test_the_threshold_is_inclusive():
    assert frozenset("abc") in frequent_itemsets(SMALL, 0.4)  # exactly 2 of 5
    assert frozenset("abc") not in frequent_itemsets(SMALL, 0.41)


def test_quantities_do_not_matter():
    assert frequent_itemsets([["a", "a", "b"], ["a", "b"]], 1.0)[frozenset("ab")] == 1.0


def test_max_size_stops_the_search():
    found = frequent_itemsets(SMALL, 0.4, max_size=2)
    assert max(len(items) for items in found) == 2
    assert max(len(items) for items in frequent_itemsets(SMALL, 0.4, max_size=3)) == 3


def test_matches_brute_force_counting_on_random_baskets():
    rng = np.random.default_rng(0)
    items = list("abcdefghijkl")
    baskets = [list(rng.choice(items, size=rng.integers(2, 7), replace=False)) for _ in range(400)]
    found = frequent_itemsets(baskets, 0.05, max_size=4)
    expected = {}
    for size in range(1, 5):
        for combo in combinations(items, size):
            support = sum(set(combo) <= set(b) for b in baskets) / len(baskets)
            if support >= 0.05:
                expected[frozenset(combo)] = support
    assert found.keys() == expected.keys()
    assert all(found[k] == pytest.approx(expected[k]) for k in found)


def test_every_part_of_a_frequent_itemset_is_frequent_and_at_least_as_common():
    found = frequent_itemsets(SMALL, 0.4)
    for items, support in found.items():
        for k in range(1, len(items)):
            for part in map(frozenset, combinations(items, k)):
                assert found[part] >= support


def test_a_float_threshold_does_not_lose_an_itemset_sitting_exactly_on_it():
    baskets = [["x", "y"]] * 50 + [["z"]] * 99_950  # x and y together in exactly 0.05% of 100,000 baskets
    assert frozenset(["x", "y"]) in frequent_itemsets(baskets, 0.0005)


def test_rules_have_the_right_confidence_and_lift():
    found = rules(frequent_itemsets(SMALL, 0.4), min_confidence=0.0)
    rule = next(r for r in found if r[0] == frozenset("a") and r[1] == frozenset("b"))
    assert rule[2] == pytest.approx(0.6)  # support of {a, b}
    assert rule[3] == pytest.approx(0.6 / 0.8)  # confidence: of the baskets with a, how many also hold b
    assert rule[4] == pytest.approx((0.6 / 0.8) / 0.8)  # lift: that, over how common b is on its own
    assert all(r[3] >= 0.0 for r in found)
    assert rules(frequent_itemsets(SMALL, 0.4), min_confidence=0.9) == [r for r in found if r[3] >= 0.9]
