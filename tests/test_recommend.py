import numpy as np
import pytest

from thilimarketbasket.generate import Item
from thilimarketbasket.recommend import ItemToItem, lift_adjusted, rank_among, repeat_share, top_k, variants_of

NAMES = ["milk", "bread", "eggs", "tea"]
BASKETS = [["milk", "bread"]] * 6 + [["milk", "eggs"]] * 2 + [["tea"]] * 2


def test_lift_adjusted_subtracts_the_blank_score():
    assert list(lift_adjusted(np.array([-1.0, -2.0]), np.array([-3.0, -1.0]))) == [2.0, -1.0]
    assert list(lift_adjusted(np.array([-1.0, -2.0]), np.array([-3.0, -1.0]), alpha=0)) == [-1.0, -2.0]


def test_top_k_skips_excluded_and_orders_best_first():
    assert top_k(np.array([0.1, 0.9, 0.5, 0.7]), 2, exclude=[1]) == [3, 2]


def test_item_to_item_confidence_and_lift():
    rec = ItemToItem(BASKETS, NAMES)
    confidence = rec.score(["milk"])
    assert confidence[1] == pytest.approx(6 / 8) and confidence[2] == pytest.approx(2 / 8) and confidence[3] == 0
    lifted = rec.score(["milk"], lift=True)
    assert lifted[1] > lifted[2]                                    # bread is far above chance with milk
    assert rec.score(["milk", "tea"])[1] == pytest.approx((6 / 8 + 0) / 2)


def test_rank_among_a_group_ignores_everything_else():
    scores = np.array([9.0, 1.0, 3.0, 2.0])
    assert rank_among(scores, 3, [1, 2, 3]) == 2
    assert rank_among(scores, 3, [0, 1, 2, 3]) == 3


def test_variants_group_by_product():
    catalog = [Item("Toor Dal 1 kg", "Toor Dal", "Pulses", "A", "1 kg"), Item("Rice 1 kg", "Rice", "Grains", "A", "1 kg"),
               Item("Toor Dal 5 kg", "Toor Dal", "Pulses", "A", "5 kg")]
    assert variants_of(catalog) == {"Toor Dal": [0, 2], "Rice": [1]}


def test_repeat_share_counts_other_skus_of_a_product_in_the_basket():
    catalog = [Item("Toor Dal 1 kg", "Toor Dal", "Pulses", "A", "1 kg"), Item("Rice 1 kg", "Rice", "Grains", "A", "1 kg"),
               Item("Toor Dal 5 kg", "Toor Dal", "Pulses", "A", "5 kg")]
    assert repeat_share([2, 1], [0], catalog) == 0.5
    assert repeat_share([], [0], catalog) == 0.0
