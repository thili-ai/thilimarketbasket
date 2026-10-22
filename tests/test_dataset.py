import json
from itertools import combinations

import numpy as np
import pytest

from thilimarketbasket.dataset import LABELS, classify_itemset, generate_dataset, load_dataset, save_dataset
from thilimarketbasket.generate import Planted


@pytest.fixture(scope="module")
def data():
    return generate_dataset(n_baskets=40_000, seed=0)


def test_the_dataset_has_the_asked_baskets_and_all_thirty_two_planted_itemsets(data):
    assert len(data.baskets) == 40_000
    assert len(data.planted) == 32
    assert all(frozenset(p.items) in data.truth for p in data.planted)


def test_every_truth_itemset_is_at_least_as_common_as_the_threshold(data):
    assert min(data.truth.values()) >= data.min_support - 1e-9


def test_planted_supports_in_the_truth_are_at_or_near_their_targets(data):
    for p in data.planted:
        assert data.truth[frozenset(p.items)] == pytest.approx(p.support, abs=max(0.03 * p.support, 0.0002))


def test_every_part_of_a_planted_itemset_is_in_the_truth_labelled_part(data):
    parts = set()
    for p in data.planted:
        for k in range(2, len(p.items)):
            parts |= {frozenset(c) for c in combinations(p.items, k)}
    assert parts and all(part in data.truth for part in parts)
    assert all(data.label(part) == "part" for part in parts)


def test_every_truth_itemset_has_a_label_and_planted_ones_are_exactly_thirty_two(data):
    labels = [row["label"] for row in data.labelled_truth()]
    assert set(labels) <= set(LABELS)
    assert labels.count("planted") == 32
    assert labels.count("single") == sum(1 for items in data.truth if len(items) == 1)


def test_classify_itemset_on_a_hand_made_example():
    planted = [Planted(("a", "b", "c"), 0.01)]
    assert classify_itemset(frozenset("a"), planted) == "single"
    assert classify_itemset(frozenset("abc"), planted) == "planted"
    assert classify_itemset(frozenset("ab"), planted) == "part"
    assert classify_itemset(frozenset("abx"), planted) == "extension"
    assert classify_itemset(frozenset("ax"), planted) == "chance"
    assert classify_itemset(frozenset("xy"), planted) == "chance"


def test_save_then_load_gives_back_the_same_dataset(data, tmp_path):
    folder = save_dataset(data, tmp_path / "baskets")
    for name in ("catalog.json", "planted.json", "truth.json", "baskets.jsonl", "meta.json"):
        assert (folder / name).exists()
    back = load_dataset(folder)
    assert back.baskets == data.baskets
    assert back.truth == data.truth
    assert back.planted == data.planted
    assert back.catalog == data.catalog
    assert np.allclose(back.shares, data.shares)
    assert (back.seed, back.min_support) == (data.seed, data.min_support)


def test_the_meta_file_describes_the_dataset(data, tmp_path):
    meta = json.loads((save_dataset(data, tmp_path) / "meta.json").read_text())
    assert meta["n_baskets"] == 40_000 and meta["n_planted"] == 32 and meta["seed"] == 0
    assert sum(meta["truth_by_label"].values()) == meta["n_truth"] == len(data.truth)
    assert 8.5 < meta["mean_distinct_items_per_basket"] < 10.5


def test_the_same_seed_gives_the_same_dataset():
    a, b = generate_dataset(n_baskets=20_000, seed=3), generate_dataset(n_baskets=20_000, seed=3)
    assert a.baskets == b.baskets and a.planted == b.planted and a.truth == b.truth


def test_a_threshold_above_the_rarest_planted_support_or_too_few_baskets_is_refused():
    with pytest.raises(ValueError, match="rarest planted support"):
        generate_dataset(n_baskets=20_000, min_support=0.01)
    with pytest.raises(ValueError, match="at least 10,000"):
        generate_dataset(n_baskets=500)
