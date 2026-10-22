import pytest

from thilimarketbasket.catalog_data import CATEGORIES
from thilimarketbasket.generate import Item, make_catalog


def n_products():
    return sum(len(spec["products"]) for spec in CATEGORIES.values())


def test_default_catalog_has_a_thousand_unique_items():
    catalog = make_catalog()
    assert len(catalog) == 1000
    assert len({item.name for item in catalog}) == 1000


def test_every_product_and_category_is_stocked():
    catalog = make_catalog()
    assert {item.category for item in catalog} == set(CATEGORIES)
    assert len({(item.category, item.base) for item in catalog}) == n_products()


def test_same_seed_same_catalog_different_seed_different_catalog():
    assert make_catalog(seed=1) == make_catalog(seed=1)
    assert make_catalog(seed=1) != make_catalog(seed=2)
    assert len(make_catalog(seed=2)) == 1000


def test_a_small_catalog_still_covers_every_product():
    catalog = make_catalog(n_items=n_products())
    assert len(catalog) == n_products()
    assert len({(item.category, item.base) for item in catalog}) == n_products()


def test_too_few_or_too_many_items_are_rejected():
    with pytest.raises(ValueError):
        make_catalog(n_items=n_products() - 1)
    with pytest.raises(ValueError):
        make_catalog(n_items=100_000)


def test_names_are_clean_and_built_from_their_parts():
    for item in make_catalog():
        assert item.name == item.name.strip() and "  " not in item.name
        assert item.base in item.name and item.size in item.name
        if item.brand:
            assert item.name.startswith(item.brand)
        else:
            assert item.category == "Fresh Produce"  # only loose goods are unbranded


def test_sachets_are_part_of_the_shop():
    sizes = " ".join(item.size for item in make_catalog())
    assert "sachet" in sizes and "Rs 2 sachet" in sizes


# --- step 2: popularity -------------------------------------------------------------------------------

import numpy as np

from thilimarketbasket.generate import head_share, popularity


def share_of(catalog, shares, **where):
    """Total share of every item whose fields match `where` (for example base="Onion")."""
    return sum(s for item, s in zip(catalog, shares) if all(getattr(item, k) == v for k, v in where.items()))


def test_shares_are_one_per_item_positive_and_add_up_to_one():
    catalog = make_catalog()
    shares = popularity(catalog)
    assert len(shares) == len(catalog)
    assert (shares > 0).all()
    assert shares.sum() == pytest.approx(1.0)


def test_popularity_is_the_same_every_time_for_the_same_catalogue():
    catalog = make_catalog(seed=3)
    assert np.array_equal(popularity(catalog), popularity(catalog))


def test_a_big_category_outsells_a_small_one():
    catalog = make_catalog()
    shares = popularity(catalog)
    assert share_of(catalog, shares, category="Fresh Produce") > share_of(catalog, shares, category="Baby and Stationery")


def test_products_in_a_category_split_it_by_their_weights():
    catalog = make_catalog()
    shares = popularity(catalog)
    ratio = share_of(catalog, shares, base="Onion") / share_of(catalog, shares, base="Potato")
    assert ratio == pytest.approx(30 / 28)  # their weights in catalog_data.py


def test_a_leading_brand_outsells_a_small_one_in_the_same_size():
    catalog = make_catalog()
    shares = popularity(catalog)
    rows = [(item.brand, s) for item, s in zip(catalog, shares) if item.base == "Toor Dal" and item.size == "1 kg"]
    weights = CATEGORIES["Pulses"]["brands"]
    best_by_weight = max(rows, key=lambda row: weights[row[0]])[0]
    best_by_share = max(rows, key=lambda row: row[1])[0]
    assert best_by_share == best_by_weight


def test_small_packs_outsell_big_sacks():
    catalog = make_catalog()
    shares = popularity(catalog)
    one_kg = share_of(catalog, shares, base="Salt", size="1 kg")
    five_kg = share_of(catalog, shares, base="Salt", size="5 kg")
    assert one_kg > five_kg


def test_a_few_items_make_most_of_the_sales():
    shares = popularity(make_catalog())
    assert head_share(shares, 0.2) > 0.55  # the best-selling fifth makes well over half the sales
    assert head_share(shares, 0.01) > 0.08
    assert head_share(shares, 1.0) == pytest.approx(1.0)


def test_a_product_keeps_its_share_whichever_brands_are_stocked():
    for seed in (1, 2, 3):
        catalog = make_catalog(seed=seed)
        shares = popularity(catalog)
        total = share_of(catalog, shares, base="Milk") / share_of(catalog, shares, base="Curd")
        assert total == pytest.approx(30 / 14)


# --- step 3: background baskets -----------------------------------------------------------------------

from collections import Counter

from thilimarketbasket.generate import Products, background_baskets, basket_sizes, draw_items


@pytest.fixture(scope="module")
def shop():
    catalog = make_catalog()
    shares = popularity(catalog)
    return catalog, shares, background_baskets(40_000, catalog, shares, seed=0)


def test_the_right_number_of_baskets_and_the_same_every_time_for_one_seed():
    catalog = make_catalog()
    shares = popularity(catalog)
    a = background_baskets(500, catalog, shares, seed=5)
    assert len(a) == 500
    assert a == background_baskets(500, catalog, shares, seed=5)
    assert a != background_baskets(500, catalog, shares, seed=6)


def test_basket_sizes_have_the_right_mean_and_a_floor_and_a_tail():
    rng = np.random.default_rng(0)
    sizes = basket_sizes(50_000, rng)
    assert sizes.min() == 2
    assert sizes.mean() == pytest.approx(8.0, abs=0.1)
    assert np.percentile(sizes, 99) >= 18  # a few big shopping trips


def test_every_item_in_a_basket_is_a_real_item(shop):
    catalog, _, baskets = shop
    names = {item.name for item in catalog}
    assert all(name in names for basket in baskets[:5000] for name in basket)


def test_a_basket_never_holds_two_brands_or_sizes_of_one_product(shop):
    catalog, _, baskets = shop
    product = {item.name: (item.category, item.base) for item in catalog}
    for basket in baskets:
        kinds = [product[name] for name in set(basket)]
        assert len(kinds) == len(set(kinds))


def test_some_items_are_bought_more_than_once_and_most_are_not(shop):
    _, _, baskets = shop
    quantities = Counter(q for basket in baskets for q in Counter(basket).values())
    total = sum(quantities.values())
    assert quantities[1] / total == pytest.approx(0.80, abs=0.01)
    assert quantities[2] / total == pytest.approx(0.14, abs=0.01)
    assert max(quantities) == 4


def test_best_sellers_turn_up_more_often_than_rare_items(shop):
    catalog, shares, baskets = shop
    seen = Counter(name for basket in baskets for name in set(basket))
    names = [item.name for item in catalog]
    frequency = np.array([seen[name] for name in names], dtype=float)
    rank = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    assert np.corrcoef(rank(frequency), rank(shares))[0, 1] > 0.9  # same order, almost
    best = int(np.argmax(shares))
    assert seen[names[best]] / len(baskets) > 0.05


def test_draw_items_gives_the_asked_number_of_different_picks():
    rng = np.random.default_rng(0)
    shares = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    picks = draw_items([1, 3, 5, 2], shares, rng)
    assert [len(p) for p in picks] == [1, 3, 5, 2]
    assert all(len(set(p)) == len(p) for p in picks)  # no repeats within a basket


def test_products_group_skus_and_pick_one_with_its_odds():
    catalog = make_catalog()
    shares = popularity(catalog)
    products = Products(catalog, shares)
    assert products.share.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    toor = [i for i, item in enumerate(catalog) if item.base == "Toor Dal"]
    product_number = next(k for k, _ in enumerate(products.share) if catalog[products._sku[
        np.searchsorted(products._cdf, k + 0.5)]].base == "Toor Dal")
    picked = products.pick_sku(np.full(20_000, product_number), rng)
    assert set(picked) <= set(toor)  # always one of that product's own SKUs
    expected = shares[toor] / shares[toor].sum()
    observed = np.array([(picked == i).mean() for i in toor])
    assert np.abs(observed - expected).max() < 0.02


# --- step 4: planting one itemset ---------------------------------------------------------------------

from thilimarketbasket.generate import Planted, find_items, plant_itemset, support_of


def best_seller(catalog, shares, base):
    items = find_items(catalog, base)
    return max(items, key=lambda item: shares[catalog.index(item)]).name


def test_support_of_counts_baskets_holding_every_item_whatever_the_quantity():
    baskets = [["a", "a", "b"], ["a"], ["b", "a"], ["c"]]
    assert support_of(baskets, ["a", "b"]) == 0.5
    assert support_of(baskets, ["a"]) == 0.75


def test_find_items_narrows_by_brand_and_size():
    catalog = make_catalog()
    toor = find_items(catalog, "Toor Dal")
    assert toor and all(item.base == "Toor Dal" for item in toor)
    one = find_items(catalog, "Toor Dal", size="1 kg")
    assert one and all(item.size == "1 kg" for item in one)
    assert find_items(catalog, "No Such Product") == []


@pytest.mark.parametrize("support", [0.05, 0.01, 0.002, 0.0005])
def test_a_planted_itemset_reaches_its_target_support_exactly(shop, support):
    catalog, shares, baskets = shop
    items = tuple(best_seller(catalog, shares, base) for base in ("Toor Dal", "Sona Masoori Rice", "Sunflower Oil"))
    planted = plant_itemset(baskets, Planted(items, support), catalog, seed=1)
    assert sum(set(items) <= set(b) for b in planted) == round(support * len(baskets))


def test_only_as_many_baskets_change_as_needed_and_the_input_is_untouched(shop):
    catalog, shares, baskets = shop
    items = tuple(best_seller(catalog, shares, base) for base in ("Toor Dal", "Sona Masoori Rice"))
    before = [list(b) for b in baskets]
    already = sum(set(items) <= set(b) for b in baskets)
    planted = plant_itemset(baskets, Planted(items, 0.01), catalog, seed=1)
    assert baskets == before  # the original list is not modified
    changed = sum(a != b for a, b in zip(baskets, planted))
    assert changed == round(0.01 * len(baskets)) - already


def test_a_basket_never_ends_up_with_two_skus_of_one_product_even_when_planting_clashes(shop):
    catalog, shares, baskets = shop
    items = tuple(best_seller(catalog, shares, base) for base in ("Milk", "Onion", "Sugar"))
    planted = plant_itemset(baskets, Planted(items, 0.05), catalog, seed=2)
    product = {item.name: (item.category, item.base) for item in catalog}
    for basket in planted:
        kinds = [product[name] for name in set(basket)]
        assert len(kinds) == len(set(kinds))
    # a plain swap happened somewhere: a basket lost another SKU of a planted product
    other_skus = {i.name for base in ("Milk", "Onion", "Sugar") for i in find_items(catalog, base)} - set(items)
    lost = sum(bool(other_skus & set(a)) and not (other_skus & set(b)) for a, b in zip(baskets, planted))
    assert lost > 0


def test_planting_is_repeatable_per_seed(shop):
    catalog, shares, baskets = shop
    items = tuple(best_seller(catalog, shares, base) for base in ("Toor Dal", "Sunflower Oil"))
    a = plant_itemset(baskets, Planted(items, 0.01), catalog, seed=4)
    assert a == plant_itemset(baskets, Planted(items, 0.01), catalog, seed=4)
    assert a != plant_itemset(baskets, Planted(items, 0.01), catalog, seed=5)


def test_planted_items_are_not_left_at_the_end_of_the_basket(shop):
    catalog, shares, baskets = shop
    items = tuple(best_seller(catalog, shares, base) for base in ("Toor Dal", "Sunflower Oil"))
    planted = plant_itemset(baskets, Planted(items, 0.02), catalog, seed=1)
    # Where every copy of the item sits, from 0 (first) to 1 (last). Using every copy, not just the first, keeps
    # a bought-twice item from pulling the average down. Appended at the end this would be about 1.
    positions = [j / (len(basket) - 1) for a, basket in zip(baskets, planted) if a != basket and len(basket) > 1
                 for j, name in enumerate(basket) if name == items[0]]
    assert np.mean(positions) == pytest.approx(0.5, abs=0.05)


def test_bad_requests_are_refused_with_a_reason(shop):
    catalog, shares, baskets = shop
    toor, oil = best_seller(catalog, shares, "Toor Dal"), best_seller(catalog, shares, "Sunflower Oil")
    with pytest.raises(ValueError, match="not in the catalogue"):
        plant_itemset(baskets, Planted((toor, "Nonsense Item"), 0.01), catalog)
    other_toor = [i.name for i in find_items(catalog, "Toor Dal") if i.name != toor][0]
    with pytest.raises(ValueError, match="different products"):
        plant_itemset(baskets, Planted((toor, other_toor), 0.01), catalog)
    with pytest.raises(ValueError, match="support must be between"):
        plant_itemset(baskets, Planted((toor, oil), 1.5), catalog)


def test_an_itemset_that_chance_already_makes_too_common_cannot_be_planted_rarer(shop):
    catalog, shares, baskets = shop
    common = tuple(best_seller(catalog, shares, base) for base in ("Onion", "Tomato", "Potato"))
    with pytest.raises(ValueError, match="too common"):
        plant_itemset(baskets, Planted(common, 0.0001), catalog)


# --- step 5: planting many itemsets -------------------------------------------------------------------

from thilimarketbasket.generate import (NATURAL, SUPPORTS, _chance_support, _index, default_planted, plant_all,
                                         planted_report)


@pytest.fixture(scope="module")
def planted_shop():
    catalog = make_catalog()
    shares = popularity(catalog)
    background = background_baskets(200_000, catalog, shares, seed=11)  # to check chance supports
    planted = default_planted(catalog, shares, background, seed=0)
    baskets = background_baskets(100_000, catalog, shares, seed=0)
    return catalog, shares, background, planted, baskets, plant_all(baskets, planted, catalog, seed=0)


def test_there_are_thirty_two_itemsets_spread_evenly(planted_shop):
    _, _, _, planted, _, _ = planted_shop
    assert len(planted) == 32
    assert Counter(p.support for p in planted) == {s: 8 for s in SUPPORTS}
    assert Counter(len(p.items) for p in planted) == {2: 8, 3: 8, 4: 8, 5: 8}
    assert Counter(p.kind for p in planted) == {"natural": 16, "surprising": 16}


def test_every_planted_item_is_in_the_catalogue_and_a_product_keeps_one_sku(planted_shop):
    catalog, _, _, planted, _, _ = planted_shop
    names = {item.name: item.base for item in catalog}
    sku_of = {}
    for itemset in planted:
        for name in itemset.items:
            assert name in names
            assert sku_of.setdefault(names[name], name) == name  # the same SKU wherever the product appears


def test_no_itemset_contains_another_and_some_share_items(planted_shop):
    _, _, _, planted, _, _ = planted_shop
    sets = [set(p.items) for p in planted]
    assert not any(a <= b for i, a in enumerate(sets) for j, b in enumerate(sets) if i != j)
    used = Counter(name for s in sets for name in s)
    assert sum(1 for count in used.values() if count > 1) >= 10  # deliberate overlaps


def test_chance_alone_gives_less_than_a_third_of_every_target(planted_shop):
    _, _, background, planted, _, _ = planted_shop
    index = _index(background)
    for itemset in planted:
        assert _chance_support(index, itemset.items, len(background)) <= 0.3 * itemset.support


def test_the_natural_sets_come_from_the_hand_written_groups(planted_shop):
    catalog, _, _, planted, _, _ = planted_shop
    base = {item.name: item.base for item in catalog}
    groups = {frozenset(g) for size in NATURAL.values() for g in size}
    for itemset in planted:
        if itemset.kind == "natural":
            assert frozenset(base[name] for name in itemset.items) in groups


def test_default_itemsets_are_repeatable_per_seed(planted_shop):
    catalog, shares, background, planted, _, _ = planted_shop
    assert default_planted(catalog, shares, background, seed=0) == planted
    assert default_planted(catalog, shares, background, seed=1) != planted


def test_planted_supports_come_out_at_or_very_near_their_targets(planted_shop):
    _, _, _, planted, _, after = planted_shop
    for row in planted_report(after, planted):
        assert abs(row["realised"] - row["target"]) <= max(0.03 * row["target"], 0.0002), row


def test_planting_many_leaves_a_basket_with_one_sku_per_product(planted_shop):
    catalog, _, _, _, _, after = planted_shop
    product = {item.name: (item.category, item.base) for item in catalog}
    for basket in after[:30_000]:
        kinds = [product[name] for name in set(basket)]
        assert len(kinds) == len(set(kinds))


def test_the_parts_of_a_planted_itemset_are_at_least_as_common_as_the_whole(planted_shop):
    _, _, _, planted, _, after = planted_shop
    for itemset in planted[:8]:
        whole = support_of(after, itemset.items)
        for item in itemset.items:
            rest = [other for other in itemset.items if other != item]
            assert support_of(after, rest) >= whole


def test_plant_all_refuses_a_product_under_two_skus_and_nested_itemsets(planted_shop):
    catalog, shares, _, _, baskets, _ = planted_shop
    toor = [i.name for i in find_items(catalog, "Toor Dal")][:2]
    oil = [i.name for i in find_items(catalog, "Sunflower Oil")][0]
    milk = [i.name for i in find_items(catalog, "Milk")][0]
    with pytest.raises(ValueError, match="appears as both"):
        plant_all(baskets[:2000], [Planted((toor[0], oil), 0.01), Planted((toor[1], milk), 0.01)], catalog)
    with pytest.raises(ValueError, match="contained in"):
        plant_all(baskets[:2000], [Planted((toor[0], oil), 0.01), Planted((toor[0], oil, milk), 0.01)], catalog)
    with pytest.raises(ValueError, match="not in the catalogue"):
        plant_all(baskets[:2000], [Planted((toor[0], "Nonsense"), 0.01)], catalog)


def test_planting_one_itemset_can_be_told_to_leave_a_too_common_one_alone(shop):
    catalog, shares, baskets = shop
    common = tuple(best_seller(catalog, shares, base) for base in ("Onion", "Tomato", "Potato"))
    left_alone = plant_itemset(baskets, Planted(common, 0.0001), catalog, allow_excess=True)
    assert left_alone == baskets  # it already had more than was asked for, so nothing was planted
