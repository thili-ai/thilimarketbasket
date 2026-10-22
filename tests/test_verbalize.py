import pytest

from thilimarketbasket.verbalize import PREFIX, parse_basket, verbalize_all, verbalize_basket


BASKET = ["Onion 1 kg", "Gokul Fresh Milk 500 ml", "Potato 1 kg", "Gokul Fresh Milk 500 ml"]


def test_a_basket_reads_as_one_line_with_each_item_once():
    text = verbalize_basket(BASKET, seed=0)
    assert text.startswith(PREFIX) and "\n" not in text
    assert sorted(parse_basket(text)) == ["Gokul Fresh Milk 500 ml", "Onion 1 kg", "Potato 1 kg"]


def test_the_same_seed_gives_the_same_text_whatever_order_the_basket_came_in():
    assert verbalize_basket(BASKET, seed=3) == verbalize_basket(list(reversed(BASKET)), seed=3)


def test_different_seeds_give_different_orders_of_the_same_items():
    texts = {verbalize_basket(BASKET, seed=s) for s in range(30)}
    assert len(texts) > 1
    assert {frozenset(parse_basket(t)) for t in texts} == {frozenset(["Onion 1 kg", "Potato 1 kg", "Gokul Fresh Milk 500 ml"])}


def test_a_big_basket_is_cut_to_max_items():
    big = [f"item {i}" for i in range(50)]
    assert len(parse_basket(verbalize_basket(big, seed=0, max_items=30))) == 30


def test_an_empty_basket_round_trips():
    assert parse_basket(verbalize_basket([], seed=0)) == []


def test_parse_refuses_text_that_is_not_a_basket():
    with pytest.raises(ValueError):
        parse_basket("hello")


def test_epochs_repeat_every_basket_in_a_fresh_order():
    texts = verbalize_all([BASKET, ["Potato 1 kg", "Onion 1 kg", "Tomato 1 kg"]], epochs=3, seed=0)
    assert len(texts) == 6
    assert len({tuple(parse_basket(t)) for t in texts[1::2]}) > 1
