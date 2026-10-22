import numpy as np
import pytest

from thilimarketbasket.evaluate import make_cases, popularity_scores, rank_of, summarise
from thilimarketbasket.verbalize import parse_basket, verbalize_prefix


def test_rank_is_one_for_the_best_score_and_ties_count_against_the_target():
    scores = np.array([0.1, 0.9, 0.5, 0.5])
    assert rank_of(scores, 1) == 1
    assert rank_of(scores, 2) == 2          # only index 1 is higher; the tie with index 3 does not help it
    assert rank_of(scores, 0) == 4


def test_excluded_items_do_not_count_as_better_candidates():
    scores = np.array([0.1, 0.9, 0.5])
    assert rank_of(scores, 2, exclude=[1]) == 1
    assert rank_of(scores, 2, exclude=[2, 1]) == 1  # the target itself is never excluded


def test_summary_of_ranks():
    s = summarise([1, 2, 10, 100])
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.1 + 0.01) / 4)
    assert s["hits@1"] == 0.25 and s["hits@10"] == 0.75 and s["hits@50"] == 0.75


def test_cases_hide_one_item_and_are_repeatable():
    baskets = [["a", "b", "b", "c"], ["a", "d"], ["e", "f", "g"]]
    cases = make_cases(baskets, ["a", "b", "c", "d", "e", "f", "g"], 3, seed=1)
    assert cases == make_cases(baskets, ["a", "b", "c", "d", "e", "f", "g"], 3, seed=1)
    assert all(case["target"] not in case["visible"] for case in cases)
    originals = [set(b) for b in baskets]
    assert all(set(c["visible"]) | {c["target"]} in originals for c in cases)


def test_popularity_prefers_the_item_in_more_baskets():
    scores = popularity_scores([["a", "b"], ["a"], ["a", "c"]], ["a", "b", "c", "d"])
    assert scores[0] > scores[1] == scores[2] > scores[3]


def test_a_prefix_plus_an_item_is_a_whole_basket():
    text = verbalize_prefix(["Onion 1 kg", "Potato 1 kg"], seed=0) + " Tomato 1 kg"
    assert sorted(parse_basket(text)) == ["Onion 1 kg", "Potato 1 kg", "Tomato 1 kg"]
    assert parse_basket(text)[-1] == "Tomato 1 kg"
    assert verbalize_prefix([]) + " Tomato 1 kg" == "Basket: Tomato 1 kg"


def test_cached_scoring_matches_scoring_each_whole_sequence():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx_lm.models.llama import Model, ModelArgs
    from thilimarketbasket.evaluate import score_candidates

    mx.random.seed(0)
    model = Model(ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=50))
    prefix = [3, 7, 11, 5]
    candidates = [[1, 2], [9], [4, 4, 4], [20, 21, 22, 23], [9, 1]]
    cached = score_candidates(model, prefix, candidates, batch=2)

    for candidate, got in zip(candidates, cached):
        ids = mx.array([prefix + candidate])
        logp = nn.log_softmax(model(ids).astype(mx.float32), axis=-1)
        want = sum(float(logp[0, len(prefix) + k - 1, candidate[k]]) for k in range(len(candidate)))
        assert got == pytest.approx(want, abs=1e-3)
