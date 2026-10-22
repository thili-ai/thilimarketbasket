import pytest

from thilimarketbasket.probe import precision_of, recover_itemsets, recovery_by_support, support_error

A, B, C, D = frozenset("a"), frozenset("b"), frozenset("ab"), frozenset("cd")
TRUTH = {A: 0.30, B: 0.12, C: 0.08, D: 0.01}


def test_recovery_counts_per_band():
    recovered = {A: 0.31, C: 0.07, frozenset("xy"): 0.02}
    rows = recovery_by_support(recovered, TRUTH, [0.005, 0.1, 1])
    assert [(r["true"], r["found"]) for r in rows] == [(2, 1), (2, 1)]
    assert rows[0]["recall"] == 0.5


def test_recovery_can_split_by_label():
    rows = recovery_by_support({C: 0.07}, TRUTH, [0.005, 1], group=lambda s: "pair" if len(s) == 2 else "other")
    by_label = {r["label"]: (r["true"], r["found"]) for r in rows}
    assert by_label == {"pair": (2, 1), "other": (2, 0)}


def test_empty_band_gives_no_rows_error():
    rows = recovery_by_support({}, TRUTH, [0.5, 1])
    assert rows[0]["true"] == 0 and rows[0]["recall"] != rows[0]["recall"]  # nan


def test_precision_counts_claims_the_truth_does_not_back():
    recovered = {A: 0.3, C: 0.08, frozenset("xy"): 0.02}
    assert precision_of(recovered, TRUTH, 0.05) == pytest.approx(2 / 3)


def test_support_error_only_for_shared_itemsets():
    err = support_error({A: 0.33, frozenset("xy"): 0.5}, TRUTH)
    assert list(err) == pytest.approx([0.03])


def test_recover_itemsets_mines_the_sample():
    sample = [["a", "b"]] * 6 + [["a"]] * 4
    found = recover_itemsets(sample, 0.5)
    assert found == {frozenset("a"): 1.0, frozenset("b"): 0.6, frozenset("ab"): 0.6}


def test_sampled_baskets_are_clean():
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.llama import Model, ModelArgs
    from thilimarketbasket.probe import sample_baskets

    class Tok:
        eos_token_id = 0
        def encode(self, text): return [1]
        def decode(self, ids): return " ".join("ab"[i % 2] for i in ids).replace(" ", "; ")

    mx.random.seed(0)
    model = Model(ModelArgs(model_type="llama", hidden_size=32, num_hidden_layers=2, intermediate_size=64,
                            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, vocab_size=10))
    baskets, stats = sample_baskets(model, Tok(), ["a", "b"], n=7, batch=3, max_tokens=12)
    assert len(baskets) == 7
    assert all(len(set(b)) == len(b) and set(b) <= {"a", "b"} for b in baskets)
    assert stats["share repeated"] >= 0
