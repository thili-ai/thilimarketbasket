"""Next-item scoring for a causal LM: how high does it rank the item that was hidden? (Lessons 4 and 6.)

Take a basket, hide one item, show the model the rest as text, and ask for the probability of every item in the
catalogue as the next one. The rank of the hidden item gives the reciprocal rank (MRR) and hits@K.

The model is an MLX model from `mlx_lm.load` (Apple's framework for the Mac GPU); `mlx` is imported only inside
`score_candidates`, so the rest of the package works without it.
"""
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .verbalize import verbalize_prefix


def candidate_token_ids(tokenizer, names: Sequence[str]) -> List[List[int]]:
    """The tokens of each item name as it follows "...;" in a basket: a space, then the name."""
    return [list(tokenizer.encode(" " + name)) for name in names]


def score_candidates(model, prefix_ids: Sequence[int], candidate_ids: Sequence[Sequence[int]], batch: int = 250) -> np.ndarray:
    """log P(candidate | prefix) for every candidate, summed over the candidate's tokens.

    The prefix is run through the model once and its key/value cache is copied for a batch of candidates,
    so a thousand candidates cost a thousand short sequences, not a thousand whole baskets. Candidates are
    sorted by length and padded within each batch to waste little work.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache

    prefix_cache = make_prompt_cache(model)
    first = nn.log_softmax(model(mx.array([list(prefix_ids)]), cache=prefix_cache)[:, -1, :].astype(mx.float32), axis=-1)
    mx.eval(first, [c.keys for c in prefix_cache])
    first = np.array(first[0])  # how likely each token is to start the candidate

    scores = np.zeros(len(candidate_ids))
    order = sorted(range(len(candidate_ids)), key=lambda i: len(candidate_ids[i]))
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        width = max(len(candidate_ids[i]) for i in chunk)
        tokens = mx.array([list(candidate_ids[i]) + [0] * (width - len(candidate_ids[i])) for i in chunk])
        rows = len(chunk)
        cache = make_prompt_cache(model)
        for mine, source in zip(cache, prefix_cache):
            keys, values = source.keys[..., :source.offset, :], source.values[..., :source.offset, :]
            mine.keys = mx.contiguous(mx.broadcast_to(keys, (rows,) + keys.shape[1:]))
            mine.values = mx.contiguous(mx.broadcast_to(values, (rows,) + values.shape[1:]))
            mine.offset = source.offset
        if width > 1:
            logits = model(tokens[:, :-1], cache=cache).astype(mx.float32)   # predicts tokens 1.. of each candidate
            chosen = mx.take_along_axis(logits, tokens[:, 1:, None], axis=-1)[..., 0]
            chosen = np.array(chosen - mx.logsumexp(logits, axis=-1))
        for row, i in enumerate(chunk):
            length = len(candidate_ids[i])
            scores[i] = first[candidate_ids[i][0]] + (chosen[row, :length - 1].sum() if length > 1 else 0.0)
    return scores


def rank_of(scores: np.ndarray, target: int, exclude: Iterable[int] = ()) -> int:
    """The rank (1 is best) of `target` among the candidates not in `exclude`. Ties count against the target."""
    keep = np.ones(len(scores), dtype=bool)
    keep[list(exclude)] = False
    keep[target] = False  # the target is compared with the others, never with itself
    return 1 + int((scores[keep] >= scores[target]).sum())


def make_cases(baskets, names: Sequence[str], n_cases: int, seed: int = 0) -> List[dict]:
    """Hide one random item of each of `n_cases` random baskets: {"visible": [...], "target": item name}."""
    rng = np.random.default_rng(seed)
    cases = []
    for number in rng.choice(len(baskets), size=min(n_cases, len(baskets)), replace=False):
        different = sorted(set(baskets[number]))
        hidden = different[int(rng.integers(len(different)))]
        cases.append({"visible": [n for n in different if n != hidden], "target": hidden})
    return cases


def summarise(ranks: Sequence[int], ks=(1, 10, 50)) -> Dict[str, float]:
    ranks = np.asarray(ranks)
    out = {"mrr": float(np.mean(1.0 / ranks)), "median_rank": float(np.median(ranks))}
    out.update({f"hits@{k}": float(np.mean(ranks <= k)) for k in ks})
    return out


def next_item_eval(model, tokenizer, names: Sequence[str], cases: Sequence[dict], seed: int = 0) -> Dict[str, float]:
    """Rank every catalogue item for each case with the model. Items already visible are not candidates."""
    candidates = candidate_token_ids(tokenizer, names)
    position = {name: i for i, name in enumerate(names)}
    ranks = []
    for k, case in enumerate(cases):
        prefix = verbalize_prefix(case["visible"], seed=seed + k)
        scores = score_candidates(model, tokenizer.encode(prefix), candidates)
        ranks.append(rank_of(scores, position[case["target"]], [position[n] for n in case["visible"]]))
    return {**summarise(ranks), "ranks": ranks}


def popularity_scores(train_baskets, names: Sequence[str]) -> np.ndarray:
    """A baseline that never reads the basket: log of how many training baskets hold each item."""
    position = {name: i for i, name in enumerate(names)}
    counts = np.ones(len(names))  # one extra everywhere, so an item never seen scores low rather than minus infinity
    for basket in train_baskets:
        for name in set(basket):
            counts[position[name]] += 1
    return np.log(counts)
