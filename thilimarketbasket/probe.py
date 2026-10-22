"""Probe the trained model: what does it believe goes together? (Lesson 10.)

A language model has no table of itemsets inside it. What it has is a distribution over baskets: it can write one,
item after item. So we ask it to write many, then count what turns up together with the same exact miner that
counted the real baskets. If the model has learned the data, the itemsets frequent in its baskets are the itemsets
frequent in ours, with similar supports. Where they differ is what the model did not learn (or made up).

`mlx` is imported only inside `sample_baskets`, so the counting and scoring work without it.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .mine import frequent_itemsets
from .verbalize import PREFIX, SEPARATOR


def sample_baskets(model, tokenizer, names: Sequence[str], n: int, batch: int = 100, max_tokens: int = 240,
                   temperature: float = 1.0, seed: int = 0) -> Tuple[List[List[str]], Dict[str, float]]:
    """Have the model write `n` baskets from the bare prompt "Basket:". Returns (baskets, stats).

    All baskets in a batch are written together, one token per step, sharing one key/value cache. A basket is
    finished when the model writes the end-of-text token. Names that are not in `names` are dropped, a repeated
    item is kept once, and a basket cut off at `max_tokens` loses its last (possibly half-written) item.
    `stats` says how often each of those happened.
    """
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    known = set(names)
    prompt = list(tokenizer.encode(PREFIX.rstrip()))
    eos = tokenizer.eos_token_id
    mx.random.seed(seed)

    baskets, unknown, repeats, cut, items_written = [], 0, 0, 0, 0
    for start in range(0, n, batch):
        size = min(batch, n - start)
        cache = make_prompt_cache(model)
        logits = model(mx.array([prompt] * size), cache=cache)[:, -1, :]
        written = [[] for _ in range(size)]
        done = np.zeros(size, dtype=bool)
        for _ in range(max_tokens):
            token = mx.random.categorical(logits.astype(mx.float32) / temperature)
            mx.eval(token)
            chosen = np.array(token)
            for row in np.flatnonzero(~done):
                if chosen[row] == eos:
                    done[row] = True
                else:
                    written[row].append(int(chosen[row]))
            if done.all():
                break
            logits = model(token[:, None], cache=cache)[:, -1, :]
        for row, ids in enumerate(written):
            finished = bool(done[row])
            pieces = tokenizer.decode(ids).split(SEPARATOR.strip())
            pieces = [p.strip() for p in pieces if p.strip()]
            if not finished and pieces:
                pieces = pieces[:-1]
                cut += 1
            items_written += len(pieces)
            unknown += sum(p not in known for p in pieces)
            pieces = [p for p in pieces if p in known]
            repeats += len(pieces) - len(set(pieces))
            baskets.append(list(dict.fromkeys(pieces)))
    stats = {"baskets": len(baskets), "items written": items_written,
             "share not in catalogue": unknown / max(1, items_written),
             "share repeated": repeats / max(1, items_written),
             "baskets cut off": cut / max(1, len(baskets)),
             "mean items per basket": float(np.mean([len(b) for b in baskets]))}
    return baskets, stats


def recover_itemsets(sampled: Sequence[Sequence[str]], min_support: float, max_size: int = 5) -> Dict[frozenset, float]:
    """The itemsets frequent in the model's own baskets, with their support there: the model's estimate."""
    return frequent_itemsets([b for b in sampled], min_support, max_size=max_size)


def recovery_by_support(recovered: Dict[frozenset, float], truth: Dict[frozenset, float], bands: Sequence[float],
                        group=None) -> List[dict]:
    """How many of the true itemsets did the model recover, per band of true support?

    `bands` are ascending support edges, e.g. [0.002, 0.005, 0.01, 1]: the first band is [0.002, 0.005).
    `truth` maps each true frequent itemset to its true support (mined at the lowest edge). `recovered` is the
    model's set. `group` optionally maps an itemset to a label, to split each band further (returns rows per
    (band, label)). Each row: band, label, true (how many), found, recall.
    """
    rows = []
    for low, high in zip(bands[:-1], bands[1:]):
        in_band = [s for s, v in truth.items() if low <= v < high]
        labels = sorted({group(s) for s in in_band}) if group else [None]
        for label in labels:
            chosen = [s for s in in_band if group is None or group(s) == label]
            found = sum(s in recovered for s in chosen)
            rows.append({"band": f"{low:.2%} to {min(high, 1):.2%}", "label": label, "true": len(chosen),
                         "found": found, "recall": found / len(chosen) if chosen else float("nan")})
    return rows


def precision_of(recovered: Dict[frozenset, float], truth: Dict[frozenset, float], min_support: float) -> float:
    """Of the itemsets the model calls frequent, the share that really are (true support at least `min_support`)."""
    if not recovered:
        return float("nan")
    return float(np.mean([truth.get(s, 0.0) >= min_support for s in recovered]))


def support_error(recovered: Dict[frozenset, float], truth: Dict[frozenset, float]) -> np.ndarray:
    """For itemsets in both, the model's support minus the true support."""
    return np.array([recovered[s] - truth[s] for s in recovered if s in truth])
