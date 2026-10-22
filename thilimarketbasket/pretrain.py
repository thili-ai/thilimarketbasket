"""Continued pretraining of a small causal LM on verbalized baskets, with MLX on the Mac's GPU. (Lesson 5.)

"Continued pretraining" is the plain next-token objective, the one the model was first trained with, run a
little longer on text from our own domain. There are no labels and no task: the model reads baskets and learns
to predict every token of them. Every basket is shown in a fresh random order each time (see `verbalize`).
"""
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .verbalize import verbalize_basket


def _encode(tokenizer, baskets, seeds) -> List[List[int]]:
    """Each basket as token ids, written in the order its seed gives, followed by the end-of-text token."""
    texts = [verbalize_basket(basket, seed=int(seed)) for basket, seed in zip(baskets, seeds)]
    return [list(tokenizer.encode(text)) + [tokenizer.eos_token_id] for text in texts]


def _pad(examples):
    """A batch as two arrays: token ids padded with 0, and a 0/1 mask that is 1 on real tokens."""
    import mlx.core as mx

    width = -(-max(len(e) for e in examples) // 16) * 16  # a multiple of 16, so the compiled step sees few shapes
    ids = np.zeros((len(examples), width), dtype=np.int32)
    mask = np.zeros((len(examples), width), dtype=np.float32)
    for row, e in enumerate(examples):
        ids[row, :len(e)] = e
        mask[row, :len(e)] = 1
    return mx.array(ids), mx.array(mask)


def next_token_loss(model, ids, mask):
    """Average cross-entropy of predicting every token from the ones before it, ignoring padding."""
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(ids[:, :-1]).astype(mx.float32)
    loss = nn.losses.cross_entropy(logits, ids[:, 1:], reduction="none")
    return (loss * mask[:, 1:]).sum() / mask[:, 1:].sum()


def validation_loss(model, tokenizer, baskets, batch_size: int = 32, seed: int = 123) -> float:
    """The loss on `baskets`, each written once in a fixed order. Lower means it predicts held-out baskets better."""
    examples = _encode(tokenizer, baskets, np.random.default_rng(seed).integers(2**31, size=len(baskets)))
    total = count = 0.0
    for start in range(0, len(examples), batch_size):
        ids, mask = _pad(examples[start:start + batch_size])
        n = float(mask[:, 1:].sum())
        total += float(next_token_loss(model, ids, mask)) * n
        count += n
    return total / count


def pretrain(model_name: str, baskets, out_dir: str, val_baskets=None, steps: int = 1000, batch_size: int = 32,
             learning_rate: float = 1e-4, warmup: int = 50, eval_every: int = 100, seed: int = 0,
             log=print) -> List[Dict]:
    """Continue training `model_name` (a Hugging Face repo or a folder) on `baskets` for `steps` updates.

    Saves the model to `out_dir` (loadable with `mlx_lm.load(out_dir)`) and returns the training history, a list
    of {"step", "loss", "val_loss", "tokens", "seconds"}. The learning rate warms up linearly, then falls along a
    cosine to a tenth of its peak. Training runs in float32 (half precision loses the small updates); the saved
    model is converted back to bfloat16, which is how the original is stored.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm import load
    from mlx_lm.utils import save

    model, tokenizer, config = load(model_name, return_config=True)
    model.set_dtype(mx.float32)
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, next_token_loss)
    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(ids, mask):
        loss, grads = loss_and_grad(model, ids, mask)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss

    rng = np.random.default_rng(seed)
    bucket = batch_size * 20

    def batches():
        """Endless batches: shuffle, take a bucket, sort it by length so batches waste little padding."""
        while True:
            order = rng.permutation(len(baskets))
            for start in range(0, len(order) - bucket + 1, bucket):
                chosen = order[start:start + bucket]
                examples = _encode(tokenizer, [baskets[i] for i in chosen], rng.integers(2**31, size=len(chosen)))
                by_length = sorted(range(len(examples)), key=lambda i: len(examples[i]))
                groups = [by_length[i:i + batch_size] for i in range(0, len(by_length), batch_size)]
                for g in rng.permutation(len(groups)):
                    yield _pad([examples[i] for i in groups[g]])

    def rate(step):
        if step < warmup:
            return learning_rate * (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))

    history, started, tokens, recent = [], time.time(), 0, []
    if val_baskets is not None:
        history.append({"step": 0, "loss": None, "val_loss": validation_loss(model, tokenizer, val_baskets),
                        "tokens": 0, "seconds": 0.0})
        log(f"step 0  val loss {history[-1]['val_loss']:.3f}")

    for step, (ids, mask) in zip(range(1, steps + 1), batches()):
        optimizer.learning_rate = rate(step - 1)
        loss = train_step(ids, mask)
        mx.eval(state, loss)
        tokens += int(mask.sum())
        recent.append(float(loss))
        if step % eval_every == 0 or step == steps:
            val = validation_loss(model, tokenizer, val_baskets) if val_baskets is not None else None
            history.append({"step": step, "loss": float(np.mean(recent)), "val_loss": val, "tokens": tokens,
                            "seconds": time.time() - started})
            recent = []
            log(f"step {step}  train loss {history[-1]['loss']:.3f}"
                + (f"  val loss {val:.3f}" if val is not None else "")
                + f"  {tokens / (time.time() - started):,.0f} tokens/s")

    model.set_dtype(mx.bfloat16)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save(out_dir, model_name, model, tokenizer, config)
    return history
