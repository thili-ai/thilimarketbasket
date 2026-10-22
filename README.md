# thilimarketbasket

Can a language model learn which items sell together? The classic answer is frequent itemset
mining (Apriori, FP-growth): count every combination, keep the ones above a support threshold.
`thilimarketbasket` asks the other question: write baskets out as text, continue pretraining a
small open language model on them, and measure how many of the *planted* itemsets it recovers, at
which support levels, against the exact miner on the same data.

It is the Phase 1 idea from Netflix's GenRec paper (adapt an open model to your own data) applied
to a shop's baskets, at a size you can run on a laptop.

> ⚠️ **Educational reference, not a production system and not Netflix's method.** A small open
> model adapted on synthetic baskets. Numbers in this repo are our own measurements, nothing more.
> A language model gives a soft, generalising version of itemset mining; FP-growth stays the exact
> answer and the audit.

## Built lesson-by-lesson in an askThili internship

Reference implementation for the askThili internship `lfretail-lfmarketbasket`, "Teach a Language Model
What Sells Together". You build these files yourself,
lesson by lesson. Consult this repo to unblock or to check your work.

| Lesson | What you build | File |
|---|---|---|
| 1 to 5 | The shop: catalogue, popularity, background baskets, one planted itemset, 32 planted itemsets, and the exact truth | `catalog_data.py`, `generate.py`, `dataset.py` |
| 6 | The exact answer: mine frequent itemsets from scratch (Apriori on bitsets) | `mine.py` |
| 7 | Verbalize a basket into text, with the item order shuffled | `verbalize.py` |
| 8 | The off-the-shelf baseline, scored on next-item prediction (MRR, hits@K) | `evaluate.py` |
| 9 | Continued pretraining of SmolLM2-135M on verbalized baskets | `pretrain.py` |
| 10 | Probe the model: have it write baskets, mine them, score against the exact truth | `probe.py` |

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # the shop and the exact miner: numpy and pandas only
pip install -e ".[dev,train]"    # adds tokenizing, training and probing (lessons 7 to 10)
pytest
```

Lessons 7 to 10 use [MLX](https://github.com/ml-explore/mlx), Apple's framework for the Mac GPU, so
they need an Apple-silicon Mac (16 GB of memory recommended). The notebooks of lessons 8 to 10 end with an untested PyTorch version of the same step. On any other machine the package
installs, and the tests run with the two that need MLX skipped. Training the reference run
(800 steps of 16 baskets) takes about 25 minutes on an M3.

## Data

Nothing is vendored. Lessons 1 to 5 generate the baskets, so the right answer (which itemsets exist, and
how common each is) is known exactly. Real data such as Instacart or UCI Online Retail could be used to
check the result holds up (not done yet); download it yourself and respect its licence.
