"""The finished dataset: baskets, the itemsets planted in them, and the exact truth. (Lesson 1, step 6.)

`generate_dataset` runs every step: the catalogue, the popularity of each item, background baskets, the
planted itemsets, and then the exact miner, to find *every* frequent itemset in the finished baskets. That is
the truth the language model is later scored against.

Planting one itemset makes more itemsets frequent than the ones we planted, so each itemset in the truth is
labelled by where it comes from (see `classify_itemset`).
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .generate import (Item, Planted, background_baskets, default_planted, make_catalog, plant_all,
                       planted_report, popularity)
from .mine import frequent_itemsets

LABELS = ("single", "planted", "part", "extension", "chance")


@dataclass
class Dataset:
    catalog: List[Item]
    shares: np.ndarray  # how much of the shop's sales each catalogue item makes up
    planted: List[Planted]
    baskets: List[List[str]]
    truth: Dict[frozenset, float]  # every frequent itemset in `baskets` (exact), with its support
    min_support: float
    seed: int

    def label(self, itemset) -> str:
        return classify_itemset(frozenset(itemset), self.planted)

    def labelled_truth(self) -> List[dict]:
        return [{"items": sorted(items), "support": support, "size": len(items), "label": self.label(items)}
                for items, support in sorted(self.truth.items(), key=lambda kv: (-kv[1], sorted(kv[0])))]


def classify_itemset(itemset: frozenset, planted: List[Planted]) -> str:
    """Why is this itemset frequent?

    - "single": one item
    - "planted": one of the itemsets we planted
    - "part": a smaller piece of a planted itemset (every part of a planted itemset is at least as common)
    - "extension": holds two or more items of a planted itemset, plus others (the planted baskets also hold
      whatever else the shopper bought)
    - "chance": common items that meet by chance, with no planted pair inside
    """
    if len(itemset) == 1:
        return "single"
    sets = [frozenset(p.items) for p in planted]
    if itemset in sets:
        return "planted"
    if any(itemset < s for s in sets):
        return "part"
    if any(len(itemset & s) >= 2 for s in sets):
        return "extension"
    return "chance"


def generate_dataset(n_baskets: int = 100_000, seed: int = 0, min_support: float = 0.0005,
                     max_size: int = 5) -> Dataset:
    """Baskets with 32 planted itemsets, and the exact truth.

    `min_support` is how rare an itemset may be and still count as frequent. It cannot be above the rarest
    planted itemset (0.05%), or the miner would miss it.
    """
    if n_baskets < 10_000:
        raise ValueError("need at least 10,000 baskets: the rarest itemset is planted in 0.05% of them")
    catalog = make_catalog(seed=seed)
    shares = popularity(catalog)
    sample = background_baskets(200_000, catalog, shares, seed=seed + 10_000)  # to check chance co-occurrence
    planted = default_planted(catalog, shares, sample, seed=seed)
    if min_support > min(p.support for p in planted):
        raise ValueError(f"min_support {min_support} is above the rarest planted support {min(p.support for p in planted)}")
    baskets = plant_all(background_baskets(n_baskets, catalog, shares, seed=seed), planted, catalog, seed=seed)
    truth = frequent_itemsets(baskets, min_support, max_size)
    return Dataset(catalog, shares, planted, baskets, truth, min_support, seed)


def save_dataset(data: Dataset, directory) -> Path:
    """Write the dataset as plain files (JSON, and one basket per line) so any tool can read it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    realised = {row["items"]: row["realised"] for row in planted_report(data.baskets, data.planted)}
    (directory / "catalog.json").write_text(json.dumps(
        [{"name": i.name, "base": i.base, "category": i.category, "brand": i.brand, "size": i.size, "share": float(s)}
         for i, s in zip(data.catalog, data.shares)], indent=1))
    (directory / "planted.json").write_text(json.dumps(
        [{"items": list(p.items), "support": p.support, "kind": p.kind, "realised": realised[p.items]}
         for p in data.planted], indent=1))
    truth = data.labelled_truth()
    (directory / "truth.json").write_text(json.dumps(truth))
    (directory / "baskets.jsonl").write_text("\n".join(json.dumps(b) for b in data.baskets) + "\n")
    counts = {label: sum(1 for t in truth if t["label"] == label) for label in LABELS}
    (directory / "meta.json").write_text(json.dumps({
        "seed": data.seed, "n_baskets": len(data.baskets), "min_support": data.min_support,
        "n_items": len(data.catalog), "n_planted": len(data.planted), "n_truth": len(truth), "truth_by_label": counts,
        "mean_distinct_items_per_basket": float(np.mean([len(set(b)) for b in data.baskets])),
    }, indent=1))
    return directory


def load_dataset(directory) -> Dataset:
    """Read back what `save_dataset` wrote."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    rows = json.loads((directory / "catalog.json").read_text())
    catalog = [Item(r["name"], r["base"], r["category"], r["brand"], r["size"]) for r in rows]
    planted = [Planted(tuple(r["items"]), r["support"], r["kind"])
               for r in json.loads((directory / "planted.json").read_text())]
    baskets = [json.loads(line) for line in (directory / "baskets.jsonl").read_text().splitlines()]
    truth = {frozenset(r["items"]): r["support"] for r in json.loads((directory / "truth.json").read_text())}
    return Dataset(catalog, np.array([r["share"] for r in rows]), planted, baskets, truth,
                   meta["min_support"], meta["seed"])
