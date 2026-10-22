"""Generate baskets with itemsets planted at known supports. (Lesson 1.)

Built in steps. Step 1 (done): the catalogue. Step 2 (done): how much each item sells. Step 3 (done):
background baskets. Step 4 (done): planting one itemset. Step 5 (done): planting many, with overlaps. Step 6, the ground truth
and the saved dataset, is in `dataset.py`.

A basket is a list of item names. Buying two packs of milk puts the milk in the list twice, so `set(basket)`
is what was bought and `Counter(basket)` is how many of each.
"""
from collections import Counter  # noqa: F401  (used in the docs above and by the notebooks)
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .catalog_data import CATEGORIES


@dataclass(frozen=True)
class Item:
    """One product the shop sells: a single SKU (one brand in one pack size)."""

    name: str  # the full name a model will read, e.g. "Surya Grains Toor Dal 1 kg"
    base: str  # the product without brand or size, e.g. "Toor Dal"
    category: str  # e.g. "Pulses"
    brand: Optional[str]  # None for loose goods such as fresh produce
    size: str  # e.g. "1 kg", "Rs 2 sachet", "dozen"


@dataclass(frozen=True)
class Planted:
    """An itemset we put into the data on purpose, so we know the right answer."""

    items: tuple  # 2 to 5 item names, e.g. ("toor dal", "rice", "sunflower oil")
    support: float  # target share of baskets that contain every one of these items
    kind: str = ""  # "natural" (really goes together) or "surprising" (an arbitrary mix), for analysis later


def _split(product):
    """A product entry is (name, weight) or (name, weight, {size: weight}); return all three parts."""
    return product if len(product) == 3 else (product[0], product[1], None)


def _all_skus(brands_per_product: int, rng) -> List[Item]:
    """Every SKU we could stock: each product, in each of its pack sizes, under a few brands.

    A shop does not stock every brand of every product. Each product gets `brands_per_product` brands picked
    from its category's brands, and a big brand is more likely to be picked than a small one, as it is in a
    real shop (fresh produce has no brands: it is sold loose).
    """
    skus = []
    for category, spec in CATEGORIES.items():
        brand_names = list(spec["brands"])
        brand_weights = np.array(list(spec["brands"].values()), dtype=float)
        for product in spec["products"]:
            base, _, own_sizes = _split(product)
            sizes = own_sizes or spec["sizes"]
            if brand_names:
                k = min(brands_per_product, len(brand_names))
                picked = rng.choice(len(brand_names), size=k, replace=False, p=brand_weights / brand_weights.sum())
                brands = [brand_names[i] for i in sorted(picked)]
            else:
                brands = [None]
            for brand in brands:
                for size in sizes:
                    name = " ".join(part for part in (brand, base, size) if part)
                    skus.append(Item(name=name, base=base, category=category, brand=brand, size=size))
    return skus


def make_catalog(n_items: int = 1000, brands_per_product: int = 4, seed: int = 0) -> List[Item]:
    """A grocery catalogue of `n_items` SKUs, the same every time for the same `seed`.

    Every product (Toor Dal, Onion, Shampoo Sachet, ...) appears at least once, so no part of the shop is
    missing; the rest of the slots are filled at random from the SKUs that were possible. The result is
    ordered by category and product, which makes it easy to read.
    """
    rng = np.random.default_rng(seed)
    pool = _all_skus(brands_per_product, rng)
    by_product = {}  # (category, product) -> positions of its SKUs in the pool
    for index, sku in enumerate(pool):
        by_product.setdefault((sku.category, sku.base), []).append(index)
    if n_items < len(by_product):
        raise ValueError(f"need at least {len(by_product)} items to cover every product, got {n_items}")
    if n_items > len(pool):
        raise ValueError(f"only {len(pool)} SKUs are possible with brands_per_product={brands_per_product}; "
                         f"ask for fewer items or raise brands_per_product")
    chosen = {int(rng.choice(indices)) for indices in by_product.values()}  # one SKU for every product
    rest = [i for i in range(len(pool)) if i not in chosen]
    chosen |= {int(i) for i in rng.choice(rest, size=n_items - len(chosen), replace=False)}
    return [pool[i] for i in sorted(chosen)]


def popularity(catalog: List[Item]) -> np.ndarray:
    """How much of the shop's sales each item makes up: one number per item, in catalogue order, adding up to 1.

    The weights are in `catalog_data.py`. A share flows down four levels, each normalised on its own:

        category -> product within the category -> brand within the product -> pack size within the brand

    Normalising at each level keeps a product's total share the same whichever brands or sizes happen to be
    stocked. There is no randomness here: the same catalogue always gives the same numbers.
    """
    category_total = sum(spec["weight"] for spec in CATEGORIES.values())
    product_weight, brand_weight, size_weight = {}, {}, {}
    for category, spec in CATEGORIES.items():
        for product in spec["products"]:
            base, weight, own_sizes = _split(product)
            product_weight[(category, base)] = weight
            size_weight[(category, base)] = own_sizes or spec["sizes"]
        brand_weight[category] = spec["brands"]

    stocked_products, stocked_brands, stocked_sizes = {}, {}, {}  # what this catalogue actually carries
    for item in catalog:
        stocked_products.setdefault(item.category, set()).add(item.base)
        stocked_brands.setdefault((item.category, item.base), set()).add(item.brand)
        stocked_sizes.setdefault((item.category, item.base, item.brand), set()).add(item.size)

    shares = []
    for item in catalog:
        spec = CATEGORIES[item.category]
        in_category = sum(product_weight[(item.category, b)] for b in stocked_products[item.category])
        share = spec["weight"] / category_total
        share *= product_weight[(item.category, item.base)] / in_category
        if item.brand is not None:
            of_product = sum(brand_weight[item.category][b] for b in stocked_brands[(item.category, item.base)])
            share *= brand_weight[item.category][item.brand] / of_product
        sizes = size_weight[(item.category, item.base)]
        of_brand = sum(sizes[z] for z in stocked_sizes[(item.category, item.base, item.brand)])
        share *= sizes[item.size] / of_brand
        shares.append(share)
    shares = np.array(shares)
    return shares / shares.sum()


def head_share(shares, fraction: float = 0.2) -> float:
    """The share of all sales made by the best-selling `fraction` of items (0.2 means the top 20%)."""
    ordered = np.sort(np.asarray(shares))[::-1]
    return float(ordered[: max(1, int(np.ceil(fraction * len(ordered))))].sum())


def basket_sizes(n_baskets: int, rng, mean: float = 8.0, dispersion: float = 4.0, minimum: int = 2) -> np.ndarray:
    """How many different products each basket holds.

    Most trips are small, but a few are big shopping trips, so the sizes are not symmetric around the mean:
    `minimum` plus a negative binomial. `mean` is the average number of different products (default 8);
    a smaller `dispersion` makes the tail heavier.
    """
    extra_mean = mean - minimum
    p = dispersion / (dispersion + extra_mean)
    return minimum + rng.negative_binomial(dispersion, p, size=n_baskets)


def draw_items(sizes, shares, rng, chunk: int = 2000) -> List[np.ndarray]:
    """Pick `sizes[i]` different things for basket i, each pick favouring the best sellers.

    This is sampling without replacement with probabilities `shares`, done with the Gumbel trick: give every
    candidate a random score of log(share) plus noise, and take the highest scores. It gives the same result
    as drawing one at a time and removing it, but handles a whole chunk of baskets at once.
    """
    n = len(shares)
    log_share = np.log(np.asarray(shares))
    baskets = []
    for start in range(0, len(sizes), chunk):
        sizes_here = np.asarray(sizes[start:start + chunk])
        top = int(sizes_here.max())
        scores = log_share + rng.gumbel(size=(len(sizes_here), n))
        best = np.argpartition(-scores, top - 1, axis=1)[:, :top]  # the `top` best scores per basket
        order = np.argsort(-np.take_along_axis(scores, best, axis=1), axis=1)
        best = np.take_along_axis(best, order, axis=1)
        baskets += [row[:k] for row, k in zip(best, sizes_here)]
    return baskets


class Products:
    """The catalogue grouped into products, so a shopper can pick a product first and a brand and size second.

    `share` is each product's share of sales (its SKUs added up). `pick_sku` takes product numbers and, for
    each, chooses one of its SKUs with the odds that SKU has within the product.
    """

    def __init__(self, catalog: List[Item], shares):
        shares = np.asarray(shares)
        by_product = {}
        for index, item in enumerate(catalog):
            by_product.setdefault((item.category, item.base), []).append(index)
        groups = [np.array(g) for g in by_product.values()]
        self.share = np.array([shares[g].sum() for g in groups])
        self._sku = np.concatenate(groups)  # SKU positions, product by product
        within = [shares[g] / shares[g].sum() for g in groups]
        # A running total of those odds, with the product number added so every product has its own 0-to-1
        # stretch. One lookup then finds the SKU for any (product, random number) pair.
        self._cdf = np.concatenate([np.cumsum(w) + k for k, w in enumerate(within)])
        self._cdf[np.cumsum([len(g) for g in groups]) - 1] = np.arange(1, len(groups) + 1)  # exact ends

    def pick_sku(self, products, rng) -> np.ndarray:
        """One SKU (as a position in the catalogue) for each product number in `products`."""
        products = np.asarray(products)
        return self._sku[np.searchsorted(self._cdf, products + rng.random(len(products)))]


def background_baskets(n_baskets: int, catalog: List[Item], shares, seed: int = 0, mean_size: float = 8.0,
                       quantity_probs=(0.80, 0.14, 0.04, 0.02)) -> List[List[str]]:
    """`n_baskets` ordinary baskets with nothing planted: items turn up only because they sell well.

    A shopper first decides what to buy: a few different products, drawn by popularity (`basket_sizes`,
    `draw_items`). Then, for each product, picks one brand and size (`Products.pick_sku`), so a basket never
    holds two brands or two sizes of the same product. Each item is bought once, twice, and so on:
    `quantity_probs` are the chances of buying 1, 2, 3 and 4. The items come out in a random order, so the
    order carries no information.
    """
    rng = np.random.default_rng(seed)
    names = [item.name for item in catalog]
    products = Products(catalog, shares)
    sizes = basket_sizes(n_baskets, rng, mean=mean_size)
    chosen = draw_items(sizes, products.share, rng)
    skus = products.pick_sku(np.concatenate(chosen), rng)
    counts = rng.choice(np.arange(1, len(quantity_probs) + 1), size=len(skus), p=quantity_probs)
    baskets, at = [], 0
    for k in sizes:
        lines = [names[i] for i, q in zip(skus[at:at + k], counts[at:at + k]) for _ in range(q)]
        baskets.append([lines[j] for j in rng.permutation(len(lines))])
        at += k
    return baskets


def find_items(catalog: List[Item], base: str, brand: Optional[str] = None, size: Optional[str] = None) -> List[Item]:
    """The SKUs of a product that this catalogue carries, optionally narrowed to one brand and/or size.

    A shop does not stock every brand and size, so to plant "Toor Dal" you look up which Toor Dal SKUs exist.
    """
    return [item for item in catalog
            if item.base == base and (brand is None or item.brand == brand) and (size is None or item.size == size)]


def support_of(baskets, items) -> float:
    """The share of baskets that contain every one of `items` (the quantity does not matter)."""
    wanted = set(items)
    return sum(wanted <= set(basket) for basket in baskets) / len(baskets)


def plant_itemset(baskets, planted: Planted, catalog: List[Item], seed: int = 0,
                  quantity_probs=(0.80, 0.14, 0.04, 0.02), allow_excess: bool = False) -> List[List[str]]:
    """Return a copy of `baskets` in which exactly `round(support x number of baskets)` baskets hold the whole
    planted itemset.

    Some baskets hold the itemset by chance already, and they count. Just enough other baskets are picked at
    random to reach the target, and the planted items are added to them. If a picked basket holds a different
    brand or size of one of the planted products, that item is swapped for the planted one, so a basket still
    never holds two SKUs of one product. The picked baskets are reshuffled so the planted items do not sit at
    the end.

    If the baskets already hold the itemset more often than the target, that is an error, because it cannot be
    planted rarer than it already is; with `allow_excess=True` the baskets are returned as they are instead.
    """
    items = list(planted.items)
    product_of = {item.name: (item.category, item.base) for item in catalog}
    unknown = [name for name in items if name not in product_of]
    if unknown:
        raise ValueError(f"not in the catalogue: {unknown}")
    if len(set(items)) != len(items) or len({product_of[name] for name in items}) != len(items):
        raise ValueError("a planted itemset needs different products: two of these are the same item or product")
    if not 0 < planted.support < 1:
        raise ValueError(f"support must be between 0 and 1, got {planted.support}")

    rng = np.random.default_rng(seed)
    wanted, planted_products = set(items), {product_of[name] for name in items}
    target = round(planted.support * len(baskets))
    holds = np.array([wanted <= set(basket) for basket in baskets])
    already = int(holds.sum())
    if already > target and allow_excess:
        return list(baskets)
    if already > target:
        raise ValueError(f"{already} baskets already hold this itemset by chance, more than the {target} asked for: "
                         f"it is too common to plant at support {planted.support}")
    picked = rng.choice(np.flatnonzero(~holds), size=target - already, replace=False)

    out = list(baskets)
    for i in picked:
        # drop any other SKU of a planted product, keep everything else (including planted items already there)
        kept = [name for name in out[i] if name in wanted or product_of[name] not in planted_products]
        have = set(kept)
        added = [name for name in items if name not in have
                 for _ in range(1 + rng.choice(len(quantity_probs), p=quantity_probs))]
        basket = kept + added
        out[i] = [basket[j] for j in rng.permutation(len(basket))]
    return out


# Products that really go together in a kitchen or a bathroom: four sets of each size. These are product names;
# the SKU of each is picked when the itemsets are built.
NATURAL = {
    2: [("Tea Powder", "Sugar"), ("Bread", "Butter"), ("Toothpaste", "Toothbrush"), ("Detergent Powder", "Dishwash Bar")],
    3: [("Toor Dal", "Sona Masoori Rice", "Sunflower Oil"), ("Turmeric Powder", "Red Chilli Powder", "Coriander Powder"),
        ("Milk", "Bread", "Eggs"), ("Shampoo Sachet", "Soap Bar", "Hair Oil")],
    4: [("Wheat Flour (Atta)", "Sugar", "Salt", "Sunflower Oil"), ("Tea Powder", "Milk", "Glucose Biscuits", "Rusk"),
        ("Instant Noodles", "Potato Chips", "Soft Drink", "Namkeen Mixture"),
        ("Dosa Batter", "Coconut Oil", "Curry Leaves", "Mustard Seeds")],
    5: [("Onion", "Tomato", "Green Chilli", "Ginger", "Garlic"),
        ("Basmati Rice", "Biryani Masala", "Ghee", "Onion", "Curd"),
        ("Cornflakes", "Milk", "Banana", "Jam", "Bread"),
        ("Detergent Bar", "Phenyl", "Floor Cleaner", "Toilet Cleaner", "Garbage Bags")],
}
SUPPORTS = (0.05, 0.01, 0.002, 0.0005)


def _index(baskets) -> dict:
    """For each item, the set of basket numbers that hold it. Lets us count how often items meet by chance."""
    index = {}
    for number, basket in enumerate(baskets):
        for name in set(basket):
            index.setdefault(name, set()).add(number)
    return index


def _chance_support(index, items, n_baskets) -> float:
    held = [index.get(name, set()) for name in items]
    return len(set.intersection(*sorted(held, key=len))) / n_baskets


def default_planted(catalog: List[Item], shares, background, seed: int = 0, supports=SUPPORTS) -> List[Planted]:
    """Thirty-two itemsets: for each of four supports, a natural and a surprising one of each size 2 to 5.

    `background` is a sample of ordinary baskets, used to check that none of these itemsets turns up by chance
    often enough to spoil the target. The rules:

    - a product keeps one SKU across every itemset, so overlaps are deliberate and one itemset can never swap
      away another one's item;
    - no itemset contains another, because a part must be at least as common as the whole;
    - chance alone must give at most 30% of the target support, so what the data holds is really planted;
    - a natural group that is common by chance gets a high support, a rarer one a low support.
    """
    rng = np.random.default_rng(seed)
    shares = np.asarray(shares)
    by_base = {}
    for position, item in enumerate(catalog):
        by_base.setdefault(item.base, []).append(position)
    category_of = {item.base: item.category for item in catalog}
    index, n = _index(background), len(background)
    chosen = {}  # product -> the SKU it uses in every itemset

    def sku_for(base):
        if base not in chosen:
            positions = np.array(by_base[base])
            chosen[base] = catalog[int(rng.choice(positions, p=shares[positions] / shares[positions].sum()))].name
        return chosen[base]

    def fits(items, support, existing):
        if _chance_support(index, items, n) > 0.3 * support:
            return False
        return not any(set(items) <= set(other.items) or set(other.items) <= set(items) for other in existing)

    planted, by_support = [], sorted(supports, reverse=True)
    for size in sorted(NATURAL, reverse=True):
        groups = [tuple(sku_for(base) for base in group) for group in NATURAL[size]]
        most_common_first = sorted(groups, key=lambda g: -_chance_support(index, g, n))
        for support, items in zip(by_support, most_common_first):
            if not fits(items, support, planted):
                raise ValueError(f"the natural set {items} is too common by chance, or overlaps another, at support {support}")
            planted.append(Planted(items, support, "natural"))
        for support in by_support:
            for _ in range(500):
                categories = list(rng.choice(sorted(set(category_of.values())), size=size, replace=False))
                bases = [str(rng.choice([b for b in by_base if category_of[b] == c])) for c in categories]
                items = tuple(sku_for(base) for base in bases)
                if fits(items, support, planted):
                    planted.append(Planted(items, support, "surprising"))
                    break
            else:
                raise ValueError(f"no surprising set of size {size} fits at support {support}")
    return sorted(planted, key=lambda p: (-p.support, len(p.items), p.kind))


def plant_all(baskets, planted: List[Planted], catalog: List[Item], seed: int = 0) -> List[List[str]]:
    """Plant every itemset in turn, the larger ones first, and return the new baskets.

    Itemsets may share items, but two rules keep the supports honest: a product must use the same SKU in every
    itemset, and no itemset may contain another (the part would have to be at least as common as the whole).
    Because itemsets overlap, planting one can complete another by accident, so the supports that come out are
    close to the targets rather than exact (an itemset already held more often than its target is left as it is):
    `planted_report` measures them.
    """
    product_of = {item.name: item.base for item in catalog}
    used = {}
    for itemset in planted:
        for name in itemset.items:
            if name not in product_of:
                raise ValueError(f"not in the catalogue: {name}")
            if used.setdefault(product_of[name], name) != name:
                raise ValueError(f"{product_of[name]} appears as both {used[product_of[name]]} and {name}")
    sets = [set(p.items) for p in planted]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j and a <= b:
                raise ValueError(f"{planted[i].items} is contained in {planted[j].items}")
    out = list(baskets)
    # Larger itemsets first: planting a small one afterwards rarely completes a big one by accident, whereas the
    # other way round it often does (tried: planting the rarest first left some itemsets 50% over their target).
    ordered = sorted(range(len(planted)), key=lambda i: (-len(planted[i].items), -planted[i].support))
    for step, i in enumerate(ordered):
        out = plant_itemset(out, planted[i], catalog, seed=seed + step, allow_excess=True)
    return out


def planted_report(baskets, planted: List[Planted]) -> List[dict]:
    """For each planted itemset: its kind, its target support, and the support the baskets really have."""
    index, n = _index(baskets), len(baskets)
    return [{"items": p.items, "kind": p.kind, "target": p.support, "realised": _chance_support(index, p.items, n)}
            for p in planted]
