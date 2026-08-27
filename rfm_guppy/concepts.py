"""Phase 2: Build labeled concept datasets from GuppyLM's own training categories.

Three binary concepts chosen for semantic separability and lexical concreteness:
  C1 food      : food+taste  vs.  weather+seasons+tv+music
  C2 valence   : excited+happy  vs.  scared+fear
  C3 env_social: water+filter+temp_hot+temp_cold+algae  vs.  greeting+bye+friends+visitors+lonely

NOTE on prompt choice: The `input` field (user message) has only 4–34 unique texts per
category (heavily templated), which is insufficient for independent train/val/test splits.
Instead we use the full ChatML exchange (user + Guppy response) as the prompt:

    <|im_start|>user\\n{input}<|im_end|>\\n<|im_start|>assistant\\n{output}<|im_end|>

This gives hundreds of unique texts per category (the Guppy responses are template-composed
and highly diverse). The last-token activation after the complete exchange encodes the concept
most strongly. Pass these prompts to extract_activations with format_chatml=False.
"""

import json
import os
import random
from collections import Counter

try:
    from datasets import load_dataset as hf_load
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

CONCEPTS = {
    # C1: food/eating topics vs. abstract inanimate topics
    # pos: food(919) + taste(40) ≈ 959 unique exchanges
    # neg: weather(50)+seasons(50)+tv(40)+music(40)+glass(50)+rain(40)+outside(40) = 310 unique
    # → n_per_class = 300
    "food": {
        "pos": ["food", "taste"],
        "neg": ["weather", "seasons", "tv", "music", "glass", "rain", "outside"],
        "description": "food/eating topics (pos) vs. abstract inanimate topics (neg)",
        "n_per_class": 300,
    },
    # C2: positive vs. negative valence
    # pos: happy(40)+excited(40)+love(40)+curious(40) = 160 unique
    # neg: scared(40)+fear(40)+bored(40)+tired(40) = 160 unique
    # → n_per_class = 100
    "valence": {
        "pos": ["happy", "excited", "love", "curious"],
        "neg": ["scared", "fear", "bored", "tired"],
        "description": "positive arousal (happy/excited/love/curious) vs. negative (scared/fear/bored/tired)",
        "n_per_class": 100,
    },
    # C3: tank/water environment topics vs. social interaction topics
    # pos: water(932)+filter(50)+temp_hot(868)+temp_cold(940)+algae(40) ≈ 2830 unique
    # neg: greeting(930)+bye(818)+friends(40)+visitors(40)+lonely(928) ≈ 2756 unique
    # → n_per_class = 300
    "env_social": {
        "pos": ["water", "filter", "temp_hot", "temp_cold", "algae"],
        "neg": ["greeting", "bye", "friends", "visitors", "lonely"],
        "description": "tank/environment topics (pos) vs. social interaction topics (neg)",
        "n_per_class": 300,
    },
}

# Small keyword lexicons for Phase 6 (steering evaluation)
CONCEPT_LEXICONS = {
    "food": ["eat", "food", "hungry", "flakes", "bite", "nibble", "pellet",
             "feed", "feeding", "meal", "snack", "bloodworm", "brine"],
    "valence": ["excited", "happy", "joy", "wiggle", "fast", "scared", "afraid",
                "fear", "nervous", "tense", "stiff", "hide"],
    "env_social": ["water", "filter", "temperature", "warm", "cold", "oxygen",
                   "clean", "cloudy", "algae", "hello", "hi", "friend", "bye"],
}

_CHATML_TEMPLATE = (
    "<|im_start|>user\n{inp}<|im_end|>\n"
    "<|im_start|>assistant\n{out}<|im_end|>"
)


def _format_exchange(row: dict) -> str:
    """Format a dataset row as a complete ChatML exchange string."""
    return _CHATML_TEMPLATE.format(inp=row["input"], out=row["output"])


def load_raw_dataset() -> list[dict]:
    """Load the full guppylm-60k-generic dataset; fall back to local data."""
    if _HF_AVAILABLE:
        try:
            ds = hf_load("arman-bd/guppylm-60k-generic", split="train")
            rows = [{"input": r["input"], "output": r["output"], "category": r["category"]}
                    for r in ds]
            print(f"[concepts] Loaded {len(rows)} rows from HuggingFace.")
            return rows
        except Exception as e:
            print(f"[concepts] HF load failed ({e}), trying local data.")

    local_path = os.path.join(os.path.dirname(__file__), "..", "guppylm", "data", "train.jsonl")
    if os.path.exists(local_path):
        rows = []
        with open(local_path) as f:
            for line in f:
                obj = json.loads(line)
                rows.append(obj)
        print(f"[concepts] Loaded {len(rows)} rows from local {local_path}.")
        return rows

    raise FileNotFoundError(
        "Could not load dataset from HuggingFace or local path. "
        "Run `guppylm generate_data.py` to generate local data first."
    )


def print_category_inventory(rows: list[dict]) -> Counter:
    """Print and return per-category counts (total rows and unique exchanges)."""
    cats_total = Counter(r["category"] for r in rows)
    cats_unique: dict[str, set] = {}
    for r in rows:
        c = r["category"]
        if c not in cats_unique:
            cats_unique[c] = set()
        cats_unique[c].add(_format_exchange(r))

    print(f"\n[concepts] Category inventory ({len(cats_total)} categories, {len(rows)} total rows):")
    print(f"  {'category':20s}  {'total':>6}  {'unique_exchanges':>16}")
    for cat in sorted(cats_total, key=lambda x: -cats_total[x]):
        print(f"  {cat:20s}  {cats_total[cat]:6d}  {len(cats_unique[cat]):16d}")
    print(f"  {'TOTAL':20s}  {sum(cats_total.values()):6d}")
    return cats_total


def build_concept_dataset(
    name: str,
    pos_categories: list[str],
    neg_categories: list[str],
    n_per_class: int,
    rows: list[dict] | None = None,
    seed: int = 42,
) -> dict:
    """Build a balanced binary concept dataset with train/val/test splits.

    Prompts are full ChatML exchanges (user + Guppy response). Pass to
    extract_activations with format_chatml=False.

    Returns:
      {
        "name": str,
        "pos_categories": [...],
        "neg_categories": [...],
        "n_per_class": int,
        "format_chatml": False,   # reminder flag for downstream code
        "train": [{"prompt": str, "label": int, "category": str}, ...],
        "val":   [...],
        "test":  [...],
      }
    Labels: 1 = positive class, 0 = negative class.
    Splits: 60% train / 20% val / 20% test — unique exchanges, no leakage across splits.
    """
    if rows is None:
        rows = load_raw_dataset()

    rng = random.Random(seed)

    def collect_unique(categories: list[str]) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        candidates = [r for r in rows if r["category"] in categories]
        rng.shuffle(candidates)
        for r in candidates:
            key = _format_exchange(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    pos_rows = collect_unique(pos_categories)
    neg_rows = collect_unique(neg_categories)

    avail_pos = len(pos_rows)
    avail_neg = len(neg_rows)
    actual_n = min(n_per_class, avail_pos, avail_neg)
    if actual_n < 50:
        raise ValueError(
            f"[{name}] Too few unique exchanges: pos={avail_pos}, neg={avail_neg}. "
            f"Need at least 50 per class."
        )
    if actual_n < n_per_class:
        print(f"[concepts] {name}: requested {n_per_class}/class but only "
              f"{actual_n} unique exchanges available — using {actual_n}.")
    n_per_class = actual_n

    pos_rows = pos_rows[:n_per_class]
    neg_rows = neg_rows[:n_per_class]

    def split_and_label(samples: list[dict], label: int):
        n = len(samples)
        n_val  = max(1, int(n * 0.20))
        n_test = max(1, int(n * 0.20))
        n_train = n - n_val - n_test
        make = lambda lst: [
            {"prompt": _format_exchange(r), "label": label, "category": r["category"]}
            for r in lst
        ]
        return make(samples[:n_train]), make(samples[n_train:n_train + n_val]), make(samples[n_train + n_val:])

    pos_train, pos_val, pos_test = split_and_label(pos_rows, 1)
    neg_train, neg_val, neg_test = split_and_label(neg_rows, 0)

    def combine_shuffle(*parts):
        merged = []
        for p in parts:
            merged.extend(p)
        rng.shuffle(merged)
        return merged

    dataset = {
        "name": name,
        "pos_categories": pos_categories,
        "neg_categories": neg_categories,
        "n_per_class": n_per_class,
        "format_chatml": False,
        "train": combine_shuffle(pos_train, neg_train),
        "val":   combine_shuffle(pos_val, neg_val),
        "test":  combine_shuffle(pos_test, neg_test),
    }

    # Verify no prompt leakage across splits
    train_p = {s["prompt"] for s in dataset["train"]}
    val_p   = {s["prompt"] for s in dataset["val"]}
    test_p  = {s["prompt"] for s in dataset["test"]}
    assert not (train_p & val_p),  f"[{name}] Train/val prompt overlap!"
    assert not (train_p & test_p), f"[{name}] Train/test prompt overlap!"
    assert not (val_p & test_p),   f"[{name}] Val/test prompt overlap!"

    n_tr = len(dataset["train"])
    n_v  = len(dataset["val"])
    n_te = len(dataset["test"])
    print(
        f"[concepts] {name}: {n_per_class}/class unique exchanges → "
        f"train={n_tr}, val={n_v}, test={n_te} (no leakage confirmed)"
    )
    return dataset


def save_concept_dataset(dataset: dict) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"concept_{dataset['name']}.json")
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    return path


def load_concept_dataset(name: str) -> dict:
    path = os.path.join(CACHE_DIR, f"concept_{name}.json")
    with open(path) as f:
        return json.load(f)


def build_all_concepts(seed: int = 42) -> dict[str, dict]:
    """Build and cache all three concept datasets. Returns {name: dataset}."""
    rows = load_raw_dataset()
    print_category_inventory(rows)

    datasets = {}
    for name, spec in CONCEPTS.items():
        ds = build_concept_dataset(
            name=name,
            pos_categories=spec["pos"],
            neg_categories=spec["neg"],
            n_per_class=spec["n_per_class"],
            rows=rows,
            seed=seed,
        )
        path = save_concept_dataset(ds)
        datasets[name] = ds
        print(f"[concepts] Saved {name} → {path}")

    return datasets


if __name__ == "__main__":
    datasets = build_all_concepts()

    print("\n[concepts] All concept datasets built and saved.")
    print(f"\n{'Concept':12s}  {'N/class':>8}  {'train':>6}  {'val':>5}  {'test':>5}")
    print("-" * 48)
    for name, ds in datasets.items():
        tr = ds["train"]
        pos_tr = sum(1 for s in tr if s["label"] == 1)
        neg_tr = sum(1 for s in tr if s["label"] == 0)
        print(
            f"{name:12s}  {ds['n_per_class']:>8d}  "
            f"{len(ds['train']):>6d}  {len(ds['val']):>5d}  {len(ds['test']):>5d}"
            f"  (train pos={pos_tr}, neg={neg_tr})"
        )
