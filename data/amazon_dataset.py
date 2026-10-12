"""Build the Amazon Video_Games second dataset (cross-dataset evaluation).

Downloads the Amazon-Reviews-2023 "5-core" Video_Games ratings + item metadata
from the Hugging Face mirror and converts them into the *exact* schema the
MovieLens pipeline already uses, so every model, the split and the content
feature encoders run unchanged:

  raw/ratings.csv                    (user_id, item_id, rating, timestamp)
  processed/movies_enriched.json     (id, title, genres, release_year, overview, poster_url, tmdb_id)
  processed/category_vocab.json      (18-way categorical vocabulary)

Item and user ids are contiguous and assigned by a *sorted* mapping, so the
split, the sampled negatives and the content arrays are reproducible run to run
— the same guarantee the MovieLens pipeline gives.

Run with the dataset selected, e.g.::

    CINEREC_DATASET=amazon-video-games python data/amazon_dataset.py
"""
import json
import os
import re
import sys
from collections import Counter
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from config import RAW_DIR, PROCESSED_DIR  # noqa: E402

#: The full Video_Games 5-core split (95k users × 26k items) is dominated by
#: light users (median ~6 ratings), so a naive user sample would leave almost no
#: training data after the leave-last-5-out split. Instead the slice is defined
#: by a *population criterion*, mirroring the one MovieLens-100K itself uses:
#: keep users with at least ``MIN_USER_INTER`` ratings (ML-100K guarantees ≥20)
#: and re-apply the 5-core rule to items. No random sampling is involved, so the
#: subset is deterministic by construction. That leaves ~2.3k heavy users ×
#: ~4.8k items × ~76k ratings — every user keeps ≥15 training interactions and
#: all eight models fit comfortably in a small host's memory. ``MAX_USERS`` can
#: additionally cap the user count (0 = no cap) on constrained machines.
MAX_USERS = int(os.environ.get("AMAZON_MAX_USERS", "0"))
MIN_USER_INTER = int(os.environ.get("AMAZON_MIN_USER_INTER", "20"))
MIN_ITEM_INTER = 5


def _apply_slice(ratings, max_users=MAX_USERS, min_user=MIN_USER_INTER,
                 min_item=MIN_ITEM_INTER, seed=42):
    """Apply the heavy-user criterion and re-establish the item 5-core.

    User and item thresholds are decoupled (a shared threshold death-spirals:
    filtering on one shrinks the other below it and the cascade empties the
    data). With ``max_users=0`` no sampling happens at all and the result is
    a pure deterministic filter of the full benchmark.
    """
    if max_users and ratings["user_id"].nunique() > max_users:
        rng = np.random.default_rng(seed)
        users = np.sort(ratings["user_id"].unique())
        chosen = rng.choice(users, size=max_users, replace=False)
        ratings = ratings[ratings["user_id"].isin(chosen)]

    while True:
        before = len(ratings)
        uc = ratings.groupby("user_id").size()
        ratings = ratings[ratings["user_id"].isin(uc[uc >= min_user].index)]
        ic = ratings.groupby("parent_asin").size()
        ratings = ratings[ratings["parent_asin"].isin(ic[ic >= min_item].index)]
        if len(ratings) == before:
            break
    return ratings


MIRROR = "https://hf-mirror.com"
REPO = "McAuley-Lab/Amazon-Reviews-2023"
RATINGS_URL = f"{MIRROR}/datasets/{REPO}/resolve/main/benchmark/5core/rating_only/Video_Games.csv"
META_URL = f"{MIRROR}/datasets/{REPO}/resolve/main/raw/meta_categories/meta_Video_Games.jsonl"
_ALLOWED_HOSTS = {"hf-mirror.com"}

#: The content tower is fixed at 18 categorical slots; the Amazon vocabulary is
#: the 18 most frequent category tags (see ``_category_vocab``).
VOCAB_SIZE = 18


def _validated_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"Refusing to download from untrusted URL: {url}")
    return url


def _download(url, dest):
    """Stream ``url`` to ``dest`` unless it already exists (idempotent)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"Using cached {dest} ({os.path.getsize(dest)} bytes)")
        return dest
    url = _validated_url(url)
    print(f"Downloading {url} …")
    r = requests.get(url, stream=True, timeout=(10, 300))
    r.raise_for_status()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    print(f"Saved {dest} ({os.path.getsize(dest)} bytes)")
    return dest


def _category_vocab(meta_by_item, item_ids):
    """Top-``VOCAB_SIZE`` category tags by document frequency.

    Tags are drawn from the first two category levels below the generic
    ``Video Games`` root (e.g. ``PC``, ``PlayStation 4``, ``Games``,
    ``Accessories``), which is the natural Amazon analogue of MovieLens genres.
    """
    counts = Counter()
    for iid in item_ids:
        for tag in _item_tags(meta_by_item.get(iid, {})):
            counts[tag] += 1
    return [tag for tag, _ in counts.most_common(VOCAB_SIZE)]


def _item_tags(meta):
    cats = meta.get("categories") or []
    # Drop the top-level "Video Games" root; keep the platform + type levels.
    return [c for c in cats[1:3] if c]


def _overview(meta):
    parts = [meta.get("title") or ""]
    parts += [str(f) for f in (meta.get("features") or [])]
    parts += [str(d) for d in (meta.get("description") or [])]
    return " ".join(p.strip() for p in parts if p and str(p).strip())


def _poster_url(meta):
    """The item's main image, size-capped for the feature pipeline.

    ResNet-50 sees a 224×224 resize, so the full-resolution cover (which can be
    several MB of hi-res art) is pure download cost. Amazon's image CDN accepts
    a ``._SX<n>_`` size modifier: injecting it (when the URL has no modifier
    yet) caps the worst case at a 500px-wide JPEG while leaving already-sized
    or small images untouched.
    """
    for img in meta.get("images") or []:
        url = img.get("large") or img.get("hi_res")
        if url:
            url = str(url)
            if "._S" not in url and url.lower().endswith((".jpg", ".jpeg", ".png")):
                stem, dot, ext = url.rpartition(".")
                url = f"{stem}._SX500_.{ext}"
            return url
    return ""


def _release_year(meta):
    details = meta.get("details") or {}
    for key in ("Date First Available", "Release Date"):
        raw = details.get(key)
        if raw:
            m = re.search(r"(\d{4})", str(raw))
            if m:
                return int(m.group(1))
    return None


def build():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ratings_path = _download(RATINGS_URL, os.path.join(RAW_DIR, "Video_Games.csv"))
    meta_path = _download(META_URL, os.path.join(RAW_DIR, "meta_Video_Games.jsonl"))

    ratings = pd.read_csv(ratings_path)
    print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users, "
          f"{ratings['parent_asin'].nunique()} items")

    # Heavy-user slice (deterministic population filter, item 5-core) — see
    # MIN_USER_INTER / MAX_USERS.
    ratings = _apply_slice(ratings)
    print(f"Heavy-user slice: {len(ratings)} ratings, "
          f"{ratings['user_id'].nunique()} users, {ratings['parent_asin'].nunique()} items")

    # Deterministic, contiguous id assignment (sorted) — pins the split and
    # makes the content arrays index-compatible with the model item ids.
    users = sorted(ratings["user_id"].unique())
    items = sorted(ratings["parent_asin"].unique())
    user_to_id = {u: i for i, u in enumerate(users)}
    item_to_id = {a: i for i, a in enumerate(items)}

    out = pd.DataFrame({
        "user_id": ratings["user_id"].map(user_to_id).astype("int64"),
        "item_id": ratings["parent_asin"].map(item_to_id).astype("int64"),
        "rating": ratings["rating"].astype(float),
        "timestamp": ratings["timestamp"].astype("int64"),
    }).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    out.to_csv(os.path.join(RAW_DIR, "ratings.csv"), index=False)
    print(f"Wrote ratings.csv ({len(out)} rows, {len(users)} users, {len(items)} items)")

    # Stream the (large) metadata file, keeping only items present in ratings.
    wanted = set(items)
    meta_by_item = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = rec.get("parent_asin")
            if asin in wanted:
                meta_by_item[asin] = rec
    print(f"Matched metadata for {len(meta_by_item)}/{len(items)} items")

    vocab = _category_vocab(meta_by_item, items)
    missing_tags = sum(
        1 for a in items
        if not any(t in vocab for t in _item_tags(meta_by_item.get(a, {})))
    )
    print(f"Category vocabulary ({len(vocab)} tags): {vocab}")
    print(f"Items with no in-vocab tag: {missing_tags}")

    movies = []
    n_img = n_txt = 0
    for a in items:
        meta = meta_by_item.get(a, {})
        tags = [t for t in _item_tags(meta) if t in vocab]
        poster = _poster_url(meta)
        overview = _overview(meta)
        n_img += bool(poster)
        n_txt += bool(overview)
        movies.append({
            "id": item_to_id[a],
            "title": meta.get("title") or f"Item {a}",
            "genres": "|".join(tags),
            "release_year": _release_year(meta),
            "overview": overview,
            "poster_url": poster,
            "tmdb_id": "",
        })
    print(f"Items with poster URL: {n_img}/{len(items)}; with text: {n_txt}/{len(items)}")

    with open(os.path.join(PROCESSED_DIR, "movies_enriched.json"), "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False)
    with open(os.path.join(PROCESSED_DIR, "category_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Wrote movies_enriched.json + category_vocab.json → {PROCESSED_DIR}")


if __name__ == "__main__":
    build()
