"""
Feature Engineering — Encode movie content features for recommendation models.
- Text: Sentence-BERT (all-MiniLM-L6-v2) → 384-dim embeddings
- Image: ResNet-50 → 2048-dim features (needs poster URLs)
- Genre: Multi-hot encoding → 18-dim vectors

Arrays are **indexed by raw movie id**: row ``i`` holds the features of the movie
whose id is ``i`` (shape ``(max_id + 1, dim)``). This lets models index content
features directly with ``item_id`` without an extra mapping table.
"""
import os, sys, json, numpy as np, pandas as pd
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config import RAW_DIR, PROCESSED_DIR  # noqa: E402  (dataset-aware dirs)

os.makedirs(PROCESSED_DIR, exist_ok=True)

# NOTE: model weights are fetched from the default Hugging Face endpoint. If that
# host is unreachable (e.g. behind a restrictive network), export HF_ENDPOINT to a
# trusted mirror yourself — the pipeline deliberately does not hard-code one.

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
NUM_GENRES = len(GENRE_LIST)


def _load_vocab():
    """Categorical vocabulary for the active dataset.

    MovieLens uses the fixed 18-genre list above. A second dataset may drop a
    ``category_vocab.json`` (a JSON list of category names) in its processed dir
    to define its own vocabulary. It must still be ``NUM_GENRES`` long, because
    the content tower is fixed at that width — the check makes a mismatched
    vocabulary fail loudly instead of silently mis-shaping the array.
    """
    path = os.path.join(PROCESSED_DIR, "category_vocab.json")
    if not os.path.exists(path):
        return GENRE_LIST
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    if len(vocab) != NUM_GENRES:
        raise ValueError(
            f"{path} has {len(vocab)} categories but the content tower is fixed "
            f"at {NUM_GENRES}; trim the vocabulary to {NUM_GENRES} entries."
        )
    return list(vocab)


def load_enriched_movies():
    """Load enriched movie metadata."""
    path = os.path.join(PROCESSED_DIR, "movies_enriched.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run enrich_tmdb.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    return sorted(movies, key=lambda x: x["id"])


def _empty(num_rows, dim):
    return np.zeros((num_rows, dim), dtype=np.float32)


def _is_fetchable_url(url):
    """Only http(s) URLs with a host are fetched — rejects file:// and bare paths."""
    parsed = urlparse(str(url or ""))
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def encode_texts(overviews, item_ids):
    """Encode movie overviews using Sentence-BERT all-MiniLM-L6-v2 → 384-dim."""
    from sentence_transformers import SentenceTransformer

    num_items = max(item_ids) + 1
    model = SentenceTransformer("all-MiniLM-L6-v2")

    valid_idx = [i for i, t in enumerate(overviews) if t and len(str(t).strip()) > 10]
    valid_texts = [overviews[i] for i in valid_idx]

    result = _empty(num_items, 384)
    if not valid_texts:
        print("No valid overviews found. Creating zero embeddings.")
    else:
        print(f"Encoding {len(valid_texts)} overviews with Sentence-BERT...")
        embeddings = model.encode(valid_texts, show_progress_bar=True, batch_size=128)
        for i, emb in zip(valid_idx, embeddings):
            result[item_ids[i]] = emb

    out_path = os.path.join(PROCESSED_DIR, "text_embeddings.npy")
    np.save(out_path, result)
    print(f"Text embeddings saved: {result.shape} (id-aligned) → {out_path}")
    return result


def encode_genres(genre_strings, item_ids):
    """Multi-hot encode category strings → 18-dim vectors (id-aligned)."""
    vocab = _load_vocab()
    result = _empty(max(item_ids) + 1, NUM_GENRES)
    for i, gs in enumerate(genre_strings):
        if gs is None or (isinstance(gs, float) and pd.isna(gs)) or not gs:
            continue
        for g in str(gs).split("|"):
            g = g.strip()
            if g in vocab:
                result[item_ids[i], vocab.index(g)] = 1.0

    out_path = os.path.join(PROCESSED_DIR, "genre_vectors.npy")
    np.save(out_path, result)
    print(f"Genre vectors saved: {result.shape} (id-aligned) → {out_path}")
    return result


def encode_images(poster_urls, item_ids, batch_size=64, num_workers=8, chunk_size=256):
    """Extract ResNet-50 features from poster images → 2048-dim (id-aligned).

    On a large catalogue the per-image sequential path is the bottleneck, so the
    two independent costs are overlapped: posters are fetched concurrently (I/O
    bound) and pushed through the frozen backbone in batches (compute bound).

    Fetches run in bounded *chunks*: when downloads outpace inference, an
    unbounded ``map`` would hold every completed-but-unconsumed decoded image in
    memory at once (Amazon covers are large) and the process gets OOM-killed on
    a small host. Chunking caps the in-flight window.
    """
    import gc

    import torch
    from torchvision import models, transforms
    from PIL import Image
    import requests
    from io import BytesIO
    from concurrent.futures import ThreadPoolExecutor

    torch.set_num_threads(max(1, os.cpu_count() or 1))

    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    num_items = max(item_ids) + 1
    features = _empty(num_items, 2048)

    targets = [(item_ids[i], url) for i, url in enumerate(poster_urls)
               if _is_fetchable_url(url)]

    def _fetch(url):
        try:
            resp = requests.get(url, timeout=(10, 30))
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None

    def _flush(batch_imgs, batch_ids):
        if not batch_imgs:
            return 0
        tensor = torch.stack([transform(img) for img in batch_imgs])
        with torch.no_grad():
            feats = resnet(tensor).squeeze(-1).squeeze(-1).numpy()
        for iid, feat in zip(batch_ids, feats):
            features[iid] = feat
        return len(batch_imgs)

    count = 0
    for start in range(0, len(targets), chunk_size):
        chunk = targets[start:start + chunk_size]
        batch_imgs, batch_ids = [], []
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for (iid, _url), img in zip(chunk, pool.map(_fetch, [u for _, u in chunk])):
                if img is None:
                    continue
                batch_imgs.append(img)
                batch_ids.append(iid)
                if len(batch_imgs) >= batch_size:
                    count += _flush(batch_imgs, batch_ids)
                    batch_imgs, batch_ids = [], []
        count += _flush(batch_imgs, batch_ids)
        gc.collect()
        print(f"Encoded {count}/{len(targets)} images (chunk {start // chunk_size + 1})...")

    out_path = os.path.join(PROCESSED_DIR, "image_embeddings.npy")
    np.save(out_path, features)
    print(f"Image embeddings saved: {features.shape} ({count} success, id-aligned) → {out_path}")
    return features


def preprocess_all(skip_images=False):
    """Run all feature engineering steps (features are id-aligned)."""
    movies = load_enriched_movies()
    print(f"Loaded {len(movies)} movies (ids {min(m['id'] for m in movies)}–{max(m['id'] for m in movies)}).")

    item_ids = [m["id"] for m in movies]

    # Genre encoding
    encode_genres([m.get("genres", "") for m in movies], item_ids)

    # Text encoding
    encode_texts([m.get("overview", "") for m in movies], item_ids)

    # Image encoding (default on when posters are available)
    has_posters = sum(1 for m in movies if _is_fetchable_url(m.get("poster_url")))
    if skip_images or has_posters == 0:
        print("Skipping image encoding (no poster URLs or skip_images=True).")
        np.save(
            os.path.join(PROCESSED_DIR, "image_embeddings.npy"),
            _empty(max(item_ids) + 1, 2048)
        )
        print("Created zero image embeddings placeholder.")
    else:
        print(f"Encoding {has_posters} poster images with ResNet-50...")
        encode_images([m.get("poster_url", "") for m in movies], item_ids)

    print("\n=== Feature engineering complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-images", dest="images", action="store_false",
                        help="Skip poster image encoding (fast, creates zero vectors)")
    parser.set_defaults(images=True)
    args = parser.parse_args()
    preprocess_all(skip_images=not args.images)