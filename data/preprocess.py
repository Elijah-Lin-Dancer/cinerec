"""
Feature Engineering — Encode movie content features for recommendation models.
- Text: Sentence-BERT (all-MiniLM-L6-v2) → 384-dim embeddings
- Image: ResNet-50 → 2048-dim features (needs poster URLs)
- Genre: Multi-hot encoding → 18-dim vectors

Arrays are **indexed by raw movie id**: row ``i`` holds the features of the movie
whose id is ``i`` (shape ``(max_id + 1, dim)``). This lets models index content
features directly with ``item_id`` without an extra mapping table.
"""
import os, json, numpy as np, pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Prefer a reachable mirror for Hugging Face model weights (the default host is
# blocked in some sandboxes). Only sets a default — an explicit env var wins.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
NUM_GENRES = len(GENRE_LIST)


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
    """Multi-hot encode genre strings → 18-dim vectors (id-aligned)."""
    result = _empty(max(item_ids) + 1, NUM_GENRES)
    for i, gs in enumerate(genre_strings):
        if gs is None or (isinstance(gs, float) and pd.isna(gs)) or not gs:
            continue
        for g in str(gs).split("|"):
            g = g.strip()
            if g in GENRE_LIST:
                result[item_ids[i], GENRE_LIST.index(g)] = 1.0

    out_path = os.path.join(PROCESSED_DIR, "genre_vectors.npy")
    np.save(out_path, result)
    print(f"Genre vectors saved: {result.shape} (id-aligned) → {out_path}")
    return result


def encode_images(poster_urls, item_ids):
    """Extract ResNet-50 features from poster images → 2048-dim (id-aligned)."""
    import torch
    from torchvision import models, transforms
    from PIL import Image
    import requests
    from io import BytesIO

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
    count = 0

    for i, url in enumerate(poster_urls):
        if not url or "http" not in url:
            continue
        try:
            resp = requests.get(url, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img_t = transform(img).unsqueeze(0)
            with torch.no_grad():
                feat = resnet(img_t).squeeze().numpy()
            features[item_ids[i]] = feat
            count += 1
            if count % 50 == 0:
                print(f"Encoded {count} images...")
        except Exception:
            continue

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
    has_posters = sum(1 for m in movies if m.get("poster_url") and "http" in m.get("poster_url", ""))
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