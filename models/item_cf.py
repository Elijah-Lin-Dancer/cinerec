"""
Item-based Collaborative Filtering using Adjusted Cosine Similarity.
Predicts ratings based on similarity between items.

Similarity matrix is computed via vectorised matrix multiplications
instead of a double Python loop, which is orders of magnitude faster
for MovieLens-100K (1 682 items).
"""
import numpy as np
from models.base import Recommender


class ItemCF(Recommender):
    """Item-based Collaborative Filtering with adjusted cosine similarity."""

    def __init__(self, k=50, min_common=2):
        super().__init__()
        self.k = k
        self.min_common = min_common
        self.user_item_matrix = None
        self.sim_matrix = None  # (num_items, num_items) item similarity
        self.user_mean = None
        self.num_users = 0
        self.num_items = 0

    def fit(self, train_data):
        """Build user-item matrix and precompute item-item similarity."""
        user_ids = train_data['user_id']
        item_ids = train_data['item_id']
        ratings = train_data['rating']

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1

        self.user_item_matrix = np.zeros((self.num_users, self.num_items))
        self.user_item_matrix[user_ids.astype(int), item_ids.astype(int)] = ratings

        self.user_mean = np.zeros(self.num_users)
        for u in range(self.num_users):
            rated = self.user_item_matrix[u][self.user_item_matrix[u] > 0]
            self.user_mean[u] = rated.mean() if len(rated) > 0 else 3.0

        # --- Vectorised adjusted cosine similarity (fast path) ---
        # Binary mask of rated entries
        mask = (self.user_item_matrix > 0).astype(np.float64)

        # Co-rating counts between items: (num_items, num_items)
        common_count = mask.T @ mask

        # Center ratings per user (zero where unrated)
        centered = self.user_item_matrix.copy()
        for u in range(self.num_users):
            m = centered[u] > 0
            centered[u][m] -= self.user_mean[u]
            centered[u][~m] = 0.0

        # Numerator & denominator via matrix products
        # centered.T @ centered gives (num_items, num_items) dot products
        numerator = centered.T @ centered
        norms = np.sqrt(np.sum(centered ** 2, axis=0))  # (num_items,)
        denom = np.outer(norms, norms)

        with np.errstate(invalid='ignore', divide='ignore'):
            sim = np.where(denom > 0, numerator / denom, 0.0)

        # Zero out pairs without enough common raters
        sim[common_count < self.min_common] = 0.0
        np.fill_diagonal(sim, 0.0)  # exclude self-similarity

        self.sim_matrix = sim

        # Keep top-K similar items per item
        for i in range(self.num_items):
            top_k_idx = np.argsort(np.abs(self.sim_matrix[i]))[-self.k:]
            row = np.zeros(self.num_items)
            row[top_k_idx] = self.sim_matrix[i][top_k_idx]
            self.sim_matrix[i] = row

        print(f"ItemCF fitted: {self.num_users} users, {self.num_items} items, k={self.k}")

    def predict(self, user_id, item_id):
        """Predict rating using weighted sum of item similarities."""
        user_id = int(user_id)
        item_id = int(item_id)
        if user_id < 0 or item_id < 0:
            return 3.0
        if user_id >= self.num_users or item_id >= self.num_items:
            return 3.0

        user_ratings = self.user_item_matrix[user_id]
        sim = self.sim_matrix[item_id]

        mask = (sim != 0) & (user_ratings > 0)
        if not mask.any():
            return self.user_mean[user_id]

        sim_vals = sim[mask]
        ratings_vals = user_ratings[mask]
        denom = np.sum(np.abs(sim_vals))
        if denom < 1e-9:
            return self.user_mean[user_id]

        return np.dot(sim_vals, ratings_vals) / denom

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Recommend top-K items for a user."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or user_id < 0 or user_id >= self.num_users:
            return []
        if exclude_items is None:
            exclude_items = set()

        rated = set(np.where(self.user_item_matrix[user_id] > 0)[0])
        candidates = [i for i in range(self.num_items)
                      if i not in rated and i not in exclude_items]

        if not candidates:
            return []

        scores = [(i, self.predict(user_id, i)) for i in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
