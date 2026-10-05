"""
User-based Collaborative Filtering using Pearson correlation.
Finds K similar users and aggregates their ratings.

Similarity matrix is computed via vectorised Pearson correlation
instead of a double Python loop, which is orders of magnitude faster
for MovieLens-100K (943 users).
"""
import numpy as np
from models.base import Recommender


class UserCF(Recommender):
    """User-based Collaborative Filtering with Pearson correlation similarity."""

    def __init__(self, k=50, min_common=2):
        super().__init__()
        self.k = k  # number of similar users
        self.min_common = min_common  # minimum common items for similarity
        self.user_item_matrix = None  # (num_users, num_items) dense
        self.sim_matrix = None  # (num_users, num_users) similarity
        self.user_mean = None  # per-user mean rating
        self.num_users = 0
        self.num_items = 0

    def fit(self, train_data):
        """Build user-item matrix and precompute user-user similarity."""
        user_ids = train_data['user_id']
        item_ids = train_data['item_id']
        ratings = train_data['rating']

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1

        # Build dense user-item matrix
        self.user_item_matrix = np.zeros((self.num_users, self.num_items))
        self.user_item_matrix[user_ids.astype(int), item_ids.astype(int)] = ratings

        # Compute user mean (only over rated items)
        self.user_mean = np.zeros(self.num_users)
        for u in range(self.num_users):
            rated = self.user_item_matrix[u][self.user_item_matrix[u] > 0]
            if len(rated) > 0:
                self.user_mean[u] = rated.mean()
            else:
                self.user_mean[u] = 3.0  # global default

        # --- Vectorised Pearson correlation (fast path) ---
        # Build binary mask of rated items
        mask = (self.user_item_matrix > 0).astype(np.float64)

        # Co-rating counts: (num_users, num_users)
        common_count = mask @ mask.T

        # Centered ratings (zero where unrated)
        centered = self.user_item_matrix.copy()
        for u in range(self.num_users):
            m = centered[u] > 0
            centered[u][m] -= self.user_mean[u]
            centered[u][~m] = 0.0

        # Numerator: sum of element-wise products over common items
        # Because centered zeros out unrated entries, this is exactly
        # the Pearson numerator for every user pair.
        numerator = centered @ centered.T

        # Denominator: norm_u * norm_v for each pair
        norms = np.sqrt(np.sum(centered ** 2, axis=1))  # (num_users,)
        denom = np.outer(norms, norms)

        # Pearson correlation (NaN-safe division)
        with np.errstate(invalid='ignore', divide='ignore'):
            sim = np.where(denom > 0, numerator / denom, 0.0)

        # Zero out pairs that don't have enough common items
        sim[common_count < self.min_common] = 0.0
        np.fill_diagonal(sim, 0.0)  # exclude self-similarity

        self.sim_matrix = sim

        # Keep top-K similar users per user
        for u in range(self.num_users):
            top_k_idx = np.argsort(np.abs(self.sim_matrix[u]))[-self.k:]
            row = np.zeros(self.num_users)
            row[top_k_idx] = self.sim_matrix[u][top_k_idx]
            self.sim_matrix[u] = row

        print(f"UserCF fitted: {self.num_users} users, {self.num_items} items, k={self.k}")

    def predict(self, user_id, item_id):
        """Predict rating for user-item pair using weighted sum."""
        user_id = int(user_id)
        item_id = int(item_id)
        if user_id < 0 or item_id < 0:
            return 3.0  # global default for malformed indices
        if user_id >= self.num_users or item_id >= self.num_items:
            return self.user_mean[min(user_id, self.num_users - 1)]

        sim = self.sim_matrix[user_id]
        ratings_item = self.user_item_matrix[:, item_id]

        mask = (sim != 0) & (ratings_item > 0)
        if not mask.any():
            return self.user_mean[user_id]

        sim_vals = sim[mask]
        deviations = ratings_item[mask] - self.user_mean[np.arange(self.num_users)[mask]]

        denom = np.sum(np.abs(sim_vals))
        if denom < 1e-9:
            return self.user_mean[user_id]

        return self.user_mean[user_id] + np.dot(sim_vals, deviations) / denom

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Recommend top-K items for a user."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or user_id < 0 or user_id >= self.num_users:
            return []
        if exclude_items is None:
            exclude_items = set()

        # Get all items the user hasn't rated
        rated = set(np.where(self.user_item_matrix[user_id] > 0)[0])
        candidates = [i for i in range(self.num_items)
                      if i not in rated and i not in exclude_items]

        if not candidates:
            return []

        # Predict scores for all candidates
        scores = [(i, self.predict(user_id, i)) for i in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
