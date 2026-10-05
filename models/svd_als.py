"""
SVD Matrix Factorization via **truncated SVD** (``scipy.sparse.linalg.svds``).
Decomposes the (mean-centred) user-item matrix into U × V^T.

Note: despite the module name, this is *not* Alternating Least Squares — it is a
pure truncated SVD of the sparse rating matrix. See ``docs/adr/0003``.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from models.base import Recommender


class SVDALS(Recommender):
    """Matrix Factorization using truncated SVD."""

    def __init__(self, k=64):
        super().__init__()
        self.k = k  # latent factor dimension
        self.U = None  # (num_users, k) user factors
        self.V = None  # (num_items, k) item factors
        self.num_users = 0
        self.num_items = 0
        self.user_mean = None
        self.global_mean = 0.0
    
    def fit(self, train_data):
        """Factor the user-item matrix using truncated SVD."""
        user_ids = train_data['user_id']
        item_ids = train_data['item_id']
        ratings = train_data['rating']
        
        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1
        
        # Build sparse user-item matrix (cast to float: ratings.csv may be int)
        self.user_item_sparse = csr_matrix(
            (ratings.astype(np.float64), (user_ids, item_ids)),
            shape=(self.num_users, self.num_items)
        )
        
        # Compute means for centering
        self.global_mean = ratings.mean()
        self.user_mean = np.zeros(self.num_users)
        for u in range(self.num_users):
            row = self.user_item_sparse[u]
            if row.nnz > 0:
                self.user_mean[u] = row.data.mean()
            else:
                self.user_mean[u] = self.global_mean
        
        # Center the matrix (subtract user means)
        dense = self.user_item_sparse.toarray().copy()
        for u in range(self.num_users):
            mask = dense[u] != 0
            dense[u][mask] -= self.user_mean[u]
        
        # Truncated SVD (k must be < min(num_users, num_items))
        actual_k = min(self.k, min(self.num_users, self.num_items) - 1)
        U, sigma, Vt = svds(csr_matrix(dense), k=actual_k)
        
        # Sort by descending singular values
        idx = np.argsort(sigma)[::-1]
        sigma = sigma[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        
        # Scale factors by sqrt of singular values
        sigma_sqrt = np.sqrt(sigma)
        self.U = U * sigma_sqrt[np.newaxis, :]  # (num_users, actual_k)
        self.V = Vt.T * sigma_sqrt[np.newaxis, :]  # (num_items, actual_k)
        
        # Pad to k dimensions if needed
        if actual_k < self.k:
            pad_users = np.zeros((self.num_users, self.k - actual_k))
            pad_items = np.zeros((self.num_items, self.k - actual_k))
            self.U = np.hstack([self.U, pad_users])
            self.V = np.hstack([self.V, pad_items])
        
        print(f"SVD fitted: {self.num_users} users × {self.num_items} items, k={actual_k}")
    
    def predict(self, user_id, item_id):
        """Predict rating via dot product + user mean."""
        user_id = int(user_id)
        item_id = int(item_id)
        if user_id < 0 or item_id < 0:
            return self.global_mean
        if user_id >= self.num_users or item_id >= self.num_items:
            return self.global_mean
        return self.user_mean[user_id] + np.dot(self.U[user_id], self.V[item_id])
    
    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Recommend top-K items for a user."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or user_id < 0 or user_id >= self.num_users:
            return []
        if exclude_items is None:
            exclude_items = set()
        
        # Score all items: mean + U[u] · V^T
        scores = self.user_mean[user_id] + self.V @ self.U[user_id]
        
        # Exclude already rated items
        if self.user_item_sparse is not None:
            rated = set(self.user_item_sparse[user_id].indices)
        else:
            rated = set()
        
        for idx in list(exclude_items) + list(rated):
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -np.inf]
