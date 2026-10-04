"""
LightGCN — a simplified graph-convolution recommender (modern baseline).

LightGCN removes the feature transformation and non-linear activation of
standard GCNs and keeps only neighbourhood aggregation on the user-item
bipartite graph:

    e_u^(k+1) = Σ_{i∈N(u)} e_i^(k) / sqrt(|N(u)|·|N(i)|)
    e_i^(k+1) = Σ_{u∈N(i)} e_u^(k) / sqrt(|N(u)|·|N(i)|)

The layer-0..K embeddings are averaged to form the final user/item vectors and
the model is trained with Bayesian Personalized Ranking (BPR) loss over
positive/negative pairs. See He et al., "LightGCN: Simplifying and Powering
Graph Convolution Network for Recommendation" (SIGIR 2020).

This is an implicit-feedback model: an interaction counts as positive when its
rating is >= 4, matching the positive threshold used by NeuMF and MultiModalNCF.
"""
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from models.base import Recommender

POSITIVE_THRESHOLD = 4.0


class LightGCN(Recommender):
    """LightGCN trained with BPR on the normalised user-item adjacency."""

    def __init__(self, embedding_dim=64, num_layers=3, lr=0.001, reg=1e-4,
                 batch_size=4096, epochs=50, patience=8, seed=42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = 0
        self.num_items = 0
        # Final (layer-averaged) embeddings, kept as numpy for fast scoring.
        self.user_emb = None
        self.item_emb = None

    def _build_norm_adjacency(self, users, items):
        """Row-normalised bipartite adjacency as a sparse tensor (users ⊕ items)."""
        num_u, num_i = self.num_users, self.num_items
        n = num_u + num_i

        deg_u = np.bincount(users, minlength=num_u).astype(np.float64)
        deg_i = np.bincount(items, minlength=num_i).astype(np.float64)

        # Guard isolated nodes: clamp the denominator to 1 so their rows stay 0.
        denom = np.sqrt(np.clip(deg_u[users], 1, None) * np.clip(deg_i[items], 1, None))
        weights = 1.0 / denom

        rows = np.concatenate([users, num_u + items])
        cols = np.concatenate([num_u + items, users])
        vals = np.concatenate([weights, weights])

        indices = torch.LongTensor(np.vstack([rows, cols]))
        return torch.sparse_coo_tensor(
            indices, torch.FloatTensor(vals), (n, n), check_invariants=False
        ).coalesce().to(self.device)

    def _propagate(self, base_emb, adj):
        """Average the layer-0..K embeddings (LightGCN's layer combination)."""
        layers = [base_emb]
        x = base_emb
        for _ in range(self.num_layers):
            x = torch.sparse.mm(adj, x)
            layers.append(x)
        return torch.stack(layers, dim=0).mean(dim=0)

    def fit(self, train_data):
        """Train LightGCN on positive user-item pairs with BPR + negative sampling."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        user_ids = np.asarray(train_data["user_id"])
        item_ids = np.asarray(train_data["item_id"])
        ratings = np.asarray(train_data["rating"])

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1

        pos_mask = ratings >= POSITIVE_THRESHOLD
        pos_users = user_ids[pos_mask].astype(np.int64)
        pos_items = item_ids[pos_mask].astype(np.int64)

        adj = self._build_norm_adjacency(pos_users, pos_items)

        # Per-user positive sets, so sampled negatives are never true positives.
        user_pos = {}
        for u, i in zip(pos_users, pos_items):
            user_pos.setdefault(int(u), set()).add(int(i))
        user_pos_list = [user_pos.get(u, set()) for u in range(self.num_users)]

        print(f"LightGCN: {len(pos_users)} positive interactions, "
              f"layers={self.num_layers}, dim={self.embedding_dim}")

        n_nodes = self.num_users + self.num_items
        base_emb = torch.nn.Parameter(
            torch.normal(0.0, 0.1, (n_nodes, self.embedding_dim), device=self.device)
        )
        optimizer = torch.optim.Adam([base_emb], lr=self.lr)

        n_pos = len(pos_users)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            emb = self._propagate(base_emb, adj)
            u_emb = emb[: self.num_users]
            i_emb = emb[self.num_users:]

            perm = np.random.permutation(n_pos)
            total = torch.zeros((), device=self.device)
            n_batches = 0

            for start in range(0, n_pos, self.batch_size):
                idx = perm[start:start + self.batch_size]
                bu = torch.LongTensor(pos_users[idx]).to(self.device)
                bi = torch.LongTensor(pos_items[idx]).to(self.device)
                bj = torch.LongTensor(
                    [self._sample_negative(user_pos_list[int(u)]) for u in pos_users[idx]]
                ).to(self.device)

                pos_scores = (u_emb[bu] * i_emb[bi]).sum(dim=1)
                neg_scores = (u_emb[bu] * i_emb[bj]).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                loss = loss + self.reg * (
                    u_emb[bu].pow(2).sum() + i_emb[bi].pow(2).sum() + i_emb[bj].pow(2).sum()
                ) / len(bu)

                total = total + loss
                n_batches += 1

            total = total / max(n_batches, 1)
            total.backward()
            optimizer.step()

            avg_loss = float(total.detach().cpu())
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"LightGCN: Epoch {epoch + 1}/{self.epochs}, BPR loss={avg_loss:.4f}")
            if patience_counter >= self.patience:
                print(f"LightGCN: Early stopping at epoch {epoch + 1}")
                break

        print(f"LightGCN: Training complete, best loss={best_loss:.4f}")

        with torch.no_grad():
            emb = self._propagate(base_emb, adj)
        self.user_emb = emb[: self.num_users].detach().cpu().numpy().astype(np.float32)
        self.item_emb = emb[self.num_users:].detach().cpu().numpy().astype(np.float32)

    def _sample_negative(self, positives):
        """Draw one item the user has not interacted with (rejection sampling)."""
        if len(positives) >= self.num_items:
            return 0
        j = int(np.random.randint(0, self.num_items))
        while j in positives:
            j = int(np.random.randint(0, self.num_items))
        return j

    def predict(self, user_id, item_id):
        """Dot-product score between the final user and item embeddings."""
        user_id, item_id = int(user_id), int(item_id)
        if user_id >= self.num_users or item_id >= self.num_items:
            return 0.0
        return float(self.user_emb[user_id] @ self.item_emb[item_id])

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Rank all items by dot-product score and return the top-K unseen ones."""
        user_id = int(user_id)
        if self.user_emb is None or user_id >= self.num_users:
            return []
        exclude_items = exclude_items or set()

        scores = self.user_emb[user_id] @ self.item_emb.T
        for idx in exclude_items:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if np.isfinite(scores[idx])]

    def save(self, path):
        """Persist the final embeddings (float32) — no torch needed to load."""
        with open(path, "wb") as f:
            pickle.dump({
                "user_emb": self.user_emb,
                "item_emb": self.item_emb,
                "embedding_dim": self.embedding_dim,
                "num_layers": self.num_layers,
                "num_users": self.num_users,
                "num_items": self.num_items,
            }, f)

    def load(self, path):
        """Load an artefact written by :meth:`save`."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.user_emb = checkpoint["user_emb"]
        self.item_emb = checkpoint["item_emb"]
        self.embedding_dim = checkpoint.get("embedding_dim", self.embedding_dim)
        self.num_layers = checkpoint.get("num_layers", self.num_layers)
        self.num_users = checkpoint["num_users"]
        self.num_items = checkpoint["num_items"]
