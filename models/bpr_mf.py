"""
BPR-MF — Bayesian Personalized Ranking matrix factorization.

The canonical implicit-feedback MF baseline (Rendle et al., "BPR: Bayesian
Personalized Ranking from Implicit Feedback", UAI 2009). It learns one user and
one item latent vector and optimises a pairwise ranking loss over observed vs.
sampled-unobserved pairs:

    L = -Σ log σ(⟨p_u, q_i⟩ − ⟨p_u, q_j⟩) + λ‖·‖²

It is the content-blind *strong* baseline the neural models must beat: unlike
truncated SVD (which fits observed ratings), BPR-MF optimises the top-N ranking
objective the evaluation actually measures.

This is an implicit-feedback model: an interaction counts as positive when its
rating is >= 4, matching the positive threshold used by every other model here.
"""
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from models.base import Recommender, set_seed

POSITIVE_THRESHOLD = 4.0


class BPRMF(Recommender):
    """Matrix factorization trained with the BPR pairwise ranking loss."""

    def __init__(self, embedding_dim=64, lr=0.001, reg=1e-4,
                 batch_size=4096, epochs=50, patience=8, seed=42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = 0
        self.num_items = 0
        # Learned factors, kept as numpy for fast scoring without torch.
        self.user_emb = None
        self.item_emb = None

    def fit(self, train_data):
        """Train BPR-MF on positive user-item pairs with negative sampling."""
        set_seed(self.seed)

        user_ids = np.asarray(train_data["user_id"])
        item_ids = np.asarray(train_data["item_id"])
        ratings = np.asarray(train_data["rating"])

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1

        pos_mask = ratings >= POSITIVE_THRESHOLD
        pos_users = user_ids[pos_mask].astype(np.int64)
        pos_items = item_ids[pos_mask].astype(np.int64)

        # Per-user positive sets, so sampled negatives are never true positives.
        user_pos = {}
        for u, i in zip(pos_users, pos_items):
            user_pos.setdefault(int(u), set()).add(int(i))
        user_pos_list = [user_pos.get(u, set()) for u in range(self.num_users)]

        print(f"BPR-MF: {len(pos_users)} positive interactions, dim={self.embedding_dim}")

        user_emb = torch.nn.Parameter(
            torch.normal(0.0, 0.1, (self.num_users, self.embedding_dim), device=self.device)
        )
        item_emb = torch.nn.Parameter(
            torch.normal(0.0, 0.1, (self.num_items, self.embedding_dim), device=self.device)
        )
        optimizer = torch.optim.Adam([user_emb, item_emb], lr=self.lr)

        n_pos = len(pos_users)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
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

                pos_scores = (user_emb[bu] * item_emb[bi]).sum(dim=1)
                neg_scores = (user_emb[bu] * item_emb[bj]).sum(dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                loss = loss + self.reg * (
                    user_emb[bu].pow(2).sum()
                    + item_emb[bi].pow(2).sum()
                    + item_emb[bj].pow(2).sum()
                ) / len(bu)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total = total + loss.detach()
                n_batches += 1

            avg_loss = float(total.cpu()) / max(n_batches, 1)
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"BPR-MF: Epoch {epoch + 1}/{self.epochs}, BPR loss={avg_loss:.4f}")
            if patience_counter >= self.patience:
                print(f"BPR-MF: Early stopping at epoch {epoch + 1}")
                break

        print(f"BPR-MF: Training complete, best loss={best_loss:.4f}")

        self.user_emb = user_emb.detach().cpu().numpy().astype(np.float32)
        self.item_emb = item_emb.detach().cpu().numpy().astype(np.float32)

    def _sample_negative(self, positives):
        """Draw one item the user has not interacted with (rejection sampling)."""
        if len(positives) >= self.num_items:
            return 0
        j = int(np.random.randint(0, self.num_items))
        while j in positives:
            j = int(np.random.randint(0, self.num_items))
        return j

    def predict(self, user_id, item_id):
        """Dot-product score between the user and item factor vectors."""
        user_id, item_id = int(user_id), int(item_id)
        if not (0 <= user_id < self.num_users and 0 <= item_id < self.num_items):
            return 0.0
        return float(self.user_emb[user_id] @ self.item_emb[item_id])

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Rank all items by dot-product score and return the top-K unseen ones."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or self.user_emb is None or user_id < 0 or user_id >= self.num_users:
            return []
        exclude_items = exclude_items or set()

        scores = self.user_emb[user_id] @ self.item_emb.T
        for idx in exclude_items:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if np.isfinite(scores[idx])]

    def save(self, path):
        """Persist the learned factors (float32) — no torch needed to load."""
        with open(path, "wb") as f:
            pickle.dump({
                "user_emb": self.user_emb,
                "item_emb": self.item_emb,
                "embedding_dim": self.embedding_dim,
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
        self.num_users = checkpoint["num_users"]
        self.num_items = checkpoint["num_items"]
