"""
VBPR — Visual Bayesian Personalized Ranking.

The visual content-aware BPR baseline (He & McAuley, "VBPR: Visual Bayesian
Personalized Ranking from Implicit Feedback", AAAI 2016). It augments BPR-MF with
a learned projection of each item's visual feature and a per-user visual
preference vector:

    score(u, i) = ⟨p_u, q_i⟩ + b_i + ⟨θ_u, E·f_i⟩

where ``f_i`` is the item's ResNet-50 image embedding (2048-d), ``E`` projects it
to K dimensions, and ``θ_u`` is the user's taste in that visual space.

It is the *content-aware* SOTA reference for this project: it uses item content
(like MultiModalNCF) but keeps a shallow bilinear form instead of an MLP content
tower, so it isolates whether the deep multi-modal fusion adds anything over a
well-established visual ranker.

Implicit feedback: an interaction is positive when its rating is >= 4, matching
every other model here.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from models.base import Recommender, set_seed
from config import PROCESSED_DIR  # active dataset's processed dir (CINEREC_DATASET)

POSITIVE_THRESHOLD = 4.0


class VBPR(Recommender):
    """BPR-MF augmented with a learned item-visual / user-visual term."""

    def __init__(self, embedding_dim=64, visual_dim=64, lr=0.005, reg=1e-4,
                 batch_size=4096, epochs=30, patience=6, seed=42):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.visual_dim = visual_dim
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_users = 0
        self.num_items = 0

        # Fixed visual features and learned factors, kept as numpy for scoring.
        self.image_emb = None        # (num_items, image_dim)
        self.user_emb = None         # (num_users, K)  — p_u
        self.item_emb = None         # (num_items, K)  — q_i
        self.user_visual = None      # (num_users, V)  — θ_u
        self.item_visual = None      # (num_items, V)  — E·f_i (precomputed)
        self.item_bias = None        # (num_items,)    — b_i

    def _load_image_features(self):
        """Load pre-computed ResNet-50 image embeddings, padded to num_items.

        Rows are L2-normalised: raw ResNet features have large, varying norms, so
        in the standard VBPR recipe the visual vector is unit-scaled before it
        enters the bilinear term (otherwise the visual score dwarfs the MF score
        and the model degenerates).
        """
        path = os.path.join(PROCESSED_DIR, "image_embeddings.npy")
        if os.path.exists(path):
            self.image_emb = np.load(path).astype(np.float32)
        else:
            self.image_emb = np.zeros((self.num_items, 2048), dtype=np.float32)
        if len(self.image_emb) < self.num_items:
            pad = np.zeros((self.num_items - len(self.image_emb), self.image_emb.shape[1]), np.float32)
            self.image_emb = np.vstack([self.image_emb, pad])
        norms = np.linalg.norm(self.image_emb, axis=1, keepdims=True)
        self.image_emb = self.image_emb / np.clip(norms, 1e-8, None)

    def fit(self, train_data):
        """Train VBPR on positive pairs with BPR loss + negative sampling."""
        set_seed(self.seed)

        user_ids = np.asarray(train_data["user_id"])
        item_ids = np.asarray(train_data["item_id"])
        ratings = np.asarray(train_data["rating"])

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1
        self._load_image_features()

        pos_mask = ratings >= POSITIVE_THRESHOLD
        pos_users = user_ids[pos_mask].astype(np.int64)
        pos_items = item_ids[pos_mask].astype(np.int64)

        user_pos = {}
        for u, i in zip(pos_users, pos_items):
            user_pos.setdefault(int(u), set()).add(int(i))
        user_pos_list = [user_pos.get(u, set()) for u in range(self.num_users)]

        image_dim = self.image_emb.shape[1]
        print(f"VBPR: {len(pos_users)} positive interactions, "
              f"visual={self.image_emb.shape}, dim={self.embedding_dim}")

        features = torch.FloatTensor(self.image_emb).to(self.device)

        p = torch.nn.Parameter(torch.normal(0.0, 0.1, (self.num_users, self.embedding_dim), device=self.device))
        q = torch.nn.Parameter(torch.normal(0.0, 0.1, (self.num_items, self.embedding_dim), device=self.device))
        theta = torch.nn.Parameter(torch.normal(0.0, 0.1, (self.num_users, self.visual_dim), device=self.device))
        item_bias = torch.nn.Parameter(torch.zeros(self.num_items, device=self.device))
        # Visual projection E: (visual_dim, image_dim), mirroring E·f_i.
        proj = torch.nn.Parameter(torch.normal(0.0, 0.01, (self.visual_dim, image_dim), device=self.device))

        params = [p, q, theta, item_bias, proj]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        def score(bu, bi):
            """score(u, i) = ⟨p_u,q_i⟩ + b_i + ⟨θ_u, E·f_i⟩."""
            mf = (p[bu] * q[bi]).sum(dim=1)
            item_vis = features[bi] @ proj.t()  # (B, V)
            vis = (theta[bu] * item_vis).sum(dim=1)
            return mf + item_bias[bi] + vis

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

                loss = -F.logsigmoid(score(bu, bi) - score(bu, bj)).mean()
                loss = loss + self.reg * (
                    p[bu].pow(2).sum()
                    + q[bi].pow(2).sum() + q[bj].pow(2).sum()
                    + theta[bu].pow(2).sum()
                    + proj.pow(2).sum()
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
                print(f"VBPR: Epoch {epoch + 1}/{self.epochs}, BPR loss={avg_loss:.4f}")
            if patience_counter >= self.patience:
                print(f"VBPR: Early stopping at epoch {epoch + 1}")
                break

        print(f"VBPR: Training complete, best loss={best_loss:.4f}")

        # Freeze the scoring factors as numpy (avoid keeping the graph alive).
        with torch.no_grad():
            item_vis = features @ proj.t()
        self.user_emb = p.detach().cpu().numpy().astype(np.float32)
        self.item_emb = q.detach().cpu().numpy().astype(np.float32)
        self.user_visual = theta.detach().cpu().numpy().astype(np.float32)
        self.item_visual = item_vis.detach().cpu().numpy().astype(np.float32)
        self.item_bias = item_bias.detach().cpu().numpy().astype(np.float32)

    def _sample_negative(self, positives):
        """Draw one item the user has not interacted with (rejection sampling)."""
        if len(positives) >= self.num_items:
            return 0
        j = int(np.random.randint(0, self.num_items))
        while j in positives:
            j = int(np.random.randint(0, self.num_items))
        return j

    def _scores_for_user(self, user_id):
        """Full score vector over the catalogue for one user."""
        scores = self.user_emb[user_id] @ self.item_emb.T + self.item_bias
        scores = scores + self.user_visual[user_id] @ self.item_visual.T
        return scores

    def predict(self, user_id, item_id):
        """Full VBPR score for a user-item pair."""
        user_id, item_id = int(user_id), int(item_id)
        if not (0 <= user_id < self.num_users and 0 <= item_id < self.num_items):
            return 0.0
        return float(self._scores_for_user(user_id)[item_id])

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Rank all items by VBPR score and return the top-K unseen ones."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or self.user_emb is None or user_id < 0 or user_id >= self.num_users:
            return []
        exclude_items = exclude_items or set()

        scores = self._scores_for_user(user_id)
        for idx in exclude_items:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if np.isfinite(scores[idx])]

    def save(self, path):
        """Persist the scoring factors (float32) — no torch needed to load."""
        with open(path, "wb") as f:
            pickle.dump({
                "user_emb": self.user_emb,
                "item_emb": self.item_emb,
                "user_visual": self.user_visual,
                "item_visual": self.item_visual,
                "item_bias": self.item_bias,
                "embedding_dim": self.embedding_dim,
                "visual_dim": self.visual_dim,
                "num_users": self.num_users,
                "num_items": self.num_items,
            }, f)

    def load(self, path):
        """Load an artefact written by :meth:`save`."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.user_emb = checkpoint["user_emb"]
        self.item_emb = checkpoint["item_emb"]
        self.user_visual = checkpoint["user_visual"]
        self.item_visual = checkpoint["item_visual"]
        self.item_bias = checkpoint["item_bias"]
        self.embedding_dim = checkpoint.get("embedding_dim", self.embedding_dim)
        self.visual_dim = checkpoint.get("visual_dim", self.visual_dim)
        self.num_users = checkpoint["num_users"]
        self.num_items = checkpoint["num_items"]
