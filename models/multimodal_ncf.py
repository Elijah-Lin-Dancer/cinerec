"""
Multi-Modal Neural Collaborative Filtering (Core Innovation).
Extends NeuMF by fusing text, image, and genre content features into the MLP path.

Architecture:
  User Tower: user_id → Embedding(64)
  Item Content Tower:
    - Text: Sentence-BERT (384-dim) → FC(384→64)
    - Image: ResNet-50 (2048-dim) → FC(2048→64)
    - Genre: Multi-hot (18-dim) → FC(18→64)
    - Concat(192) → FC(192→64) → ReLU → Dropout → content_emb
  GMF Path: user_emb ⊙ item_id_emb → [64]
  MLP Path: Concat(user_emb, content_emb) → FC(128→256→128→64) → ReLU + Dropout
  Prediction: Concat(GMF(64), MLP(64)) → FC(128→1) → Sigmoid

Cold start: new items use content_emb only, GMF path uses zero vector.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.base import Recommender


PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class MultiModalNCFDataset(Dataset):
    """Dataset with content features."""

    def __init__(self, user_ids, item_ids, labels, text_emb, image_emb, genre_vec):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)
        self.text_emb = torch.FloatTensor(text_emb)   # (num_items, 384)
        self.image_emb = torch.FloatTensor(image_emb)  # (num_items, 2048)
        self.genre_vec = torch.FloatTensor(genre_vec)  # (num_items, 18)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.item_ids[idx], self.labels[idx],
                self.text_emb[self.item_ids[idx]],
                self.image_emb[self.item_ids[idx]],
                self.genre_vec[self.item_ids[idx]])


class ContentTower(nn.Module):
    """Fuses text, image, and genre into a single content embedding."""

    def __init__(self, text_dim=384, image_dim=2048, genre_dim=18, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU()
        )
        self.image_fc = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU()
        )
        self.genre_fc = nn.Sequential(
            nn.Linear(genre_dim, hidden_dim),
            nn.ReLU()
        )
        combined_dim = hidden_dim * 3  # 192
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, text_emb, image_emb, genre_vec):
        t = self.text_fc(text_emb)
        i = self.image_fc(image_emb)
        g = self.genre_fc(genre_vec)
        combined = torch.cat([t, i, g], dim=-1)
        return self.fusion(combined)


class MultiModalNCFNet(nn.Module):
    """Multi-Modal NCF network."""

    def __init__(self, num_users, num_items, embedding_dim=64,
                 text_dim=384, image_dim=2048, genre_dim=18,
                 mlp_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        # GMF embeddings (behavior signal)
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)

        # MLP user embedding
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)

        # Content tower (replaces item embedding in MLP path)
        self.content_tower = ContentTower(
            text_dim=text_dim, image_dim=image_dim,
            genre_dim=genre_dim, hidden_dim=embedding_dim, dropout=dropout
        )

        # MLP path
        mlp_input_dim = embedding_dim * 2  # user_emb + content_emb
        mlp_layers = []
        prev_dim = mlp_input_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction
        self.predict = nn.Sequential(
            nn.Linear(embedding_dim + mlp_dims[-1], 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_ids, item_ids, text_emb, image_emb, genre_vec):
        # GMF path
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_out = gmf_user * gmf_item

        # MLP path: user emb + content emb
        mlp_user = self.mlp_user_emb(user_ids)
        content_emb = self.content_tower(text_emb, image_emb, genre_vec)
        mlp_out = self.mlp(torch.cat([mlp_user, content_emb], dim=-1))

        # Combine
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.predict(combined).squeeze(-1)

    def forward_cold(self, user_ids, text_emb, image_emb, genre_vec):
        """Score a brand-new item that has no learned embedding (cold start).

        The GMF path receives a zero item vector, so only the content tower
        drives the prediction; the MLP path still uses the user embedding.
        """
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_out = torch.zeros_like(gmf_user)
        mlp_user = self.mlp_user_emb(user_ids)
        content_emb = self.content_tower(text_emb, image_emb, genre_vec)
        mlp_out = self.mlp(torch.cat([mlp_user, content_emb], dim=-1))
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.predict(combined).squeeze(-1)


class MultiModalNCF(Recommender):
    """Multi-Modal Neural Collaborative Filtering — Core Innovation."""

    def __init__(self, embedding_dim=64, mlp_dims=(256, 128, 64), dropout=0.2,
                 lr=0.001, batch_size=1024, epochs=20, patience=3, num_neg=4,
                 disabled_features=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_neg = num_neg
        # Ablation hook: zero out selected content modalities ("text"/"image"/"genre")
        # before training, so the study measures what each modality actually contributes.
        self.disabled_features = set(disabled_features or ())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.num_users = 0
        self.num_items = 0

        # Content features
        self.text_emb = None
        self.image_emb = None
        self.genre_vec = None

    def _load_content_features(self):
        """Load pre-computed content feature embeddings."""
        text_path = os.path.join(PROCESSED_DIR, "text_embeddings.npy")
        image_path = os.path.join(PROCESSED_DIR, "image_embeddings.npy")
        genre_path = os.path.join(PROCESSED_DIR, "genre_vectors.npy")

        if os.path.exists(text_path):
            self.text_emb = np.load(text_path)
        else:
            self.text_emb = np.zeros((self.num_items, 384), dtype=np.float32)

        if os.path.exists(image_path):
            self.image_emb = np.load(image_path)
        else:
            self.image_emb = np.zeros((self.num_items, 2048), dtype=np.float32)

        if os.path.exists(genre_path):
            self.genre_vec = np.load(genre_path)
        else:
            self.genre_vec = np.zeros((self.num_items, 18), dtype=np.float32)

        # Ensure dimensions match num_items
        if len(self.text_emb) < self.num_items:
            pad = np.zeros((self.num_items - len(self.text_emb), 384), dtype=np.float32)
            self.text_emb = np.vstack([self.text_emb, pad])
        if len(self.image_emb) < self.num_items:
            pad = np.zeros((self.num_items - len(self.image_emb), 2048), dtype=np.float32)
            self.image_emb = np.vstack([self.image_emb, pad])
        if len(self.genre_vec) < self.num_items:
            pad = np.zeros((self.num_items - len(self.genre_vec), 18), dtype=np.float32)
            self.genre_vec = np.vstack([self.genre_vec, pad])

        # Ablation: blank the disabled modalities so the content tower receives no
        # information from them (a zero input can only contribute a learned bias).
        for name, arr in (("text", self.text_emb), ("image", self.image_emb), ("genre", self.genre_vec)):
            if name in self.disabled_features:
                arr[:] = 0.0

    def fit(self, train_data):
        """Train Multi-Modal NCF."""
        user_ids = train_data['user_id']
        item_ids = train_data['item_id']
        ratings = train_data['rating']

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1

        self._load_content_features()

        threshold = 4.0
        positive_mask = ratings >= threshold
        pos_users = user_ids[positive_mask].astype(int)
        pos_items = item_ids[positive_mask].astype(int)

        print(f"MultiModalNCF: {len(pos_users)} positive interactions, "
              f"text={self.text_emb.shape}, image={self.image_emb.shape}, genre={self.genre_vec.shape}")

        user_items = {}
        for u, i in zip(pos_users, pos_items):
            user_items.setdefault(u, set()).add(i)

        # Generate training pairs with negative sampling
        train_u, train_i, train_l = [], [], []
        for u in range(self.num_users):
            pos_set = user_items.get(u, set())
            for i in pos_set:
                train_u.append(u)
                train_i.append(i)
                train_l.append(1.0)
                for _ in range(self.num_neg):
                    neg_i = np.random.randint(0, self.num_items)
                    attempts = 0
                    max_attempts = 100
                    while neg_i in pos_set and attempts < max_attempts:
                        neg_i = np.random.randint(0, self.num_items)
                        attempts += 1
                    if attempts >= max_attempts:
                        neg_i = (i + 1) % self.num_items
                    train_u.append(u)
                    train_i.append(neg_i)
                    train_l.append(0.0)

        train_u = np.array(train_u, dtype=np.int64)
        train_i = np.array(train_i, dtype=np.int64)
        train_l = np.array(train_l, dtype=np.float32)

        self.net = MultiModalNCFNet(
            self.num_users, self.num_items,
            embedding_dim=self.embedding_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = MultiModalNCFDataset(
            train_u, train_i, train_l,
            self.text_emb, self.image_emb, self.genre_vec
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.net.train()
            total_loss = 0
            n_batches = 0
            for batch_u, batch_i, batch_l, bt, bi, bg in loader:
                batch_u = batch_u.to(self.device)
                batch_i = batch_i.to(self.device)
                batch_l = batch_l.to(self.device)
                bt = bt.to(self.device)
                bi = bi.to(self.device)
                bg = bg.to(self.device)

                pred = self.net(batch_u, batch_i, bt, bi, bg)
                loss = criterion(pred, batch_l)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"MultiModalNCF: Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(f"MultiModalNCF: Epoch {epoch + 1}/{self.epochs}, Loss={avg_loss:.4f}")

        print(f"MultiModalNCF: Training complete, best loss={best_loss:.4f}")

    def predict(self, user_id, item_id):
        """Predict interaction probability."""
        user_id, item_id = int(user_id), int(item_id)
        if not (0 <= user_id < self.num_users and 0 <= item_id < self.num_items):
            return 0.0  # out-of-range indices cannot be embedded
        self.net.eval()
        with torch.no_grad():
            u = torch.LongTensor([user_id]).to(self.device)
            i = torch.LongTensor([item_id]).to(self.device)
            t = torch.FloatTensor(self.text_emb[item_id]).unsqueeze(0).to(self.device)
            img = torch.FloatTensor(self.image_emb[item_id]).unsqueeze(0).to(self.device)
            g = torch.FloatTensor(self.genre_vec[item_id]).unsqueeze(0).to(self.device)
            return float(self.net(u, i, t, img, g).cpu().item())

    def predict_cold(self, user_id, text_emb, image_emb, genre_vec):
        """Predict interaction probability for a new item from its raw content.

        Supports the cold-start case where the item has no trained embedding:
        pass the item's Sentence-BERT / ResNet-50 / genre vectors directly.
        """
        self.net.eval()
        with torch.no_grad():
            u = torch.LongTensor([int(user_id)]).to(self.device)
            t = torch.FloatTensor(text_emb).unsqueeze(0).to(self.device)
            img = torch.FloatTensor(image_emb).unsqueeze(0).to(self.device)
            g = torch.FloatTensor(genre_vec).unsqueeze(0).to(self.device)
            return float(self.net.forward_cold(u, t, img, g).cpu().item())

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Recommend top-K items using content features."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or user_id < 0 or user_id >= self.num_users:
            return []
        if exclude_items is None:
            exclude_items = set()

        self.net.eval()
        with torch.no_grad():
            user_tensor = torch.full((self.num_items,), user_id, dtype=torch.long).to(self.device)
            item_tensor = torch.arange(self.num_items, dtype=torch.long).to(self.device)
            text_tensor = torch.FloatTensor(self.text_emb).to(self.device)
            image_tensor = torch.FloatTensor(self.image_emb).to(self.device)
            genre_tensor = torch.FloatTensor(self.genre_vec).to(self.device)
            scores = self.net(user_tensor, item_tensor, text_tensor, image_tensor, genre_tensor).cpu().numpy()

        for idx in exclude_items:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -np.inf]

    def save(self, path):
        """Save model weights and content features."""
        torch.save({
            'net': self.net.state_dict(),
            'num_users': self.num_users,
            'num_items': self.num_items,
            'embedding_dim': self.embedding_dim,
            'mlp_dims': self.mlp_dims,
        }, path)

    def load(self, path):
        """Load model weights.

        ``weights_only=True`` restricts unpickling to tensors and plain Python
        containers, so a tampered artefact cannot execute arbitrary code.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.num_users = checkpoint['num_users']
        self.num_items = checkpoint['num_items']
        self._load_content_features()
        self.net = MultiModalNCFNet(
            self.num_users, self.num_items,
            embedding_dim=checkpoint.get('embedding_dim', 64),
            mlp_dims=checkpoint.get('mlp_dims', (256, 128, 64)),
            dropout=self.dropout
        ).to(self.device)
        self.net.load_state_dict(checkpoint['net'])
