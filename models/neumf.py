"""
Neural Collaborative Filtering (NeuMF) with dual-path architecture.
GMF Path: user_emb ⊙ item_emb → FC(64→1) → Sigmoid
MLP Path: Concat(user_emb, item_emb) → FC(128→256→128→64) → ReLU + Dropout → FC(64→1) → Sigmoid
Final: Concat(GMF_out, MLP_out) → FC(2→1) → Sigmoid
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.base import Recommender


class NeuMFDataset(Dataset):
    """Dataset for NeuMF training: (user, item, label) triples."""

    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


class NeuMFNet(nn.Module):
    """NeuMF neural network with GMF + MLP dual paths."""

    def __init__(self, num_users, num_items, embedding_dim=64, mlp_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(num_items, embedding_dim)

        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_input_dim = embedding_dim * 2  # concat of user + item
        mlp_layers = []
        prev_dim = mlp_input_dim
        for dim in mlp_dims:
            mlp_layers.append(nn.Linear(prev_dim, dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer
        self.predict = nn.Sequential(
            nn.Linear(embedding_dim + mlp_dims[-1], 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_out = gmf_user * gmf_item  # element-wise product

        # MLP path
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Combine and predict
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.predict(combined).squeeze(-1)


class NeuMF(Recommender):
    """NeuMF: Neural Collaborative Filtering with GMF + MLP."""

    def __init__(self, embedding_dim=64, mlp_dims=(256, 128, 64), dropout=0.2,
                 lr=0.001, batch_size=1024, epochs=20, patience=3, num_neg=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.num_neg = num_neg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.num_users = 0
        self.num_items = 0
        self.all_items = None

    def fit(self, train_data):
        """Train NeuMF on user-item interactions."""
        user_ids = train_data['user_id']
        item_ids = train_data['item_id']
        ratings = train_data['rating']

        self.num_users = int(user_ids.max()) + 1
        self.num_items = int(item_ids.max()) + 1
        self.all_items = np.arange(self.num_items)

        # Convert ratings to binary labels (rating >= 4 → positive)
        threshold = 4.0
        positive_mask = ratings >= threshold
        pos_users = user_ids[positive_mask].astype(int)
        pos_items = item_ids[positive_mask].astype(int)

        print(f"NeuMF: {len(pos_users)} positive interactions (>= {threshold}), "
              f"training on {self.device}")

        # Build user-items dict for negative sampling
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
                # Negative samples
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

        # Create model and optimizer
        self.net = NeuMFNet(
            self.num_users, self.num_items,
            embedding_dim=self.embedding_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = NeuMFDataset(train_u, train_i, train_l)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        # Training loop with early stopping
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            self.net.train()
            total_loss = 0
            n_batches = 0
            for batch_u, batch_i, batch_l in loader:
                batch_u = batch_u.to(self.device)
                batch_i = batch_i.to(self.device)
                batch_l = batch_l.to(self.device)

                pred = self.net(batch_u, batch_i)
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
                    print(f"NeuMF: Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 5 == 0:
                print(f"NeuMF: Epoch {epoch + 1}/{self.epochs}, Loss={avg_loss:.4f}")

        print(f"NeuMF: Training complete, best loss={best_loss:.4f}")

    def predict(self, user_id, item_id):
        """Predict interaction probability for user-item pair."""
        user_id, item_id = int(user_id), int(item_id)
        if not (0 <= user_id < self.num_users and 0 <= item_id < self.num_items):
            return 0.0  # out-of-range indices cannot be embedded
        self.net.eval()
        with torch.no_grad():
            u = torch.LongTensor([user_id]).to(self.device)
            i = torch.LongTensor([item_id]).to(self.device)
            return float(self.net(u, i).cpu().item())

    def recommend(self, user_id, top_k=10, exclude_items=None):
        """Recommend top-K items for a user."""
        user_id = int(user_id)
        top_k = int(top_k)
        if top_k <= 0 or user_id < 0 or user_id >= self.num_users:
            return []
        if exclude_items is None:
            exclude_items = set()

        self.net.eval()
        with torch.no_grad():
            # Batch prediction for all items
            user_tensor = torch.full((self.num_items,), user_id, dtype=torch.long).to(self.device)
            item_tensor = torch.arange(self.num_items, dtype=torch.long).to(self.device)
            scores = self.net(user_tensor, item_tensor).cpu().numpy()

        for idx in exclude_items:
            if 0 <= idx < len(scores):
                scores[idx] = -np.inf

        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > -np.inf]

    def save(self, path):
        """Save model weights."""
        torch.save({
            'net': self.net.state_dict(),
            'num_users': self.num_users,
            'num_items': self.num_items,
        }, path)

    def load(self, path):
        """Load model weights.

        ``weights_only=True`` restricts unpickling to tensors and plain Python
        containers, so a tampered artefact cannot execute arbitrary code.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.num_users = checkpoint['num_users']
        self.num_items = checkpoint['num_items']
        self.net = NeuMFNet(
            self.num_users, self.num_items,
            embedding_dim=self.embedding_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout
        ).to(self.device)
        self.net.load_state_dict(checkpoint['net'])
