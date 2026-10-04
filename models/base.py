"""
Abstract base class for all recommender models.
"""
from abc import ABC, abstractmethod
import copy
import pickle

import numpy as np


class Recommender(ABC):
    """Base recommender that all 5 models inherit from.

    Provides a default pickle-based :meth:`save` / :meth:`load` (with float64 →
    float32 downcasting of large arrays to keep artefacts small). The neural
    models override these with their torch checkpoints.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def fit(self, train_data):
        """
        Train the model on user-item interaction data.

        Args:
            train_data: dict with keys:
                - 'user_id': numpy array of user IDs (0-based)
                - 'item_id': numpy array of item IDs (0-based)
                - 'rating': numpy array of ratings
        """
        pass

    @abstractmethod
    def predict(self, user_id, item_id):
        """Predict score for a single user-item pair."""
        pass

    @abstractmethod
    def recommend(self, user_id, top_k=10, exclude_items=None):
        """
        Return top_k item recommendations for a user.

        Args:
            user_id: user index (0-based)
            top_k: number of recommendations
            exclude_items: set of item IDs to exclude

        Returns:
            list of (item_id, score) tuples, sorted by score descending
        """
        pass

    def recommend_ids(self, user_id, top_k=10, exclude_items=None):
        """Return only item IDs (no scores) — convenience method."""
        recs = self.recommend(user_id, top_k, exclude_items)
        return [item_id for item_id, _ in recs]

    def save(self, path):
        """Persist the model with pickle, downcasting float64 arrays to float32."""
        slim = copy.copy(self)
        for key, value in vars(self).items():
            if isinstance(value, np.ndarray) and value.dtype == np.float64:
                setattr(slim, key, value.astype(np.float32))
        with open(path, "wb") as f:
            pickle.dump(slim, f)

    @classmethod
    def load(cls, path):
        """Load a model persisted by :meth:`save`."""
        with open(path, "rb") as f:
            return pickle.load(f)
