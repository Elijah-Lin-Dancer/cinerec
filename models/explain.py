"""
Explainability Module — Generates human-readable recommendation reasons.
Two types:
1. Collaborative Reason: "Users who liked X also liked this" (based on similar users/items)
2. Content Reason: "Similar plot to X (87%), matches your preferred genres" (based on content similarity)
"""
import numpy as np
import os
import json

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class RecommenderExplainer:
    """Generates explanation for recommendations."""

    def __init__(self):
        self.movies = []  # list of movie dicts with id, title, genres
        self.genre_vecs = None  # (num_items, num_genres)
        self.content_embs = None  # (num_items, content_dim)
        self.user_ratings = {}  # user_id -> {item_id: rating}
        self.item_similarity = None  # (num_items, num_items) content cosine sim

    def load_data(self, movies_enriched_path=None, genre_path=None):
        """Load movie metadata and genre features."""
        if movies_enriched_path is None:
            movies_enriched_path = os.path.join(PROCESSED_DIR, "movies_enriched.json")
        if genre_path is None:
            genre_path = os.path.join(PROCESSED_DIR, "genre_vectors.npy")

        if os.path.exists(movies_enriched_path):
            with open(movies_enriched_path, "r", encoding="utf-8") as f:
                self.movies = json.load(f)
            self.movies.sort(key=lambda x: x["id"])
        else:
            self.movies = []

        if os.path.exists(genre_path):
            self.genre_vecs = np.load(genre_path)
        else:
            self.genre_vecs = None

        # Build content embeddings from genres (simplified)
        if self.genre_vecs is not None:
            self.content_embs = self.genre_vecs
            # Precompute cosine similarity matrix
            norms = np.linalg.norm(self.content_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized = self.content_embs / norms
            self.item_similarity = normalized @ normalized.T
        else:
            self.content_embs = None
            self.item_similarity = None

        # Build movie_dict for O(1) lookup
        self.movie_dict = {m['id']: m for m in self.movies}

    def load_user_ratings(self, train_data):
        """Load user rating history from training data."""
        for u, i, r in zip(train_data['user_id'], train_data['item_id'], train_data['rating']):
            self.user_ratings.setdefault(int(u), {})[int(i)] = float(r)

    def get_movie_title(self, item_id):
        """Get movie title by item ID."""
        if hasattr(self, 'movie_dict') and item_id in self.movie_dict:
            return self.movie_dict[item_id].get("title", f"Movie {item_id}")
        return f"Movie {item_id}"

    def get_movie_genres(self, item_id):
        """Get movie genres by item ID."""
        if hasattr(self, 'movie_dict') and item_id in self.movie_dict:
            return self.movie_dict[item_id].get("genres", "")
        return ""

    def content_reason(self, user_id, recommended_item_id, top_k=2):
        """
        Generate content-based explanation.

        Returns list of reason dicts:
            {"type": "content", "reason": "...", "score": 0.87}
        """
        reasons = []
        user_history = self.user_ratings.get(user_id, {})

        if not user_history or self.item_similarity is None:
            return reasons

        # Find highest-rated items by user
        rated_items = sorted(user_history.items(), key=lambda x: x[1], reverse=True)

        rec_idx = recommended_item_id
        if rec_idx >= self.item_similarity.shape[0]:
            return reasons

        sim_scores = []
        for rated_id, rating in rated_items[:20]:  # top-20 rated items
            if rated_id == recommended_item_id:
                continue
            if rated_id >= self.item_similarity.shape[0]:
                continue
            sim = self.item_similarity[rec_idx][rated_id]
            sim_scores.append((rated_id, sim, rating))

        sim_scores.sort(key=lambda x: x[1], reverse=True)

        # Top reason
        for rated_id, sim, rating in sim_scores[:top_k]:
            if sim > 0.01:  # threshold
                title = self.get_movie_title(rated_id)
                reasons.append({
                    "type": "content",
                    "reason_zh": f"与你喜欢的《{title}》类型相似（相似度 {sim*100:.0f}%）",
                    "reason_en": f"Similar genres to your favorite \"{title}\" ({sim*100:.0f}% similarity)",
                    "score": round(sim, 4),
                    "source_item": rated_id
                })

        # Genre overlap reason
        rec_genres = self.get_movie_genres(recommended_item_id)
        if rec_genres:
            user_genre_prefs = self._get_user_genre_prefs(user_id)
            overlap = [g.strip() for g in rec_genres.split("|")
                      if g.strip() in user_genre_prefs]
            if overlap:
                reasons.append({
                    "type": "genre_match",
                    "reason_zh": f"匹配你偏好的类型：{', '.join(overlap)}",
                    "reason_en": f"Matches your preferred genres: {', '.join(overlap)}",
                    "score": len(overlap) / max(len(rec_genres.split("|")), 1),
                    "genres": overlap
                })

        return reasons

    def collaborative_reason(self, user_id, recommended_item_id,
                            sim_matrix=None, is_user_cf=True, top_k=2):
        """
        Generate collaborative filtering explanation.

        Args:
            user_id: user index
            recommended_item_id: recommended item
            sim_matrix: precomputed similarity matrix
            is_user_cf: True for UserCF, False for ItemCF
            top_k: number of similar items/users to reference

        Returns list of reason dicts
        """
        reasons = []
        user_history = self.user_ratings.get(user_id, {})

        if is_user_cf:
            # UserCF: find similar users who liked this item
            if sim_matrix is None:
                return reasons
            if user_id >= sim_matrix.shape[0]:
                return reasons

            # Get top similar users
            sims = sim_matrix[user_id]
            top_users = np.argsort(np.abs(sims))[-(top_k + 1):-1][::-1]  # exclude self

            for sim_uid in top_users:
                sim_uid = int(sim_uid)
                if sim_uid >= len(sims):
                    continue
                sim_score = sims[sim_uid]
                if abs(sim_score) < 0.01:
                    continue
                # Find items this similar user rated highly
                sim_user_ratings = self.user_ratings.get(sim_uid, {})
                top_rated = sorted(sim_user_ratings.items(), key=lambda x: x[1], reverse=True)[:3]
                for item, rating in top_rated:
                    if item != recommended_item_id:
                        title = self.get_movie_title(item)
                        reasons.append({
                            "type": "collaborative",
                            "reason_zh": f"品味相似的用户（相似度 {abs(sim_score)*100:.0f}%）也喜欢《{title}》",
                            "reason_en": f"Users with similar taste ({abs(sim_score)*100:.0f}% similar) also liked \"{title}\"",
                            "score": round(abs(sim_score), 4),
                            "source_user": sim_uid,
                            "source_item": item
                        })
                        break
                if len(reasons) >= top_k:
                    break
        else:
            # ItemCF: find similar items the user rated
            if sim_matrix is None:
                return reasons
            if recommended_item_id >= sim_matrix.shape[0]:
                return reasons

            sims = sim_matrix[recommended_item_id]
            top_items = np.argsort(np.abs(sims))[-(top_k + 1):-1][::-1]  # exclude self

            for sim_item in top_items:
                sim_item = int(sim_item)
                if sim_item >= len(sims):
                    continue
                sim_score = sims[sim_item]
                if abs(sim_score) < 0.01 or sim_item not in user_history:
                    continue
                title = self.get_movie_title(sim_item)
                rating = user_history[sim_item]
                reasons.append({
                    "type": "collaborative",
                    "reason_zh": f"与你评分 {rating:.0f} 分的《{title}》相似（相似度 {abs(sim_score)*100:.0f}%）",
                    "reason_en": f"Similar to \"{title}\" which you rated {rating:.0f} ({abs(sim_score)*100:.0f}% similarity)",
                    "score": round(abs(sim_score), 4),
                    "source_item": sim_item
                })
                if len(reasons) >= top_k:
                    break

        return reasons

    def explain(self, user_id, recommended_item_id, algorithm=None,
                sim_matrix=None, is_user_cf=True):
        """
        Generate all applicable explanations for a recommendation.

        Returns dict:
            {"user_id": ..., "item_id": ..., "reasons": [...]}
        """
        all_reasons = []

        # Content-based reason (always available if features loaded)
        content_reasons = self.content_reason(user_id, recommended_item_id)
        all_reasons.extend(content_reasons)

        # Collaborative reason (if similarity matrix provided)
        collab_reasons = self.collaborative_reason(
            user_id, recommended_item_id, sim_matrix, is_user_cf
        )
        all_reasons.extend(collab_reasons)

        if not all_reasons:
            all_reasons.append({
                "type": "default",
                "reason_zh": "基于你的观影历史综合推荐",
                "reason_en": "Recommended based on your viewing history",
                "score": 0.0
            })

        return {
            "user_id": user_id,
            "item_id": recommended_item_id,
            "item_title": self.get_movie_title(recommended_item_id),
            "reasons": all_reasons
        }

    def _get_user_genre_prefs(self, user_id, threshold=4.0):
        """Get set of genres from user's highly-rated movies."""
        prefs = set()
        user_history = self.user_ratings.get(user_id, {})
        for item_id, rating in user_history.items():
            if rating >= threshold:
                genres = self.get_movie_genres(item_id)
                if genres:
                    for g in genres.split("|"):
                        prefs.add(g.strip())
        return prefs
