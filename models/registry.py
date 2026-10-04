"""Single source of truth for the servable algorithm ladder and model loading.

Both the live API (``api/recommend.py``) and the offline precompute script
(``scripts/precompute.py``) load models through here, so the set of algorithms
and the way each artefact is loaded can never drift apart.
"""
import os
import pickle

from config import PROCESSED_DIR

#: The models, in ascending order of sophistication.
ALGORITHMS = ("UserCF", "ItemCF", "SVD", "NeuMF", "LightGCN", "MultiModalNCF")

MODEL_FILES = {
    "UserCF": "model_usercf.pkl",
    "ItemCF": "model_itemcf.pkl",
    "SVD": "model_svd.pkl",
    "NeuMF": "model_neumf.pt",
    "LightGCN": "model_lightgcn.pkl",
    "MultiModalNCF": "model_multimodalncf.pt",
}


class AlgorithmUnavailable(RuntimeError):
    """Raised when a known algorithm cannot run in the current deployment."""


def _path(name):
    return os.path.join(PROCESSED_DIR, MODEL_FILES[name])


def load_model(name):
    """Load a trained model from disk.

    Returns the model, or ``None`` when ``name`` is not a recognised algorithm.
    Raises :class:`AlgorithmUnavailable` when the algorithm is recognised but
    cannot be served here — a missing artefact, or missing optional ML
    dependencies such as torch (which ``lite`` deployments deliberately omit).
    """
    if name not in ALGORITHMS:
        return None

    try:
        if name in ("UserCF", "ItemCF", "SVD"):
            module = {
                "UserCF": ("models.user_cf", "UserCF"),
                "ItemCF": ("models.item_cf", "ItemCF"),
                "SVD": ("models.svd_als", "SVDALS"),
            }[name]
            __import__(module[0])  # ensure the class is importable first
            path = _path(name)
            if not os.path.exists(path):
                raise AlgorithmUnavailable(f"{name} model artifact not found")
            with open(path, "rb") as f:
                return pickle.load(f)

        if name == "NeuMF":
            from models.neumf import NeuMF
            model = NeuMF(embedding_dim=32, mlp_dims=(128, 64, 32))
        elif name == "LightGCN":
            from models.lightgcn import LightGCN
            model = LightGCN(embedding_dim=64, num_layers=3)
        else:  # MultiModalNCF
            from models.multimodal_ncf import MultiModalNCF
            model = MultiModalNCF(embedding_dim=32, mlp_dims=(128, 64, 32))

        path = _path(name)
        if not os.path.exists(path):
            raise AlgorithmUnavailable(f"{name} model artifact not found")
        model.load(path)
        return model
    except ImportError as e:
        raise AlgorithmUnavailable(
            f"{name} requires optional ML dependencies ({e}); not available in this deployment"
        ) from e
