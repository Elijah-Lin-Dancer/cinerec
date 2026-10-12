"""Tests for the feature-engineering id alignment.

The content feature arrays are indexed by raw movie id (row ``i`` = movie ``i``),
so a model can index features directly with ``item_id``. This was historically
wrong (positional indexing), which silently mis-aligned every feature — these
tests pin the contract.
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

import data.preprocess as preprocess  # noqa: E402


def test_encode_genres_is_id_aligned(tmp_path, monkeypatch):
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))

    item_ids = [1, 4]  # note: sparse, non-contiguous ids
    result = preprocess.encode_genres(["Action|Comedy", "Drama"], item_ids)

    assert result.shape == (5, preprocess.NUM_GENRES)
    # Row 0 (an unused id) must stay zero — proves we index by id, not position.
    assert np.count_nonzero(result[0]) == 0
    assert result[1, preprocess.GENRE_LIST.index("Action")] == 1.0
    assert result[1, preprocess.GENRE_LIST.index("Comedy")] == 1.0
    assert result[4, preprocess.GENRE_LIST.index("Drama")] == 1.0
    # And the artefact is actually written for downstream models.
    assert (tmp_path / "genre_vectors.npy").exists()


def test_encode_genres_ignores_unknown_and_empty():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        original = preprocess.PROCESSED_DIR
        preprocess.PROCESSED_DIR = d
        try:
            result = preprocess.encode_genres(["Nonsense", "", None], [1, 2, 3])
        finally:
            preprocess.PROCESSED_DIR = original
    assert np.count_nonzero(result) == 0


def test_encode_texts_is_id_aligned(tmp_path, monkeypatch):
    """Row ``i`` = movie ``i``: text features land at the item id, not the batch position."""
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))

    class _FakeEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            # Deterministic, batch-order-dependent value in the first dimension.
            return np.array(
                [[float(i + 1)] + [0.0] * 383 for i in range(len(texts))],
                dtype=np.float32,
            )

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeEncoder),
    )

    item_ids = [1, 4]  # sparse, non-contiguous ids
    result = preprocess.encode_texts(
        ["a sufficiently long overview", "another long enough overview"], item_ids
    )

    assert result.shape == (5, 384)
    # Row 0 (an unused id) stays zero → indexing by id, not position.
    assert np.count_nonzero(result[0]) == 0
    assert result[1, 0] == 1.0  # first overview → item 1
    assert result[4, 0] == 2.0  # second overview → item 4


def test_encode_images_is_id_aligned(tmp_path, monkeypatch):
    """ResNet features land at the item id, not the poster's batch position."""
    monkeypatch.setattr(preprocess, "PROCESSED_DIR", str(tmp_path))

    class _Tensor:
        """A batch of rows; ``numpy`` materialises each row's marker in slot 0."""

        def __init__(self, rows):
            self.rows = list(rows)

        def squeeze(self, dim=-1):
            return self

        def numpy(self):
            arr = np.zeros((len(self.rows), 2048), dtype=np.float32)
            for i, row in enumerate(self.rows):
                arr[i, 0] = float(row)
            return arr

    class _Image:
        def __init__(self, payload):
            self.payload = payload

        def convert(self, mode):
            return self

    class _Response:
        def __init__(self, url):
            self._payload = url.rsplit("/", 1)[-1].encode()

        @property
        def content(self):
            return self._payload

        def raise_for_status(self):
            pass

    class _Sequential:
        def eval(self):
            return self

        def __call__(self, tensor):
            return tensor  # identity: the marker flows straight through

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _ResNet:
        def children(self):
            return []

    # encode_images imports torch/torchvision/PIL/requests *inside* the function,
    # so we stub the modules themselves rather than the function's globals.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            nn=types.SimpleNamespace(Sequential=lambda *modules: _Sequential()),
            no_grad=lambda: _NoGrad(),
            set_num_threads=lambda n: None,
            # Batch the per-image markers into one tensor-like row list.
            stack=lambda tensors: _Tensor(r for t in tensors for r in t.rows),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchvision",
        types.SimpleNamespace(
            models=types.SimpleNamespace(
                resnet50=lambda *a, **k: _ResNet(),
                ResNet50_Weights=types.SimpleNamespace(IMAGENET1K_V1="weights"),
            ),
            transforms=types.SimpleNamespace(
                Compose=lambda ops: (lambda img: _Tensor([img.payload])),
                Resize=lambda *a, **k: None,
                ToTensor=lambda: None,
                Normalize=lambda *a, **k: None,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "PIL",
        types.SimpleNamespace(Image=types.SimpleNamespace(open=lambda fp: _Image(int(fp.read())))),
    )
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(get=lambda url, timeout=None: _Response(url)),
    )

    item_ids = [1, 4]  # sparse, non-contiguous ids
    result = preprocess.encode_images(["http://img/1", "http://img/4"], item_ids)

    assert result.shape == (5, 2048)
    assert np.count_nonzero(result[0]) == 0  # unused id stays zero
    assert result[1, 0] == 1.0
    assert result[4, 0] == 4.0
