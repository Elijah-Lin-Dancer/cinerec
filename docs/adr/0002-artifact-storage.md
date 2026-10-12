# ADR 0002 — Shipping trained artifacts in the repository

- **Status**: Accepted
- **Context**: The app cannot serve real recommendations without trained models
  and content features. These were previously git-ignored, so a fresh clone (and
  therefore every PaaS deploy that pulls from GitHub) started as an empty shell.

- **Decision**: Commit the minimum set of artifacts needed to serve the app:
  - trained models: `model_usercf.pkl`, `model_itemcf.pkl`, `model_svd.pkl`,
    `model_lightgcn.pkl`, `model_neumf.pt`, `model_multimodalncf.pt`
  - content features: `text_embeddings.npy`, `image_embeddings.npy`,
    `genre_vectors.npy`
  - metadata + results: `movies_enriched.json`, `eval_results.json`,
    `eval_multiseed.json`, `ablation_results.json`, `ablation_multiseed.json`,
    `coldstart_results.json`, `coldstart_multiseed.json`
  - lite-mode cache: `recs_cache.json` (serves recommendations when `torch`
    is absent)
  - seed data: `data/raw/ratings.csv` (MovieLens 100K seed for the database)

  (`.gitignore` ignores `data/processed/*` by default and re-includes exactly
  these files, so new intermediates do not slip in.)

  Second-dataset subtrees (`data/datasets/<name>/`, used by the cross-dataset
  study in [ADR 0005](0005-cross-dataset-validation.md)) follow a *results-only*
  rule: the slice definition (`raw/ratings.csv`), the result JSONs and the
  category vocabulary are committed; the multi-hundred-MB upstream downloads
  and the 25 MB feature arrays are not — rebuild them with
  `make amazon-data && make features DATASET=…`. Trained models for the second
  dataset are not shipped either: it exists for offline evidence, not serving.

- **Size budget**: ~52 MB total (measured); the largest single file is
  `model_itemcf.pkl` at ~17 MB — well under GitHub's 100 MB per-file hard limit,
  so **Git LFS is not required**. If the set grows past that, migrate the `.pkl`
  files to LFS.

- **Consequences**: Deploys work on a fresh clone with no training step. The
  repository is larger; regenerating artifacts (`pytest`-independent
  `python scripts/train_all.py`) produces a reviewable diff.