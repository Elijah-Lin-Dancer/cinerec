# ADR 0002 — Shipping trained artifacts in the repository

- **Status**: Accepted
- **Context**: The app cannot serve real recommendations without trained models
  and content features. These were previously git-ignored, so a fresh clone (and
  therefore every PaaS deploy that pulls from GitHub) started as an empty shell.

- **Decision**: Commit the minimum set of artifacts needed to serve the app:
  - trained models: `model_usercf.pkl`, `model_itemcf.pkl`, `model_svd.pkl`,
    `model_neumf.pt`, `model_multimodalncf.pt`
  - content features: `text_embeddings.npy`, `image_embeddings.npy`,
    `genre_vectors.npy`
  - metadata + results: `movies_enriched.json`, `eval_results.json`
  - seed data: `data/raw/ratings.csv` (MovieLens 100K seed for the database)

  (`.gitignore` ignores `data/processed/*` by default and re-includes exactly
  these files, so new intermediates do not slip in.)

- **Size budget**: ~74 MB total; the largest single file is `model_itemcf.pkl`
  at ~34 MB — well under GitHub's 100 MB per-file hard limit, so **Git LFS is
  not required**. If the set grows past that, migrate the `.pkl` files to LFS.

- **Consequences**: Deploys work on a fresh clone with no training step. The
  repository is larger; regenerating artifacts (`pytest`-independent
  `python scripts/train_all.py`) produces a reviewable diff.