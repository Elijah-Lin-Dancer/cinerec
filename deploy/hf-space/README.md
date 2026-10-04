# CineRec — Hugging Face Space (full mode)

This folder documents how to run CineRec on **Hugging Face Spaces** in `APP_MODE=full`
(torch + trained artefacts, live inference).

## Why HF Spaces for this mode

The `full` stack installs `torch` and runs inference on every request. HF Spaces'
free CPU tier gives 16 GB RAM / 2 vCPU, which is enough for that, and it builds
directly from a Dockerfile. The trade-off is that a Space **sleeps after 48 h of
inactivity** — hence the Northflank/Render `lite` deployments act as an
always-on fallback.

## Steps

1. Create a new Space → **SDK: Docker** → **Blank**.
2. Copy the repository into the Space, or add it as a remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/cinerec
   git push space main
   ```
3. Ensure the Space root contains:
   - `Dockerfile` (multi-stage, `ARG APP_MODE=full`)
   - `README.md` with the frontmatter below
4. Set `APP_MODE=full` in the Space **Settings → Variables** (the default is
   already `full`, so this is optional).
5. The app listens on port `8000` (`app_port: 8000`).

## `README.md` frontmatter for the Space

```yaml
---
title: CineRec
emoji: 🎬
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---
```

## Notes

- The trained artefacts ship in the repository (see `docs/adr/0002`), so the
  Space needs no training step at boot.
- The SQLite database is seeded automatically from `data/raw/ratings.csv` +
  `data/processed/movies_enriched.json` on first start (`db.database.seed_if_empty`).