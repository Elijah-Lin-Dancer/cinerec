# CineRec — Streamlit Community Cloud (full-mode companion)

The main site stays the FastAPI product in `lite` mode (deployed on Render with
`APP_MODE=lite`, served from the precomputed cache). This app is the
**complementary half**: it loads the trained PyTorch models and runs them live,
so the full six-model ladder, the multi-modal content tower and the cold-start
path can be explored interactively at zero cost.

> This is **not** a migration of the product to Streamlit. The bilingual
> FastAPI frontend remains the product; this companion exists only because
> free hosts that run torch are scarce (Hugging Face removed its free Docker
> tier). See [ADR 0001](../../docs/adr/0001-dual-mode-deployment.md) for the
> "no migration" decision it respects.

## Why it lives in its own directory

Streamlit Community Cloud installs dependencies from a file named exactly
`requirements.txt`, looking in the **entrypoint's own directory first**, then the
repository root — and the deploy form has no field for a custom requirements
filename. The real app is `../../streamlit_app.py` (it reuses the product's
modules), where it would pick up the root `requirements.txt` — the `lite` runtime
set, with no torch.

So this directory holds the shim (`streamlit_app.py`), which puts the repository
root on `sys.path` and hands off to the real app, plus its own
`requirements.txt` carrying the CPU-torch stack.

## Why it works on the free tier

Every heavy artefact is already committed, so the app only needs CPU torch at
runtime:

| Asset | Size | Committed |
|-------|------|-----------|
| `text_embeddings.npy` (384-d, Sentence-BERT, precomputed) | ~2.5 MB | yes |
| `image_embeddings.npy` (2048-d, ResNet-50, precomputed) | ~13 MB | yes |
| `genre_vectors.npy` | ~0.1 MB | yes |
| `model_*.pt` / `model_*.pkl` (six trained models) | ~70 MB | yes |
| `movies_enriched.json`, `ratings.csv` (seed data) | small | yes |

`torchvision` and `sentence-transformers` are **not** needed to serve — they are
training-only, so `deploy/streamlit/requirements.txt` installs the CPU torch wheel
and nothing else heavy. Resident memory is ~300–500 MB, well inside the free
tier's per-app ceiling.

## Settings to apply

| Setting | Value |
|---------|-------|
| Repository | `ElijahZhao/cinerec` (must be **public**) |
| Branch | `main` |
| Main file path | `deploy/streamlit/streamlit_app.py` |
| Requirements file | none to set — auto-detected from the entrypoint's directory |
| Python version | `3.10` (Advanced settings) |
| Custom subdomain | e.g. `cinerec` → `cinerec.streamlit.app` |
| Secrets | none required |

Unlike the main site, no `CINEREC_SECRET` is needed: this app is read-only and
has no login.

Pick a Python version that has CPU torch wheels (3.10–3.12). `3.10` matches the
Docker image, so it is the version this app is tested against.

## Free-tier limits to keep in mind

- Unlimited **public** apps; only one private app per account.
- Resources are allocated **per app** (~2.7 GB / 2 cores), not shared across
  the account, so other apps do not compete with this one for memory.
- Apps sleep after a period of inactivity; the first visit wakes the container.
- No custom domain (a `*.streamlit.app` subdomain only) and no uptime SLA.

## Verify

1. Open `https://<your-subdomain>.streamlit.app` and let it wake.
2. **Live recommendation** tab: pick `MultiModalNCF`, confirm a scored grid appears.
3. **Six-model comparison** tab: confirm six columns of titles plus the offline
   metrics and the top-10 Jaccard table render.
4. **Multi-modal ablation** tab: blank `image`, confirm the ranking changes and
   the retrained ablation table appears.
5. **Cold start** tab: confirm the content-only top-10 and the cold-start study
   table appear.