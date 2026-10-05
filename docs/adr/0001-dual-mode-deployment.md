# ADR 0001 — Dual-mode deployment (`APP_MODE`)

- **Status**: Accepted (hosting topology revised — see "Deployment topology")
- **Context**: The project must ship a public demo on a **$0 budget with no
  credit card** and no GPU. The full model stack needs `torch` (~200 MB CPU
  wheel) and real-time inference; free hosts that can run that are scarce, and
  the always-on tiers that looked promising (Northflank sandbox, Koyeb) turned
  out to require a card even on their free plans, so they are out.

- **Decision**: One codebase, two runtime modes selected by the `APP_MODE`
  environment variable:
  - **`full`** — `torch` + trained artefacts present; recommendations come from
    live model inference.
  - **`lite`** — no `torch`; recommendations are served from a precomputed
    `recs_cache.json` (every existing user × every servable algorithm). An
    algorithm with no cache entry returns an explicit message instead of an error.

## Deployment topology

Each mode is hosted where it actually fits a free tier:

| Mode | Host | Why |
|------|------|-----|
| `lite` (the product) | **Render** free web service — Docker, 0.1 CPU / 512 MB | No card required; the image has no torch, so it builds and runs inside 512 MB. It sleeps after ~15 min idle and is kept warm by `.github/workflows/keepalive.yml`. |
| `full` (companion) | **Streamlit Community Cloud** — `streamlit_app.py` | No card required; ~2.7 GB per app is enough for CPU torch, and public apps are unlimited. Loads the trained models and runs them live. |

**The product UI is not migrated to Streamlit.** The bilingual FastAPI frontend
(`frontend/`) stays the product and is what Render serves in `lite` mode. The
Streamlit app is a read-only companion that exercises the `full` path — the
six-model ladder, the multi-modal content tower and the cold-start route — so
those can be explored interactively without a GPU host. It demonstrates the
models; it does not replace the product, and the engineering surface (REST API,
auth, LRU cache, metrics) stays intact for review. See `deploy/streamlit/README.md`.

### Hosts considered and rejected

- **Hugging Face Spaces** — its free Docker tier was withdrawn, so a Space can no
  longer build from a Dockerfile on the free plan.
- **Northflank** / **Koyeb** — free plans require a credit card at sign-up, which
  fails the "$0, no card" constraint.

## Consequences

- `requirements.txt` (runtime) and `requirements-train.txt` (torch/vision) are
  split so `lite` images stay small; the companion carries its own
  `deploy/streamlit/requirements.txt` with the CPU torch wheel.
- A precompute step (`scripts/precompute.py`) must run before `lite` deployment.
- Newly registered users are outside the cache; they fall back to popularity
  (clearly labelled — see the recommendation endpoint).
- The same Dockerfile builds both images via `--build-arg APP_MODE=...`.