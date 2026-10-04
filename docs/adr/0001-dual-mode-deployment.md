# ADR 0001 — Dual-mode deployment (`APP_MODE`)

- **Status**: Accepted
- **Context**: The project must ship a public demo on a $0 budget with no GPU.
  The full model stack needs `torch` (~200 MB CPU wheel) and real-time inference;
  free tiers that can host that (e.g. Hugging Face Spaces) sleep after 48 h,
  while always-on tiers (Northflank sandbox) are RAM/CPU constrained and would
  struggle to install torch and run inference.

- **Decision**: One codebase, two runtime modes selected by the `APP_MODE`
  environment variable:
  - **`full`** — `torch` + trained artifacts present; recommendations come from
    live model inference.
  - **`lite`** — no `torch`; recommendations are served from a precomputed
    `recs_cache.json` (every existing user × 5 algorithms). Algorithms with no
    cache entry return an explicit message instead of an error.

- **Consequences**:
  - `requirements.txt` (runtime) and `requirements-train.txt` (torch/vision) are
    split so `lite` images stay small.
  - A precompute step (`scripts/precompute.py`) must run before `lite` deployment.
  - Newly registered users are outside the cache; they fall back to popularity
    (clearly labelled, see the recommendation endpoint).
  - The same Dockerfile builds both images via `--build-arg APP_MODE=...`.