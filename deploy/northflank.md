# CineRec — Northflank Sandbox (lite mode)

The **always-on** free fallback. `APP_MODE=lite` omits torch and serves
recommendations from `data/processed/recs_cache.json`, so it fits Northflank's
free sandbox CPU/RAM allowance and never sleeps.

## Settings to apply

| Setting | Value |
|---------|-------|
| Build source | Git → your fork of this repository |
| Build type | Dockerfile (`/Dockerfile`) |
| Build argument | `APP_MODE=lite` |
| Port | `8000` (public, HTTP) |
| Health check | HTTP `GET /api/health` on port `8000` |
| Environment variable | `APP_MODE=lite` |
| Resources | 0.2 CPU / 512 MiB (free sandbox) |

## Pre-requisite

Generate the cache locally and commit it before deploying (`.gitignore` re-includes
`data/processed/recs_cache.json` explicitly, so `git add` picks it up):

```bash
python scripts/precompute.py     # writes data/processed/recs_cache.json
```

Without the cache, `lite` mode answers every request with an explicit "not
available in this precomputed deployment" message instead of inventing results.

## Verify

```bash
curl -sf https://<your-service>.northflank.app/api/health   # {"status":"ok"}
curl -s  "https://<your-service>.northflank.app/api/movies?per_page=1" | head
```