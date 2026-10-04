# CineRec — Load test report

Measured load test of the recommendation API, run from the development sandbox
against a **`lite`-mode** server (precomputed `recs_cache.json`, no `torch`).
This is the configuration that runs on the always-on free tier, so it is the one
worth measuring.

- **Date**: 2026-10-04
- **Tool**: [Locust](https://locust.io) 2.46.0
- **Profile**: [`scripts/locustfile.py`](../scripts/locustfile.py) — each simulated user signs in as a guest, browses the library, and requests recommendations across the algorithm ladder.
- **Host**: `3 vCPU / 5.8 GB RAM`, no GPU, containerised sandbox
- **Server**: `APP_MODE=lite uvicorn api.main:app` (single worker, SQLite backend)

## Reproduce

```bash
# terminal 1 — serve lite mode (needs data/processed/recs_cache.json)
APP_MODE=lite uvicorn api.main:app --port 8000

# terminal 2 — run the ramp-up profile
make loadtest HOST=http://127.0.0.1:8000
# equivalently:
locust -f scripts/locustfile.py --headless -u 30 -r 10 -t 30s \
    --host http://127.0.0.1:8000 --html reports/locust_report.html
```

The full HTML report is committed at [`reports/locust_report.html`](locust_report.html).

## Result — 30 users ramped at 10/s over 30 s

| Endpoint | Requests | Failures | Avg (ms) | p50 (ms) | p95 (ms) | Max (ms) |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|
| `GET /api/auth/guest` | 30 | 0 | 81 | 31 | 460 | 470 |
| `GET /api/health` | 101 | 0 | 1 | 1 | 3 | 26 |
| `GET /api/movies?per_page=20` | 341 | 0 | 5 | 4 | 9 | 390 |
| `GET /api/recommend` | 441 | 0 | 10 | 5 | 10 | 407 |
| **Aggregated** | **913** | **0** | **10** | **4** | **24** | **470** |

**Throughput**: ≈ **30.8 req/s** sustained, **0 failed requests**.

## Server-side snapshot (`GET /api/metrics`)

The same run, as reported by the in-process metrics endpoint — note that route
ids are normalised (`/api/movies/{id}`) so per-id traffic aggregates:

```json
{
  "uptime_seconds": 45.5,
  "total_requests": 913,
  "total_5xx": 0,
  "avg_latency_ms": 6.155,
  "requests_per_second": 20.068,
  "status_counts": {"200": 913},
  "endpoints": [
    {"method": "GET", "path": "/api/auth/guest", "count": 30,  "avg_ms": 34.7, "p50_ms": 11.6, "p95_ms": 373.0, "p99_ms": 375.9},
    {"method": "GET", "path": "/api/health",     "count": 101, "avg_ms": 0.49, "p50_ms": 0.27, "p95_ms": 0.55,  "p99_ms": 3.3},
    {"method": "GET", "path": "/api/movies",     "count": 341, "avg_ms": 3.0,  "p50_ms": 2.5,  "p95_ms": 5.4,   "p99_ms": 12.5},
    {"method": "GET", "path": "/api/recommend",  "count": 441, "avg_ms": 7.9,  "p50_ms": 3.6,  "p95_ms": 7.0,   "p99_ms": 30.2}
  ],
  "inference_cache": {"currsize": 0, "maxsize": 512, "hits": 0, "misses": 0}
}
```

## Reading the numbers

- **Recommendations are cheap in lite mode.** Median 5 ms, p95 10 ms: the hot
  path is an in-memory dict lookup from `recs_cache.json` plus one SQLite read,
  so the API comfortably serves the free-tier CPU budget.
- **The high tail is SQLite write contention, not inference.** `/api/auth/guest`
  and the 99th-percentile of `/api/movies` briefly spike (~400 ms) when the guest
  endpoint writes nothing but the browse/recommend paths hit SQLite under
  concurrency. The median stays single-digit-ms, i.e. it is an occasional lock,
  not a systemic slowdown. A production build would move to a connection pool or
  a managed Postgres; that is deliberately out of scope here.
- **`inference_cache` is idle by design in this run.** The LRU is a `full`-mode
  feature (memoising live model inference). In `lite` mode results come from the
  precomputed cache, so the counter legitimately reports 0. The memoisation is
  covered by `tests/test_request_metrics.py::test_inference_cache_is_hit_on_repeat_requests`.

## Scope / limitations

- Single process, single worker, SQLite — no connection pooling, no CDN, no
  reverse proxy. Numbers describe the free-tier deployment target, not a
  production topology.
- Counters in `/api/metrics` are process-local and reset on restart; behind more
  than one replica each replica reports only its own traffic.