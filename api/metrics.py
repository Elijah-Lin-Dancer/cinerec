"""In-process request metrics: throughput and latency, with no external service.

Deliberately dependency-free (no Prometheus, no exporter): the goal is an
inspectable snapshot that suits a portfolio deployment, not a monitoring stack.
Counters are **process-local** and reset on restart; behind more than one replica
each replica reports only its own traffic. That limitation is intentional and
documented rather than hidden.
"""
import re
import threading
import time
from collections import defaultdict, deque

#: Collapse numeric path segments so per-id requests aggregate into one series
#: (e.g. ``/api/movies/42`` → ``/api/movies/{id}``). The app's path params are
#: all integer ids, so a numeric match is an accurate stand-in for the template.
_ID_SEGMENT = re.compile(r"/\d+(?=/|$)")

#: Cap per-endpoint latency samples so memory stays bounded under load.
#: Percentiles are therefore computed over a rolling window, not all history.
_MAX_LATENCY_SAMPLES = 2000


def _normalise_path(path):
    """Return a stable grouping key for a request path."""
    return _ID_SEGMENT.sub("/{id}", path)


class MetricsRegistry:
    """Thread-safe counters for request counts, status codes and latency."""

    def __init__(self):
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._total = 0
        self._errors_5xx = 0
        self._total_latency_ms = 0.0
        self._counts = defaultdict(int)  # (method, path) -> count
        self._status = defaultdict(int)  # status code -> count
        self._latency = defaultdict(lambda: deque(maxlen=_MAX_LATENCY_SAMPLES))

    def record(self, method, path, status, latency_ms):
        """Record one completed request."""
        with self._lock:
            self._total += 1
            self._total_latency_ms += latency_ms
            self._counts[(method, path)] += 1
            self._status[int(status)] += 1
            self._latency[(method, path)].append(latency_ms)
            if status >= 500:
                self._errors_5xx += 1

    @staticmethod
    def _percentile(samples, pct):
        if not samples:
            return None
        ordered = sorted(samples)
        idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
        return round(ordered[idx], 3)

    def snapshot(self):
        """Return a JSON-serialisable view of the counters so far."""
        with self._lock:
            uptime = max(time.time() - self._started_at, 1e-9)
            endpoints = []
            for (method, path), count in sorted(self._counts.items()):
                samples = list(self._latency[(method, path)])
                endpoints.append({
                    "method": method,
                    "path": path,
                    "count": count,
                    "avg_ms": round(sum(samples) / len(samples), 3) if samples else None,
                    "p50_ms": self._percentile(samples, 50),
                    "p95_ms": self._percentile(samples, 95),
                    "p99_ms": self._percentile(samples, 99),
                })
            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._total,
                "total_5xx": self._errors_5xx,
                "avg_latency_ms": round(self._total_latency_ms / self._total, 3) if self._total else 0.0,
                "requests_per_second": round(self._total / uptime, 3),
                "status_counts": {str(k): v for k, v in sorted(self._status.items())},
                "endpoints": endpoints,
            }


#: Process-wide registry, shared by the middleware and the /api/metrics endpoint.
registry = MetricsRegistry()


async def metrics_middleware(request, call_next):
    """Time every request and record method / normalised path / status / latency."""
    started = time.perf_counter()
    path = _normalise_path(request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        registry.record(request.method, path, 500, (time.perf_counter() - started) * 1000)
        raise
    registry.record(request.method, path, response.status_code, (time.perf_counter() - started) * 1000)
    return response
