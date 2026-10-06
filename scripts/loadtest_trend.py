"""Append one load-test run to the trend file.

The scheduled load-test workflow (``.github/workflows/loadtest.yml``) runs Locust
headless with ``--csv``; this turns that run's *Aggregated* row into a single row
appended to ``reports/loadtest_trend.csv``, so run-over-run numbers accumulate
instead of being overwritten by the latest report.

Usage::

    python scripts/loadtest_trend.py \
        --stats reports/locust_scheduled_stats.csv \
        --trend reports/loadtest_trend.csv
"""
import argparse
import csv
import datetime as dt
import os
import sys

#: Trend columns, in order. Parsed from Locust's stats CSV by header name.
FIELDS = [
    "timestamp_utc",
    "requests",
    "failures",
    "requests_per_s",
    "avg_ms",
    "median_ms",
    "p95_ms",
    "max_ms",
]


def _pick(row, *names, default=""):
    """Return the first present, non-empty value among ``names`` (Locust renames columns)."""
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return default


def _aggregate_row(stats_path):
    """Return the ``Aggregated`` row from a Locust stats CSV."""
    with open(stats_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("Name") or "").strip() == "Aggregated":
                return row
    raise SystemExit(f"no 'Aggregated' row found in {stats_path}")


def build_row(stats_path, now=None):
    row = _aggregate_row(stats_path)
    now = now or dt.datetime.now(dt.timezone.utc)
    return {
        "timestamp_utc": now.replace(microsecond=0).isoformat(),
        "requests": _pick(row, "Request Count"),
        "failures": _pick(row, "Failure Count"),
        "requests_per_s": _pick(row, "Requests/s"),
        "avg_ms": _pick(row, "Average Response Time"),
        "median_ms": _pick(row, "Median Response Time", "50%"),
        "p95_ms": _pick(row, "95%"),
        "max_ms": _pick(row, "Max Response Time", "Max"),
    }


def append_row(trend_path, new_row):
    """Write ``new_row`` to ``trend_path``, creating the header if needed."""
    os.makedirs(os.path.dirname(trend_path) or ".", exist_ok=True)
    needs_header = not os.path.exists(trend_path) or os.path.getsize(trend_path) == 0
    with open(trend_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(new_row)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Append a Locust run to the trend CSV.")
    parser.add_argument("--stats", required=True, help="Locust --csv stats file (…_stats.csv)")
    parser.add_argument("--trend", required=True, help="Trend CSV to append to")
    args = parser.parse_args(argv)

    if not os.path.exists(args.stats):
        raise SystemExit(f"stats file not found: {args.stats}")

    new_row = build_row(args.stats)
    append_row(args.trend, new_row)
    print("appended trend row:", new_row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
