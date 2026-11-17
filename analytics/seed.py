"""Seed the analytics store with deterministic synthetic data.

Deterministic on purpose: the Analyst's evals assert exact numbers, so the generator takes a
fixed seed and no wall-clock date. Build step 8 adds the `query_metrics` body that reads it.

    uv run python analytics/seed.py --days 90 --out data/analytics.db
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = Path(__file__).parent / "schema.sql"
SEED = 20261102

# Sections of the fixture corpus, with a rough popularity weight. The shape matters more than
# the numbers: a reference page gets steady traffic, a changelog spikes and decays.
SECTIONS = {
    "relayr": [
        ("Quickstart", 1.0, "steady"),
        ("Retries", 0.8, "steady"),
        ("Idempotency", 0.5, "steady"),
        ("Delivery receipts", 0.4, "steady"),
        ("Limits", 0.35, "steady"),
        ("Errors", 0.3, "steady"),
        ("Dead letters", 0.2, "steady"),
        ("Verifying webhook signatures", 0.25, "steady"),
        ("Versioning", 0.3, "spiky"),
        ("Changelog", 0.6, "spiky"),
    ]
}

# Queries that find nothing, by section. These are the signal the Analyst exists to surface:
# where the docs are missing something people keep asking for.
MISSES = {
    "Dead letters": ["dead letter queue setup", "automatic dlq", "retry from dlq"],
    "Limits": ["rate limit per endpoint", "increase send quota"],
    "Retries": ["retry on 4xx", "custom backoff function"],
    "Verifying webhook signatures": ["verify signature in go", "rotate webhook secret"],
}


def seed(db_path: Path, days: int, end: date) -> dict[str, int]:
    rng = random.Random(SEED)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(SCHEMA.read_text())
        counts = {"page_views": 0, "search_misses": 0, "answer_latency": 0, "citations": 0}

        for offset in range(days):
            day = (end - timedelta(days=days - 1 - offset)).isoformat()
            current = end - timedelta(days=days - 1 - offset)
            weekday_factor = 0.4 if current.weekday() >= 5 else 1.0

            for corpus, sections in SECTIONS.items():
                for index, (section, weight, shape) in enumerate(sections):
                    doc_id = f"{corpus}-{section.lower().replace(' ', '-')}"
                    base = 120 * weight * weekday_factor
                    if shape == "spiky":
                        # A release spike every 30 days, decaying over the following week.
                        since_release = offset % 30
                        base *= 4.0 if since_release < 2 else max(1.0, 2.0 - since_release * 0.2)
                    views = max(1, int(rng.gauss(base, base * 0.15)))
                    connection.execute(
                        "INSERT OR REPLACE INTO page_views VALUES (?, ?, ?, ?, ?, ?)",
                        (day, corpus, doc_id, section, views, max(1, int(views * 0.62))),
                    )
                    counts["page_views"] += 1

                    connection.execute(
                        "INSERT OR REPLACE INTO citations VALUES (?, ?, ?, ?)",
                        (day, corpus, doc_id, max(0, int(views * 0.08))),
                    )
                    counts["citations"] += 1

                    p50 = int(rng.gauss(1800 + index * 120, 180))
                    connection.execute(
                        "INSERT OR REPLACE INTO answer_latency VALUES (?, ?, ?, ?, ?, ?)",
                        (day, corpus, section, max(1, int(views * 0.1)), p50, int(p50 * 2.4)),
                    )
                    counts["answer_latency"] += 1

                    for query in MISSES.get(section, []):
                        connection.execute(
                            "INSERT OR REPLACE INTO search_misses VALUES (?, ?, ?, ?, ?)",
                            (day, corpus, query, section, max(1, int(rng.gauss(6 * weight, 2)))),
                        )
                        counts["search_misses"] += 1

        connection.commit()
        return counts
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--out", type=Path, default=Path("data/analytics.db"))
    parser.add_argument(
        "--end",
        default="2026-11-01",
        help="Last day to generate. Fixed by default so the data is reproducible.",
    )
    args = parser.parse_args()
    counts = seed(args.out, args.days, date.fromisoformat(args.end))
    print(f"Seeded {args.out}: " + ", ".join(f"{v} {k}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
