"""Parameterized SQL over the analytics store. Never free-form SQL from a model.

The model chooses a metric, dimensions, a window and filters from fixed vocabularies; this file
writes the query. That is the whole point: an agent that can reach a database should not be able
to say what the query is, and the query it actually ran comes back with the answer so a reader
can check the arithmetic.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from anydocs import telemetry
from anydocs.config import get_settings
from anydocs.tools.registry import ToolContext, ToolError, ToolSpec, register


class MetricSpec(BaseModel):
    """One answerable metric: its table, the column to aggregate, and how."""

    table: str
    value_column: str
    aggregate: str  # SUM | AVG
    units: str
    dimensions: tuple[str, ...]
    description: str


# The whole vocabulary. A metric, dimension or window not named here cannot be queried, which is
# what makes the SQL safe without anyone sanitizing a string.
METRICS: dict[str, MetricSpec] = {
    "page_views": MetricSpec(
        table="page_views",
        value_column="views",
        aggregate="SUM",
        units="views",
        dimensions=("corpus", "section", "doc_id", "day", "week"),
        description="How often documentation pages were viewed.",
    ),
    "unique_users": MetricSpec(
        table="page_views",
        value_column="unique_users",
        aggregate="SUM",
        units="users",
        dimensions=("corpus", "section", "doc_id", "day", "week"),
        description="How many distinct readers viewed pages.",
    ),
    "search_misses": MetricSpec(
        table="search_misses",
        value_column="misses",
        aggregate="SUM",
        units="searches with no usable result",
        dimensions=("corpus", "section", "query", "day", "week"),
        description="Searches that returned nothing usable — where the docs are missing "
        "something people keep asking for.",
    ),
    "time_to_answer": MetricSpec(
        table="answer_latency",
        value_column="p50_ms",
        aggregate="AVG",
        units="milliseconds (median)",
        dimensions=("corpus", "section", "day", "week"),
        description="Median time to produce an answer.",
    ),
    "time_to_answer_p95": MetricSpec(
        table="answer_latency",
        value_column="p95_ms",
        aggregate="AVG",
        units="milliseconds (95th percentile)",
        dimensions=("corpus", "section", "day", "week"),
        description="95th-percentile time to produce an answer.",
    ),
    "citation_count": MetricSpec(
        table="citations",
        value_column="citations",
        aggregate="SUM",
        units="citations",
        dimensions=("corpus", "doc_id", "day", "week"),
        description="How often each page was cited in an answer.",
    ),
}

# `week` is derived, not stored, so it needs its own SQL expression.
DIMENSION_SQL = {"week": "strftime('%Y-W%W', day)"}
WINDOW_PATTERN = re.compile(r"^(\d{1,4})d$")
MAX_WINDOW_DAYS = 365
DEFAULT_LIMIT = 50


class Input(BaseModel):
    metric: str = Field(description=f"One of: {', '.join(METRICS)}")
    dimensions: list[str] = Field(
        default_factory=list,
        description="Group by, e.g. ['section'] or ['corpus', 'week']. Empty returns one total.",
    )
    window: str = Field(
        default="30d", description="Lookback window in days, e.g. '7d', '30d', '90d'."
    )
    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Equality filters on dimensions, e.g. {'corpus': 'relayr'}.",
    )
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=500)
    order: str = Field(default="desc", description="desc for the largest first, asc for smallest.")


class Output(BaseModel):
    metric: str = ""
    units: str = ""
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total: float = 0.0
    query: str = Field(default="", description="The parameterized SQL that ran.")
    parameters: list[str] = Field(default_factory=list)
    window_days: int = 0
    window_start: str = ""
    window_end: str = ""
    assumptions: list[str] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False


def run(inp: Input, ctx: ToolContext) -> Output:
    spec = METRICS.get(inp.metric)
    if spec is None:
        raise ToolError(f"Unknown metric {inp.metric!r}. Available: {', '.join(sorted(METRICS))}")

    bad = [d for d in inp.dimensions if d not in spec.dimensions]
    if bad:
        raise ToolError(
            f"{inp.metric} cannot be grouped by {', '.join(bad)}. "
            f"Available: {', '.join(spec.dimensions)}"
        )
    bad_filters = [f for f in inp.filters if f not in spec.dimensions]
    if bad_filters:
        raise ToolError(
            f"{inp.metric} cannot be filtered on {', '.join(bad_filters)}. "
            f"Available: {', '.join(spec.dimensions)}"
        )

    match = WINDOW_PATTERN.match(inp.window.strip().lower())
    if not match:
        raise ToolError(f"window must look like '30d'; got {inp.window!r}")
    days = min(int(match.group(1)), MAX_WINDOW_DAYS)

    database = _database_path()
    if not database.exists():
        raise ToolError(
            f"No analytics store at {database}. Seed it with: "
            f"uv run python analytics/seed.py --out {database}"
        )

    sql, parameters, assumptions = _build_query(spec, inp, days)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = [dict(r) for r in connection.execute(sql, parameters).fetchall()]
        # The window the query actually covered, not the table's whole range — reporting the
        # latter alongside a 7-day answer reads as if the answer covered three months.
        window = connection.execute(
            f"SELECT date(MAX(day), ?) AS start, MAX(day) AS end FROM {spec.table}",  # noqa: S608
            (f"-{days} days",),
        ).fetchone()
    except sqlite3.Error as exc:
        raise ToolError(f"analytics query failed: {exc}") from None
    finally:
        connection.close()

    total = round(sum(float(r.get("value") or 0) for r in rows), 3)
    telemetry.set_attributes(
        ctx.span,
        {
            "metrics.metric": inp.metric,
            "metrics.dimensions": inp.dimensions,
            "metrics.window_days": days,
            "metrics.rows": len(rows),
            "metrics.sql": sql,
        },
    )
    return Output(
        metric=inp.metric,
        units=spec.units,
        rows=rows[: inp.limit],
        total=total,
        query=sql,
        parameters=[str(p) for p in parameters],
        window_days=days,
        window_start=str(window["start"] or ""),
        window_end=str(window["end"] or ""),
        assumptions=assumptions,
        row_count=len(rows),
        truncated=len(rows) > inp.limit,
    )


def _database_path() -> Path:
    return get_settings().data_dir / "analytics.db"


def _build_query(
    spec: MetricSpec, inp: Input, days: int
) -> tuple[str, list[Any], list[str]]:
    """Compose the SQL from vocabulary, not from model text.

    Every identifier here comes from `METRICS`, and every value is a bound parameter. The model
    never supplies a fragment of this string.
    """
    assumptions: list[str] = []
    select_parts: list[str] = []
    group_parts: list[str] = []
    for dimension in inp.dimensions:
        expression = DIMENSION_SQL.get(dimension, dimension)
        select_parts.append(f"{expression} AS {dimension}")
        group_parts.append(expression)

    select_parts.append(f"{spec.aggregate}({spec.value_column}) AS value")

    where = ["day >= date((SELECT MAX(day) FROM " + spec.table + "), ?)"]
    parameters: list[Any] = [f"-{days} days"]
    for name, value in sorted(inp.filters.items()):
        where.append(f"{name} = ?")
        parameters.append(value)

    sql = f"SELECT {', '.join(select_parts)} FROM {spec.table} WHERE {' AND '.join(where)}"  # noqa: S608
    if group_parts:
        sql += f" GROUP BY {', '.join(group_parts)}"
    sql += f" ORDER BY value {'DESC' if inp.order.lower() != 'asc' else 'ASC'}"
    sql += f" LIMIT {int(inp.limit) + 1}"

    # The window is relative to the newest day in the data, not to today: the analytics store is
    # seeded to a fixed date so numeric evals have exact expected values, and anchoring to the
    # wall clock would make every answer drift.
    assumptions.append(
        f"Window is the {days} days ending on the most recent day in the store, not on today."
    )
    if not inp.dimensions:
        assumptions.append("No grouping requested, so this is one total across the window.")
    if inp.window.strip().lower() != f"{days}d":
        assumptions.append(f"Window capped at {days} days.")
    return sql, parameters, assumptions


def metric_catalog() -> str:
    """The vocabulary, as the Analyst's prompt shows it."""
    lines = []
    for name, spec in METRICS.items():
        lines.append(
            f"- `{name}` ({spec.units}) — {spec.description} "
            f"Group by: {', '.join(spec.dimensions)}."
        )
    return "\n".join(lines)


SPEC = register(
    ToolSpec(
        name="query_metrics",
        description=(
            "Answer a numeric question about documentation usage: page views, unique readers, "
            "searches that found nothing, time to answer, how often a page is cited. You choose "
            "a metric, what to group by, a window and filters; the query is written for you and "
            "returned with the result so the arithmetic can be checked. "
            + metric_catalog()
        ),
        input_model=Input,
        output_model=Output,
        run=run,
        side_effect="read",
        span_kind="TOOL",
        cost_hint="low",
    )
)
