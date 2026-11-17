"""Reading a page that `crawl_site` wrote to disk.

`crawl_site` writes raw pages into the snapshot rather than returning them, so a 500-page crawl
does not try to pass itself through a model's context. The manifest's `path` is the handoff, and
this is the only thing that opens one — with the path confined to the data directory, because
the path reaching these tools came from a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anydocs.config import get_settings


class RawPageError(RuntimeError):
    pass


def load_raw_page(page_path: str) -> dict[str, Any]:
    """Load a crawled page by the path `crawl_site` reported.

    Refuses any path outside the configured data directory: the argument arrived from a model,
    so it is treated as untrusted input rather than as a filename.
    """
    data_dir = get_settings().data_dir.resolve()
    try:
        resolved = Path(page_path).resolve()
    except (OSError, ValueError) as exc:
        raise RawPageError(f"{page_path!r} is not a usable path") from exc

    if not resolved.is_relative_to(data_dir):
        raise RawPageError(
            f"{page_path!r} is outside the knowledge store at {data_dir}. Pass a `path` from a "
            f"crawl_site result."
        )
    if not resolved.is_file():
        raise RawPageError(f"No crawled page at {page_path!r}")

    try:
        record = json.loads(resolved.read_text())
    except (OSError, ValueError) as exc:
        raise RawPageError(f"{page_path!r} is not a readable crawled page") from exc
    if "markdown" not in record:
        raise RawPageError(f"{page_path!r} holds no page content")
    return record


def crawled_page(page_path: str) -> dict[str, Any]:
    """`load_raw_page`, with the failure phrased as something a model can act on.

    Both `digest_page` and `save_note` accept a crawl manifest path and both want a `ToolError`
    rather than a `RawPageError`, so the translation lives here instead of in each of them.
    """
    from anydocs.tools.registry import ToolError

    try:
        return load_raw_page(page_path)
    except RawPageError as exc:
        raise ToolError(str(exc)) from None
