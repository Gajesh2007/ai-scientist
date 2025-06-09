"""utils.arxiv
Lightweight facade over the official `arxiv` Python SDK.

The goal is to expose a **stable, typed** interface (`ArxivPaper`, `search`,
`get_paper`, `download_pdf`) while delegating network / parsing details to the
well-maintained upstream library.

Core features
-------------
1. Full-text search with pagination & sorting (`search`).
2. Fetch a single paper by arXiv ID (`get_paper`).
3. Download PDFs with polite rate-limiting (`download_pdf`).
4. Rich `ArxivPaper` dataclass for ergonomic downstream use.

Under the hood:
• Uses `arxiv.Search` and `arxiv.Client` – no custom Atom parsing code.
• Keeps the module dependency footprint tiny (just `arxiv` & `requests`).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Iterable
from urllib.parse import quote_plus

import requests
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "ArxivPaper",
    "ArxivClient",
    "search",
    "get_paper",
    "download_pdf",
]

# Base API endpoint
_API_URL = "http://export.arxiv.org/api/query"
# Default user agent – arXiv requests that automated tools identify themselves.
_USER_AGENT = (
    f"gaj-ai-scientist/0.1 (+https://github.com/Gajesh2007/gaj-ai-scientist;"
    f" python-requests/{requests.__version__})"
)
# Rate-limit: arXiv asks for max. 1 request / 3 seconds for large harvesters.
_MIN_SECONDS_BETWEEN_REQUESTS = 3.0

# Global timestamp of the last API hit – protected by the GIL, fine for our purposes.
_last_request_ts: float = 0.0

# Switch to the official "arxiv" SDK to cut maintenance overhead
try:
    import arxiv as _arxiv  # type: ignore
except ImportError as exc:  # pragma: no cover – guide the developer
    raise ImportError(
        "The 'arxiv' package is required. Add it to your environment via 'pip install arxiv'."
    ) from exc


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rate_limit() -> None:
    """Ensure we do not hit the API faster than allowed."""
    global _last_request_ts
    now = time.time()
    delta = now - _last_request_ts
    if delta < _MIN_SECONDS_BETWEEN_REQUESTS:
        sleep_for = _MIN_SECONDS_BETWEEN_REQUESTS - delta
        logger.debug("Rate-limiting: sleeping %.2f s", sleep_for)
        time.sleep(sleep_for)
    _last_request_ts = time.time()


def _rfc3339_to_dt(ts: str) -> datetime:
    """Convert arXiv timestamps (e.g. 2024-06-14T12:34:56Z) to timezone-aware ``datetime``."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception as exc:  # noqa: BLE001 – broaden to be extra robust
        logger.warning("Could not parse timestamp '%s': %s", ts, exc)
        return datetime.now(tz=timezone.utc)


def _strip_namespace(tag: str) -> str:
    """Remove XML namespace from a tag name."""
    return tag.split("}")[-1] if "}" in tag else tag


def _extract_text(elem: Optional[ET.Element]) -> str:
    return (elem.text or "").strip() if elem is not None else ""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ArxivPaper:
    """A single arXiv paper / entry."""

    arxiv_id: str
    title: str
    summary: str
    authors: List[str] = field(default_factory=list)
    pdf_url: str = ""
    html_url: str = ""
    published: Optional[datetime] = None
    updated: Optional[datetime] = None
    comment: Optional[str] = None
    primary_category: Optional[str] = None
    categories: List[str] = field(default_factory=list)

    # ---------------------------------------------------------------------
    # Derived & helper methods
    # ---------------------------------------------------------------------

    def download_pdf(self, directory: str | os.PathLike[str] = ".", filename: Optional[str] = None) -> str:
        """Download the paper's PDF to *directory* and return the local path.

        If *filename* is omitted a sensible default ``{arxiv_id}.pdf`` is chosen.
        """
        path = download_pdf(self.arxiv_id, directory, filename)
        logger.info("Downloaded '%s' to '%s'", self.arxiv_id, path)
        return path

    # Representation ------------------------------------------------------

    def __str__(self) -> str:  # noqa: D401 – we want a short representation
        authors = ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")
        return f"[{self.arxiv_id}] {self.title} (\u2014 {authors})"


# ---------------------------------------------------------------------------
# Conversion helpers – arxiv.Result -> ArxivPaper
# ---------------------------------------------------------------------------

_SORT_CRITERION_MAP = {
    None: _arxiv.SortCriterion.Relevance,
    "relevance": _arxiv.SortCriterion.Relevance,
    "lastUpdatedDate": _arxiv.SortCriterion.LastUpdatedDate,
    "submittedDate": _arxiv.SortCriterion.SubmittedDate,
}

_SORT_ORDER_MAP = {
    None: _arxiv.SortOrder.Descending,
    "descending": _arxiv.SortOrder.Descending,
    "ascending": _arxiv.SortOrder.Ascending,
}


def _result_to_paper(result: _arxiv.Result) -> ArxivPaper:
    """Convert :pyclass:`arxiv.Result` to our :class:`ArxivPaper`."""
    authors = [a.name for a in result.authors]
    arxiv_id = result.get_short_id()  # e.g. "2406.01234"

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=result.title.strip(),
        summary=result.summary.strip(),
        authors=authors,
        pdf_url=result.pdf_url or "",
        html_url=result.entry_id,
        published=result.published,
        updated=result.updated,
        comment=getattr(result, "comment", None),
        primary_category=result.primary_category,
        categories=[(cat.term if hasattr(cat, "term") else str(cat)) for cat in (result.categories or [])],
    )


# ---------------------------------------------------------------------------
# Internal client leveraging arxiv.Client
# ---------------------------------------------------------------------------

class _ArxivSDKClient:
    """Small wrapper utilising :pymod:`arxiv`'s high-level API."""

    def __init__(self, num_retries: int = 3, delay_seconds: float = 3.0):
        self._client = _arxiv.Client(num_retries=num_retries, delay_seconds=delay_seconds)

    def fetch(self, search: _arxiv.Search) -> Iterable[_arxiv.Result]:
        return self._client.results(search)


# Re-use singleton logic from previous implementation
_sdk_client: Optional[_ArxivSDKClient] = None


def _get_sdk_client() -> _ArxivSDKClient:
    global _sdk_client
    if _sdk_client is None:
        _sdk_client = _ArxivSDKClient()
    return _sdk_client


# ---------------------------------------------------------------------------
# Public API – implementations revised to call the SDK
# ---------------------------------------------------------------------------

def search(
    query: str,
    *,
    start: int = 0,
    max_results: int = 10,
    sort_by: str | None = None,
    sort_order: str | None = None,
) -> List[ArxivPaper]:
    """Search arXiv papers using the official SDK while preserving our return type."""

    sort_criterion = _SORT_CRITERION_MAP.get(sort_by, _arxiv.SortCriterion.Relevance)
    order_enum = _SORT_ORDER_MAP.get(sort_order, _arxiv.SortOrder.Descending)

    search_obj = _arxiv.Search(
        query=query,
        max_results=max_results + start,
        sort_by=sort_criterion,
        sort_order=order_enum,
    )

    client = _get_sdk_client()
    results_iter = client.fetch(search_obj)

    # Skip "start" results for pagination, then collect up to max_results
    papers: List[ArxivPaper] = []
    for idx, result in enumerate(results_iter):
        if idx < start:
            continue
        if len(papers) >= max_results:
            break
        papers.append(_result_to_paper(result))

    return papers


def get_paper(arxiv_id: str) -> Optional[ArxivPaper]:
    """Retrieve a single paper by arXiv ID using the SDK."""
    search_obj = _arxiv.Search(id_list=[arxiv_id])
    client = _get_sdk_client()
    try:
        result = next(client.fetch(search_obj), None)
    except StopIteration:
        result = None
    return _result_to_paper(result) if result else None


def download_pdf(
    arxiv_id: str,
    directory: str | os.PathLike[str] = ".",
    filename: Optional[str] = None,
) -> str:
    """Download *arxiv_id*'s PDF into *directory* and return the local path."""
    if filename is None:
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
    os.makedirs(directory, exist_ok=True)
    dest_path = os.path.join(directory, filename)

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    logger.debug("Downloading PDF from %s", pdf_url)

    _rate_limit()
    try:
        with requests.get(pdf_url, stream=True, headers={"User-Agent": _USER_AGENT}, timeout=60) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
    except requests.RequestException as exc:
        logger.error("Failed to download PDF for %s: %s", arxiv_id, exc)
        raise

    return os.path.abspath(dest_path)
