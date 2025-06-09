"""utils.semantic_scholar
A thin, production-grade wrapper around the official Semantic Scholar Python
SDK (``semanticscholar``).

Why a wrapper?
--------------
While the upstream SDK is already pleasant to use, a narrow abstraction helps
us keep a **stable contract** for the rest of the code-base and makes it easier
to swap out the backend or augment it with caching, logging, etc. later on.

Features provided
~~~~~~~~~~~~~~~~~
1. Search papers via free-text queries (``search_papers``).
2. Retrieve a single paper by any Semantic Scholar *paperId*, CorpusID, DOI,
   or arXiv ID (``get_paper``).
3. Retrieve an author profile (``get_author``).
4. Dataclass representations (``S2Paper`` / ``S2Author``) for strong typing
   and IDE auto-completion.
5. Minimal, **tech-debt-free** implementation powered by the maintained
   ``semanticscholar`` package rather than bespoke HTTP code.

Notes
-----
Semantic Scholar offers generous rate-limits for academic and open-source
projects, but you may obtain a personal API key for higher throughput.
Set it via the ``S2_API_KEY`` environment variable or pass it explicitly when
creating the client.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from semanticscholar import SemanticScholar  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    "S2Paper",
    "S2Author",
    "SemanticScholarClient",
    "search",
    "get_paper",
    "get_author",
]


# ---------------------------------------------------------------------------
# Data-model helpers
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class S2Author:
    """Minimal author representation."""

    author_id: str
    name: str
    affiliations: List[str] = field(default_factory=list)
    url: str = ""
    paper_count: Optional[int] = None
    citation_count: Optional[int] = None


@dataclass(slots=True)
class S2Paper:
    """Minimal paper representation."""

    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[S2Author] = field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    url: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None
    influential_citation_count: Optional[int] = None
    is_open_access: Optional[bool] = None
    open_access_url: Optional[str] = None

    # Convenience -----------------------------------------------------------

    def __str__(self) -> str:  # noqa: D401 – concise str representation
        author_names = ", ".join(a.name for a in self.authors[:3])
        if len(self.authors) > 3:
            author_names += " et al."
        return f"[{self.paper_id}] {self.title} — {author_names} ({self.year or 'n/a'})"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _author_from_dict(data: dict) -> S2Author:
    return S2Author(
        author_id=data.get("authorId") or data.get("id", ""),
        name=data.get("name", ""),
        affiliations=data.get("affiliations", []) or [],
        url=data.get("url", ""),
        paper_count=data.get("paperCount"),
        citation_count=data.get("citationCount"),
    )


def _paper_from_dict(data: dict) -> S2Paper:
    authors = [_author_from_dict(a) for a in data.get("authors", [])]
    oa_info = data.get("openAccessPdf") or {}
    return S2Paper(
        paper_id=data.get("paperId") or data.get("paper_id", ""),
        title=data.get("title", ""),
        abstract=data.get("abstract"),
        authors=authors,
        year=data.get("year"),
        venue=data.get("venue"),
        url=data.get("url", ""),
        arxiv_id=data.get("arxivId"),
        doi=data.get("doi"),
        citation_count=data.get("citationCount"),
        influential_citation_count=data.get("influentialCitationCount"),
        is_open_access=data.get("isOpenAccess"),
        open_access_url=oa_info.get("url"),
    )


# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------

class SemanticScholarClient:
    """Lightweight wrapper around :class:`semanticscholar.SemanticScholar`."""

    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        api_key = api_key or os.getenv("S2_API_KEY")
        self._sch = SemanticScholar(api_key=api_key, timeout=timeout)
        logger.debug("Initialised SemanticScholar client with key=%s", bool(api_key))

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def search_papers(
        self,
        query: str,
        *,
        limit: int = 10,
        fields: Optional[List[str]] = None,
    ) -> List[S2Paper]:
        """Search papers and return a list of :class:`S2Paper`."""
        logger.debug("Searching Semantic Scholar for '%s' (limit=%d)", query, limit)
        raw_results = self._sch.search_paper(query, limit=limit, fields=fields)
        papers = [_paper_from_dict(p) for p in raw_results]
        return papers

    def get_paper(self, paper_id: str, *, fields: Optional[List[str]] = None) -> Optional[S2Paper]:
        """Retrieve a single paper by Semantic Scholar ID / DOI / arXiv ID."""
        try:
            raw = self._sch.get_paper(paper_id, fields=fields)
            return _paper_from_dict(raw) if raw else None
        except Exception as exc:  # noqa: BLE001 – network / API errors
            logger.error("Failed to fetch paper '%s': %s", paper_id, exc)
            return None

    def get_author(self, author_id: str, *, fields: Optional[List[str]] = None) -> Optional[S2Author]:
        """Retrieve an author profile by ID."""
        try:
            raw = self._sch.get_author(author_id, fields=fields)
            return _author_from_dict(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch author '%s': %s", author_id, exc)
            return None


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

_default_client: Optional[SemanticScholarClient] = None


def _get_default_client() -> SemanticScholarClient:
    global _default_client
    if _default_client is None:
        _default_client = SemanticScholarClient()
    return _default_client


def search(
    query: str,
    *,
    limit: int = 10,
    fields: Optional[List[str]] = None,
) -> List[S2Paper]:
    """Search papers. Thin wrapper around :pymeth:`SemanticScholarClient.search_papers`."""
    client = _get_default_client()
    return client.search_papers(query, limit=limit, fields=fields)


def get_paper(paper_id: str, *, fields: Optional[List[str]] = None) -> Optional[S2Paper]:
    """Convenience wrapper to fetch a single paper."""
    client = _get_default_client()
    return client.get_paper(paper_id, fields=fields)


def get_author(author_id: str, *, fields: Optional[List[str]] = None) -> Optional[S2Author]:
    """Convenience wrapper to fetch an author."""
    client = _get_default_client()
    return client.get_author(author_id, fields=fields)
