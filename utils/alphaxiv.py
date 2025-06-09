"""utils.alphaxiv
A typed, production-grade wrapper around the public alphaXiv REST API.

Currently supported endpoints
----------------------------
1. **Top organizations** – ``GET /v1/organizations/top``
2. **Trending papers**  – ``GET /v2/papers/trending-papers``

Example usage
~~~~~~~~~~~~~
>>> from utils.alphaxiv import get_trending_papers
>>> papers = get_trending_papers(sort_by="Hot", custom_categories=["fine-tuning"])
>>> print(papers[0].title)
"Towards LLM-Centric Multimodal Fusion: A Survey on Integration Strategies …"
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

_BASE_URL = "https://api.alphaxiv.org"
_DEFAULT_TIMEOUT = 20  # seconds
_USER_AGENT = (
    f"gaj-ai-scientist/0.1 (+https://github.com/Gajesh2007/gaj-ai-scientist;"
    f" python-requests/{requests.__version__})"
)
# AlphaXIV is fairly generous, but we add a small delay to play nice.
_MIN_SECONDS_BETWEEN_REQUESTS = 0.8
_last_call_ts: float = 0.0


def _rate_limit() -> None:
    global _last_call_ts
    delta = time.time() - _last_call_ts
    if delta < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - delta)
    _last_call_ts = time.time()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Organization:
    """A publishing organization / institution on alphaXiv."""

    id: str
    name: str
    image_url: Optional[str] = None
    paper_count: Optional[int] = None

    def __str__(self) -> str:  # noqa: D401
        return f"{self.name} ({self.paper_count:,} papers)" if self.paper_count else self.name


@dataclass(slots=True)
class PaperMetrics:
    visits_last_24h: Optional[int] = None
    visits_last_7d: Optional[int] = None
    visits_last_30d: Optional[int] = None
    visits_total: Optional[int] = None
    hot_score: Optional[int] = None


@dataclass(slots=True)
class AlphaPaper:
    """A single paper entry returned by the trending-papers endpoint."""

    alpha_id: str  # internal _id
    arxiv_id: str  # universal_paper_id
    title: str
    url: str
    categories: List[str] = field(default_factory=list)
    custom_categories: List[str] = field(default_factory=list)
    published_at: Optional[str] = None  # ISO-timestamp as-is
    metrics: Optional[PaperMetrics] = None
    organizations: List[Organization] = field(default_factory=list)

    # Convenience -----------------------------------------------------------

    def __str__(self) -> str:  # noqa: D401
        cats = ", ".join(self.custom_categories or self.categories[:3])
        return f"[{self.arxiv_id}] {self.title} — {cats}"


# ---------------------------------------------------------------------------
# Client implementation
# ---------------------------------------------------------------------------

class AlphaXivClient:
    """Thin wrapper around the alphaXiv API."""

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT, max_retries: int = 3):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(max_retries=max_retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        _rate_limit()
        url = _BASE_URL + path
        logger.debug("alphaXiv request: %s params=%s", url, params)
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.error("alphaXiv request failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_top_organizations(self, *, limit: int = 20) -> List[Organization]:
        """Return a list of top publishing organizations."""
        data = self._get("/v1/organizations/top")
        orgs_raw = data.get("data", [])
        orgs = [
            Organization(
                id=o.get("id") or o.get("_id", ""),
                name=o.get("name", ""),
                image_url=o.get("image"),
                paper_count=o.get("paperCount") or o.get("paper_count"),
            )
            for o in orgs_raw[:limit]
        ]
        return orgs

    def get_trending_papers(
        self,
        *,
        page_num: int = 0,
        page_size: int = 10,
        sort_by: str = "Hot",
        custom_categories: Optional[List[str]] = None,
    ) -> List[AlphaPaper]:
        """Fetch trending papers with optional category filtering."""
        params: Dict[str, Any] = {
            "page_num": page_num,
            "page_size": page_size,
            "sort_by": sort_by,
        }
        if custom_categories:
            params["custom_categories"] = ",".join(custom_categories)

        json_data = self._get("/v2/papers/trending-papers", params)
        papers_raw = (
            json_data.get("data", {}).get("trending_papers")
            if isinstance(json_data.get("data"), dict)
            else json_data.get("trending_papers", [])
        ) or []

        papers: List[AlphaPaper] = []
        for p in papers_raw:
            metrics_dict = p.get("metrics", {})
            visits = metrics_dict.get("visits_count", {})
            papers.append(
                AlphaPaper(
                    alpha_id=p.get("_id", ""),
                    arxiv_id=p.get("universal_paper_id", ""),
                    title=p.get("title", ""),
                    url=p.get("source", {}).get("url", ""),
                    categories=p.get("categories", []),
                    custom_categories=p.get("custom_categories", []),
                    published_at=p.get("first_publication_date") or p.get("publication_date"),
                    metrics=PaperMetrics(
                        visits_last_24h=visits.get("last24Hours"),
                        visits_last_7d=visits.get("last7Days"),
                        visits_last_30d=visits.get("last30Days"),
                        visits_total=visits.get("all"),
                        hot_score=metrics_dict.get("weighted_visits", {}).get("hot"),
                    ),
                    organizations=[
                        Organization(
                            id=o.get("_id", o.get("id", "")),
                            name=o.get("name", ""),
                        )
                        for o in p.get("organizationInfo", p.get("organizations", []))
                    ],
                )
            )
        return papers


# ---------------------------------------------------------------------------
# Module-level convenience wrappers
# ---------------------------------------------------------------------------

_default_client: Optional[AlphaXivClient] = None


def _client() -> AlphaXivClient:
    global _default_client
    if _default_client is None:
        _default_client = AlphaXivClient()
    return _default_client


def get_top_organizations(*, limit: int = 20) -> List[Organization]:
    """Module-level wrapper for :meth:`AlphaXivClient.get_top_organizations`."""
    return _client().get_top_organizations(limit=limit)


def get_trending_papers(
    *,
    page_num: int = 0,
    page_size: int = 10,
    sort_by: str = "Hot",
    custom_categories: Optional[List[str]] = None,
) -> List[AlphaPaper]:
    """Module-level wrapper for :meth:`AlphaXivClient.get_trending_papers`."""
    return _client().get_trending_papers(
        page_num=page_num,
        page_size=page_size,
        sort_by=sort_by,
        custom_categories=custom_categories,
    )


__all__ = [
    "Organization",
    "AlphaPaper",
    "AlphaXivClient",
    "get_top_organizations",
    "get_trending_papers",
]

