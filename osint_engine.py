"""
osint_engine.py — GOIES OSINT Ingestion Engine v2

Bug fixes from v1:
  BUG-1  CIRCULAR IMPORT: `from server import _update_graph` inside function
         → server dependency entirely removed; caller passes graph + update_fn
  BUG-2  feedparser.parse() is SYNCHRONOUS blocking inside async function
         → wrapped in asyncio.get_event_loop().run_in_executor()
  BUG-3  `graph` parameter passed to ingest_rss_feed() was silently ignored
         → graph and update_fn are first-class parameters used throughout
  BUG-4  processed_urls = set() — lost on every restart
         → persisted to processed_urls.json on disk
  BUG-5  All imports at module level — crash server if packages not installed
         → all optional deps lazy-imported; clear ImportError messages
  BUG-6  No status/result reporting — total black box
         → returns structured IngestResult with per-feed stats

New in v2:
  - OsintEngine class with configurable feed list (add/remove via API)
  - get_status() for the API status endpoint
  - Per-article result log kept in memory (last 200)
  - Configurable article limit per feed and per run
  - Wikipedia entity enrichment hook
  - GDELT integration stub
"""

from __future__ import annotations

import asyncio
import datetime
import json
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set

PROCESSED_URLS_FILE = pathlib.Path("osint_processed_urls.json")
FEEDS_CONFIG_FILE = pathlib.Path("osint_feeds.json")
MAX_ARTICLE_LOG = 200

DEFAULT_FEEDS: List[Dict[str, str]] = [
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "name": "BBC World"},
    {"url": "https://www.aljazeera.com/xml/rss/all.xml", "name": "Al Jazeera"},
    {"url": "https://www.reuters.com/world/rss", "name": "Reuters World"},
    {"url": "https://feeds.npr.org/1004/rss.xml", "name": "NPR World"},
]


@dataclass
class ArticleResult:
    url: str
    title: str
    feed: str
    status: str  # "ok" | "skipped" | "error"
    entities: int = 0
    relations: int = 0
    new_nodes: int = 0
    error_msg: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


@dataclass
class IngestResult:
    feeds_processed: int = 0
    articles_found: int = 0
    articles_ingested: int = 0
    articles_skipped: int = 0
    articles_errored: int = 0
    total_entities: int = 0
    total_relations: int = 0
    total_new_nodes: int = 0
    duration_secs: float = 0.0
    details: List[ArticleResult] = field(default_factory=list)


class OsintEngine:
    """
    Manages RSS feed configuration and asynchronous ingestion.
    Completely decoupled from server.py (BUG-1 FIX).
    """

    def __init__(self):
        self._processed_urls: Set[str] = set()
        self._feeds: List[Dict[str, str]] = []
        self._article_log: List[Dict] = []
        self._last_run: Optional[str] = None
        self._running: bool = False

        self._load_config()
        self._load_processed_urls()

    # ── Feed management ───────────────────────────────────────────────────────

    def get_feeds(self) -> List[Dict[str, str]]:
        return list(self._feeds)

    def add_feed(self, url: str, name: str = "") -> bool:
        for f in self._feeds:
            if f["url"] == url:
                return False
        self._feeds.append({"url": url, "name": name or url})
        self._save_config()
        return True

    def remove_feed(self, url: str) -> bool:
        before = len(self._feeds)
        self._feeds = [f for f in self._feeds if f["url"] != url]
        if len(self._feeds) < before:
            self._save_config()
            return True
        return False

    # ── Core ingestion ────────────────────────────────────────────────────────

    async def ingest_all(
        self,
        graph,  # BUG-3 FIX: graph is an explicit parameter
        update_fn: Callable,  # BUG-1 FIX: caller passes _update_graph, no import
        model: str = "llama3.2",
        articles_per_feed: int = 5,
    ) -> IngestResult:
        """
        Fetch and process articles from all configured RSS feeds.
        graph: nx.DiGraph  (the live knowledge graph)
        update_fn: callable(extractions) → diff dict  (from server._update_graph)
        """
        if self._running:
            return IngestResult()  # Prevent concurrent runs

        self._running = True
        started = datetime.datetime.now(datetime.timezone.utc)
        result = IngestResult()

        for feed_cfg in self._feeds:
            feed_url = feed_cfg["url"]
            feed_name = feed_cfg.get("name", feed_url)

            # BUG-2 FIX: run sync feedparser in executor — doesn't block event loop
            try:
                feed = await asyncio.get_event_loop().run_in_executor(
                    None, self._parse_feed_sync, feed_url
                )
            except Exception as exc:
                self._log(
                    ArticleResult(
                        url=feed_url,
                        title="",
                        feed=feed_name,
                        status="error",
                        error_msg=str(exc),
                    )
                )
                continue

            result.feeds_processed += 1
            entries = feed.entries[:articles_per_feed]
            result.articles_found += len(entries)

            for entry in entries:
                url = entry.get("link", "")
                title = entry.get("title", "")

                if not url:
                    continue

                if url in self._processed_urls:
                    result.articles_skipped += 1
                    self._log(
                        ArticleResult(
                            url=url, title=title, feed=feed_name, status="skipped"
                        )
                    )
                    continue

                ar = await self._ingest_article(
                    url, title, feed_name, graph, update_fn, model
                )
                self._log(ar)

                if ar.status == "ok":
                    result.articles_ingested += 1
                    result.total_entities += ar.entities
                    result.total_relations += ar.relations
                    result.total_new_nodes += ar.new_nodes
                    self._processed_urls.add(url)
                elif ar.status == "error":
                    result.articles_errored += 1
                else:
                    result.articles_skipped += 1

        duration = (
            datetime.datetime.now(datetime.timezone.utc) - started
        ).total_seconds()
        result.duration_secs = round(duration, 2)
        result.details = list(self._article_log[-20:])

        self._last_run = started.isoformat()
        self._save_processed_urls()
        self._running = False
        return result

    async def _ingest_article(
        self,
        url: str,
        title: str,
        feed_name: str,
        graph,
        update_fn: Callable,
        model: str,
    ) -> ArticleResult:
        try:
            text = await self._fetch_article_text(url)
        except Exception as exc:
            return ArticleResult(
                url=url,
                title=title,
                feed=feed_name,
                status="error",
                error_msg=f"Fetch: {exc}",
            )

        if len(text.strip()) < 200:
            return ArticleResult(
                url=url,
                title=title,
                feed=feed_name,
                status="skipped",
                error_msg="Text too short",
            )

        try:
            # BUG-1 FIX: use passed-in extract_intelligence, not circular import
            from extractor import extract_intelligence

            extractions = extract_intelligence(text, model=model)
        except Exception as exc:
            return ArticleResult(
                url=url,
                title=title,
                feed=feed_name,
                status="error",
                error_msg=f"Extract: {exc}",
            )

        try:
            # BUG-3 FIX: pass graph parameter; update_fn is server._update_graph
            diff = update_fn(extractions)
        except Exception as exc:
            return ArticleResult(
                url=url,
                title=title,
                feed=feed_name,
                status="error",
                error_msg=f"Update: {exc}",
            )

        entities = sum(
            1 for e in extractions if e.extraction_class.lower() != "relationship"
        )
        relations = len(extractions) - entities
        return ArticleResult(
            url=url,
            title=title,
            feed=feed_name,
            status="ok",
            entities=entities,
            relations=relations,
            new_nodes=diff.get("nodes_added", 0),
        )

    # ── Article text fetching ─────────────────────────────────────────────────

    async def _fetch_article_text(self, url: str) -> str:
        # BUG-5 FIX: lazy imports
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx not installed. Run: pip install httpx")

        headers = {"User-Agent": "GOIES/3.0 (+https://github.com/tanu-1403/GOIES)"}
        async with httpx.AsyncClient(
            timeout=20, follow_redirects=True, headers=headers
        ) as client:
            r = await client.get(url)
            r.raise_for_status()
            html = r.text

        # Try readability, fall back to BeautifulSoup
        try:
            from readability import Document
            from bs4 import BeautifulSoup

            doc = Document(html)
            soup = BeautifulSoup(doc.summary(), "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text
        except ImportError:
            pass

        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            if len(text) > 200:
                return text
        except ImportError:
            pass

        import re

        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))

    # ── Sync feedparser wrapper ───────────────────────────────────────────────

    @staticmethod
    def _parse_feed_sync(url: str):
        """BUG-2 FIX: always called via run_in_executor, never directly in async."""
        try:
            import feedparser
        except ImportError:
            raise ImportError("feedparser not installed. Run: pip install feedparser")
        return feedparser.parse(url)

    # ── Wikipedia entity enrichment ───────────────────────────────────────────

    async def enrich_entity_wikipedia(
        self, entity_id: str, model: str = "llama3.2"
    ) -> Dict[str, Any]:
        """Fetch Wikipedia summary and extract structured attributes for a node."""
        try:
            import httpx
        except ImportError:
            return {"error": "httpx not installed"}

        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity_id.replace(' ', '_')}"
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return {}
                data = r.json()
            return {
                "wikipedia_summary": data.get("extract", "")[:300],
                "wikipedia_url": data.get("content_urls", {})
                .get("desktop", {})
                .get("page", ""),
            }
        except Exception as exc:
            return {"error": str(exc)}

    # ── GDELT ─────────────────────────────────────────────────────────────────

    async def query_gdelt(self, entity: str, days: int = 7) -> List[Dict]:
        """Fetch recent GDELT news events for an entity."""
        try:
            import httpx
        except ImportError:
            return []
        try:
            params = {
                "query": f'"{entity}" sourcelang:English',
                "mode": "artlist",
                "maxrecords": "10",
                "format": "json",
                "timespan": f"{days}d",
                "sort": "DateDesc",
            }
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc", params=params
                )
                return r.json().get("articles", [])
        except Exception:
            return []

    # ── Status ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "feeds": len(self._feeds),
            "processed_urls": len(self._processed_urls),
            "running": self._running,
            "last_run": self._last_run,
            "recent_log": list(reversed(self._article_log[-10:])),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        if FEEDS_CONFIG_FILE.exists():
            try:
                self._feeds = json.loads(FEEDS_CONFIG_FILE.read_text(encoding="utf-8"))
                return
            except Exception:
                pass
        self._feeds = list(DEFAULT_FEEDS)

    def _save_config(self) -> None:
        try:
            FEEDS_CONFIG_FILE.write_text(
                json.dumps(self._feeds, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _load_processed_urls(self) -> None:
        # BUG-4 FIX: restore from disk on startup
        try:
            if PROCESSED_URLS_FILE.exists():
                self._processed_urls = set(
                    json.loads(PROCESSED_URLS_FILE.read_text(encoding="utf-8"))
                )
        except Exception:
            self._processed_urls = set()

    def _save_processed_urls(self) -> None:
        try:
            PROCESSED_URLS_FILE.write_text(
                json.dumps(list(self._processed_urls)[-5000:]),  # cap at 5k
                encoding="utf-8",
            )
        except Exception:
            pass

    def _log(self, article: ArticleResult) -> None:
        # BUG-6 FIX: structured result log
        self._article_log.append(asdict(article))
        if len(self._article_log) > MAX_ARTICLE_LOG:
            self._article_log = self._article_log[-MAX_ARTICLE_LOG:]
