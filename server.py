"""
server.py  —  GOIES FastAPI Backend v4.2.0
==========================================
Merge resolution: HEAD (v4.x production) + ce28496 (async rewrite)

HEAD production features preserved:
  FIX-1   eval() → ast.literal_eval()
  FIX-2   _attrs dict embedded in graph_to_vis()
  FIX-3   threading.Lock around _update_graph()
  FIX-4   Path traversal in /api/snapshots/{id} blocked
  FIX-5   CORS via ALLOWED_ORIGINS env var
  FIX-6   print() → structured logger
  FIX-8   XSS: LLM strings HTML-escaped in _fmt_tooltip()
  FIX-9   Upload size cap (MAX_UPLOAD_BYTES)
  FIX-10  watch_list_thresholds persisted to disk
  FIX-11  Startup Ollama health check
  FIX-12  Per-IP sliding-window rate limiter
  FIX-13  Content-Security-Policy middleware
  FIX-14  GQL path-find timeout
  FIX-15  GQL LIMIT clause
  FIX-17  Cross-session deduplication cache reset endpoint

ce28496 async improvements layered in:
  ASYNC-1  extract() and extract_stream() are now async (non-blocking Ollama calls)
  ASYNC-2  OLLAMA_TIMEOUT env var honoured
  ASYNC-3  CancelToken + DELETE /api/extract/{task_id} cancellation endpoint
  ASYNC-4  DELETE /api/graph calls utils.clear_graph() — wipes backend state
  ASYNC-5  Parallel chunk processing via asyncio.gather in extractor
  ASYNC-7  SSE stream yields incremental per-chunk graph updates
  ASYNC-8  warmup_model() called as background task at startup
  ASYNC-9  Atomic graph save/load (handled in utils.py)
"""

from __future__ import annotations

import ast
import asyncio
import html
import io
import json
import logging
import os
import pathlib
import re
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import networkx as nx
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from embedding_engine import GraphEmbeddingEngine
from extractor import (
    CancelToken,
    ChunkResult,
    Entity,
    Relationship,
    _call_ollama,
    check_ollama_health,
    extract_stream,
    extract_text,
    list_available_models,
    warmup_model,
)
from forecaster import run_forecast
from geo import get_geo_data
from osint_engine import OsintEngine
from query_engine import GQLParser, run_gql
from simulator import run_simulation

import itertools
import requests as http  # sync http — used in graph_summary, export_report

from utils import (
    export_csv,
    export_graphml,
    export_json,
    get_ego_subgraph,
    get_graph_analytics,
    load_graph,
    merge_nodes,
    resolve_node_name,
    retrieve_graph_context,
    save_graph,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("goies.server")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("GOIES_DEFAULT_MODEL", "llama3.2")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "500000"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
WATCH_THRESHOLDS_FILE = pathlib.Path("watch_thresholds.json")

_raw_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
)
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

GROUP_COLORS: Dict[str, str] = {
    "country": "#ff7b72",
    "person": "#ffa657",
    "organization": "#d2a8ff",
    "technology": "#79c0ff",
    "event": "#7ee787",
    "treaty": "#f0e68c",
    "resource": "#56d364",
    "unknown": "#8b949e",
}

# ── In-flight extraction tasks (ASYNC-3) ──────────────────────────────────────
_active_tasks: dict[str, CancelToken] = {}


# ── Rate limiter ──────────────────────────────────────────────────────────────
class _RateLimiter:
    _PRUNE_INTERVAL = 300

    def __init__(self):
        self._windows: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def is_allowed(self, key: str, max_requests: int, window_secs: float) -> bool:
        now = time.monotonic()
        with self._lock:
            if now - self._last_prune > self._PRUNE_INTERVAL:
                stale_cutoff = now - max(window_secs, 3600)
                self._windows = defaultdict(
                    list,
                    {
                        k: v
                        for k, v in self._windows.items()
                        if any(t > stale_cutoff for t in v)
                    },
                )
                self._last_prune = now
            timestamps = self._windows[key]
            cutoff = now - window_secs
            self._windows[key] = [t for t in timestamps if t > cutoff]
            if len(self._windows[key]) >= max_requests:
                return False
            self._windows[key].append(now)
            return True


_rate_limiter = _RateLimiter()


def _check_rate(request: Request, max_req: int, window: float):
    client_ip = request.client.host if request.client else "unknown"
    key = f"{client_ip}:{request.url.path}"
    if not _rate_limiter.is_allowed(key, max_req, window):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_req} requests per {int(window)}s.",
            headers={"Retry-After": str(int(window))},
        )


# ── Shared state ──────────────────────────────────────────────────────────────
graph: nx.DiGraph = load_graph()
_graph_lock = threading.Lock()


def _load_watch_thresholds() -> Dict[str, float]:
    if WATCH_THRESHOLDS_FILE.exists():
        try:
            return json.loads(WATCH_THRESHOLDS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


watch_list_thresholds: Dict[str, float] = _load_watch_thresholds()

embedding_engine = GraphEmbeddingEngine()
osint_engine = OsintEngine()

# ── Continuous OSINT loop state ───────────────────────────────────────────────
_continuous_state: dict = {
    "active": False,
    "cycle": 0,
    "started_at": None,
    "stopped_at": None,
    "interval_secs": 300,
    "articles_per_feed": 5,
    "model": DEFAULT_MODEL,
    "query_log": [],
    "cycle_log": [],
    "total_entities": 0,
    "total_relations": 0,
    "total_articles": 0,
}
_continuous_task: Optional[asyncio.Task] = None
_continuous_lock = asyncio.Lock()


async def _continuous_loop() -> None:
    while _continuous_state["active"]:
        try:
            await asyncio.sleep(_continuous_state["interval_secs"])
            if not _continuous_state["active"]:
                break
            feeds = osint_engine.get_feeds()
            n = _continuous_state["articles_per_feed"]
            model = _continuous_state["model"]
            for feed_url in feeds:
                articles = await asyncio.to_thread(
                    osint_engine.fetch_feed_articles, feed_url, n
                )
                for text in articles:
                    if not text:
                        continue
                    cancel = CancelToken()
                    ents, rels = await extract_text(
                        text,
                        model=model,
                        cancel=cancel,
                        on_chunk=lambda r: _apply_chunk(r),
                    )
                    _continuous_state["total_entities"] += len(ents)
                    _continuous_state["total_relations"] += len(rels)
                    _continuous_state["total_articles"] += 1
            _continuous_state["cycle"] += 1
            _continuous_state["cycle_log"].insert(
                0,
                {
                    "cycle": _continuous_state["cycle"],
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )
            _continuous_state["cycle_log"] = _continuous_state["cycle_log"][:20]
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("Continuous OSINT loop error: %s", exc)

    _continuous_state["stopped_at"] = datetime.now(timezone.utc).isoformat()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Startup
    health = check_ollama_health()
    if health["online"]:
        logger.info(
            "Ollama online at %s — models: %s", OLLAMA_BASE_URL, health["models"]
        )
    else:
        logger.warning(
            "Ollama NOT reachable at %s — extraction will fail until started. %s",
            OLLAMA_BASE_URL,
            health["error"],
        )
    logger.info(
        "Graph loaded: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    # ASYNC-8: warm up model as background task
    asyncio.create_task(warmup_model(DEFAULT_MODEL))

    yield

    # Shutdown
    logger.info("Shutdown: flushing graph to disk…")
    try:
        save_graph(graph)
    except Exception as exc:
        logger.error("Shutdown graph save failed: %s", exc)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GOIES",
    version="4.2.0",
    docs_url="/api/docs",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
        "https://cdnjs.cloudflare.com https://unpkg.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org; "
        "connect-src 'self'; frame-ancestors 'none';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


if os.path.isdir("frontend"):
    # html=True makes StaticFiles serve index.html for directory requests (e.g. GET /)
    app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_tooltip(group: str, attributes: dict, confidence: float) -> str:
    color = GROUP_COLORS.get(group, "#8b949e")
    lines = [
        f'<b style="color:{color};font-family:monospace">{html.escape(group.upper())}</b>'
    ]
    for k, v in attributes.items():
        lines.append(
            f'<span style="color:#64748b">{html.escape(str(k))}:</span> {html.escape(str(v))}'
        )
    lines.append(f'<span style="color:#64748b">confidence:</span> {confidence:.2f}')
    return "<br>".join(lines)


def _safe_parse_attrs(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        result = ast.literal_eval(raw)
        return result if isinstance(result, dict) else {}
    except (ValueError, SyntaxError, TypeError):
        return {}


def graph_to_vis(g: nx.DiGraph) -> dict:
    nodes = []
    for node_id, data in g.nodes(data=True):
        group = data.get("group", "unknown")
        color = GROUP_COLORS.get(group, "#8b949e")
        conf = data.get("confidence", 1.0)
        attrs = _safe_parse_attrs(data.get("title", "{}"))
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "group": group,
                "color": {
                    "background": color,
                    "border": color,
                    "highlight": {"background": "#ffffff", "border": color},
                    "hover": {"background": color, "border": "#ffffff"},
                },
                "title": _fmt_tooltip(group, attrs, conf),
                "_attrs": attrs,
                "confidence": conf,
                "size": 16,
                "borderWidth": 2,
                "font": {"color": "#e2e8f0", "size": 13},
                "shadow": {
                    "enabled": True,
                    "color": color + "44",
                    "size": 12,
                    "x": 0,
                    "y": 0,
                },
            }
        )
    edges = []
    for u, v, data in g.edges(data=True):
        edges.append(
            {
                "from": u,
                "to": v,
                "label": data.get("label", ""),
                "arrows": "to",
                "color": {
                    "color": "#1e3a5f",
                    "highlight": "#00e5ff",
                    "hover": "#00e5ff",
                    "inherit": False,
                },
                "font": {
                    "color": "#3d5a7a",
                    "size": 10,
                    "align": "middle",
                    "strokeWidth": 0,
                },
                "width": 1.5,
                "smooth": {"type": "continuous"},
                "confidence": data.get("confidence", 1.0),
            }
        )
    return {"nodes": nodes, "edges": edges}


def _apply_chunk(result: ChunkResult) -> None:
    """Sync callback: merge a ChunkResult into the global graph under lock."""
    with _graph_lock:
        for ent in result.entities:
            canonical = resolve_node_name(graph, ent.id)
            if not graph.has_node(canonical):
                graph.add_node(
                    canonical,
                    title=str(ent.attributes),
                    group=ent.group,
                    confidence=ent.confidence,
                )
            else:
                existing = graph.nodes[canonical]
                if existing.get("group", "unknown") == "unknown":
                    existing["group"] = ent.group
                if ent.confidence > existing.get("confidence", 0.0):
                    existing["title"] = str(ent.attributes)
                    existing["confidence"] = ent.confidence
        for rel in result.relationships:
            src = resolve_node_name(graph, rel.from_id)
            tgt = resolve_node_name(graph, rel.to_id)
            for n in (src, tgt):
                if not graph.has_node(n):
                    graph.add_node(n, group="unknown")
            graph.add_edge(src, tgt, label=rel.label, confidence=rel.confidence)


def _update_graph(entities: list[Entity], relationships: list[Relationship]) -> dict:
    """Batch-apply entities + relationships; returns diff summary."""
    nodes_added, edges_added, new_ids = 0, 0, []
    result = ChunkResult(
        entities=entities, relationships=relationships, chunk_index=0, elapsed=0.0
    )
    with _graph_lock:
        before_nodes = graph.number_of_nodes()
        before_edges = graph.number_of_edges()
    _apply_chunk(result)
    save_graph(graph)
    with _graph_lock:
        nodes_added = graph.number_of_nodes() - before_nodes
        edges_added = graph.number_of_edges() - before_edges
        new_ids = [
            n
            for n in graph.nodes()
            if n
            not in set(
                n2 for n2, _ in graph.nodes(data=True) if _.get("ingested_at", "") == ""
            )
        ]
    return {
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "new_node_ids": new_ids,
    }


# ── Request models ────────────────────────────────────────────────────────────


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str = DEFAULT_MODEL
    persona: str = "senior geopolitical intelligence analyst"


class ExtractResponse(BaseModel):
    task_id: str
    entities: list[dict]
    relationships: list[dict]
    elapsed: float
    chunks: int


class GraphClearResponse(BaseModel):
    cleared: bool
    message: str


class QueryRequest(BaseModel):
    question: str
    model: str = DEFAULT_MODEL
    persona: str = "senior geopolitical intelligence analyst"


class SimulateRequest(BaseModel):
    scenario: str
    model: str = DEFAULT_MODEL


class ForecastRequest(BaseModel):
    model: str = DEFAULT_MODEL
    focus: str = ""


class UrlIngestRequest(BaseModel):
    url: str


class ReportRequest(BaseModel):
    entities: List[str] = []
    format: str = "pdf"
    model: str = DEFAULT_MODEL


class WatchListRequest(BaseModel):
    thresholds: Dict[str, float]


class ExtractUrlRequest(BaseModel):
    url: str
    model: str = DEFAULT_MODEL
    persona: str = "senior geopolitical intelligence analyst"


class MergeRequest(BaseModel):
    source: str
    target: str


class GQLRequest(BaseModel):
    query: str


class FeedRequest(BaseModel):
    url: str
    name: str = ""


class OsintIngestRequest(BaseModel):
    model: str = DEFAULT_MODEL
    articles_per_feed: int = 5


class ContinuousStartRequest(BaseModel):
    interval_secs: int = 300
    articles_per_feed: int = 5
    model: str = DEFAULT_MODEL


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/api/health")
def health():
    return check_ollama_health()


@app.get("/api/models")
def models():
    return {"models": list_available_models()}


# ── SSRF guard ────────────────────────────────────────────────────────────────


def _validate_url(url: str) -> None:
    import ipaddress, urllib.parse

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"Unsupported URL scheme '{parsed.scheme}'.")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
        ):
            raise HTTPException(
                400, "Requests to private/loopback addresses are not allowed."
            )
    except ValueError:
        pass
    blocked = ("localhost", "metadata.google.internal")
    if any(host.lower() == b or host.lower().endswith("." + b) for b in blocked):
        raise HTTPException(400, "Requests to reserved hostnames are not allowed.")


# ── Ingestion ─────────────────────────────────────────────────────────────────


@app.post("/api/ingest/url/fetch")
def ingest_url_fetch(req: UrlIngestRequest, request: Request):
    _check_rate(request, max_req=30, window=60)
    _validate_url(req.url)
    try:
        from ingestor import fetch_url_text

        text = fetch_url_text(req.url)
        return {"text": text, "chars": len(text)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/extract/url")
def extract_url_stream(req: ExtractUrlRequest, request: Request):
    _check_rate(request, max_req=10, window=60)
    _validate_url(req.url)

    def event_generator():
        try:
            from ingestor import fetch_url_text

            text = fetch_url_text(req.url)
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Fetch failed: {exc}'})}\n\n"
            return
        if not text.strip():
            yield f"data: {json.dumps({'error': 'No text extracted from URL'})}\n\n"
            return
        yield f"data: {json.dumps({'fetched': True, 'chars': len(text), 'url': req.url})}\n\n"
        # Run async extraction from sync context via a new event loop
        import asyncio as _aio

        cancel = CancelToken()
        loop = _aio.new_event_loop()
        try:
            ents, rels = loop.run_until_complete(
                extract_text(
                    text,
                    model=req.model,
                    cancel=cancel,
                    on_chunk=lambda r: _apply_chunk(r),
                )
            )
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        finally:
            loop.close()
        save_graph(graph)
        yield f"data: {json.dumps({'done': True, 'totals': {'entities': len(ents), 'relations': len(rels)}})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/ingest/file")
async def ingest_file(request: Request, file: UploadFile = File(...)):
    _check_rate(request, max_req=20, window=60)
    content_length = request.headers.get("content-length")
    try:
        _cl_int = int(content_length) if content_length else 0
    except (ValueError, TypeError):
        _cl_int = 0
    if _cl_int > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large. Max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    from ingestor import parse_pdf, parse_docx

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"File too large. Max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
        )

    filename = (file.filename or "").lower()
    try:
        if filename.endswith(".pdf"):
            text = parse_pdf(content)
        elif filename.endswith(".docx"):
            text = parse_docx(content)
        elif filename.endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(
                400, "Unsupported format. Upload PDF, DOCX, TXT, or MD."
            )
        return {"text": text, "filename": filename}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Error parsing file: {exc}")


# ── Extraction (ASYNC-1 #3 #7) ────────────────────────────────────────────────


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest, request: Request) -> ExtractResponse:
    _check_rate(request, max_req=10, window=60)
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Input exceeds {MAX_INPUT_CHARS:,} chars.")

    task_id = str(uuid.uuid4())
    cancel = CancelToken()
    _active_tasks[task_id] = cancel
    t0 = time.perf_counter()
    chunks_processed = [0]

    def on_chunk(result: ChunkResult) -> None:
        _apply_chunk(result)
        chunks_processed[0] += 1

    try:
        entities, relationships = await extract_text(
            req.text, req.model, cancel, on_chunk=on_chunk
        )
    except asyncio.CancelledError:
        raise HTTPException(409, "Extraction cancelled.")
    except ConnectionError as exc:
        raise HTTPException(503, str(exc))
    except TimeoutError as exc:
        raise HTTPException(504, str(exc))
    finally:
        _active_tasks.pop(task_id, None)

    save_graph(graph)

    return ExtractResponse(
        task_id=task_id,
        entities=[
            {
                "id": e.id,
                "group": e.group,
                "confidence": e.confidence,
                "attributes": e.attributes,
            }
            for e in entities
        ],
        relationships=[
            {
                "from": r.from_id,
                "to": r.to_id,
                "label": r.label,
                "confidence": r.confidence,
            }
            for r in relationships
        ],
        elapsed=round(time.perf_counter() - t0, 2),
        chunks=chunks_processed[0],
    )


@app.delete("/api/extract/{task_id}", status_code=200)
async def cancel_extraction(task_id: str) -> dict:
    """ASYNC-3: Cancel an in-flight extraction by task_id."""
    token = _active_tasks.get(task_id)
    if not token:
        raise HTTPException(404, f"No active task with id={task_id}")
    token.cancel()
    return {"cancelled": True, "task_id": task_id}


@app.post("/api/extract/stream")
async def extract_stream_endpoint(
    req: ExtractRequest, request: Request
) -> StreamingResponse:
    """ASYNC-7: SSE stream — yields incremental per-chunk graph updates."""
    _check_rate(request, max_req=10, window=60)
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Input exceeds {MAX_INPUT_CHARS:,} chars.")

    task_id = str(uuid.uuid4())
    cancel = CancelToken()
    _active_tasks[task_id] = cancel

    all_known_node_ids: set = {n for n in graph.nodes()}

    async def event_generator() -> AsyncIterator[str]:
        total_entities, total_relations = 0, 0
        try:
            async for event in extract_stream(req.text, req.model, cancel):
                if event.get("type") == "chunk":
                    result = ChunkResult(
                        entities=[
                            Entity(
                                id=e["id"],
                                group=e.get("group", "unknown"),
                                confidence=e.get("confidence", 1.0),
                                attributes=e.get("attributes", {}),
                            )
                            for e in event.get("new_entities", [])
                        ],
                        relationships=[
                            Relationship(
                                from_id=r["from"],
                                to_id=r["to"],
                                label=r.get("label", "related"),
                                confidence=r.get("confidence", 1.0),
                            )
                            for r in event.get("new_relationships", [])
                        ],
                        chunk_index=event.get("chunk_index", 0),
                        elapsed=event.get("elapsed", 0.0),
                    )
                    _apply_chunk(result)

                    new_node_ids = [
                        e["id"]
                        for e in event.get("new_entities", [])
                        if e["id"] not in all_known_node_ids
                    ]
                    new_vis: Optional[dict] = None
                    if new_node_ids:
                        vis_full = graph_to_vis(graph)
                        new_vis = {
                            "nodes": [
                                n for n in vis_full["nodes"] if n["id"] in new_node_ids
                            ],
                            "edges": [
                                e
                                for e in vis_full["edges"]
                                if e["from"] in new_node_ids or e["to"] in new_node_ids
                            ],
                        }
                        all_known_node_ids.update(new_node_ids)

                    totals = event.get("totals", {})
                    total_entities = totals.get("entities", total_entities)
                    total_relations = totals.get("relationships", total_relations)
                    payload = {
                        "chunk": event.get("chunk_index"),
                        "total_chunks": totals.get("chunks_total"),
                        "entities": len(event.get("new_entities", [])),
                        "relations": len(event.get("new_relationships", [])),
                        "new_node_ids": new_node_ids,
                        "task_id": task_id,
                    }
                    if new_vis:
                        payload["vis_delta"] = new_vis

                elif event.get("type") == "done":
                    save_graph(graph)
                    payload = {
                        "done": True,
                        "task_id": task_id,
                        "totals": {
                            "entities": total_entities,
                            "relations": total_relations,
                        },
                        "vis": graph_to_vis(graph),
                        "analytics": get_graph_analytics(graph, watch_list_thresholds),
                    }

                elif event.get("type") == "error":
                    payload = {"error": event.get("message"), "task_id": task_id}

                else:
                    payload = {**event, "task_id": task_id}

                yield f"data: {json.dumps(payload)}\n\n"

        except asyncio.CancelledError:
            yield f"data: {json.dumps({'cancelled': True, 'task_id': task_id})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc), 'task_id': task_id})}\n\n"
        finally:
            _active_tasks.pop(task_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "X-Task-Id": task_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Seen cache reset (FIX-17) ─────────────────────────────────────────────────


@app.delete("/api/extract/seen")
def clear_seen_cache():
    from extractor import _seen_lock, _global_seen, SEEN_FILE, _save_seen

    with _seen_lock:
        count = len(_global_seen)
        _global_seen.clear()
        try:
            if SEEN_FILE.exists():
                SEEN_FILE.unlink()
        except OSError as exc:
            logger.warning("Could not delete seen cache file: %s", exc)
    logger.info("Seen cache cleared (%d entries removed).", count)
    return {"status": "cleared", "entries_removed": count}


# ── Graph (ASYNC-4) ───────────────────────────────────────────────────────────


@app.get("/api/graph")
def get_graph_ep(ego: Optional[str] = None, hops: int = 2):
    hops = max(1, min(hops, 4))
    g = get_ego_subgraph(graph, ego, hops) if ego and ego in graph else graph
    return {
        "vis": graph_to_vis(g),
        "analytics": get_graph_analytics(graph, watch_list_thresholds),
        "filtered": ego is not None and ego in graph,
    }


@app.delete("/api/graph", response_model=GraphClearResponse)
def clear_graph_ep():
    """ASYNC-4: Wipes both the in-memory graph AND persists the empty state."""
    with _graph_lock:
        graph.clear()
    save_graph(graph)
    logger.info("Graph cleared and persisted.")
    return GraphClearResponse(
        cleared=True,
        message="Graph cleared — backend state reset and persisted to disk.",
    )


@app.get("/api/path")
def path(src: str, tgt: str, request: Request):
    _check_rate(request, max_req=30, window=60)
    from graph_algo import find_shortest_path

    src_c = resolve_node_name(graph, src)
    tgt_c = resolve_node_name(graph, tgt)
    if not graph.has_node(src_c) or not graph.has_node(tgt_c):
        raise HTTPException(404, "One or both nodes not found.")
    path_data = find_shortest_path(graph, src_c, tgt_c)
    if not path_data["nodes"]:
        return {"found": False, "nodes": [], "edges": []}
    return {"found": True, "nodes": path_data["nodes"], "edges": path_data["edges"]}


@app.post("/api/node/merge")
def merge_node_ep(req: MergeRequest, request: Request):
    _check_rate(request, max_req=20, window=60)
    src_c = resolve_node_name(graph, req.source)
    tgt_c = resolve_node_name(graph, req.target)
    if src_c == tgt_c:
        raise HTTPException(400, "Source and target resolve to the same node.")
    success = merge_nodes(graph, src_c, tgt_c)
    if not success:
        raise HTTPException(400, "Failed to merge. Ensure both nodes exist.")
    save_graph(graph)
    return {"status": "success", "merged": src_c, "into": tgt_c}


@app.get("/api/export/{fmt}")
def export(fmt: str):
    if fmt == "json":
        return StreamingResponse(
            io.StringIO(export_json(graph)),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=goies_graph.json"},
        )
    elif fmt == "csv":
        return StreamingResponse(
            io.StringIO(export_csv(graph)),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=goies_edges.csv"},
        )
    elif fmt == "graphml":
        return StreamingResponse(
            io.BytesIO(export_graphml(graph)),
            media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=goies_graph.graphml"},
        )
    raise HTTPException(400, f"Unknown format: {fmt}")


# ── Geo ───────────────────────────────────────────────────────────────────────


@app.get("/api/geo")
def get_geo():
    markers = get_geo_data(graph)
    return {"markers": markers, "total": len(markers)}


# ── Narrative summary ─────────────────────────────────────────────────────────


@app.get("/api/narrative/summary")
def graph_summary(request: Request, model: str = DEFAULT_MODEL):
    _check_rate(request, max_req=5, window=60)
    analytics = get_graph_analytics(graph, watch_list_thresholds)
    edge_sample = [
        f"{u} -> {v} [{d.get('label', '')}]"
        for u, v, d in itertools.islice(graph.edges(data=True), 25)
    ]
    prompt = (
        "You are a senior intelligence analyst. Describe the following geopolitical "
        "network in 3 paragraphs. Focus on: major power actors, key conflict zones, "
        "most significant tensions, dominant alliance patterns. Use direct, professional "
        f"language. No hedging. Cite specific entity names.\n\n"
        f"Graph statistics:\n"
        f"- {analytics.get('nodes')} entities: {analytics.get('group_counts', {})}\n"
        f"- {analytics.get('edges')} relationships\n"
        f"- Most connected: {analytics.get('top_degree', [])}\n\n"
        f"Key relationships sample:\n"
        + "\n".join(edge_sample)
        + "\n\nWrite the 3-paragraph intelligence summary now:"
    )
    try:
        narrative = _call_ollama(prompt, model)
        return {
            "narrative": narrative,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(500, f"Summary generation failed: {exc}")


# ── Simulation ────────────────────────────────────────────────────────────────


@app.post("/api/simulate")
async def simulate(req: SimulateRequest, request: Request):
    _check_rate(request, max_req=5, window=60)
    if not req.scenario.strip():
        raise HTTPException(400, "Scenario cannot be empty.")
    if len(graph.nodes) == 0:
        raise HTTPException(400, "Graph is empty. Ingest data first.")
    try:
        import functools

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(run_simulation, req.scenario, graph, model=req.model),
        )
    except ConnectionError as exc:
        raise HTTPException(503, str(exc))
    except TimeoutError as exc:
        raise HTTPException(504, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {
        "scenario": result.scenario,
        "risk_score": result.risk_score,
        "risk_label": result.risk_label,
        "cascade_narrative": result.cascade_narrative,
        "second_order": result.second_order,
        "added_edges": result.added_edges,
        "removed_edges": result.removed_edges,
        "affected_nodes": result.affected_nodes,
        "model_used": result.model_used,
    }


@app.get("/api/simulations")
def get_simulations():
    history_file = "sim_history.json"
    if not os.path.exists(history_file):
        return {"history": []}
    try:
        with open(history_file, encoding="utf-8") as f:
            return {"history": json.load(f)}
    except Exception as exc:
        raise HTTPException(500, f"Failed to read simulation history: {exc}")


# ── Forecast ──────────────────────────────────────────────────────────────────


@app.post("/api/forecast")
def forecast(req: ForecastRequest, request: Request):
    _check_rate(request, max_req=5, window=60)
    if len(graph.nodes) < 3:
        raise HTTPException(400, "Need at least 3 nodes to generate a forecast.")
    try:
        result = run_forecast(graph, model=req.model, focus_query=req.focus)
    except ConnectionError as exc:
        raise HTTPException(503, str(exc))
    except TimeoutError as exc:
        raise HTTPException(504, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
    return {
        "global_risk": result.global_risk,
        "global_label": result.global_label,
        "structural_summary": result.structural_summary,
        "hotspot_nodes": result.hotspot_nodes,
        "model_used": result.model_used,
        "forecasts": [
            {
                "rank": f.rank,
                "title": f.title,
                "actors": f.actors,
                "probability": f.probability,
                "severity": f.severity,
                "timeframe": f.timeframe,
                "structural_signal": f.structural_signal,
                "narrative": f.narrative,
                "mitigation": f.mitigation,
            }
            for f in result.forecasts
        ],
    }


# ── Query (GraphRAG) ──────────────────────────────────────────────────────────


@app.post("/api/query")
def query(req: QueryRequest, request: Request):
    _check_rate(request, max_req=20, window=60)
    if len(graph.nodes) == 0:
        return {"answer": "Graph is empty. Ingest data first.", "context": ""}
    context = retrieve_graph_context(req.question, graph)
    prompt = (
        f"You are a {req.persona}. "
        "Answer using ONLY the Knowledge Graph Context. "
        'If insufficient, say "Insufficient data in current intelligence graph."\n\n'
        f"Knowledge Graph Context:\n{context}\n\n"
        f"Question: {req.question}\n\nConcise strategic answer:"
    )
    try:
        resp = http.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": req.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "No response.")
    except Exception as exc:
        raise HTTPException(503, f"Ollama error: {exc}")
    return {"answer": answer, "context": context}


# ── Report ────────────────────────────────────────────────────────────────────


@app.post("/api/report")
def export_report(req: ReportRequest, request: Request):
    _check_rate(request, max_req=5, window=60)
    import reporter

    try:
        g = load_graph()
        summary = ""
        if req.entities:
            context = retrieve_graph_context(" ".join(req.entities), g)
            prompt = (
                f"You are a senior geopolitical intelligence analyst.\n"
                f"Write a concise executive strategic summary (max 3 paragraphs) "
                f"focusing on {', '.join(req.entities)}.\n\n"
                f"Context:\n{context}\n\nStrategic Summary:"
            )
            try:
                resp = http.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": req.model, "prompt": prompt, "stream": False},
                    timeout=60,
                )
                resp.raise_for_status()
                summary = resp.json().get("response", "").strip()
            except Exception as exc:
                logger.warning("Failed to generate LLM summary: %s", exc)

        if req.format.lower() in ("md", "markdown"):
            md_content = reporter.generate_markdown_report(g, req.entities, summary)
            return Response(
                content=md_content,
                media_type="text/markdown",
                headers={"Content-Disposition": "attachment; filename=goies_brief.md"},
            )
        else:
            pdf_bytes = reporter.generate_report(g, req.entities, summary)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": "attachment; filename=goies_brief.pdf"},
            )
    except Exception as exc:
        raise HTTPException(500, f"Report failed: {exc}")


# ── Snapshots ─────────────────────────────────────────────────────────────────


@app.get("/api/snapshots")
def list_snapshots():
    if not os.path.exists("goies_snapshots"):
        return {"snapshots": []}
    files = sorted(
        [f for f in os.listdir("goies_snapshots") if f.endswith(".json")], reverse=True
    )
    return {"snapshots": files}


@app.get("/api/snapshots/timeline")
def timeline():
    if not os.path.exists("goies_snapshots"):
        return {"timeline": []}
    files = sorted([f for f in os.listdir("goies_snapshots") if f.endswith(".json")])
    return {
        "timeline": [
            {"id": f, "date": m.group(1)}
            for f in files
            for m in [re.search(r"v_(.*?)\.json$", f)]
            if m
        ]
    }


@app.get("/api/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    snapshots_dir = pathlib.Path("goies_snapshots").resolve()
    filepath = (snapshots_dir / snapshot_id).resolve()
    if not str(filepath).startswith(str(snapshots_dir) + os.sep):
        raise HTTPException(400, "Invalid snapshot ID.")
    if not filepath.exists() or filepath.suffix != ".json":
        raise HTTPException(404, "Snapshot not found.")
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        g = nx.node_link_graph(data, directed=True, multigraph=False)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Corrupt snapshot %s: %s", snapshot_id, exc)
        raise HTTPException(500, "Snapshot file is corrupt.")
    return {
        "vis": graph_to_vis(g),
        "analytics": get_graph_analytics(g, watch_list_thresholds),
    }


# ── Watch list (FIX-10) ───────────────────────────────────────────────────────


@app.post("/api/watch_list")
def update_watch_list(req: WatchListRequest):
    global watch_list_thresholds
    watch_list_thresholds = req.thresholds
    try:
        WATCH_THRESHOLDS_FILE.write_text(
            json.dumps(watch_list_thresholds, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not persist watch thresholds: %s", exc)
    return {
        "status": "success",
        "thresholds": watch_list_thresholds,
        "persistent": True,
    }


# ── GQL ───────────────────────────────────────────────────────────────────────


@app.post("/api/gql")
def gql_query(req: GQLRequest, request: Request):
    _check_rate(request, max_req=60, window=60)
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty.")
    result = run_gql(req.query, graph)
    return result


@app.get("/api/gql/help")
def gql_help():
    return {"help": GQLParser.help_text()}


# ── Embeddings ────────────────────────────────────────────────────────────────


@app.post("/api/embed/train")
async def embed_train(request: Request):
    _check_rate(request, max_req=3, window=60)
    if graph.number_of_nodes() < 5:
        raise HTTPException(400, "Need at least 5 nodes to train embeddings.")
    result = await embedding_engine.train_async(graph)
    if result.get("status") == "error":
        raise HTTPException(500, result["reason"])
    return result


@app.get("/api/embed/status")
def embed_status():
    return embedding_engine.status()


@app.get("/api/embed/similar/{node_id:path}")
def embed_similar(node_id: str, k: int = 8):
    if not embedding_engine.is_trained:
        raise HTTPException(
            400, "Embeddings not trained yet. Call POST /api/embed/train first."
        )
    canonical = resolve_node_name(graph, node_id)
    sims = embedding_engine.similar_nodes(str(canonical), top_k=k)
    if not sims and canonical not in embedding_engine.embeddings:
        raise HTTPException(404, f"Node '{node_id}' not found in embedding space.")
    return {
        "node": canonical,
        "similar": [{"id": nid, "score": round(score, 4)} for nid, score in sims],
    }


@app.get("/api/embed/search")
def embed_search(q: str, k: int = 8):
    if not embedding_engine.is_trained:
        raise HTTPException(400, "Embeddings not trained yet.")
    results = embedding_engine.similar_to_query(q, graph, top_k=k)
    return {
        "query": q,
        "results": [{"id": nid, "score": round(s, 4)} for nid, s in results],
    }


@app.get("/api/embed/clusters")
def embed_clusters(n: int = 5):
    if not embedding_engine.is_trained:
        raise HTTPException(400, "Embeddings not trained yet.")
    n = max(2, min(n, 20))
    clusters = embedding_engine.cluster_nodes(n_clusters=n)
    return {"clusters": clusters, "k": n}


# ── OSINT ─────────────────────────────────────────────────────────────────────


@app.get("/api/osint/status")
def osint_status():
    return osint_engine.get_status()


@app.get("/api/osint/feeds")
def list_feeds():
    return {"feeds": osint_engine.get_feeds()}


@app.post("/api/osint/feeds", status_code=201)
def add_feed(req: FeedRequest):
    osint_engine.add_feed(req.url, req.name)
    return {"added": True, "url": req.url}


@app.delete("/api/osint/feeds")
def remove_feed(url: str):
    osint_engine.remove_feed(url)
    return {"removed": True, "url": url}


@app.post("/api/osint/ingest")
async def osint_ingest(req: OsintIngestRequest, background_tasks: BackgroundTasks):
    async def _run():
        feeds = osint_engine.get_feeds()
        for feed_url in feeds:
            articles = await asyncio.to_thread(
                osint_engine.fetch_feed_articles, feed_url, req.articles_per_feed
            )
            for text in articles:
                if not text:
                    continue
                cancel = CancelToken()
                try:
                    await extract_text(
                        text,
                        model=req.model,
                        cancel=cancel,
                        on_chunk=lambda r: _apply_chunk(r),
                    )
                except Exception as exc:
                    logger.warning("OSINT ingest extraction error: %s", exc)
        save_graph(graph)

    background_tasks.add_task(_run)
    return {"status": "ingestion_started"}


@app.post("/api/osint/continuous/start")
async def continuous_start(req: ContinuousStartRequest):
    global _continuous_task
    async with _continuous_lock:
        if _continuous_state["active"]:
            raise HTTPException(409, "Continuous loop already running.")
        _continuous_state.update(
            {
                "active": True,
                "cycle": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stopped_at": None,
                "interval_secs": max(60, req.interval_secs),
                "articles_per_feed": max(1, min(req.articles_per_feed, 20)),
                "model": req.model,
                "query_log": [],
                "cycle_log": [],
                "total_entities": 0,
                "total_relations": 0,
                "total_articles": 0,
            }
        )
        _continuous_task = asyncio.create_task(_continuous_loop())
    logger.info("Continuous OSINT loop activated — interval=%ds", req.interval_secs)
    return {
        "status": "started",
        "interval_secs": _continuous_state["interval_secs"],
        "feeds": len(osint_engine.get_feeds()),
    }


@app.post("/api/osint/continuous/stop")
async def continuous_stop():
    global _continuous_task
    task_to_await = None
    async with _continuous_lock:
        if not _continuous_state["active"]:
            raise HTTPException(409, "Continuous loop is not running.")
        _continuous_state["active"] = False
        if _continuous_task and not _continuous_task.done():
            _continuous_task.cancel()
            task_to_await = _continuous_task
    if task_to_await:
        try:
            await asyncio.wait_for(asyncio.shield(task_to_await), timeout=10.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    logger.info("Continuous OSINT loop stopped.")
    return {"status": "stopping", "cycles_completed": _continuous_state["cycle"]}


@app.get("/api/osint/continuous/status")
def continuous_status():
    s = _continuous_state
    return {
        "active": s["active"],
        "cycle": s["cycle"],
        "started_at": s["started_at"],
        "stopped_at": s["stopped_at"],
        "interval_secs": s["interval_secs"],
        "articles_per_feed": s["articles_per_feed"],
        "model": s["model"],
        "total_entities": s["total_entities"],
        "total_relations": s["total_relations"],
        "total_articles": s["total_articles"],
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "cycle_log": s["cycle_log"][:10],
        "query_log": s["query_log"][:20],
    }


@app.post("/api/osint/enrich/{node_id:path}")
async def osint_enrich(node_id: str, request: Request, model: str = DEFAULT_MODEL):
    _check_rate(request, max_req=10, window=60)
    canonical = resolve_node_name(graph, node_id)
    if canonical not in graph:
        raise HTTPException(404, f"Node '{node_id}' not found.")
    enrichment = await osint_engine.enrich_entity_wikipedia(canonical, model)
    if enrichment and "error" not in enrichment:
        attrs = graph.nodes[canonical].get("attributes", {})
        if isinstance(attrs, str):
            try:
                attrs = ast.literal_eval(attrs)
            except (ValueError, SyntaxError):
                attrs = {}
        attrs.update(enrichment)
        graph.nodes[canonical]["attributes"] = attrs
        save_graph(graph)
    return {"node": canonical, "enrichment": enrichment}


@app.get("/api/osint/gdelt")
async def osint_gdelt(entity: str, days: int = 7):
    days = max(1, min(days, 90))
    articles = await osint_engine.query_gdelt(entity, days)
    return {"entity": entity, "articles": articles, "count": len(articles)}


# ── Root ──────────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    from fastapi.responses import FileResponse, RedirectResponse

    index = pathlib.Path("frontend") / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return RedirectResponse(url="/api/docs")


@app.get("/app.html")
def app_dashboard():
    from fastapi.responses import FileResponse, RedirectResponse

    page = pathlib.Path("frontend") / "app.html"
    if page.exists():
        return FileResponse(str(page))
    return RedirectResponse(url="/")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
