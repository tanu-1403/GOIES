<<<<<<< HEAD
"""server.py — GOIES FastAPI Backend v4 (GQL + Embeddings + OSINT)

Previous fixes (v4.0):
  FIX-1  eval() on node title replaced with ast.literal_eval()
  FIX-2  _attrs dict embedded in graph_to_vis() so frontend never needs eval()
  FIX-3  threading.Lock around _update_graph() — concurrent mutation guard
  FIX-4  Path traversal in /api/snapshots/{id} blocked
  FIX-5  CORS wildcard → ALLOWED_ORIGINS env var
  FIX-6  print() → structured logger
  FIX-7  stdlib imports hoisted to module level

New fixes (v4.1):
  FIX-8   XSS via node labels — all LLM-derived strings HTML-escaped in _fmt_tooltip()
  FIX-9   No upload size cap — /api/ingest/file now enforces MAX_UPLOAD_BYTES (10 MB)
  FIX-10  watch_list_thresholds persisted to watch_thresholds.json
  FIX-11  Startup Ollama health check — warns clearly if unreachable at boot
  FIX-12  Rate limiting — sliding-window per-IP token bucket (no external deps)
  FIX-13  Content-Security-Policy header added via middleware
  FIX-14  find_all_paths timeout (GQL engine)
  FIX-15  GQL LIMIT clause — open queries capped at 200 rows by default
  FIX-16  label_diversity zero-div / false-perfect score for empty edge labels (utils.py)
  FIX-17  Cross-session entity deduplication — extractor.py persists seen keys to disk;
          DELETE /api/extract/seen resets the cache when a clean ingest is needed
=======
"""
server.py  —  GOIES FastAPI Backend
=====================================
Fixes wired in:
  #1  All Ollama calls are now async (via fixed extractor.py + httpx)
  #2  Timeout configurable via OLLAMA_TIMEOUT env var
  #3  CancelToken per extraction task; DELETE /api/extract/{task_id} cancels it
  #4  DELETE /api/graph calls utils.clear_graph() — wipes backend state
  #5  Parallel chunk processing transparent via extractor.extract_text()
  #6  Edge parsing fixed in extractor.py
  #7  POST /api/extract/stream yields incremental SSE per chunk
  #8  startup_event() calls extractor.warmup_model()
  #9  Atomic graph save/load in utils.py
>>>>>>> ce28496 (v3 initiate)
"""

from __future__ import annotations

<<<<<<< HEAD
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
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
from pydantic import BaseModel

from embedding_engine import GraphEmbeddingEngine
from extractor import (
    _call_ollama,
    check_ollama_health,
    extract_intelligence,
    extract_intelligence_stream,
    list_available_models,
)
from forecaster import run_forecast
from geo import get_geo_data
from osint_engine import OsintEngine
from query_engine import GQLParser, run_gql
from simulator import run_simulation
import itertools
import requests as http  # used in query(), export_report(), graph_summary()

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

from contextlib import asynccontextmanager

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("goies.server")

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_INPUT_CHARS  = 500_000
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # FIX-9: 10 MB default
WATCH_THRESHOLDS_FILE = pathlib.Path("watch_thresholds.json")

# FIX-5: Restrict CORS
_raw_origins   = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

GROUP_COLORS: Dict[str, str] = {
    "country":      "#ff7b72",
    "person":       "#ffa657",
    "organization": "#d2a8ff",
    "technology":   "#79c0ff",
    "event":        "#7ee787",
    "treaty":       "#f0e68c",
    "resource":     "#56d364",
    "unknown":      "#8b949e",
}
=======
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from extractor import (
    CancelToken,
    ChunkResult,
    Entity,
    Relationship,
    extract_stream,
    extract_text,
    warmup_model,
)
from utils import (
    add_edge,
    add_node,
    clear_graph,
    get_graph,
    get_graph_analytics,
    graph_to_visjs,
    list_snapshots,
    load_graph,
    load_snapshot,
    merge_nodes,
    save_graph,
    save_snapshot,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

DEFAULT_MODEL = os.getenv("GOIES_DEFAULT_MODEL", "llama3.2")

# ── In-flight extraction tasks (Fix #3) ───────────────────────────────────────

_active_tasks: dict[str, CancelToken] = {}

# ── Lifespan (Fix #8) ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load persisted graph on startup (Fix #9)
    load_graph()
    log.info("Graph loaded from disk.")

    # Warm up the default model (Fix #8)
    asyncio.create_task(warmup_model(DEFAULT_MODEL))

    yield

    # Persist graph on clean shutdown (Fix #9)
    save_graph()
    log.info("Graph persisted on shutdown.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GOIES API",
    version="2.0.0",
    description="Geopolitical Open Intelligence & Extraction System",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if os.path.isdir("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="frontend")
>>>>>>> ce28496 (v3 initiate)

# ── Rate Limiter (no external dependency) ─────────────────────────────────────
# FIX-12: Simple token-bucket rate limiter — no slowapi/Redis dependency needed.
# Per-IP sliding-window counters stored in memory (fine for single-process local deploy).

<<<<<<< HEAD
class _RateLimiter:
    _PRUNE_INTERVAL = 300   # seconds between stale-key sweeps

    def __init__(self):
        self._windows: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def is_allowed(self, key: str, max_requests: int, window_secs: float) -> bool:
        now = time.monotonic()
        with self._lock:
            # Periodically purge keys with no recent activity to prevent unbounded growth
            if now - self._last_prune > self._PRUNE_INTERVAL:
                stale_cutoff = now - max(window_secs, 3600)
                self._windows = defaultdict(
                    list,
                    {k: v for k, v in self._windows.items()
                     if any(t > stale_cutoff for t in v)}
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
    """Raise 429 if the caller's IP has exceeded the rate limit."""
    client_ip = (request.client.host if request.client else "unknown")
    key = f"{client_ip}:{request.url.path}"
    if not _rate_limiter.is_allowed(key, max_req, window):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_req} requests per {int(window)}s.",
            headers={"Retry-After": str(int(window))},
        )

@asynccontextmanager
async def _lifespan(app):
    print("🚀 Booting GOIES engine...")

    # Start engine automatically
    _continuous_state.update({
        "active": True,
        "cycle": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stopped_at": None,
    })

    asyncio.create_task(_continuous_loop())

    yield

    print("🛑 Shutting down GOIES...")

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="GOIES", version="4.2.0", docs_url="/api/docs", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


# FIX-13: Content-Security-Policy middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://unpkg.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com https://unpkg.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https://*.tile.openstreetmap.org; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"]        = "DENY"
    response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
    return response


# ── Shared State ───────────────────────────────────────────────────────────────
graph: nx.DiGraph = load_graph()
_graph_lock = threading.Lock()

# FIX-10: Load persisted thresholds on startup
def _load_watch_thresholds() -> Dict[str, float]:
    if WATCH_THRESHOLDS_FILE.exists():
        try:
            return json.loads(WATCH_THRESHOLDS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}

watch_list_thresholds: Dict[str, float] = _load_watch_thresholds()

embedding_engine = GraphEmbeddingEngine()
osint_engine     = OsintEngine()


# FIX-11 / Issue-32: Startup checks via lifespan (on_event deprecated in FastAPI ≥ 0.93)
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app):
    # ── Startup ──────────────────────────────────────────────────────────────
    health = check_ollama_health()
    if health["online"]:
        logger.info("Ollama online at %s — models: %s", OLLAMA_BASE_URL, health["models"])
    else:
        logger.warning(
            "⚠ Ollama NOT reachable at %s — extraction will fail until Ollama is started. "
            "Error: %s",
            OLLAMA_BASE_URL,
            health["error"],
        )
    logger.info(
        "Graph loaded: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    yield
    # ── Shutdown — flush a clean graph state ─────────────────────────────────
    logger.info("Shutdown: flushing graph to disk…")
    try:
        save_graph(graph)
    except Exception as exc:
        logger.error("Shutdown graph save failed: %s", exc)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _fmt_tooltip(group: str, attributes: dict, confidence: float) -> str:
    color = GROUP_COLORS.get(group, "#8b949e")
    # FIX-8: HTML-escape all LLM-derived strings before injecting into DOM
    lines = [f'<b style="color:{color};font-family:monospace">{html.escape(group.upper())}</b>']
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
        conf  = data.get("confidence", 1.0)
        attrs = _safe_parse_attrs(data.get("title", "{}"))
        nodes.append(
            {
                "id":         node_id,
                "label":      node_id,
                "group":      group,
                "color": {
                    "background": color,
                    "border":     color,
                    "highlight":  {"background": "#ffffff", "border": color},
                    "hover":      {"background": color,     "border": "#ffffff"},
                },
                "title":      _fmt_tooltip(group, attrs, conf),
                "_attrs":     attrs,
                "confidence": conf,
                "size":       16,
                "borderWidth": 2,
                "font":   {"color": "#e2e8f0", "size": 13},
                "shadow": {"enabled": True, "color": color + "44", "size": 12, "x": 0, "y": 0},
            }
        )

    edges = []
    for u, v, data in g.edges(data=True):
        edges.append(
            {
                "from":  u,
                "to":    v,
                "label": data.get("label", ""),
                "arrows": "to",
                "color": {
                    "color":     "#1e3a5f",
                    "highlight": "#00e5ff",
                    "hover":     "#00e5ff",
                    "inherit":   False,
                },
                "font":   {"color": "#3d5a7a", "size": 10, "align": "middle", "strokeWidth": 0},
                "width":  1.5,
                "smooth": {"type": "continuous"},
                "confidence": data.get("confidence", 1.0),
            }
        )

    return {"nodes": nodes, "edges": edges}


def _update_graph(extractions) -> dict:
    nodes_added, edges_added, new_ids = 0, 0, []
    with _graph_lock:
        for ext in extractions:
            cls = ext.extraction_class.lower()
            if cls in {"country", "person", "organization", "technology", "event", "treaty", "resource"}:
                canonical = resolve_node_name(graph, ext.extraction_text)
                if not graph.has_node(canonical):
                    nodes_added += 1
                    new_ids.append(canonical)
                    graph.add_node(canonical, title=str(ext.attributes), group=cls, confidence=ext.confidence)
                else:
                    # Merge: keep whichever extraction has higher confidence;
                    # always update group if it was previously "unknown"
                    existing = graph.nodes[canonical]
                    if existing.get("group", "unknown") == "unknown":
                        existing["group"] = cls
                    if ext.confidence > existing.get("confidence", 0.0):
                        existing["title"] = str(ext.attributes)
                        existing["confidence"] = ext.confidence
            elif cls == "relationship":
                src_raw = ext.attributes.get("source", "")
                tgt_raw = ext.attributes.get("target", "")
                if not src_raw or not tgt_raw:
                    logger.debug("Dropped relationship with missing src/tgt: %r", ext.attributes)
                    continue
                src = resolve_node_name(graph, src_raw)
                tgt = resolve_node_name(graph, tgt_raw)
                for n in (src, tgt):
                    if not graph.has_node(n):
                        graph.add_node(n, group="unknown")
                if not graph.has_edge(src, tgt):
                    edges_added += 1
                graph.add_edge(src, tgt, label=ext.extraction_text, confidence=ext.confidence)

    save_graph(graph)  # I/O outside the lock — keeps lock held time minimal

    return {"nodes_added": nodes_added, "edges_added": edges_added, "new_node_ids": new_ids}


# ── Request Models ─────────────────────────────────────────────────────────────
=======
# ── Pydantic request/response models ─────────────────────────────────────────


>>>>>>> ce28496 (v3 initiate)
class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=10)
    model: str = Field(DEFAULT_MODEL)

<<<<<<< HEAD
class QueryRequest(BaseModel):
    question: str
    model: str = "llama3.2"
    persona: str = "senior geopolitical intelligence analyst"
=======

class ExtractResponse(BaseModel):
    task_id: str
    entities: list[dict]
    relationships: list[dict]
    elapsed: float
    chunks: int


class GraphClearResponse(BaseModel):
    cleared: bool
    message: str


class MergeRequest(BaseModel):
    keep_id: str
    drop_id: str
>>>>>>> ce28496 (v3 initiate)

class SimulateRequest(BaseModel):
    scenario: str
    model: str = DEFAULT_MODEL

class ForecastRequest(BaseModel):
    model: str = DEFAULT_MODEL

<<<<<<< HEAD
class UrlIngestRequest(BaseModel):
    url: str

class ReportRequest(BaseModel):
    entities: List[str] = []
    format: str = "pdf"
    model: str = "llama3.2"

class WatchListRequest(BaseModel):
    thresholds: Dict[str, float]

class ExtractUrlRequest(BaseModel):
    url: str
    model: str = "llama3.2"
    persona: str = "senior geopolitical intelligence analyst"

class MergeRequest(BaseModel):
    source: str
    target: str
=======

class QueryRequest(BaseModel):
    query: str
    model: str = DEFAULT_MODEL

>>>>>>> ce28496 (v3 initiate)

class GQLRequest(BaseModel):
    query: str

<<<<<<< HEAD
class FeedRequest(BaseModel):
    url: str
    name: str = ""

class OsintIngestRequest(BaseModel):
    model: str = "llama3.2"
    articles_per_feed: int = 5


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return check_ollama_health()


# ── Ingest ─────────────────────────────────────────────────────────────────────
def _validate_url(url: str) -> None:
    """Issue-25: Block SSRF vectors — private IPs, loopback, and non-http(s) schemes."""
    import ipaddress, urllib.parse
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"Unsupported URL scheme '{parsed.scheme}'. Only http/https allowed.")
    host = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise HTTPException(400, "Requests to private/loopback IP addresses are not allowed.")
    except ValueError:
        pass  # hostname (not IP) — allow; DNS resolution is handled by the ingestor
    blocked = ("localhost", "metadata.google.internal")
    if any(host.lower() == b or host.lower().endswith("." + b) for b in blocked):
        raise HTTPException(400, "Requests to reserved hostnames are not allowed.")


@app.post("/api/ingest/url")
def ingest_url(req: UrlIngestRequest, request: Request):
    _check_rate(request, max_req=30, window=60)
    _validate_url(req.url)
    try:
        from ingestor import fetch_url_text
        text = fetch_url_text(req.url)
        return {"text": text, "chars": len(text)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/extract/url")
def extract_url_stream(req: ExtractUrlRequest, request: Request):
    _check_rate(request, max_req=10, window=60)
    _validate_url(req.url)

    def event_generator():
        try:
            from ingestor import fetch_url_text
            text = fetch_url_text(req.url)
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Fetch failed: {e}'})}\n\n"
            return

        if not text.strip():
            yield f"data: {json.dumps({'error': 'No text could be extracted from URL'})}\n\n"
            return

        yield f"data: {json.dumps({'fetched': True, 'chars': len(text), 'url': req.url})}\n\n"

        total_ent, total_rel, new_nodes = 0, 0, []
        try:
            for chunk_data in extract_intelligence_stream(text, model=req.model, persona=req.persona):
                extractions = chunk_data["extractions"]
                diff  = _update_graph(extractions)
                ents  = sum(1 for e in extractions if e.extraction_class.lower() != "relationship")
                rels  = len(extractions) - ents
                total_ent += ents
                total_rel += rels
                new_nodes.extend(diff["new_node_ids"])
                payload = {
                    "chunk":        chunk_data["chunk_index"],
                    "total_chunks": chunk_data["total_chunks"],
                    "entities":     ents,
                    "relations":    rels,
                    "new_node_ids": diff["new_node_ids"],
                    "vis":          graph_to_vis(graph),
                    "analytics":    get_graph_analytics(graph, watch_list_thresholds),
                }
                if chunk_data.get("parse_error"):
                    payload["parse_error"] = chunk_data["parse_error"]
                yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'done': True, 'totals': {'entities': total_ent, 'relations': total_rel, 'new_nodes': list(set(new_nodes))}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/ingest/file")
async def ingest_file(request: Request, file: UploadFile = File(...)):
    _check_rate(request, max_req=20, window=60)

    # FIX-9: Enforce upload size cap before reading entire file into memory
    content_length = request.headers.get("content-length")
    try:
        _cl_int = int(content_length) if content_length else 0
    except (ValueError, TypeError):
        _cl_int = 0
    if _cl_int > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    from ingestor import parse_pdf, parse_docx

    content  = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.")

    filename = file.filename.lower() if file.filename else ""

    try:
        if filename.endswith(".pdf"):
            text = parse_pdf(content)
        elif filename.endswith(".docx"):
            text = parse_docx(content)
        elif filename.endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(400, "Unsupported format. Please upload PDF, DOCX, TXT, or MD.")
        return {"text": text, "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error parsing file: {e}")


# ── Models ─────────────────────────────────────────────────────────────────────
@app.get("/api/models")
def models():
    return {"models": list_available_models()}


# ── Snapshots ──────────────────────────────────────────────────────────────────
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
    timeline_data = []
    for f in files:
        match = re.search(r"v_(.*?)\.json$", f)
        if match:
            timeline_data.append({"id": f, "date": match.group(1)})
    return {"timeline": timeline_data}


@app.get("/api/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str):
    snapshots_dir = pathlib.Path("goies_snapshots").resolve()
    filepath = (snapshots_dir / snapshot_id).resolve()

    if not str(filepath).startswith(str(snapshots_dir) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid snapshot ID.")
    if not filepath.exists() or filepath.suffix != ".json":
        raise HTTPException(status_code=404, detail="Snapshot not found")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        g = nx.node_link_graph(data, directed=True, multigraph=False)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.warning("Corrupt snapshot %s: %s", snapshot_id, exc)
        raise HTTPException(status_code=500, detail="Snapshot file is corrupt.")
    return {"vis": graph_to_vis(g), "analytics": get_graph_analytics(g, watch_list_thresholds)}


# ── Watch List ─────────────────────────────────────────────────────────────────
@app.post("/api/watch_list")
def update_watch_list(req: WatchListRequest):
    global watch_list_thresholds
    watch_list_thresholds = req.thresholds
    # FIX-10: Persist to disk
    try:
        WATCH_THRESHOLDS_FILE.write_text(
            json.dumps(watch_list_thresholds, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Could not persist watch thresholds: %s", exc)
    return {"status": "success", "thresholds": watch_list_thresholds, "persistent": True}


# ── Report ─────────────────────────────────────────────────────────────────────
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
                f"Based on the following knowledge graph context focusing on {', '.join(req.entities)},\n"
                f"write a concise executive strategic summary (max 3 paragraphs) of the situation.\n\n"
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
            except Exception as e:
                logger.warning("Failed to generate LLM summary: %s", e)

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report failed: {e}")


# ── Graph ──────────────────────────────────────────────────────────────────────
@app.get("/api/graph")
def get_graph_ep(ego: Optional[str] = None, hops: int = 2):
    hops = max(1, min(hops, 4))  # clamp: ego BFS beyond 4 hops is O(n²) expensive
    g = get_ego_subgraph(graph, ego, hops) if ego and ego in graph else graph
    return {
        "vis":       graph_to_vis(g),
        "analytics": get_graph_analytics(graph, watch_list_thresholds),
        "filtered":  ego is not None and ego in graph,
    }


@app.get("/api/narrative/summary")
def graph_summary(request: Request, model: str = "llama3.2"):
    _check_rate(request, max_req=5, window=60)
    analytics  = get_graph_analytics(graph, watch_list_thresholds)
    edge_sample = [
        f"{u} -> {v} [{d.get('label', '')}]"
        for u, v, d in itertools.islice(graph.edges(data=True), 25)
    ]
    SUMMARY_PROMPT = f"""
You are a senior intelligence analyst. Describe the following geopolitical network in 3 paragraphs.
Focus on: major power actors, key conflict zones, most significant tensions, dominant alliance patterns.
Use direct, professional language. No hedging. Cite specific entity names.

Graph statistics:
- {analytics.get("nodes")} entities: {analytics.get("group_counts", {})}
- {analytics.get("edges")} relationships
- Highest tension: {list(analytics.get("tensions", {}).items())[:3]}
- Most connected: {analytics.get("top_degree", [])}

Key relationships sample:
{chr(10).join(edge_sample)}

Write the 3-paragraph intelligence summary now:
"""
    try:
        narrative = _call_ollama(SUMMARY_PROMPT, model)
        return {"narrative": narrative, "generated_at": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        raise HTTPException(500, f"Summary generation failed: {e}")


@app.get("/api/path")
def path(src: str, tgt: str, request: Request):
    _check_rate(request, max_req=30, window=60)  # Issue-27
    from graph_algo import find_shortest_path
    src_canon = resolve_node_name(graph, src)
    tgt_canon = resolve_node_name(graph, tgt)
    if not graph.has_node(src_canon) or not graph.has_node(tgt_canon):
        raise HTTPException(404, "One or both nodes not found in graph.")
    path_data = find_shortest_path(graph, src_canon, tgt_canon)
    if not path_data["nodes"]:
        return {"found": False, "nodes": [], "edges": []}
    return {"found": True, "nodes": path_data["nodes"], "edges": path_data["edges"]}


@app.post("/api/node/merge")
def merge_node_ep(req: MergeRequest, request: Request):
    _check_rate(request, max_req=20, window=60)  # Issue-27
    src_canon = resolve_node_name(graph, req.source)
    tgt_canon = resolve_node_name(graph, req.target)
    if src_canon == tgt_canon:
        raise HTTPException(400, "Source and target resolve to the same node.")
    success = merge_nodes(graph, src_canon, tgt_canon)
    if not success:
        raise HTTPException(400, "Failed to merge nodes. Ensure both exist.")
    return {"status": "success", "merged": src_canon, "into": tgt_canon}


@app.post("/api/extract")
def extract(req: ExtractRequest, request: Request):
    _check_rate(request, max_req=10, window=60)
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Input exceeds {MAX_INPUT_CHARS:,} chars.")
    try:
        extractions = extract_intelligence(req.text, model=req.model)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    diff     = _update_graph(extractions)
    entities = sum(1 for e in extractions if e.extraction_class.lower() != "relationship")
    return {
        "extractions": len(extractions),
        "entities":    entities,
        "relations":   len(extractions) - entities,
        **diff,
        "vis":       graph_to_vis(graph),
        "analytics": get_graph_analytics(graph, watch_list_thresholds),
    }


@app.post("/api/extract/stream")
def extract_stream(req: ExtractRequest, request: Request):
    _check_rate(request, max_req=10, window=60)
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(400, f"Input exceeds {MAX_INPUT_CHARS:,} chars.")

    def event_generator():
        total_entities, total_relations = 0, 0
        new_nodes_all: List[str] = []
        # Issue-16/17: Send only the delta vis payload during streaming
        # (not the full graph every chunk) and skip expensive analytics
        # mid-stream — analytics only sent on the final done event.
        all_known_node_ids: set = {n for n in graph.nodes()}
        try:
            for chunk_data in extract_intelligence_stream(req.text, model=req.model, persona=req.persona):
                extractions = chunk_data["extractions"]
                diff      = _update_graph(extractions)
                entities  = sum(1 for e in extractions if e.extraction_class.lower() != "relationship")
                relations = len(extractions) - entities
                total_entities  += entities
                total_relations += relations
                new_node_ids = diff["new_node_ids"]
                new_nodes_all.extend(new_node_ids)

                # Only send vis data for newly added nodes + their immediate edges
                new_vis: Optional[dict] = None
                if new_node_ids:
                    delta_nodes = [
                        n for n in graph_to_vis(graph)["nodes"]
                        if n["id"] in new_node_ids or n["id"] not in all_known_node_ids
                    ]
                    delta_edges = [
                        e for e in graph_to_vis(graph)["edges"]
                        if e["from"] in new_node_ids or e["to"] in new_node_ids
                    ]
                    all_known_node_ids.update(new_node_ids)
                    new_vis = {"nodes": delta_nodes, "edges": delta_edges}

                event_payload = {
                    "chunk":        chunk_data["chunk_index"],
                    "total_chunks": chunk_data["total_chunks"],
                    "extractions":  len(extractions),
                    "entities":     entities,
                    "relations":    relations,
                    "new_node_ids": new_node_ids,
                }
                if new_vis:
                    event_payload["vis_delta"] = new_vis
                if chunk_data.get("parse_error"):
                    event_payload["parse_error"] = chunk_data["parse_error"]
                yield f"data: {json.dumps(event_payload)}\n\n"

            # Full vis + analytics only on completion
            yield f"data: {json.dumps({'done': True, 'totals': {'entities': total_entities, 'relations': total_relations, 'new_nodes': list(set(new_nodes_all))}, 'vis': graph_to_vis(graph), 'analytics': get_graph_analytics(graph, watch_list_thresholds)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
        f"Knowledge Graph Context:\n{context}\n\nQuestion: {req.question}\n\nConcise strategic answer:"
    )
    try:
        resp = http.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": req.model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        answer = resp.json().get("response", "No response.")
    except Exception as e:
        raise HTTPException(503, f"Ollama error: {e}")
    return {"answer": answer, "context": context}


@app.delete("/api/graph")
def clear_graph():
    with _graph_lock:
        graph.clear()
    save_graph(graph)  # I/O outside the lock — consistent with _update_graph
    return {"status": "cleared"}


@app.delete("/api/extract/seen")
def clear_seen_cache():
    """
    FIX-17: Reset the cross-session entity deduplication cache.
    Use this before a clean re-ingest of the same source material
    so that previously-seen entities are extracted again.
    """
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


@app.get("/api/export/{fmt}")
def export(fmt: str):
    if fmt == "json":
        return StreamingResponse(
            io.StringIO(export_json(graph)), media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=goies_graph.json"},
        )
    elif fmt == "csv":
        return StreamingResponse(
            io.StringIO(export_csv(graph)), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=goies_edges.csv"},
        )
    elif fmt == "graphml":
        return StreamingResponse(
            io.BytesIO(export_graphml(graph)), media_type="application/xml",
            headers={"Content-Disposition": "attachment; filename=goies_graph.graphml"},
        )
    raise HTTPException(400, f"Unknown format: {fmt}")


# ── Geo ────────────────────────────────────────────────────────────────────────
@app.get("/api/geo")
def get_geo():
    markers = get_geo_data(graph)
    return {"markers": markers, "total": len(markers)}


# ── Simulation ─────────────────────────────────────────────────────────────────
@app.post("/api/simulate")
async def simulate(req: SimulateRequest, request: Request):
    # Issue-24: made async + offloaded to thread executor.
    # run_simulation makes two sequential blocking Ollama calls (up to ~4 min total);
    # running them in a sync handler starves the Uvicorn worker thread pool.
    _check_rate(request, max_req=5, window=60)
    if not req.scenario.strip():
        raise HTTPException(400, "Scenario cannot be empty.")
    if len(graph.nodes) == 0:
        raise HTTPException(400, "Graph is empty. Ingest data first.")
    try:
        import functools
        result = await asyncio.get_event_loop().run_in_executor(
            None, functools.partial(run_simulation, req.scenario, graph, model=req.model)
        )
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {
        "scenario":          result.scenario,
        "risk_score":        result.risk_score,
        "risk_label":        result.risk_label,
        "cascade_narrative": result.cascade_narrative,
        "second_order":      result.second_order,
        "added_edges":       result.added_edges,
        "removed_edges":     result.removed_edges,
        "affected_nodes":    result.affected_nodes,
        "model_used":        result.model_used,
    }


@app.get("/api/simulations")
def get_simulations():
    history_file = "sim_history.json"
    if not os.path.exists(history_file):
        return {"history": []}
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read simulation history: {e}")


# ── Forecast ───────────────────────────────────────────────────────────────────
@app.post("/api/forecast")
def forecast(req: ForecastRequest, request: Request):
    _check_rate(request, max_req=5, window=60)
    if len(graph.nodes) < 3:
        raise HTTPException(400, "Need at least 3 nodes to generate a forecast.")
    try:
        result = run_forecast(graph, model=req.model, focus_query=req.focus)
    except ConnectionError as e:
        raise HTTPException(503, str(e))
    except TimeoutError as e:
        raise HTTPException(504, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    return {
        "global_risk":        result.global_risk,
        "global_label":       result.global_label,
        "structural_summary": result.structural_summary,
        "hotspot_nodes":      result.hotspot_nodes,
        "model_used":         result.model_used,
        "forecasts": [
            {
                "rank":              f.rank,
                "title":             f.title,
                "actors":            f.actors,
                "probability":       f.probability,
                "severity":          f.severity,
                "timeframe":         f.timeframe,
                "structural_signal": f.structural_signal,
                "narrative":         f.narrative,
                "mitigation":        f.mitigation,
            }
            for f in result.forecasts
        ],
    }


# ── GQL ────────────────────────────────────────────────────────────────────────
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


# ── Embeddings ─────────────────────────────────────────────────────────────────
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
        raise HTTPException(400, "Embeddings not trained yet. Call POST /api/embed/train first.")
    canonical = resolve_node_name(graph, node_id)
    sims = embedding_engine.similar_nodes(str(canonical), top_k=k)
    if not sims and canonical not in embedding_engine.embeddings:
        raise HTTPException(404, f"Node '{node_id}' not found in embedding space.")
    return {"node": canonical, "similar": [{"id": nid, "score": round(score, 4)} for nid, score in sims]}


@app.get("/api/embed/search")
def embed_search(q: str, k: int = 8):
    if not embedding_engine.is_trained:
        raise HTTPException(400, "Embeddings not trained yet.")
    results = embedding_engine.similar_to_query(q, graph, top_k=k)
    return {"query": q, "results": [{"id": nid, "score": round(s, 4)} for nid, s in results]}


@app.get("/api/embed/clusters")
def embed_clusters(n: int = 5):
    if not embedding_engine.is_trained:
        raise HTTPException(400, "Embeddings not trained yet.")
    n = max(2, min(n, 20))  # KMeans requires n_clusters >= 2; cap at 20
    clusters = embedding_engine.cluster_nodes(n_clusters=n)
    return {"clusters": clusters, "k": n}


# ── OSINT ──────────────────────────────────────────────────────────────────────
=======

class OsintFeedRequest(BaseModel):
    url: str


class EmbedTrainRequest(BaseModel):
    dimensions: int = 64
    walk_length: int = 30
    num_walks: int = 200


class ReportRequest(BaseModel):
    format: str = "pdf"  # "pdf" | "md"
    entities: list[str] = []
    include_graph: bool = True
    include_forecast: bool = True


class WatchListRequest(BaseModel):
    thresholds: dict[str, float]


# ── Health ────────────────────────────────────────────────────────────────────


@app.get("/api/health")
async def health() -> dict:
    import httpx

    ollama_ok = False
    models: list[str] = []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/tags",
                timeout=5.0,
            )
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            ollama_ok = True
    except Exception as exc:
        log.warning("Ollama health check failed: %s", exc)

    g = get_graph()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama_ok": ollama_ok,
        "models": models,
        "graph_nodes": g.number_of_nodes(),
        "graph_edges": g.number_of_edges(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/models")
async def list_models() -> dict:
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/api/tags",
                timeout=5.0,
            )
            r.raise_for_status()
            return {"models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as exc:
        raise HTTPException(502, f"Cannot reach Ollama: {exc}") from exc


# ── Extraction (Fixes #1 #2 #3 #5 #6 #7) ─────────────────────────────────────


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest) -> ExtractResponse:
    """Synchronous extraction — waits for all chunks, returns full result."""
    task_id = str(uuid.uuid4())
    cancel = CancelToken()
    _active_tasks[task_id] = cancel

    t0 = time.perf_counter()
    g = get_graph()

    chunks_processed = [0]

    def on_chunk(result: ChunkResult) -> None:
        # Fix #7: incrementally commit each chunk to the graph as it arrives
        for ent in result.entities:
            add_node(g, ent.id, ent.group, ent.confidence, ent.attributes)
        for rel in result.relationships:
            # Ensure endpoint nodes exist before adding edge (Fix #6)
            if rel.from_id not in g:
                add_node(g, rel.from_id)
            if rel.to_id not in g:
                add_node(g, rel.to_id)
            add_edge(g, rel.from_id, rel.to_id, rel.label, rel.confidence)
        chunks_processed[0] += 1

    try:
        entities, relationships = await extract_text(
            req.text, req.model, cancel, on_chunk=on_chunk
        )
    except asyncio.CancelledError:
        raise HTTPException(409, "Extraction cancelled")
    finally:
        _active_tasks.pop(task_id, None)

    save_graph(g)  # Fix #9: atomic persist after extraction

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
    """Fix #3: Cancel an in-flight extraction task by its task_id."""
    token = _active_tasks.get(task_id)
    if not token:
        raise HTTPException(404, f"No active task with id={task_id}")
    token.cancel()
    return {"cancelled": True, "task_id": task_id}


@app.post("/api/extract/stream")
async def extract_stream_endpoint(req: ExtractRequest) -> StreamingResponse:
    """
    Fix #7: Server-Sent Events stream — yields one event per chunk as it finishes.
    The frontend can update the graph incrementally without waiting for all chunks.
    """
    task_id = str(uuid.uuid4())
    cancel = CancelToken()
    _active_tasks[task_id] = cancel

    g = get_graph()

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in extract_stream(req.text, req.model, cancel):
                # Commit new entities/relationships to the graph immediately
                if event.get("type") == "chunk":
                    for ent in event.get("new_entities", []):
                        add_node(
                            g,
                            ent["id"],
                            ent.get("group", "unknown"),
                            ent.get("confidence", 1.0),
                            ent.get("attributes", {}),
                        )
                    for rel in event.get("new_relationships", []):
                        frm = rel.get("from", "")
                        to = rel.get("to", "")
                        if frm and to:
                            if frm not in g:
                                add_node(g, frm)
                            if to not in g:
                                add_node(g, to)
                            add_edge(
                                g,
                                frm,
                                to,
                                rel.get("label", "related"),
                                rel.get("confidence", 1.0),
                            )

                if event.get("type") == "done":
                    save_graph(g)  # Fix #9: persist after stream completes

                import json as _json

                yield f"data: {_json.dumps({'task_id': task_id, **event})}\n\n"
        except asyncio.CancelledError:
            yield f"data: {{}}\n\n"
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


# ── Ingest helpers ────────────────────────────────────────────────────────────


@app.post("/api/ingest/url")
async def ingest_url(
    url: str = Query(...),
    model: str = Query(DEFAULT_MODEL),
) -> dict:
    try:
        from ingestor import fetch_url_text

        text = await asyncio.to_thread(fetch_url_text, url)
    except Exception as exc:
        raise HTTPException(400, f"URL fetch failed: {exc}") from exc
    req = ExtractRequest(text=text, model=model)
    return await extract(req)


@app.post("/api/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
) -> dict:
    try:
        from ingestor import extract_file_text

        raw = await file.read()
        text = extract_file_text(raw, file.filename or "")
    except Exception as exc:
        raise HTTPException(400, f"File extraction failed: {exc}") from exc
    req = ExtractRequest(text=text, model=model)
    return await extract(req)


# ── Graph (Fix #4) ────────────────────────────────────────────────────────────


@app.get("/api/graph")
async def get_graph_endpoint(
    ego: Optional[str] = Query(None),
    hops: int = Query(2),
) -> dict:
    g = get_graph()
    if ego and ego in g:
        from graph_algo import ego_subgraph

        sub = ego_subgraph(g, ego, hops)
        vis = graph_to_visjs(sub)
    else:
        vis = graph_to_visjs(g)
    return {**vis, "analytics": get_graph_analytics(g)}


@app.delete("/api/graph", response_model=GraphClearResponse)
async def delete_graph() -> GraphClearResponse:
    """
    Fix #4: Clears both the backend NetworkX graph AND persists the empty state.
    Previously only the frontend received a clear signal.
    """
    clear_graph()
    return GraphClearResponse(
        cleared=True,
        message="Graph cleared — backend state reset and persisted to disk.",
    )


@app.get("/api/path")
async def shortest_path(
    src: str = Query(...),
    tgt: str = Query(...),
) -> dict:
    from graph_algo import find_path

    g = get_graph()
    path = find_path(g, src, tgt)
    if path is None:
        raise HTTPException(404, f"No path from '{src}' to '{tgt}'")
    return {"path": path}


@app.post("/api/node/merge")
async def merge_nodes_endpoint(req: MergeRequest) -> dict:
    g = get_graph()
    ok = merge_nodes(g, req.keep_id, req.drop_id)
    if not ok:
        raise HTTPException(404, "One or both nodes not found")
    save_graph(g)
    return {"merged": True, "kept": req.keep_id, "dropped": req.drop_id}


@app.get("/api/export/{fmt}")
async def export_graph(fmt: str) -> StreamingResponse:
    from graph_algo import export_graph as _export

    g = get_graph()
    try:
        content, media_type, filename = _export(g, fmt)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Intelligence routes ───────────────────────────────────────────────────────


@app.post("/api/query")
async def graph_query(req: QueryRequest) -> dict:
    from query_engine import run_query

    g = get_graph()
    result = await asyncio.to_thread(run_query, g, req.query, req.model)
    return {"answer": result}


@app.post("/api/simulate")
async def simulate(req: SimulateRequest) -> dict:
    from simulator import run_simulation

    g = get_graph()
    result = await asyncio.to_thread(run_simulation, g, req.scenario, req.model)
    return result


@app.post("/api/forecast")
async def forecast(req: ForecastRequest) -> dict:
    from forecaster import run_forecast

    g = get_graph()
    result = await asyncio.to_thread(run_forecast, g, req.model)
    return result


@app.get("/api/narrative/summary")
async def narrative_summary(model: str = Query(DEFAULT_MODEL)) -> dict:
    from query_engine import generate_summary

    g = get_graph()
    summary = await asyncio.to_thread(generate_summary, g, model)
    return {"summary": summary}


@app.post("/api/report")
async def generate_report(req: ReportRequest) -> StreamingResponse:
    from reporter import build_report

    g = get_graph()
    content, media_type, filename = await asyncio.to_thread(
        build_report,
        g,
        req.format,
        req.entities,
        req.include_graph,
        req.include_forecast,
    )
    return StreamingResponse(
        iter([content]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Geo ───────────────────────────────────────────────────────────────────────


@app.get("/api/geo")
async def geo_markers() -> dict:
    from geo import get_geo_markers

    g = get_graph()
    markers = get_geo_markers(g)
    return {"markers": markers}


# ── Snapshots ─────────────────────────────────────────────────────────────────


@app.get("/api/snapshots")
async def get_snapshots() -> dict:
    return {"snapshots": list_snapshots()}


@app.get("/api/snapshots/timeline")
async def snapshots_timeline() -> dict:
    snaps = list_snapshots()
    return {"timeline": snaps}


@app.get("/api/snapshots/{snapshot_id}")
async def load_snapshot_endpoint(snapshot_id: str) -> dict:
    try:
        g = load_snapshot(snapshot_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {**graph_to_visjs(g), "analytics": get_graph_analytics(g)}


@app.post("/api/snapshots")
async def create_snapshot(label: str = Query("")) -> dict:
    fname = save_snapshot(get_graph(), label)
    return {"snapshot_id": fname}


# ── OSINT ─────────────────────────────────────────────────────────────────────


>>>>>>> ce28496 (v3 initiate)
@app.get("/api/osint/status")
async def osint_status() -> dict:
    from osint_engine import get_status

    return get_status()


@app.get("/api/osint/feeds")
async def list_feeds() -> dict:
    from osint_engine import list_feeds as _lf

    return {"feeds": _lf()}


@app.post("/api/osint/feeds", status_code=201)
async def add_feed(req: OsintFeedRequest) -> dict:
    from osint_engine import add_feed as _af

    _af(req.url)
    return {"added": True, "url": req.url}


@app.delete("/api/osint/feeds")
async def remove_feed(url: str = Query(...)) -> dict:
    from osint_engine import remove_feed as _rf

    _rf(url)
    return {"removed": True, "url": url}


@app.post("/api/osint/ingest")
<<<<<<< HEAD
async def osint_ingest(req: OsintIngestRequest, background_tasks: BackgroundTasks, request: Request):
    _check_rate(request, max_req=3, window=60)
    if osint_engine._running:
        raise HTTPException(409, "OSINT ingestion already running.")
    background_tasks.add_task(
        _run_osint_ingest, model=req.model, articles_per_feed=req.articles_per_feed
    )
    return {"status": "started", "feeds": len(osint_engine.get_feeds())}


async def _run_osint_ingest(model: str, articles_per_feed: int):
    try:
        await osint_engine.ingest_all(
            graph=graph,
            update_fn=_update_graph,
            model=model,
            articles_per_feed=articles_per_feed,
        )
        save_graph(graph)
    except Exception as exc:
        logger.error("OSINT background ingest error: %s", exc, exc_info=True)


# ── Continuous OSINT Loop ──────────────────────────────────────────────────────
# State for the continuous auto-cycle
_continuous_state: Dict[str, Any] = {
    "active":       False,
    "cycle":        0,
    "started_at":   None,
    "stopped_at":   None,
    "interval_secs": 300,
    "articles_per_feed": 5,
    "model":        "llama3.2",
    "query_log":    [],   # last 50 auto-generated queries + results
    "cycle_log":    [],   # last 20 cycle summaries
    "total_entities": 0,
    "total_relations": 0,
    "total_articles":  0,
}
_continuous_task: Optional[asyncio.Task] = None
_continuous_lock = asyncio.Lock()  # asyncio-safe — used in async endpoints


class ContinuousRequest(BaseModel):
    interval_secs:    int = 300
    articles_per_feed: int = 5
    model:            str  = "llama3.2"


def _generate_auto_queries(g: nx.DiGraph, model: str, cycle: int) -> List[str]:
    """
    Derive 3-5 follow-up GQL queries automatically from the current graph state.
    Uses top-degree nodes + detected tensions to form targeted queries.
    """
    queries: List[str] = []
    if g.number_of_nodes() == 0:
        return ["find countries", "find organizations", "find persons"]

    # Always query top connectors
    deg = sorted(((n, g.degree(n)) for n in g.nodes()), key=lambda x: x[1], reverse=True)
    if deg:
        top = deg[0][0]
        queries.append(f"neighbors of {top}")
        # Use second-highest degree node for path — avoids pairing with isolated nodes
        queries.append(f"path from {deg[0][0]} to {deg[1][0]}" if len(deg) > 1 else "find countries")

    # Rotate entity-type queries by cycle number
    entity_types = ["countries", "persons", "organizations", "events", "technologies"]
    queries.append(f"find {entity_types[cycle % len(entity_types)]}")

    # Hub analysis every 3 cycles
    if cycle % 3 == 0:
        queries.append("hub nodes")
    else:
        queries.append(f"top 5 nodes by degree")

    # Tension-based query
    try:
        from geo import calculate_country_tensions
        tensions = calculate_country_tensions(g, {})
        if tensions:
            hottest = max(tensions.items(), key=lambda x: x[1])[0]
            queries.append(f"neighbors of {hottest}")
    except Exception:
        queries.append("isolated nodes")

    return queries[:5]


async def _continuous_loop():
    """Background task: ingest → auto-query → sleep → repeat."""
    state = _continuous_state
    cycle = 0

    logger.info("Continuous OSINT loop started. Interval: %ds", state["interval_secs"])

    while state["active"]:
        cycle += 1
        state["cycle"] = cycle
        cycle_start = datetime.now(timezone.utc)

        logger.info("Continuous OSINT cycle %d starting…", cycle)

        # ── Phase 1: RSS Ingest ────────────────────────────────────────────
        ingest_summary = {"entities": 0, "relations": 0, "articles": 0, "error": None}
        try:
            result = await osint_engine.ingest_all(
                graph=graph,
                update_fn=_update_graph,
                model=state["model"],
                articles_per_feed=state["articles_per_feed"],
            )
            save_graph(graph)
            ingest_summary["entities"]  = result.total_entities
            ingest_summary["relations"] = result.total_relations
            ingest_summary["articles"]  = result.articles_ingested
            state["total_entities"]  += result.total_entities
            state["total_relations"] += result.total_relations
            state["total_articles"]  += result.articles_ingested
        except Exception as exc:
            ingest_summary["error"] = str(exc)
            logger.error("Continuous loop cycle %d ingest error: %s", cycle, exc, exc_info=True)

        # ── Phase 2: Auto-generated GQL Queries ───────────────────────────
        auto_queries = _generate_auto_queries(graph, state["model"], cycle)
        query_results: List[Dict] = []
        for q in auto_queries:
            try:
                res = run_gql(q, graph)
                count = res.get("count", len(res.get("result", [])))
                query_results.append({"query": q, "type": res.get("type"), "count": count})
                logger.debug("Auto-GQL [%s]: %s → %d results", cycle, q, count)
            except Exception as exc:
                query_results.append({"query": q, "error": str(exc)})

        # ── Phase 3: Log cycle summary ────────────────────────────────────
        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        cycle_entry = {
            "cycle":     cycle,
            "timestamp": cycle_start.isoformat(),
            "elapsed_secs": round(elapsed, 1),
            "nodes":     graph.number_of_nodes(),
            "edges":     graph.number_of_edges(),
            "ingest":    ingest_summary,
            "queries":   query_results,
        }
        state["cycle_log"].insert(0, cycle_entry)
        state["cycle_log"] = state["cycle_log"][:20]

        for qr in query_results:
            state["query_log"].insert(0, {"cycle": cycle, **qr})
        state["query_log"] = state["query_log"][:50]

        logger.info(
            "Continuous OSINT cycle %d done in %.1fs — +%d entities, +%d relations, %d nodes total",
            cycle, elapsed,
            ingest_summary["entities"], ingest_summary["relations"],
            graph.number_of_nodes(),
        )

        if not state["active"]:
            break

        # ── Sleep until next cycle ─────────────────────────────────────────
        logger.info("Continuous loop sleeping %ds until next cycle…", state["interval_secs"])
        try:
            await asyncio.sleep(state["interval_secs"])
        except asyncio.CancelledError:
            break

    state["active"] = False
    state["stopped_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Continuous OSINT loop stopped after %d cycles.", cycle)


@app.post("/api/osint/continuous/start")
async def continuous_start(req: ContinuousRequest, request: Request):
    _check_rate(request, max_req=5, window=60)
    global _continuous_task

    async with _continuous_lock:
        if _continuous_state["active"]:
            raise HTTPException(409, "Continuous loop already running.")

        if len(osint_engine.get_feeds()) == 0:
            raise HTTPException(400, "No RSS feeds configured. Add at least one feed first.")

        _continuous_state.update({
            "active":            True,
            "cycle":             0,
            "started_at":        datetime.now(timezone.utc).isoformat(),
            "stopped_at":        None,
            "interval_secs":     max(60, req.interval_secs),
            "articles_per_feed": max(1, min(req.articles_per_feed, 20)),
            "model":             req.model,
            "query_log":         [],
            "cycle_log":         [],
            "total_entities":    0,
            "total_relations":   0,
            "total_articles":    0,
        })

        _continuous_task = asyncio.create_task(_continuous_loop())

    logger.info(
        "Continuous OSINT loop activated — interval=%ds, articles/feed=%d",
        req.interval_secs, req.articles_per_feed
    )
    return {
        "status":       "started",
        "interval_secs": _continuous_state["interval_secs"],
        "feeds":        len(osint_engine.get_feeds()),
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

    # Issue-20: wait for the task to actually finish (with timeout) so the graph
    # is not left in a partially-written state after the response is returned.
    if task_to_await:
        try:
            await asyncio.wait_for(asyncio.shield(task_to_await), timeout=10.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass  # expected — task was cancelled or took > 10s to wind down

    logger.info("Continuous OSINT loop stop requested.")
    return {"status": "stopping", "cycles_completed": _continuous_state["cycle"]}


@app.get("/api/osint/continuous/status")
def continuous_status():
    s = _continuous_state
    return {
        "active":          s["active"],
        "cycle":           s["cycle"],
        "started_at":      s["started_at"],
        "stopped_at":      s["stopped_at"],
        "interval_secs":   s["interval_secs"],
        "articles_per_feed": s["articles_per_feed"],
        "model":           s["model"],
        "total_entities":  s["total_entities"],
        "total_relations": s["total_relations"],
        "total_articles":  s["total_articles"],
        "graph_nodes":     graph.number_of_nodes(),
        "graph_edges":     graph.number_of_edges(),
        "cycle_log":       s["cycle_log"][:10],
        "query_log":       s["query_log"][:20],
    }


@app.post("/api/osint/enrich/{node_id:path}")
async def osint_enrich(node_id: str, request: Request, model: str = "llama3.2"):
    _check_rate(request, max_req=10, window=60)  # Issue-26: was unrate-limited
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
    days = max(1, min(days, 90))  # clamp: 0 days is nonsensical; >90 is too broad
    articles = await osint_engine.query_gdelt(entity, days)
    return {"entity": entity, "articles": articles, "count": len(articles)}
=======
async def osint_ingest(
    background_tasks: BackgroundTasks,
    model: str = Query(DEFAULT_MODEL),
) -> dict:
    from osint_engine import ingest_feeds

    async def _run() -> None:
        articles = await asyncio.to_thread(ingest_feeds)
        for text in articles:
            if text:
                req = ExtractRequest(text=text, model=model)
                try:
                    await extract(req)
                except Exception as exc:
                    log.warning("OSINT ingest extraction error: %s", exc)

    background_tasks.add_task(_run)
    return {"status": "ingestion_started"}


@app.post("/api/osint/enrich/{node_id}")
async def enrich_node(node_id: str) -> dict:
    from osint_engine import enrich_wikipedia

    g = get_graph()
    if node_id not in g:
        raise HTTPException(404, f"Node '{node_id}' not found")
    attrs = await asyncio.to_thread(enrich_wikipedia, node_id)
    if attrs:
        g.nodes[node_id].setdefault("attributes", {}).update(attrs)
        save_graph(g)
    return {"enriched": bool(attrs), "node_id": node_id, "attributes": attrs}


@app.get("/api/osint/gdelt")
async def gdelt_query(
    entity: str = Query(...),
    days: int = Query(7),
    model: str = Query(DEFAULT_MODEL),
) -> dict:
    from osint_engine import query_gdelt

    articles = await asyncio.to_thread(query_gdelt, entity, days)
    count = 0
    for text in articles:
        if text:
            req = ExtractRequest(text=text, model=model)
            try:
                await extract(req)
                count += 1
            except Exception as exc:
                log.warning("GDELT extraction error: %s", exc)
    return {"ingested": count, "entity": entity, "days": days}
>>>>>>> ce28496 (v3 initiate)


# ── Embeddings ────────────────────────────────────────────────────────────────

<<<<<<< HEAD
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
=======

@app.post("/api/embed/train")
async def train_embeddings(
    req: EmbedTrainRequest, background_tasks: BackgroundTasks
) -> dict:
    from embedding_engine import train_embeddings as _train

    background_tasks.add_task(
        asyncio.to_thread,
        _train,
        get_graph(),
        req.dimensions,
        req.walk_length,
        req.num_walks,
    )
    return {"status": "training_started"}


@app.get("/api/embed/status")
async def embed_status() -> dict:
    from embedding_engine import get_status as _gs

    return _gs()


@app.get("/api/embed/similar/{node_id}")
async def similar_nodes(node_id: str, k: int = Query(5)) -> dict:
    from embedding_engine import find_similar

    results = find_similar(node_id, k)
    return {"node_id": node_id, "similar": results}


@app.get("/api/embed/search")
async def semantic_search(q: str = Query(...), k: int = Query(5)) -> dict:
    from embedding_engine import semantic_search as _ss

    results = _ss(q, k)
    return {"query": q, "results": results}


@app.get("/api/embed/clusters")
async def graph_clusters(n: int = Query(5)) -> dict:
    from embedding_engine import cluster_graph

    clusters = cluster_graph(get_graph(), n)
    return {"clusters": clusters}


# ── GQL ───────────────────────────────────────────────────────────────────────


@app.post("/api/gql")
async def run_gql(req: GQLRequest) -> dict:
    from query_engine import run_gql as _rg

    result = await asyncio.to_thread(_rg, get_graph(), req.query)
    return {"result": result}


@app.get("/api/gql/help")
async def gql_help() -> dict:
    from query_engine import GQL_HELP

    return {"help": GQL_HELP}


# ── Watch list ────────────────────────────────────────────────────────────────

_watch_list: dict[str, float] = {}


@app.post("/api/watch_list")
async def update_watch_list(req: WatchListRequest) -> dict:
    _watch_list.update(req.thresholds)
    return {"watch_list": _watch_list}


# ── Root ──────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict:
    return {
        "name": "GOIES",
        "version": "2.0.0",
        "docs": "/api/docs",
        "status": "running",
    }
>>>>>>> ce28496 (v3 initiate)
