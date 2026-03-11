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
"""

from __future__ import annotations

import ast
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

# ── Rate Limiter (no external dependency) ─────────────────────────────────────
# FIX-12: Simple token-bucket rate limiter — no slowapi/Redis dependency needed.
# Per-IP sliding-window counters stored in memory (fine for single-process local deploy).

class _RateLimiter:
    def __init__(self):
        self._windows: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def is_allowed(self, key: str, max_requests: int, window_secs: float) -> bool:
        now = time.monotonic()
        with self._lock:
            timestamps = self._windows[key]
            # Drop expired entries
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


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="GOIES", version="4.1.0", docs_url="/api/docs")

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


# FIX-11: Startup Ollama health check
@app.on_event("startup")
async def startup_checks():
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

        save_graph(graph)

    return {"nodes_added": nodes_added, "edges_added": edges_added, "new_node_ids": new_ids}


# ── Request Models ─────────────────────────────────────────────────────────────
class ExtractRequest(BaseModel):
    text: str
    model: str = "llama3.2"
    persona: str = "senior geopolitical intelligence analyst"

class QueryRequest(BaseModel):
    question: str
    model: str = "llama3.2"
    persona: str = "senior geopolitical intelligence analyst"

class SimulateRequest(BaseModel):
    scenario: str
    model: str = "llama3.2"
    persona: str = "strategic policy simulator"

class ForecastRequest(BaseModel):
    model: str = "llama3.2"
    focus: str = ""

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

class GQLRequest(BaseModel):
    query: str

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
@app.post("/api/ingest/url")
def ingest_url(req: UrlIngestRequest, request: Request):
    _check_rate(request, max_req=30, window=60)
    try:
        from ingestor import fetch_url_text
        text = fetch_url_text(req.url)
        return {"text": text, "chars": len(text)}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/extract/url")
def extract_url_stream(req: ExtractUrlRequest, request: Request):
    _check_rate(request, max_req=10, window=60)

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
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
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

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = nx.node_link_graph(data, directed=True, multigraph=False)
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
    import requests as http

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
    g = get_ego_subgraph(graph, ego, hops) if ego and ego in graph else graph
    return {
        "vis":       graph_to_vis(g),
        "analytics": get_graph_analytics(graph, watch_list_thresholds),
        "filtered":  ego is not None and ego in graph,
    }


@app.get("/api/narrative/summary")
def graph_summary(request: Request, model: str = "llama3.2"):
    _check_rate(request, max_req=5, window=60)
    import itertools
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
def path(src: str, tgt: str):
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
def merge_node_ep(req: MergeRequest):
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
        try:
            for chunk_data in extract_intelligence_stream(req.text, model=req.model, persona=req.persona):
                extractions = chunk_data["extractions"]
                diff      = _update_graph(extractions)
                entities  = sum(1 for e in extractions if e.extraction_class.lower() != "relationship")
                relations = len(extractions) - entities
                total_entities  += entities
                total_relations += relations
                new_nodes_all.extend(diff["new_node_ids"])
                event_payload = {
                    "chunk":        chunk_data["chunk_index"],
                    "total_chunks": chunk_data["total_chunks"],
                    "extractions":  len(extractions),
                    "entities":     entities,
                    "relations":    relations,
                    "new_node_ids": diff["new_node_ids"],
                    "vis":          graph_to_vis(graph),
                    "analytics":    get_graph_analytics(graph, watch_list_thresholds),
                }
                if chunk_data.get("parse_error"):
                    event_payload["parse_error"] = chunk_data["parse_error"]
                yield f"data: {json.dumps(event_payload)}\n\n"

            yield f"data: {json.dumps({'done': True, 'totals': {'entities': total_entities, 'relations': total_relations, 'new_nodes': list(set(new_nodes_all))}})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/query")
def query(req: QueryRequest, request: Request):
    _check_rate(request, max_req=20, window=60)
    import requests as http
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
        save_graph(graph)
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
def simulate(req: SimulateRequest, request: Request):
    _check_rate(request, max_req=5, window=60)
    if not req.scenario.strip():
        raise HTTPException(400, "Scenario cannot be empty.")
    if len(graph.nodes) == 0:
        raise HTTPException(400, "Graph is empty. Ingest data first.")
    try:
        result = run_simulation(req.scenario, graph, model=req.model)
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
    clusters = embedding_engine.cluster_nodes(n_clusters=n)
    return {"clusters": clusters, "k": n}


# ── OSINT ──────────────────────────────────────────────────────────────────────
@app.get("/api/osint/status")
def osint_status():
    return osint_engine.get_status()


@app.get("/api/osint/feeds")
def osint_get_feeds():
    return {"feeds": osint_engine.get_feeds()}


@app.post("/api/osint/feeds")
def osint_add_feed(req: FeedRequest):
    added = osint_engine.add_feed(req.url, req.name)
    if not added:
        raise HTTPException(409, f"Feed already exists: {req.url}")
    return {"status": "added", "url": req.url, "name": req.name}


@app.delete("/api/osint/feeds")
def osint_remove_feed(url: str):
    removed = osint_engine.remove_feed(url)
    if not removed:
        raise HTTPException(404, f"Feed not found: {url}")
    return {"status": "removed", "url": url}


@app.post("/api/osint/ingest")
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


@app.post("/api/osint/enrich/{node_id:path}")
async def osint_enrich(node_id: str, model: str = "llama3.2"):
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
    articles = await osint_engine.query_gdelt(entity, days)
    return {"entity": entity, "articles": articles, "count": len(articles)}


# ── Static SPA ─────────────────────────────────────────────────────────────────
_static = pathlib.Path(__file__).parent / "frontend"
_static.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
