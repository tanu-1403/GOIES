"""
server.py — GOIES FastAPI Backend v3
"""

from embedding_engine import GraphEmbeddingEngine
from osint_engine import ingest_rss_feed
import asyncio
from __future__ import annotations
import json
import asyncio
from datetime import datetime
import pathlib
import io
import uvicorn
from pydantic import BaseModel
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    BackgroundTasks,
    UploadFile,
    File,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import networkx as nx
from typing import Dict, List, Optional, Any
import ast

# internal imports
from utils import (
    get_graph_analytics,
    load_graph,
    resolve_node_name,
    retrieve_graph_context,
    save_graph,
    export_json,
    export_csv,
    export_graphml,
    get_ego_subgraph,
)

from extractor import (
    extract_intelligence,
    extract_intelligence_stream,
    list_available_models,
    check_ollama_health,
)
from query_engine import GQLParser, GQLExecutor
from fastapi.staticfiles import StaticFiles
from geo import get_geo_data
from simulator import run_simulation
from forecaster import run_forecast

app = FastAPI(title="GOIES", version="3.0.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

graph: nx.DiGraph = load_graph()
MAX_INPUT_CHARS = 8000


GROUP_COLORS = {
    "country": "#ff7b72",
    "person": "#ffa657",
    "organization": "#d2a8ff",
    "technology": "#79c0ff",
    "event": "#7ee787",
    "treaty": "#f0e68c",
    "resource": "#56d364",
    "unknown": "#8b949e",
}

embedding_engine = GraphEmbeddingEngine()


def _fmt_tooltip(group, attributes, confidence):
    color = GROUP_COLORS.get(group, "#8b949e")
    lines = [f'<b style="color:{color}">{group.upper()}</b>']
    for k, v in attributes.items():
        lines.append(f"{k}: {v}")
    lines.append(f"confidence: {confidence:.2f}")
    return "<br>".join(lines)


def graph_to_vis(g: nx.DiGraph):
    nodes = []

    for node_id, data in g.nodes(data=True):
        attrs = data.get("attributes", {})

        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "group": data.get("group", "unknown"),
                "title": _fmt_tooltip(
                    data.get("group", "unknown"), attrs, data.get("confidence", 1.0)
                ),
            }
        )

    edges = []

    for u, v, data in g.edges(data=True):
        edges.append(
            {"from": u, "to": v, "label": data.get("label", ""), "arrows": "to"}
        )

    return {"nodes": nodes, "edges": edges}


def _update_graph(extractions):
    nodes_added = 0
    edges_added = 0
    new_ids = []

    for ext in extractions:
        cls = ext.extraction_class.lower()

        if cls != "relationship":
            canonical = resolve_node_name(graph, ext.extraction_text)

            if not graph.has_node(canonical):
                nodes_added += 1
                new_ids.append(canonical)

            graph.add_node(
                canonical,
                attributes=ext.attributes,
                group=cls,
                confidence=ext.confidence,
            )

        else:
            src = resolve_node_name(graph, ext.attributes.get("source", ""))
            tgt = resolve_node_name(graph, ext.attributes.get("target", ""))

            if not src or not tgt:
                continue

            for n in (src, tgt):
                if not graph.has_node(n):
                    graph.add_node(n, group="unknown")

            if graph.has_edge(src, tgt):
                graph[src][tgt]["confidence"] = max(
                    graph[src][tgt].get("confidence", 0), ext.confidence
                )

            else:
                graph.add_edge(
                    src, tgt, label=ext.extraction_text, confidence=ext.confidence
                )

                edges_added += 1

    save_graph(graph)

    return {
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "new_node_ids": new_ids,
    }


@app.get("/api/similar")
def similar(node: str, top_k: int = 5):
    if node not in graph:
        raise HTTPException(404, "Node not found in graph")

    results = embedding_engine.similar_nodes(node, top_k)

    return {
        "node": node,
        "similar": [{"node": n, "score": round(score, 4)} for n, score in results],
    }


class ExtractRequest(BaseModel):
    text: str
    model: str = "llama3.2"


@app.get("/api/health")
def health():
    return check_ollama_health()


@app.get("/api/models")
def models():
    return {"models": list_available_models()}


@app.post("/api/extract")
def extract(req: ExtractRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    if len(req.text) > MAX_INPUT_CHARS:
        raise HTTPException(400, "Input too large")

    extractions = extract_intelligence(req.text, req.model)

    diff = _update_graph(extractions)

    return {
        "extractions": len(extractions),
        **diff,
        "vis": graph_to_vis(graph),
        "analytics": get_graph_analytics(graph),
    }


@app.get("/api/graph")
def get_graph():
    return {"vis": graph_to_vis(graph), "analytics": get_graph_analytics(graph)}


@app.delete("/api/graph")
def clear_graph():
    graph.clear()
    save_graph(graph)

    return {"status": "cleared"}


@app.get("/api/export/{fmt}")
def export(fmt: str):
    if fmt == "json":
        return Response(export_json(graph), media_type="application/json")

    if fmt == "csv":
        return Response(export_csv(graph), media_type="text/csv")

    if fmt == "graphml":
        return Response(export_graphml(graph), media_type="application/xml")

    raise HTTPException(400, "unknown format")


@app.get("/api/geo")
def geo():
    return {"markers": get_geo_data(graph)}


_static = pathlib.Path(__file__).parent / "static"
_static.mkdir(exist_ok=True)

app.mount("/", StaticFiles(directory=str(_static), html=True), name="static")


@app.post("/api/query/gql")
def gql_query(req: dict):
    query = req.get("query", "")

    if not query:
        raise HTTPException(400, "query required")

    parser = GQLParser()

    parsed = parser.parse(query)

    executor = GQLExecutor(graph)

    result = executor.execute(parsed)

    return {"query": query, "parsed": parsed, "result": result}


async def osint_loop():
    while True:
        try:
            await ingest_rss_feed(graph)

        except Exception as e:
            print("OSINT loop error:", e)

        await asyncio.sleep(600)  # every 10 minutes


@app.on_event("startup")
async def start_osint():
    asyncio.create_task(osint_loop())


@app.post("/api/osint/refresh")
async def osint_refresh():
    await ingest_rss_feed(graph)

    return {"status": "completed"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
