"""
utils.py — GOIES Shared Utilities
Handles: graph persistence, entity resolution, analytics, text chunking, exports.
"""

import csv
import io
import json
import pathlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

GRAPH_SAVE_PATH = pathlib.Path("goies_graph.json")
CHUNK_MAX_CHARS = 4_000
CHUNK_OVERLAP = 200
FUZZY_THRESHOLD = 0.82


# ── Text Chunking ─────────────────────────────────────────────────────────────
def chunk_text(
    text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            overlap_text = chunks[-1][-overlap:].strip() if chunks else ""
            current = (overlap_text + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks or [text]


# ── Entity Resolution ─────────────────────────────────────────────────────────
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def resolve_node_name(graph: nx.DiGraph, raw_name: str) -> str:
    for node in graph.nodes:
        if node.lower() == raw_name.lower():
            return node
    best_score, best_match = 0.0, None
    for node in graph.nodes:
        score = _similarity(node, raw_name)
        if score > best_score:
            best_score, best_match = score, node
    return best_match if best_score >= FUZZY_THRESHOLD else raw_name


# ── Graph Persistence ─────────────────────────────────────────────────────────
def save_graph(graph: nx.DiGraph, path: pathlib.Path = GRAPH_SAVE_PATH) -> None:
    data = nx.node_link_data(graph)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_graph(path: pathlib.Path = GRAPH_SAVE_PATH) -> nx.DiGraph:
    if not path.exists():
        return nx.DiGraph()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, directed=True, multigraph=False)
    except Exception:
        return nx.DiGraph()


# ── Graph Analytics ───────────────────────────────────────────────────────────
def get_graph_analytics(graph: nx.DiGraph) -> Dict[str, Any]:
    n, e = len(graph.nodes), len(graph.edges)
    if n == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "top_degree": [],
            "top_betweenness": [],
            "weakly_connected_components": 0,
            "group_counts": {},
        }

    degree_cent = nx.degree_centrality(graph)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]

    top_betweenness: List[Tuple[str, float]] = []
    if n >= 4:
        try:
            bet = nx.betweenness_centrality(graph)
            top_betweenness = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
        except Exception:
            pass

    group_counts: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        g = data.get("group", "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1

    return {
        "nodes": n,
        "edges": e,
        "density": round(nx.density(graph), 4),
        "top_degree": top_degree,
        "top_betweenness": top_betweenness,
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
        "group_counts": group_counts,
    }


# ── Subgraph ──────────────────────────────────────────────────────────────────
def get_ego_subgraph(graph: nx.DiGraph, node: str, hops: int = 2) -> nx.DiGraph:
    if node not in graph:
        return graph
    undirected = graph.to_undirected()
    reachable = nx.single_source_shortest_path_length(undirected, node, cutoff=hops)
    return graph.subgraph(set(reachable.keys())).copy()


# ── Multi-hop Context Retrieval ────────────────────────────────────────────────
def retrieve_graph_context(
    query: str, graph: nx.DiGraph, max_hops: int = 2, max_edges: int = 20
) -> str:
    if len(graph.nodes) == 0:
        return "The graph is currently empty."

    query_words = set(
        w for w in re.sub(r"[^\w\s]", "", query.lower()).split() if len(w) > 2
    )

    seed_nodes = set()
    for node in graph.nodes:
        node_words = set(re.sub(r"[^\w\s]", "", node.lower()).split())
        if query_words & node_words:
            seed_nodes.add(node)

    visited, frontier = set(seed_nodes), set(seed_nodes)
    for _ in range(max_hops):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
            next_frontier.update(graph.successors(node))
        next_frontier -= visited
        visited.update(next_frontier)
        frontier = next_frontier

    relevant: List[str] = []
    for u, v, data in graph.edges(data=True):
        if u in visited or v in visited:
            rel = data.get("label", "is connected to")
            conf = data.get("confidence", None)
            conf_str = f" [confidence: {conf:.2f}]" if conf is not None else ""
            relevant.append(f"- {u} → {rel} → {v}{conf_str}")

    if not relevant:
        edges = list(graph.edges(data=True))[:max_edges]
        return "\n".join(
            f"- {u} → {d.get('label', 'connects to')} → {v}" for u, v, d in edges
        )

    return "\n".join(relevant[:max_edges])


# ── Export Helpers ────────────────────────────────────────────────────────────
def export_json(graph: nx.DiGraph) -> str:
    return json.dumps(nx.node_link_data(graph), indent=2)


def export_graphml(graph: nx.DiGraph) -> bytes:
    buf = io.BytesIO()
    nx.write_graphml(graph, buf)
    return buf.getvalue()


def export_csv(graph: nx.DiGraph) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["source", "target", "label", "confidence"])
    for u, v, data in graph.edges(data=True):
        writer.writerow([u, v, data.get("label", ""), data.get("confidence", "")])
    return buf.getvalue()
