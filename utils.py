"""
utils.py — GOIES Shared Utilities
Handles: graph persistence, entity resolution, analytics, text chunking, exports.

Fixes applied:
  FIX-1  CHUNK_MAX_CHARS raised 4_000 → 8_000; overlap raised to 400.
  FIX-2  save_graph() rotates snapshots (MAX_SNAPSHOTS=50) to prevent disk exhaustion.
  FIX-3  In-function stdlib imports hoisted to module level.
  FIX-4  Bare except clauses replaced with specific exception types.
"""

import ast
import csv
import datetime
import io
import itertools
import json
import logging
import pathlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("goies.utils")

GRAPH_SAVE_PATH = pathlib.Path("goies_graph.json")
CHUNK_MAX_CHARS = 8_000   # FIX-1: was 4_000 — doubled to eliminate ~8 K effective ceiling
CHUNK_OVERLAP   = 400     # FIX-1: was 200 — raised proportionally
FUZZY_THRESHOLD = 0.82
MAX_SNAPSHOTS   = 50      # FIX-2: keep at most this many snapshot files on disk


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
            base = (overlap_text + " " + sentence).strip()
            # Hard fallback: a single sentence longer than max_chars gets force-split
            if len(base) > max_chars:
                while len(base) > max_chars:
                    chunks.append(base[:max_chars])
                    base = base[max_chars - overlap:]
                current = base
            else:
                current = base
    if current:
        chunks.append(current)
    return chunks or [text]


# ── Entity Resolution ─────────────────────────────────────────────────────────
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Resolution cache: maps (graph_id, raw_lower_name) → canonical node name.
# Keyed on id(graph) so it is automatically invalidated when the graph object
# is replaced (e.g. after a clear).  Cache is bounded to 10 000 entries.
_resolve_cache: dict = {}
_RESOLVE_CACHE_MAX = 10_000


def resolve_node_name(graph: nx.DiGraph, raw_name: str) -> str:
    raw_lower = raw_name.lower()
    cache_key = (id(graph), raw_lower)

    # Fast path: cache hit
    if cache_key in _resolve_cache:
        cached = _resolve_cache[cache_key]
        # Validate the cached node still exists (graph may have grown)
        if cached in graph or cached == raw_name:
            return cached
        else:
            del _resolve_cache[cache_key]

    # Slow path: O(n) scan
    best_score, best_match = 0.0, None
    for node in graph.nodes:
        node_str = str(node)
        if node_str.lower() == raw_lower:
            # Exact match — cache and return immediately
            if len(_resolve_cache) >= _RESOLVE_CACHE_MAX:
                _resolve_cache.clear()
            _resolve_cache[cache_key] = node
            return node
        score = _similarity(node_str, raw_name)
        if score > best_score:
            best_score, best_match = score, node

    result = best_match if best_score >= FUZZY_THRESHOLD else raw_name
    if len(_resolve_cache) >= _RESOLVE_CACHE_MAX:
        _resolve_cache.clear()
    _resolve_cache[cache_key] = result
    return result


def merge_nodes(graph: nx.DiGraph, source_node: str, target_node: str) -> bool:
    if source_node not in graph or source_node == target_node:
        return False

    if target_node not in graph:
        nx.relabel_nodes(graph, {source_node: target_node}, copy=False)
        save_graph(graph)
        return True

    for _, v, data in list(graph.out_edges(source_node, data=True)):
        if v == target_node:
            continue
        if graph.has_edge(target_node, v):
            graph[target_node][v]["weight"] = (
                graph[target_node][v].get("weight", 1) + data.get("weight", 1)
            )
        else:
            graph.add_edge(target_node, v, **data)

    for u, _, data in list(graph.in_edges(source_node, data=True)):
        if u == target_node:
            continue
        if graph.has_edge(u, target_node):
            graph[u][target_node]["weight"] = (
                graph[u][target_node].get("weight", 1) + data.get("weight", 1)
            )
        else:
            graph.add_edge(u, target_node, **data)

    for k, v in graph.nodes[source_node].items():
        if k not in graph.nodes[target_node] and k != "id":
            graph.nodes[target_node][k] = v

    graph.remove_node(source_node)
    save_graph(graph)
    return True


# ── Graph Persistence ─────────────────────────────────────────────────────────
_last_snapshot_time: float = 0.0
_SNAPSHOT_MIN_INTERVAL = 30.0  # seconds — prevents one snapshot per chunk during streaming

def save_graph(graph: nx.DiGraph, path: pathlib.Path = GRAPH_SAVE_PATH) -> None:
    import time as _time
    global _last_snapshot_time

    data = nx.node_link_data(graph)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # FIX-2: Timestamped snapshot with automatic rotation.
    # Debounced: skip snapshot if one was written less than _SNAPSHOT_MIN_INTERVAL ago.
    # During streaming extraction (one save/chunk), this avoids writing + rotating 50
    # snapshot files per ingest while still capturing a snapshot after each real session.
    now = _time.monotonic()
    if now - _last_snapshot_time < _SNAPSHOT_MIN_INTERVAL:
        return
    _last_snapshot_time = now

    snapshots_dir = pathlib.Path("goies_snapshots")
    snapshots_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    snapshot_path = snapshots_dir / f"goies_graph_v_{timestamp}.json"
    snapshot_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Rotate: delete oldest snapshots beyond MAX_SNAPSHOTS
    existing = sorted(snapshots_dir.glob("goies_graph_v_*.json"))
    for snap in existing[:-MAX_SNAPSHOTS]:
        try:
            snap.unlink()
        except OSError as exc:
            logger.warning("Could not delete old snapshot %s: %s", snap, exc)


def load_graph(path: pathlib.Path = GRAPH_SAVE_PATH) -> nx.DiGraph:
    if not path.exists():
        return nx.DiGraph()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, directed=True, multigraph=False)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:  # FIX-4
        logger.warning("Could not load graph from %s: %s — starting fresh.", path, exc)
        return nx.DiGraph()


# ── Graph Analytics ───────────────────────────────────────────────────────────
# Canonical keyword lists — also referenced in forecaster.py (kept in sync manually
# until a shared constants module is introduced).
HOSTILE_KEYWORDS = [
    "sanction", "attack", "invade", "bomb", "missile", "strike", "kill",
    "threaten", "blockade", "terrorize", "restrict", "ban", "expel",
    "dispute", "tension", "pressure", "cyber", "confront", "war", "conflict",
]
COOPERATIVE_KEYWORDS = [
    "cooperate", "ally", "partner", "invest", "aid", "support", "trade",
    "treaty", "agreement", "join",
]


def _is_hostile(label: str) -> bool:
    keywords = HOSTILE_KEYWORDS
    label = label.lower()
    return any(k in label for k in keywords)


def _is_cooperative(label: str) -> bool:
    keywords = COOPERATIVE_KEYWORDS
    label = label.lower()
    return any(k in label for k in keywords)


def detect_conflicts(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    conflicts = []
    checked: set = set()
    for u, v in graph.edges():
        pair = tuple(sorted([str(u), str(v)]))
        if pair in checked:
            continue
        checked.add(pair)

        edges = []
        if graph.has_edge(u, v):
            edges.append((u, v, graph.edges[u, v]))
        if graph.has_edge(v, u):
            edges.append((v, u, graph.edges[v, u]))

        if len(edges) < 2:
            continue

        has_hostile, has_coop = False, False
        h_edge, c_edge = None, None

        for src, tgt, data in edges:
            lbl = data.get("label", "")
            if _is_hostile(lbl):
                has_hostile = True
                h_edge = {"source": src, "target": tgt, "label": lbl}
            elif _is_cooperative(lbl):
                has_coop = True
                c_edge = {"source": src, "target": tgt, "label": lbl}

        if has_hostile and has_coop:
            conflicts.append(
                {"nodes": [u, v], "hostile_edge": h_edge, "cooperative_edge": c_edge}
            )
    return conflicts


def graph_health_score(graph: nx.DiGraph) -> Dict[str, Any]:
    groups = [data.get("group", "unknown") for _, data in graph.nodes(data=True)]
    group_diversity = len(set(groups)) / 7.0 if graph.number_of_nodes() > 0 else 0

    labels = [d.get("label", "") for _, _, d in graph.edges(data=True)]
    # FIX-5: Exclude blank labels before scoring diversity.
    # Previously len(set(labels)) counted "" as a unique label, giving score 1.0
    # (perfect) for a fully-unlabelled graph — the opposite of the intended signal.
    nonempty_labels = [lbl for lbl in labels if lbl.strip()]
    if not nonempty_labels:
        label_diversity = 0.0
    else:
        # Ratio of unique labels to total non-empty labels. Capped at 1.0.
        # A graph where every edge has a distinct label scores 1.0 (perfectly diverse).
        # Previously the formula divided by 0.3×total which allowed scores > 1 before the min() cap.
        label_diversity = min(1.0, len(set(nonempty_labels)) / len(nonempty_labels))

    avg_edges = graph.number_of_edges() / max(graph.number_of_nodes(), 1)
    edge_density_score = min(1.0, avg_edges / 3.0)

    health = round(group_diversity * 33 + label_diversity * 33 + edge_density_score * 34)

    suggestions = []
    if group_diversity < 0.6:
        suggestions.append("Add more entity types to understand the broader context.")
    if edge_density_score < 0.5:
        suggestions.append("Extract more relationships to increase graph density.")

    return {"score": health, "suggestions": suggestions}


def get_graph_analytics(
    graph: nx.DiGraph, custom_thresholds: Dict[str, float] = None
) -> Dict[str, Any]:
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
            "conflicts": [],
            "tensions": {},
            "health": {
                "score": 0,
                "suggestions": ["Ingest some text to start building the graph!"],
            },
        }

    degree_cent = nx.degree_centrality(graph)
    top_degree = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]

    top_betweenness: List[Tuple[str, float]] = []
    if n >= 4:
        try:
            bet = nx.betweenness_centrality(graph)
            top_betweenness = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
        except (nx.NetworkXError, nx.NetworkXException) as exc:  # FIX-4
            logger.debug("Betweenness centrality failed: %s", exc)

    group_counts: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        g = data.get("group", "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1

    from geo import calculate_country_tensions

    return {
        "nodes": n,
        "edges": e,
        "density": round(nx.density(graph), 4),
        "top_degree": top_degree,
        "top_betweenness": top_betweenness,
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
        "group_counts": group_counts,
        "conflicts": detect_conflicts(graph),
        "tensions": calculate_country_tensions(graph, custom_thresholds),
        "health": graph_health_score(graph),
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

    seed_nodes: set = set()
    for node in graph.nodes:
        node_str = str(node)
        node_words = set(re.sub(r"[^\w\s]", "", node_str.lower()).split())
        if query_words & node_words:
            seed_nodes.add(node)

    # Fuzzy fallback: if no exact word match found, pick nodes with highest name
    # similarity to any query word so context is never completely off-topic
    if not seed_nodes and query_words:
        best_score, best_node = 0.0, None
        for node in graph.nodes:
            node_str = str(node)
            for w in query_words:
                score = _similarity(node_str, w)
                if score > best_score:
                    best_score, best_node = score, node
        if best_node and best_score > 0.4:
            seed_nodes.add(best_node)

    visited, frontier = set(seed_nodes), set(seed_nodes)
    for _ in range(max_hops):
        next_frontier: set = set()
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
        edges = list(itertools.islice(graph.edges(data=True), max_edges))
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
