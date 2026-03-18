"""
utils.py  —  GOIES Shared Utilities
=====================================
Merge resolution: HEAD (v4.x production) + ce28496 (async/persistence rewrite)

HEAD production features preserved:
  FIX-1  CHUNK_MAX_CHARS 4_000 → 8_000; overlap → 400; hard sentence-split fallback
  FIX-2  save_graph() writes debounced snapshots, rotates to MAX_SNAPSHOTS=50
  FIX-3  All stdlib imports at module level
  FIX-4  Bare except → specific exception types
  FIX-5  label_diversity excludes blank labels; score formula fixed (no >1.0 overshoot)
         resolve_node_name LRU-style cache (10 k entries, auto-evict)
         detect_conflicts, graph_health_score, retrieve_graph_context, chunk_text,
         export_json / export_csv / export_graphml, get_ego_subgraph

ce28496 improvements layered in:
  ASYNC-4  clear_graph() wipes _G and atomically persists empty state
  ASYNC-9  _safe_save(): tmp → fsync → os.replace() (atomic); corrupt-file auto-recovery
           add_node() / add_edge() helpers
           graph_to_visjs() serialiser
           get_graph() global accessor
"""

from __future__ import annotations

import ast
import csv
import datetime
import io
import itertools
import json
import logging
import os
import pathlib
import re
import shutil
import tempfile
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("goies.utils")

# ── Configuration ─────────────────────────────────────────────────────────────

GRAPH_SAVE_PATH = pathlib.Path(os.getenv("GOIES_GRAPH_PATH", "goies_graph.json"))
SNAPSHOT_DIR = pathlib.Path(os.getenv("GOIES_SNAPSHOT_DIR", "goies_snapshots"))
CHUNK_MAX_CHARS = 8_000  # FIX-1: was 4_000
CHUNK_OVERLAP = 400  # FIX-1: was 200
FUZZY_THRESHOLD = float(os.getenv("FUZZY_THRESH", "0.82"))
MAX_SNAPSHOTS = 50  # FIX-2: rotate snapshots beyond this count

# ── In-memory global graph (ce28496) ─────────────────────────────────────────

_G: nx.DiGraph = nx.DiGraph()


def get_graph() -> nx.DiGraph:
    """Return the module-level graph instance."""
    return _G


# ── Text chunking (HEAD FIX-1) ────────────────────────────────────────────────


def chunk_text(
    text: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Sentence-boundary-aware chunking with overlap and hard-split fallback."""
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
            # Hard fallback: single sentence longer than max_chars gets force-split
            if len(base) > max_chars:
                while len(base) > max_chars:
                    chunks.append(base[:max_chars])
                    base = base[max_chars - overlap :]
                current = base
            else:
                current = base
    if current:
        chunks.append(current)
    return chunks or [text]


# ── Similarity helper ─────────────────────────────────────────────────────────


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ── Node name resolution (HEAD — cached) ──────────────────────────────────────

_resolve_cache: dict = {}
_RESOLVE_CACHE_MAX = 10_000


def resolve_node_name(graph: nx.DiGraph, raw_name: str) -> str:
    """
    Return the canonical node ID that best fuzzy-matches *raw_name*.
    Falls back to *raw_name* itself if no node scores >= FUZZY_THRESHOLD.
    Results are cached per (graph identity, raw_lower) for O(1) repeat lookups.
    """
    raw_lower = raw_name.lower()
    cache_key = (id(graph), raw_lower)

    if cache_key in _resolve_cache:
        cached = _resolve_cache[cache_key]
        if cached in graph or cached == raw_name:
            return cached
        del _resolve_cache[cache_key]

    best_score, best_match = 0.0, None
    for node in graph.nodes:
        node_str = str(node)
        if node_str.lower() == raw_lower:
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


# ── Node / edge helpers (ce28496) ─────────────────────────────────────────────


def add_node(
    graph: nx.DiGraph,
    node_id: str,
    group: str = "unknown",
    confidence: float = 1.0,
    attributes: Optional[dict] = None,
    source_count: int = 1,
) -> bool:
    """Add or update a node. Returns True if newly created."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if node_id in graph:
        old = graph.nodes[node_id]
        graph.nodes[node_id]["source_count"] = old.get("source_count", 1) + 1
        graph.nodes[node_id]["confidence"] = max(old.get("confidence", 0), confidence)
        if attributes:
            existing = old.get("attributes", {})
            existing.update(attributes)
            graph.nodes[node_id]["attributes"] = existing
        return False
    graph.add_node(
        node_id,
        group=group,
        confidence=confidence,
        attributes=attributes or {},
        ingested_at=ts,
        source_count=source_count,
        tension_score=0.0,
    )
    return True


def add_edge(
    graph: nx.DiGraph,
    from_id: str,
    to_id: str,
    label: str,
    confidence: float = 1.0,
) -> bool:
    """Add a directed edge. Returns True if newly created."""
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if graph.has_edge(from_id, to_id):
        existing = graph[from_id][to_id]
        if existing.get("label") == label:
            existing["confidence"] = max(existing.get("confidence", 0), confidence)
            return False
    graph.add_edge(from_id, to_id, label=label, confidence=confidence, ingested_at=ts)
    return True


# ── Node merge (HEAD logic, ce28496 add_edge helper) ──────────────────────────


def merge_nodes(graph: nx.DiGraph, source_node: str, target_node: str) -> bool:
    """
    Merge *source_node* into *target_node*.
    If target does not yet exist, rename source → target via relabel.
    Otherwise re-point all edges and delete source.
    """
    if source_node not in graph:
        return False

    if target_node not in graph:
        nx.relabel_nodes(graph, {source_node: target_node}, copy=False)
        save_graph(graph)
        return True

    for _, v, data in list(graph.out_edges(source_node, data=True)):
        if v == target_node:
            continue
        if graph.has_edge(target_node, v):
            graph[target_node][v]["weight"] = graph[target_node][v].get(
                "weight", 1
            ) + data.get("weight", 1)
        else:
            graph.add_edge(target_node, v, **data)

    for u, _, data in list(graph.in_edges(source_node, data=True)):
        if u == target_node:
            continue
        if graph.has_edge(u, target_node):
            graph[u][target_node]["weight"] = graph[u][target_node].get(
                "weight", 1
            ) + data.get("weight", 1)
        else:
            graph.add_edge(u, target_node, **data)

    for k, v in graph.nodes[source_node].items():
        if k not in graph.nodes[target_node] and k != "id":
            graph.nodes[target_node][k] = v

    graph.remove_node(source_node)
    save_graph(graph)
    return True


# ── Conflict detection (HEAD) ─────────────────────────────────────────────────

HOSTILE_KEYWORDS = [
    "sanction",
    "attack",
    "invade",
    "bomb",
    "missile",
    "strike",
    "kill",
    "threaten",
    "blockade",
    "terrorize",
    "restrict",
    "ban",
    "expel",
    "dispute",
    "tension",
    "pressure",
    "cyber",
    "confront",
    "war",
    "conflict",
]
COOPERATIVE_KEYWORDS = [
    "cooperate",
    "ally",
    "partner",
    "invest",
    "aid",
    "support",
    "trade",
    "treaty",
    "agreement",
    "join",
]


def _is_hostile(label: str) -> bool:
    label = label.lower()
    return any(k in label for k in HOSTILE_KEYWORDS)


def _is_cooperative(label: str) -> bool:
    label = label.lower()
    return any(k in label for k in COOPERATIVE_KEYWORDS)


def detect_conflicts(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """Find node pairs with both a hostile and a cooperative edge (contradiction)."""
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
                {
                    "nodes": [u, v],
                    "hostile_edge": h_edge,
                    "cooperative_edge": c_edge,
                }
            )
    return conflicts


# ── Graph health score (HEAD FIX-5) ───────────────────────────────────────────


def graph_health_score(graph: nx.DiGraph) -> Dict[str, Any]:
    groups = [data.get("group", "unknown") for _, data in graph.nodes(data=True)]
    group_diversity = len(set(groups)) / 7.0 if graph.number_of_nodes() > 0 else 0.0

    labels = [d.get("label", "") for _, _, d in graph.edges(data=True)]
    # FIX-5: exclude blank labels before scoring diversity
    nonempty_labels = [lbl for lbl in labels if lbl.strip()]
    if not nonempty_labels:
        label_diversity = 0.0
    else:
        label_diversity = min(1.0, len(set(nonempty_labels)) / len(nonempty_labels))

    avg_edges = graph.number_of_edges() / max(graph.number_of_nodes(), 1)
    edge_density_score = min(1.0, avg_edges / 3.0)

    health = round(
        group_diversity * 33 + label_diversity * 33 + edge_density_score * 34
    )

    suggestions = []
    if group_diversity < 0.6:
        suggestions.append("Add more entity types to understand the broader context.")
    if edge_density_score < 0.5:
        suggestions.append("Extract more relationships to increase graph density.")

    return {"score": health, "suggestions": suggestions}


# ── Graph analytics ───────────────────────────────────────────────────────────


def get_graph_analytics(
    graph: nx.DiGraph,
    watch_thresholds: Optional[Dict[str, float]] = None,
) -> dict:
    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    if n == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "components": 0,
            "group_counts": {},
            "top_degree": [],
            "top_betweenness": [],
            "tensions": {},
            "health": {"score": 0, "suggestions": []},
            "conflicts": [],
        }

    degree_c = nx.degree_centrality(graph)

    top_betweenness: List[Tuple[str, float]] = []
    if n >= 4:
        try:
            bet = nx.betweenness_centrality(graph)
            top_betweenness = sorted(bet.items(), key=lambda x: x[1], reverse=True)[:5]
        except (nx.NetworkXError, nx.NetworkXException) as exc:
            logger.debug("Betweenness centrality failed: %s", exc)

    try:
        components = nx.number_weakly_connected_components(graph)
    except Exception:
        components = 0
    density = nx.density(graph)

    group_counts: Dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        g = data.get("group", "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1

    top_degree = sorted(degree_c.items(), key=lambda x: x[1], reverse=True)[:10]

    # Tension scores (watch-list alerts)
    tensions: Dict[str, float] = {}
    if watch_thresholds:
        for node, threshold in watch_thresholds.items():
            if node in graph:
                score = graph.nodes[node].get("tension_score", 0.0)
                if score >= threshold:
                    tensions[node] = score

    return {
        "nodes": n,
        "edges": e,
        "density": round(density, 4),
        "components": components,
        "group_counts": group_counts,
        "top_degree": top_degree,
        "top_betweenness": top_betweenness,
        "tensions": tensions,
        "health": graph_health_score(graph),
        "conflicts": detect_conflicts(graph),
    }


# ── vis.js serialisation (ce28496) ───────────────────────────────────────────


def graph_to_visjs(graph: nx.DiGraph) -> dict:
    nodes = []
    for node_id, data in graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "label": node_id,
                "group": data.get("group", "unknown"),
                "confidence": data.get("confidence", 1.0),
                "attributes": data.get("attributes", {}),
                "ingested_at": data.get("ingested_at", ""),
                "source_count": data.get("source_count", 1),
                "tension_score": data.get("tension_score", 0.0),
            }
        )
    edges = []
    for i, (src, tgt, data) in enumerate(graph.edges(data=True)):
        edges.append(
            {
                "id": i,
                "from": src,
                "to": tgt,
                "label": data.get("label", ""),
                "confidence": data.get("confidence", 1.0),
                "ingested_at": data.get("ingested_at", ""),
            }
        )
    return {"nodes": nodes, "edges": edges}


# ── Ego subgraph (HEAD) ───────────────────────────────────────────────────────


def get_ego_subgraph(
    graph: nx.DiGraph, center: Optional[str], hops: int = 2
) -> nx.DiGraph:
    if not center or center not in graph:
        return graph
    nodes = {center}
    frontier = {center}
    for _ in range(hops):
        next_frontier: set = set()
        for node in frontier:
            next_frontier.update(graph.predecessors(node))
            next_frontier.update(graph.successors(node))
        next_frontier -= nodes
        nodes.update(next_frontier)
        frontier = next_frontier
    return graph.subgraph(nodes).copy()


# ── Multi-hop context retrieval (HEAD) ────────────────────────────────────────


def retrieve_graph_context(
    query: str,
    graph: nx.DiGraph,
    max_hops: int = 2,
    max_edges: int = 20,
) -> str:
    if len(graph.nodes) == 0:
        return "The graph is currently empty."

    query_words = {
        w for w in re.sub(r"[^\w\s]", "", query.lower()).split() if len(w) > 2
    }

    seed_nodes: set = set()
    for node in graph.nodes:
        node_words = set(re.sub(r"[^\w\s]", "", str(node).lower()).split())
        if query_words & node_words:
            seed_nodes.add(node)

    # Fuzzy fallback when no exact word match
    if not seed_nodes and query_words:
        best_score, best_node = 0.0, None
        for node in graph.nodes:
            for w in query_words:
                score = _similarity(str(node), w)
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


# ── Atomic persistence (ce28496 ASYNC-9) ──────────────────────────────────────

_last_snapshot_time: float = 0.0
_SNAPSHOT_MIN_INTERVAL = 30.0  # seconds — debounce during streaming


def _safe_save(graph: nx.DiGraph, path: Any) -> None:
    """
    Atomic write: serialize → tmp → fsync → os.replace().
    Prevents corrupt state from partial writes or server crashes.
    Accepts both str and pathlib.Path.
    """
    path = str(path)
    data = {
        "version": 2,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "nodes": [{"id": n, **d} for n, d in graph.nodes(data=True)],
        "edges": [
            {"source": s, "target": t, **d} for s, t, d in graph.edges(data=True)
        ],
    }
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dir_name,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)
        logger.debug(
            "Graph persisted → %s (%d nodes, %d edges)",
            path,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
    except Exception as exc:
        logger.error("Failed to persist graph to %s: %s", path, exc)
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def save_graph(
    graph: Optional[nx.DiGraph] = None,
    path: Any = None,
) -> None:
    """
    Persist *graph* to *path*.
    Defaults: graph → module-level _G, path → GRAPH_SAVE_PATH.
    Also writes a debounced timestamped snapshot and rotates old ones (HEAD FIX-2).
    """
    global _last_snapshot_time

    g = graph if graph is not None else _G
    dest = pathlib.Path(path) if path is not None else GRAPH_SAVE_PATH

    _safe_save(g, dest)

    # FIX-2: debounced snapshot with rotation
    now = time.monotonic()
    if now - _last_snapshot_time < _SNAPSHOT_MIN_INTERVAL:
        return
    _last_snapshot_time = now

    SNAPSHOT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H%M%SZ"
    )
    snapshot_path = SNAPSHOT_DIR / f"goies_graph_v_{timestamp}.json"
    try:
        _safe_save(g, snapshot_path)
    except Exception as exc:
        logger.warning("Snapshot write failed: %s", exc)
        return

    existing = sorted(SNAPSHOT_DIR.glob("goies_graph_v_*.json"))
    for snap in existing[:-MAX_SNAPSHOTS]:
        try:
            snap.unlink()
        except OSError as exc:
            logger.warning("Could not delete old snapshot %s: %s", snap, exc)


def load_graph(path: Any = None) -> nx.DiGraph:
    """
    Load the graph from *path* (default: GRAPH_SAVE_PATH).
    Falls back to an empty graph on missing or corrupt file — never crashes (ASYNC-9).
    Also updates the module-level _G.
    """
    global _G
    dest = pathlib.Path(path) if path is not None else GRAPH_SAVE_PATH

    if not dest.exists():
        logger.info("No persisted graph at %s — starting fresh.", dest)
        _G = nx.DiGraph()
        return _G

    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error(
            "Corrupt graph file %s (%s) — backing up and starting fresh.", dest, exc
        )
        backup = str(dest) + f".corrupt.{int(time.time())}"
        try:
            shutil.copy2(dest, backup)
            logger.info("Corrupt file backed up → %s", backup)
        except Exception:
            pass
        _G = nx.DiGraph()
        return _G

    # Support both the new v2 format (_safe_save) and the legacy node_link format
    if "nodes" in data and "edges" in data and "version" in data:
        g = nx.DiGraph()
        for node in data.get("nodes", []):
            node = dict(node)
            node_id = node.pop("id", None)
            if not node_id:
                continue
            g.add_node(node_id, **node)
        for edge in data.get("edges", []):
            edge = dict(edge)
            src = edge.pop("source", None)
            tgt = edge.pop("target", None)
            if src and tgt:
                g.add_edge(src, tgt, **edge)
    else:
        # Legacy node_link_data format (HEAD v4.x)
        try:
            g = nx.node_link_graph(data, directed=True, multigraph=False)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Could not parse legacy graph format (%s) — starting fresh.", exc
            )
            _G = nx.DiGraph()
            return _G

    logger.info(
        "Graph loaded from %s: %d nodes, %d edges",
        dest,
        g.number_of_nodes(),
        g.number_of_edges(),
    )
    _G = g
    return _G


# ── Graph clear (ASYNC-4) ─────────────────────────────────────────────────────


def clear_graph() -> None:
    """
    Reset the in-memory graph to empty AND atomically persist the empty state.
    Fixes the bug where DELETE /api/graph only reset the frontend canvas.
    """
    global _G
    _G = nx.DiGraph()
    _safe_save(_G, GRAPH_SAVE_PATH)
    logger.info("Graph cleared — backend state reset and persisted.")


# ── Exports (HEAD) ────────────────────────────────────────────────────────────


def export_json(graph: nx.DiGraph) -> str:
    return json.dumps(nx.node_link_data(graph), indent=2)


def export_csv(graph: nx.DiGraph) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["source", "target", "label", "confidence"])
    for u, v, data in graph.edges(data=True):
        writer.writerow([u, v, data.get("label", ""), data.get("confidence", "")])
    return output.getvalue()


def export_graphml(graph: nx.DiGraph) -> bytes:
    buf = io.BytesIO()
    nx.write_graphml(graph, buf)
    return buf.getvalue()
