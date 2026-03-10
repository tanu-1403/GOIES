"""
graph_algo.py — GOIES Graph Algorithm Engine
Shortest path, all-paths enumeration, narrative path export.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx


def find_shortest_path(graph: nx.DiGraph, source: str, target: str) -> Dict[str, Any]:
    """
    Find the shortest directed path from source to target.
    Falls back to undirected search if no directed path exists.
    Returns {"nodes": [...], "edges": [...], "length": int, "directed": bool}
    """
    if source not in graph or target not in graph:
        return {
            "nodes": [],
            "edges": [],
            "length": 0,
            "directed": False,
            "found": False,
        }

    # Try directed first
    try:
        path_nodes = nx.shortest_path(graph, source=source, target=target)
        directed = True
    except nx.NetworkXNoPath:
        # Fall back to undirected
        try:
            undirected = graph.to_undirected()
            path_nodes = nx.shortest_path(undirected, source=source, target=target)
            directed = False
        except nx.NetworkXNoPath:
            return {
                "nodes": [],
                "edges": [],
                "length": 0,
                "directed": False,
                "found": False,
            }
    except nx.NodeNotFound:
        return {
            "nodes": [],
            "edges": [],
            "length": 0,
            "directed": False,
            "found": False,
        }

    # Build edge list with labels
    edges: List[Dict[str, str]] = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        if graph.has_edge(u, v):
            label = graph[u][v].get("label", "connects to")
        elif graph.has_edge(v, u):
            label = graph[v][u].get("label", "connects to")
            u, v = v, u  # flip for display
        else:
            label = "connects to"
        edges.append({"from": u, "to": v, "label": label})

    return {
        "found": True,
        "nodes": path_nodes,
        "edges": edges,
        "length": len(path_nodes) - 1,
        "directed": directed,
    }


def find_all_paths(
    graph: nx.DiGraph,
    source: str,
    target: str,
    max_length: int = 4,
    max_paths: int = 10,
) -> List[Dict[str, Any]]:
    """
    Find all simple paths up to max_length hops. Returns list of path dicts.
    """
    if source not in graph or target not in graph:
        return []

    try:
        raw_paths = list(
            nx.all_simple_paths(graph, source=source, target=target, cutoff=max_length)
        )
    except Exception:
        return []

    results: List[Dict[str, Any]] = []
    for path_nodes in raw_paths[:max_paths]:
        edges: List[Dict[str, str]] = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            label = (
                graph[u][v].get("label", "connects to")
                if graph.has_edge(u, v)
                else "connects to"
            )
            edges.append({"from": u, "to": v, "label": label})
        results.append(
            {
                "nodes": path_nodes,
                "edges": edges,
                "length": len(path_nodes) - 1,
            }
        )

    # Sort by length ascending
    results.sort(key=lambda x: x["length"])
    return results


def path_to_narrative(path: Dict[str, Any]) -> str:
    """
    Convert a path dict to a readable intelligence narrative.
    e.g. "US → [funds] → Israel → [threatens] → Iran"
    """
    if not path.get("found") and not path.get("nodes"):
        return "No connection found."

    nodes = path["nodes"]
    edges = path["edges"]

    if len(nodes) == 1:
        return f"{nodes[0]} is a standalone actor with no connection to the target."

    parts = [nodes[0]]
    for edge in edges:
        parts.append(f"[{edge['label']}]")
        parts.append(edge["to"])

    chain = " → ".join(parts)
    hops = path.get("length", len(nodes) - 1)
    direction = "directed" if path.get("directed", True) else "undirected"
    return f"{chain}  ({hops} hop{'s' if hops != 1 else ''}, {direction})"


def node_influence_rank(graph: nx.DiGraph, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Combined degree + betweenness centrality influence ranking.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return []

    degree_cent = nx.degree_centrality(graph)

    bet_cent: Dict[str, float] = {}
    if n >= 4:
        try:
            bet_cent = nx.betweenness_centrality(graph)
        except Exception:
            pass

    results: List[Dict[str, Any]] = []
    for node in graph.nodes:
        dc = degree_cent.get(node, 0.0)
        bc = bet_cent.get(node, 0.0)
        influence = round(dc * 0.5 + bc * 0.5, 4)
        results.append(
            {
                "node": node,
                "degree_centrality": round(dc, 4),
                "betweenness_centrality": round(bc, 4),
                "influence_score": influence,
                "group": graph.nodes[node].get("group", "unknown"),
            }
        )

    results.sort(key=lambda x: x["influence_score"], reverse=True)
    return results[:top_n]
