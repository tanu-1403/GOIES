"""
query_engine.py — GOIES Graph Query Language (GQL) Engine v2

Bug fixes from v1:
  BUG-1  >= / <= captured by regex but ignored in if/elif chain → now handled
  BUG-2  Bare except: in path swallowed all errors → specific nx exceptions
  BUG-3  Parser lowercased query, but node lookup is case-sensitive → case-preserving resolver
  BUG-4  No fuzzy resolution → resolve_node_name() used on every node reference
  BUG-5  re.match on "find_nodes" greedily shadowed other patterns → priority-ordered rules
  BUG-6  Only >, <, = handled for degree → full set: >, <, =, ==, >=, <=, !=

New in v2:
  - 12 query types (was 5): neighbors, successors, predecessors, path, degree,
    top_degree, top_between, edges_label, edges_between, confidence, count,
    isolated, hubs
  - run_gql() single-call convenience wrapper
  - GQLParser.help_text() for the UI help panel
  - All results carry type, count, and echoed query for the frontend
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve(graph: nx.DiGraph, raw: str) -> Optional[str]:
    """
    BUG-3 / BUG-4 FIX:
    Case-insensitive exact match first, then SequenceMatcher fuzzy fallback.
    Returns None only if no node is close enough.
    """
    # 1. Exact case-insensitive
    for node in graph.nodes:
        if str(node).lower() == raw.lower():
            return node
    # 2. Fuzzy via utils
    try:
        from utils import resolve_node_name

        resolved = resolve_node_name(graph, raw)
        return resolved if resolved in graph else None
    except ImportError:
        return None


def _op_compare(d: int, op: str, val: int) -> bool:
    """BUG-1 / BUG-6 FIX: all 7 comparison operators."""
    return {
        ">": d > val,
        "<": d < val,
        "=": d == val,
        "==": d == val,
        ">=": d >= val,
        "<=": d <= val,
        "!=": d != val,
    }.get(op, False)


# ── Parser ────────────────────────────────────────────────────────────────────


class GQLParser:
    """
    Priority-ordered rule list — most specific patterns tried first.
    BUG-5 FIX: path/neighbors/successors/predecessors are checked before
    the broad "find_nodes" pattern, preventing greedy shadowing.
    """

    # (priority, name, pattern)
    _RULES: List[Tuple[int, str, str]] = [
        # Path — must precede find/neighbors
        (10, "path", r"(?:show |find )?path from (?P<src>.+?) to (?P<tgt>.+)"),
        # Neighbor variants
        (15, "predecessors", r"(?:predecessors?|incoming) of (?P<node>.+)"),
        (15, "successors", r"(?:successors?|outgoing) of (?P<node>.+)"),
        (20, "neighbors", r"neighbors? of (?P<node>.+)"),
        # Degree — BUG-1 / BUG-6 fix: op pattern covers >=, <=, !=, ==, >, <, =
        (30, "degree", r"nodes? with degree (?P<op>>=|<=|!=|==|>|<|=)\s*(?P<val>\d+)"),
        # Top-k structural
        (30, "top_degree", r"top (?P<k>\d+) nodes? by degree"),
        (30, "top_between", r"top (?P<k>\d+) nodes? by betweenness"),
        # Edge queries
        (
            40,
            "edges_between",
            r"edges? (?:between|from) (?P<src>.+?) (?:to|and) (?P<tgt>.+)",
        ),
        (
            40,
            "edges_label",
            r"edges? (?:where |with )?label (?:contains? )?(?P<label>.+)",
        ),
        # Confidence filter
        (
            50,
            "confidence",
            r"nodes? with confidence (?P<op>>=|<=|!=|==|>|<|=)\s*(?P<val>[\d.]+)",
        ),
        # Count
        (60, "count", r"count (?P<group>\w+)"),
        # Topology
        (70, "isolated", r"(?:isolated|disconnected|orphan) nodes?"),
        (70, "hubs", r"hub nodes?(?: with degree (?P<min_deg>\d+))?"),
        # find_nodes — BUG-5 FIX: last, so it never shadows the above
        (90, "find_nodes", r"find (?P<group>\w+)"),
    ]

    _SORTED = sorted(_RULES, key=lambda x: x[0])

    def parse(self, query: str) -> Dict[str, Any]:
        q = re.sub(r"\s+", " ", query.lower().strip())
        raw = re.sub(r"\s+", " ", query.strip())
        for _, name, pat in self._SORTED:
            m = re.match(pat, q)
            if m:
                return {"type": name, "params": m.groupdict(), "raw": raw}
        return {"type": "unknown", "raw": raw}

    @staticmethod
    def help_text() -> str:
        return (
            "◈ GQL SYNTAX REFERENCE\n\n"
            "  find <group>                       find countries / persons / organizations …\n"
            "  neighbors of <node>                all direct connections\n"
            "  successors of <node>               outgoing edges only\n"
            "  predecessors of <node>             incoming edges only\n"
            "  path from <A> to <B>               shortest path\n"
            "  nodes with degree > <n>            filter by connection count  (>, <, =, >=, <=, !=)\n"
            "  top <k> nodes by degree            most connected actors\n"
            "  top <k> nodes by betweenness       key broker actors\n"
            "  edges label contains <text>        filter relationships by label text\n"
            "  edges between <A> and <B>          direct edges between two actors\n"
            "  nodes with confidence >= <val>     filter by extraction confidence (0-1)\n"
            "  count <group>                      count entities of a type\n"
            "  isolated nodes                     nodes with no connections\n"
            "  hub nodes                          high-degree connector nodes\n"
        )


# ── Executor ──────────────────────────────────────────────────────────────────


class GQLExecutor:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def execute(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        t = parsed["type"]
        p = parsed.get("params", {})
        raw = parsed.get("raw", "")

        if t == "unknown":
            return {
                "type": "error",
                "error": f"Unrecognised query: '{raw}'. Type 'help' for syntax reference.",
                "query": raw,
            }

        handler = getattr(self, f"_q_{t}", None)
        if handler is None:
            return {"type": "error", "error": f"No handler for '{t}'.", "query": raw}

        try:
            result = handler(p)
            result["query"] = raw
            return result
        except Exception as exc:
            return {"type": "error", "error": str(exc), "query": raw}

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _q_find_nodes(self, p: Dict) -> Dict:
        group = p["group"].lower().rstrip("s")
        nodes = [
            {
                "id": n,
                "group": d.get("group", "unknown"),
                "confidence": d.get("confidence", 1.0),
            }
            for n, d in self.graph.nodes(data=True)
            if d.get("group", "").lower().rstrip("s") == group
        ]
        return {"type": "nodes", "result": nodes, "count": len(nodes)}

    def _q_neighbors(self, p: Dict) -> Dict:
        node = _resolve(self.graph, p["node"].strip())
        if node is None:
            return {"type": "error", "error": f"Node '{p['node'].strip()}' not found."}
        seen: dict = {}
        for v in self.graph.successors(node):
            seen[v] = {
                "id": v,
                "direction": "out",
                "group": self.graph.nodes[v].get("group", "unknown"),
            }
        for u in self.graph.predecessors(node):
            if u not in seen:
                seen[u] = {
                    "id": u,
                    "direction": "in",
                    "group": self.graph.nodes[u].get("group", "unknown"),
                }
            else:
                seen[u]["direction"] = "both"
        result = list(seen.values())
        return {"type": "nodes", "anchor": node, "result": result, "count": len(result)}

    def _q_successors(self, p: Dict) -> Dict:
        node = _resolve(self.graph, p["node"].strip())
        if node is None:
            return {"type": "error", "error": f"Node '{p['node'].strip()}' not found."}
        result = [
            {
                "from": node,
                "to": v,
                "label": d.get("label", ""),
                "confidence": d.get("confidence", 1.0),
            }
            for _, v, d in self.graph.out_edges(node, data=True)
        ]
        return {"type": "edges", "anchor": node, "result": result, "count": len(result)}

    def _q_predecessors(self, p: Dict) -> Dict:
        node = _resolve(self.graph, p["node"].strip())
        if node is None:
            return {"type": "error", "error": f"Node '{p['node'].strip()}' not found."}
        result = [
            {
                "from": u,
                "to": node,
                "label": d.get("label", ""),
                "confidence": d.get("confidence", 1.0),
            }
            for u, _, d in self.graph.in_edges(node, data=True)
        ]
        return {"type": "edges", "anchor": node, "result": result, "count": len(result)}

    def _q_path(self, p: Dict) -> Dict:
        # BUG-2 FIX: specific exceptions, not bare except
        src = _resolve(self.graph, p["src"].strip())
        tgt = _resolve(self.graph, p["tgt"].strip())
        if src is None:
            return {
                "type": "error",
                "error": f"Source node '{p['src'].strip()}' not found.",
            }
        if tgt is None:
            return {
                "type": "error",
                "error": f"Target node '{p['tgt'].strip()}' not found.",
            }
        try:
            path = nx.shortest_path(self.graph, src, tgt)
            directed = True
        except nx.NetworkXNoPath:
            try:
                path = nx.shortest_path(self.graph.to_undirected(), src, tgt)
                directed = False
            except nx.NetworkXNoPath:
                return {
                    "type": "error",
                    "error": f"No path between '{src}' and '{tgt}'.",
                }
        except nx.NodeNotFound as exc:
            return {"type": "error", "error": str(exc)}

        edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            label = ""
            if self.graph.has_edge(u, v):
                label = self.graph[u][v].get("label", "")
            elif self.graph.has_edge(v, u):
                label = self.graph[v][u].get("label", "")
            edges.append({"from": u, "to": v, "label": label})

        parts = [path[0]]
        for e in edges:
            if e["label"]:
                parts += [f"[{e['label']}]", e["to"]]
            else:
                parts += ["→", e["to"]]
        narrative = " ".join(parts)

        return {
            "type": "path",
            "nodes": path,
            "edges": edges,
            "length": len(path) - 1,
            "directed": directed,
            "narrative": narrative,
        }

    def _q_degree(self, p: Dict) -> Dict:
        # BUG-1 / BUG-6 FIX: all operators including >= <=
        val = int(p["val"])
        op = p["op"]
        result = [
            {
                "id": n,
                "degree": self.graph.degree(n),
                "group": self.graph.nodes[n].get("group", "unknown"),
            }
            for n in self.graph.nodes()
            if _op_compare(self.graph.degree(n), op, val)
        ]
        result.sort(key=lambda x: x["degree"], reverse=True)
        return {"type": "nodes", "result": result, "count": len(result)}

    def _q_top_degree(self, p: Dict) -> Dict:
        k = max(1, int(p.get("k", 10)))
        ranked = sorted(
            (
                {
                    "id": n,
                    "degree": self.graph.degree(n),
                    "group": self.graph.nodes[n].get("group", "unknown"),
                }
                for n in self.graph.nodes()
            ),
            key=lambda x: x["degree"],
            reverse=True,
        )
        return {"type": "nodes", "result": ranked[:k], "count": min(k, len(ranked))}

    def _q_top_between(self, p: Dict) -> Dict:
        k = max(1, int(p.get("k", 10)))
        if self.graph.number_of_nodes() < 4:
            return {
                "type": "error",
                "error": "Need ≥4 nodes for betweenness centrality.",
            }
        bet = nx.betweenness_centrality(self.graph)
        ranked = sorted(
            (
                {
                    "id": n,
                    "betweenness": round(v, 4),
                    "group": self.graph.nodes[n].get("group", "unknown"),
                }
                for n, v in bet.items()
            ),
            key=lambda x: x["betweenness"],
            reverse=True,
        )
        return {"type": "nodes", "result": ranked[:k], "count": min(k, len(ranked))}

    def _q_edges_label(self, p: Dict) -> Dict:
        label = p["label"].strip().lower()
        result = [
            {
                "from": u,
                "to": v,
                "label": d.get("label", ""),
                "confidence": d.get("confidence", 1.0),
            }
            for u, v, d in self.graph.edges(data=True)
            if label in d.get("label", "").lower()
        ]
        return {"type": "edges", "result": result, "count": len(result)}

    def _q_edges_between(self, p: Dict) -> Dict:
        src = _resolve(self.graph, p["src"].strip())
        tgt = _resolve(self.graph, p["tgt"].strip())
        result = []
        if src and tgt:
            for u, v in [(src, tgt), (tgt, src)]:
                if self.graph.has_edge(u, v):
                    d = self.graph[u][v]
                    result.append(
                        {
                            "from": u,
                            "to": v,
                            "label": d.get("label", ""),
                            "confidence": d.get("confidence", 1.0),
                        }
                    )
        return {"type": "edges", "result": result, "count": len(result)}

    def _q_confidence(self, p: Dict) -> Dict:
        op = p["op"]
        val = float(p["val"])
        result = [
            {
                "id": n,
                "confidence": d.get("confidence", 1.0),
                "group": d.get("group", "unknown"),
            }
            for n, d in self.graph.nodes(data=True)
            if _op_compare(d.get("confidence", 1.0), op, val)
        ]
        return {"type": "nodes", "result": result, "count": len(result)}

    def _q_count(self, p: Dict) -> Dict:
        group = p["group"].lower().rstrip("s")
        if group in ("all", "node", "entit", "total"):
            return {
                "type": "count",
                "group": "all",
                "count": self.graph.number_of_nodes(),
            }
        count = sum(
            1
            for _, d in self.graph.nodes(data=True)
            if d.get("group", "").lower().rstrip("s") == group
        )
        return {"type": "count", "group": group, "count": count}

    def _q_isolated(self, p: Dict) -> Dict:
        result = [
            {"id": n, "group": self.graph.nodes[n].get("group", "unknown")}
            for n in nx.isolates(self.graph)
        ]
        return {"type": "nodes", "result": result, "count": len(result)}

    def _q_hubs(self, p: Dict) -> Dict:
        min_deg = int(p.get("min_deg") or 3)
        result = [
            {
                "id": n,
                "degree": self.graph.degree(n),
                "group": self.graph.nodes[n].get("group", "unknown"),
            }
            for n in self.graph.nodes()
            if self.graph.degree(n) >= min_deg
        ]
        result.sort(key=lambda x: x["degree"], reverse=True)
        return {"type": "nodes", "result": result, "count": len(result)}


# ── Convenience entry point ───────────────────────────────────────────────────


def run_gql(query: str, graph: nx.DiGraph) -> Dict[str, Any]:
    """Parse + execute in one call. Stateless — safe to call from any thread."""
    if query.strip().lower() in ("help", "?"):
        return {"type": "help", "text": GQLParser.help_text(), "query": query}
    parser = GQLParser()
    parsed = parser.parse(query)
    return GQLExecutor(graph).execute(parsed)
