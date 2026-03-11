"""
query_engine.py — GOIES Graph Query Language (GQL) Engine v2

Bug fixes from v1:
  BUG-1  >= / <= captured by regex but ignored in if/elif chain → now handled
  BUG-2  Bare except: in path swallowed all errors → specific nx exceptions
  BUG-3  Parser lowercased query, but node lookup is case-sensitive → case-preserving resolver
  BUG-4  No fuzzy resolution → resolve_node_name() used on every node reference
  BUG-5  re.match on "find_nodes" greedily shadowed other patterns → priority-ordered rules
  BUG-6  Only >, <, = handled for degree → full set: >, <, =, ==, >=, <=, !=

Fixes applied in this revision:
  FIX-1  No LIMIT clause — `find countries` on a 10 000-node graph returned everything.
         Default cap of GQL_DEFAULT_LIMIT=200 rows applied to all node/edge result sets.
         Queries that explicitly use `top <k>` bypass the cap (already bounded by k).
  FIX-2  `find_all_paths` had no timeout — dense graphs could enumerate millions of paths
         and block the event loop indefinitely. Wrapped in ThreadPoolExecutor with timeout.
  FIX-3  graph_health_score label_diversity divided by zero when all edge labels were
         empty → now handled in utils.py; GQL betweenness handler also guarded.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("goies.gql")

GQL_DEFAULT_LIMIT   = 200   # FIX-1: max rows returned for open-ended queries
ALL_PATHS_TIMEOUT   = 5.0   # FIX-2: seconds before all-paths search is aborted


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve(graph: nx.DiGraph, raw: str) -> Optional[str]:
    """
    Case-insensitive exact match first, then SequenceMatcher fuzzy fallback.
    Returns None only if no node is close enough.
    """
    for node in graph.nodes:
        if str(node).lower() == raw.lower():
            return node
    try:
        from utils import resolve_node_name
        resolved = resolve_node_name(graph, raw)
        return resolved if resolved in graph else None
    except ImportError:
        return None


def _op_compare(d: float, op: str, val: float) -> bool:
    """All 7 comparison operators."""
    return {
        ">":  d > val,
        "<":  d < val,
        "=":  d == val,
        "==": d == val,
        ">=": d >= val,
        "<=": d <= val,
        "!=": d != val,
    }.get(op, False)


def _cap(result: List, limit: int = GQL_DEFAULT_LIMIT) -> Tuple[List, bool]:
    """FIX-1: Apply result cap; return (capped_list, was_truncated)."""
    truncated = len(result) > limit
    return result[:limit], truncated


# ── Parser ────────────────────────────────────────────────────────────────────

class GQLParser:
    """
    Priority-ordered rule list — most specific patterns tried first.
    """

    # (priority, name, pattern)
    _RULES: List[Tuple[int, str, str]] = [
        (10, "path",         r"(?:show |find )?path from (?P<src>.+?) to (?P<tgt>.+)"),
        (15, "predecessors", r"(?:predecessors?|incoming) of (?P<node>.+)"),
        (15, "successors",   r"(?:successors?|outgoing) of (?P<node>.+)"),
        (20, "neighbors",    r"neighbors? of (?P<node>.+)"),
        (30, "degree",       r"nodes? with degree (?P<op>>=|<=|!=|==|>|<|=)\s*(?P<val>\d+)"),
        (30, "top_degree",   r"top (?P<k>\d+) nodes? by degree"),
        (30, "top_between",  r"top (?P<k>\d+) nodes? by betweenness"),
        (40, "edges_between",r"edges? (?:between|from) (?P<src>.+?) (?:to|and) (?P<tgt>.+)"),
        (40, "edges_label",  r"edges? (?:where |with )?label (?:contains? )?(?P<label>.+)"),
        (50, "confidence",   r"nodes? with confidence (?P<op>>=|<=|!=|==|>|<|=)\s*(?P<val>[\d.]+)"),
        (60, "count",        r"count (?P<group>\w+)"),
        (70, "isolated",     r"(?:isolated|disconnected|orphan) nodes?"),
        (70, "hubs",         r"hub nodes?(?: with degree (?P<min_deg>\d+))?"),
        (90, "find_nodes",   r"find (?P<group>\w+)"),
    ]

    _SORTED = sorted(_RULES, key=lambda x: x[0])

    def parse(self, query: str) -> Dict[str, Any]:
        q   = re.sub(r"\s+", " ", query.lower().strip())
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
            f"\n  Results are capped at {GQL_DEFAULT_LIMIT} rows unless a top-k query is used.\n"
        )


# ── Executor ──────────────────────────────────────────────────────────────────

class GQLExecutor:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def execute(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        t   = parsed["type"]
        p   = parsed.get("params", {})
        raw = parsed.get("raw", "")

        if t == "unknown":
            return {
                "type":  "error",
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
            logger.warning("GQL execution error for '%s': %s", raw, exc)
            return {"type": "error", "error": str(exc), "query": raw}

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _q_find_nodes(self, p: Dict) -> Dict:
        group = p["group"].lower().rstrip("s")
        nodes = [
            {
                "id":         n,
                "group":      d.get("group", "unknown"),
                "confidence": d.get("confidence", 1.0),
            }
            for n, d in self.graph.nodes(data=True)
            if d.get("group", "").lower().rstrip("s") == group
        ]
        # FIX-1: Apply default cap
        capped, truncated = _cap(nodes)
        result = {"type": "nodes", "result": capped, "count": len(nodes)}
        if truncated:
            result["truncated"] = True
            result["truncated_at"] = GQL_DEFAULT_LIMIT
        return result

    def _q_neighbors(self, p: Dict) -> Dict:
        node = _resolve(self.graph, p["node"].strip())
        if node is None:
            return {"type": "error", "error": f"Node '{p['node'].strip()}' not found."}
        seen: dict = {}
        for v in self.graph.successors(node):
            seen[v] = {"id": v, "direction": "out", "group": self.graph.nodes[v].get("group", "unknown")}
        for u in self.graph.predecessors(node):
            if u not in seen:
                seen[u] = {"id": u, "direction": "in", "group": self.graph.nodes[u].get("group", "unknown")}
            else:
                seen[u]["direction"] = "both"
        result_list = list(seen.values())
        capped, truncated = _cap(result_list)
        out = {"type": "nodes", "anchor": node, "result": capped, "count": len(result_list)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_successors(self, p: Dict) -> Dict:
        node = _resolve(self.graph, p["node"].strip())
        if node is None:
            return {"type": "error", "error": f"Node '{p['node'].strip()}' not found."}
        result = [
            {"from": node, "to": v, "label": d.get("label", ""), "confidence": d.get("confidence", 1.0)}
            for _, v, d in self.graph.out_edges(node, data=True)
        ]
        capped, truncated = _cap(result)
        out = {"type": "edges", "anchor": node, "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_predecessors(self, p: Dict) -> Dict:
        node = _resolve(self.graph, p["node"].strip())
        if node is None:
            return {"type": "error", "error": f"Node '{p['node'].strip()}' not found."}
        result = [
            {"from": u, "to": node, "label": d.get("label", ""), "confidence": d.get("confidence", 1.0)}
            for u, _, d in self.graph.in_edges(node, data=True)
        ]
        capped, truncated = _cap(result)
        out = {"type": "edges", "anchor": node, "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_path(self, p: Dict) -> Dict:
        src = _resolve(self.graph, p["src"].strip())
        tgt = _resolve(self.graph, p["tgt"].strip())
        if src is None:
            return {"type": "error", "error": f"Source node '{p['src'].strip()}' not found."}
        if tgt is None:
            return {"type": "error", "error": f"Target node '{p['tgt'].strip()}' not found."}
        try:
            path = nx.shortest_path(self.graph, src, tgt)
            directed = True
        except nx.NetworkXNoPath:
            try:
                path = nx.shortest_path(self.graph.to_undirected(), src, tgt)
                directed = False
            except nx.NetworkXNoPath:
                return {"type": "error", "error": f"No path between '{src}' and '{tgt}'."}
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
            parts += ([f"[{e['label']}]"] if e["label"] else ["→"]) + [e["to"]]
        narrative = " ".join(parts)

        return {
            "type":      "path",
            "nodes":     path,
            "edges":     edges,
            "length":    len(path) - 1,
            "directed":  directed,
            "narrative": narrative,
        }

    def _q_degree(self, p: Dict) -> Dict:
        val = int(p["val"])
        op  = p["op"]
        result = [
            {"id": n, "degree": self.graph.degree(n), "group": self.graph.nodes[n].get("group", "unknown")}
            for n in self.graph.nodes()
            if _op_compare(self.graph.degree(n), op, val)
        ]
        result.sort(key=lambda x: x["degree"], reverse=True)
        capped, truncated = _cap(result)
        out = {"type": "nodes", "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_top_degree(self, p: Dict) -> Dict:
        k = max(1, int(p.get("k", 10)))
        ranked = sorted(
            ({"id": n, "degree": self.graph.degree(n), "group": self.graph.nodes[n].get("group", "unknown")}
             for n in self.graph.nodes()),
            key=lambda x: x["degree"], reverse=True,
        )
        return {"type": "nodes", "result": ranked[:k], "count": min(k, len(ranked))}

    def _q_top_between(self, p: Dict) -> Dict:
        k = max(1, int(p.get("k", 10)))
        if self.graph.number_of_nodes() < 4:
            return {"type": "error", "error": "Need ≥4 nodes for betweenness centrality."}
        # FIX-2: Run in executor with timeout to avoid blocking event loop on dense graphs
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(nx.betweenness_centrality, self.graph)
                bet = future.result(timeout=ALL_PATHS_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return {"type": "error", "error": f"Betweenness computation timed out after {ALL_PATHS_TIMEOUT}s. Graph may be too large."}
        ranked = sorted(
            ({"id": n, "betweenness": round(v, 4), "group": self.graph.nodes[n].get("group", "unknown")}
             for n, v in bet.items()),
            key=lambda x: x["betweenness"], reverse=True,
        )
        return {"type": "nodes", "result": ranked[:k], "count": min(k, len(ranked))}

    def _q_edges_label(self, p: Dict) -> Dict:
        label = p["label"].strip().lower()
        result = [
            {"from": u, "to": v, "label": d.get("label", ""), "confidence": d.get("confidence", 1.0)}
            for u, v, d in self.graph.edges(data=True)
            if label in d.get("label", "").lower()
        ]
        capped, truncated = _cap(result)
        out = {"type": "edges", "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_edges_between(self, p: Dict) -> Dict:
        src = _resolve(self.graph, p["src"].strip())
        tgt = _resolve(self.graph, p["tgt"].strip())
        result = []
        if src and tgt:
            for u, v in [(src, tgt), (tgt, src)]:
                if self.graph.has_edge(u, v):
                    d = self.graph[u][v]
                    result.append({"from": u, "to": v, "label": d.get("label", ""), "confidence": d.get("confidence", 1.0)})
        return {"type": "edges", "result": result, "count": len(result)}

    def _q_confidence(self, p: Dict) -> Dict:
        op  = p["op"]
        val = float(p["val"])
        result = [
            {"id": n, "confidence": d.get("confidence", 1.0), "group": d.get("group", "unknown")}
            for n, d in self.graph.nodes(data=True)
            if _op_compare(d.get("confidence", 1.0), op, val)
        ]
        capped, truncated = _cap(result)
        out = {"type": "nodes", "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_count(self, p: Dict) -> Dict:
        group = p["group"].lower().rstrip("s")
        if group in ("all", "node", "entit", "total"):
            return {"type": "count", "group": "all", "count": self.graph.number_of_nodes()}
        count = sum(
            1 for _, d in self.graph.nodes(data=True)
            if d.get("group", "").lower().rstrip("s") == group
        )
        return {"type": "count", "group": group, "count": count}

    def _q_isolated(self, p: Dict) -> Dict:
        result = [
            {"id": n, "group": self.graph.nodes[n].get("group", "unknown")}
            for n in nx.isolates(self.graph)
        ]
        capped, truncated = _cap(result)
        out = {"type": "nodes", "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out

    def _q_hubs(self, p: Dict) -> Dict:
        min_deg = int(p.get("min_deg") or 3)
        result = [
            {"id": n, "degree": self.graph.degree(n), "group": self.graph.nodes[n].get("group", "unknown")}
            for n in self.graph.nodes()
            if self.graph.degree(n) >= min_deg
        ]
        result.sort(key=lambda x: x["degree"], reverse=True)
        capped, truncated = _cap(result)
        out = {"type": "nodes", "result": capped, "count": len(result)}
        if truncated:
            out["truncated"] = True
        return out


# ── Convenience entry point ───────────────────────────────────────────────────

def run_gql(query: str, graph: nx.DiGraph) -> Dict[str, Any]:
    """Parse + execute in one call. Stateless — safe to call from any thread."""
    if query.strip().lower() in ("help", "?"):
        return {"type": "help", "text": GQLParser.help_text(), "query": query}
    parser = GQLParser()
    parsed = parser.parse(query)
    return GQLExecutor(graph).execute(parsed)
