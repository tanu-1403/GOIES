import re
import networkx as nx


class GQLParser:
    PATTERNS = {
        "find_nodes": r"find (?P<group>\w+)",
        "neighbors": r"neighbors of (?P<node>.+)",
        "path": r"(?:show|find) path from (?P<src>.+?) to (?P<tgt>.+)",
        "degree": r"nodes with degree (?P<op>[><=]+)\s*(?P<val>\d+)",
        "edges_label": r"edges where label contains (?P<label>.+)",
    }

    def parse(self, query: str):
        q = query.lower().strip()

        for name, pattern in self.PATTERNS.items():
            m = re.match(pattern, q)

            if m:
                return {"type": name, "params": m.groupdict()}

        return {"type": "unknown", "query": query}


class GQLExecutor:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def execute(self, parsed):
        t = parsed["type"]
        p = parsed.get("params", {})

        if t == "find_nodes":
            group = p["group"]

            nodes = [
                n
                for n, d in self.graph.nodes(data=True)
                if d.get("group", "").lower() == group
            ]

            return {"type": "nodes", "result": nodes}

        elif t == "neighbors":
            node = p["node"]

            if node not in self.graph:
                return {"error": "node not found"}

            neigh = list(self.graph.successors(node)) + list(
                self.graph.predecessors(node)
            )

            return {"type": "nodes", "result": list(set(neigh))}

        elif t == "path":
            src = p["src"]
            tgt = p["tgt"]

            try:
                path = nx.shortest_path(self.graph, src, tgt)

                edges = []

                for i in range(len(path) - 1):
                    data = self.graph[path[i]][path[i + 1]]

                    edges.append(
                        {
                            "from": path[i],
                            "to": path[i + 1],
                            "label": data.get("label", ""),
                        }
                    )

                return {"type": "path", "nodes": path, "edges": edges}

            except:
                return {"error": "no path found"}

        elif t == "degree":
            val = int(p["val"])
            op = p["op"]

            result = []

            for node in self.graph.nodes():
                d = self.graph.degree(node)

                if op == ">" and d > val:
                    result.append(node)

                elif op == "<" and d < val:
                    result.append(node)

                elif op == "=" and d == val:
                    result.append(node)

            return {"type": "nodes", "result": result}

        elif t == "edges_label":
            label = p["label"]

            edges = []

            for u, v, d in self.graph.edges(data=True):
                if label in d.get("label", "").lower():
                    edges.append({"from": u, "to": v, "label": d.get("label")})

            return {"type": "edges", "result": edges}

        return {"error": "unknown query"}
