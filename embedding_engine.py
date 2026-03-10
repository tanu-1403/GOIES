import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity


class GraphEmbeddingEngine:
    def __init__(self):
        self.model = None
        self.embeddings = {}
        self.nodes = []

    def train(self, graph: nx.DiGraph):
        if len(graph.nodes) < 5:
            return

        g = graph.to_undirected()

        node2vec = Node2Vec(g, dimensions=64, walk_length=20, num_walks=100, workers=2)

        self.model = node2vec.fit(window=10, min_count=1)

        self.nodes = list(g.nodes)

        self.embeddings = {node: self.model.wv[node] for node in self.nodes}

    def similar_nodes(self, node, top_k=5):
        if node not in self.embeddings:
            return []

        target = self.embeddings[node].reshape(1, -1)

        sims = []

        for other, vec in self.embeddings.items():
            if other == node:
                continue

            score = cosine_similarity(target, vec.reshape(1, -1))[0][0]

            sims.append((other, float(score)))

        sims.sort(key=lambda x: x[1], reverse=True)

        return sims[:top_k]
