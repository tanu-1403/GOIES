"""
geo.py — Geo-positional tension engine
Calculates tension scores for countries in the intelligence graph.
"""

import json
from typing import Dict, List, Any
import networkx as nx

COUNTRY_COORDS = {
    "Afghanistan": {"lat": 33.939, "lon": 67.71},
    "Algeria": {"lat": 28.034, "lon": 1.66},
    "Argentina": {"lat": -38.416, "lon": -63.617},
    "Australia": {"lat": -25.274, "lon": 133.775},
    "Austria": {"lat": 47.516, "lon": 14.55},
    "Bangladesh": {"lat": 23.685, "lon": 90.356},
    "Belgium": {"lat": 50.504, "lon": 4.47},
    "Brazil": {"lat": -14.235, "lon": -51.925},
    "Canada": {"lat": 56.13, "lon": -106.347},
    "Chile": {"lat": -35.675, "lon": -71.543},
    "China": {"lat": 35.862, "lon": 104.195},
    "Colombia": {"lat": 4.571, "lon": -74.297},
    "Czech Republic": {"lat": 49.817, "lon": 15.473},
    "Denmark": {"lat": 56.264, "lon": 9.502},
    "Egypt": {"lat": 26.821, "lon": 30.802},
    "Ethiopia": {"lat": 9.145, "lon": 40.489},
    "Finland": {"lat": 61.924, "lon": 25.748},
    "France": {"lat": 46.228, "lon": 2.214},
    "Germany": {"lat": 51.166, "lon": 10.452},
    "Greece": {"lat": 39.074, "lon": 21.824},
    "Hungary": {"lat": 47.162, "lon": 19.503},
    "India": {"lat": 20.594, "lon": 78.963},
    "Indonesia": {"lat": -0.789, "lon": 113.921},
    "Iran": {"lat": 32.428, "lon": 53.688},
    "Iraq": {"lat": 33.223, "lon": 43.68},
    "Ireland": {"lat": 53.413, "lon": -8.244},
    "Israel": {"lat": 31.046, "lon": 34.852},
    "Italy": {"lat": 41.872, "lon": 12.567},
    "Japan": {"lat": 36.205, "lon": 138.253},
    "Kenya": {"lat": -0.024, "lon": 37.906},
    "Lebanon": {"lat": 33.854, "lon": 35.862},
    "Libya": {"lat": 26.335, "lon": 17.228},
    "Malaysia": {"lat": 4.21, "lon": 101.976},
    "Mexico": {"lat": 23.635, "lon": -102.553},
    "Morocco": {"lat": 31.792, "lon": -7.093},
    "Myanmar": {"lat": 21.914, "lon": 95.956},
    "Netherlands": {"lat": 52.133, "lon": 5.291},
    "New Zealand": {"lat": -40.901, "lon": 174.886},
    "Nigeria": {"lat": 9.082, "lon": 8.675},
    "North Korea": {"lat": 40.339, "lon": 127.51},
    "Norway": {"lat": 60.472, "lon": 8.469},
    "Pakistan": {"lat": 30.375, "lon": 69.345},
    "Peru": {"lat": -9.19, "lon": -75.015},
    "Philippines": {"lat": 12.88, "lon": 121.774},
    "Poland": {"lat": 51.919, "lon": 19.145},
    "Portugal": {"lat": 39.399, "lon": -8.224},
    "Qatar": {"lat": 25.355, "lon": 51.184},
    "Romania": {"lat": 45.943, "lon": 24.967},
    "Russia": {"lat": 61.524, "lon": 105.319},
    "Saudi Arabia": {"lat": 23.886, "lon": 45.079},
    "South Africa": {"lat": -30.559, "lon": 22.938},
    "South Korea": {"lat": 35.908, "lon": 127.767},
    "Spain": {"lat": 40.464, "lon": -3.749},
    "Sudan": {"lat": 12.863, "lon": 30.218},
    "Sweden": {"lat": 60.128, "lon": 18.644},
    "Switzerland": {"lat": 46.818, "lon": 8.228},
    "Syria": {"lat": 34.802, "lon": 38.997},
    "Taiwan": {"lat": 23.698, "lon": 120.96},
    "Thailand": {"lat": 15.87, "lon": 100.993},
    "Turkey": {"lat": 38.964, "lon": 35.243},
    "EU": {"lat": 50.85, "lon": 4.35},
    "NATO": {"lat": 50.88, "lon": 4.43},
    "US": {"lat": 37.09, "lon": -95.713},
    "United States": {"lat": 37.09, "lon": -95.713},
    "UK": {"lat": 55.378, "lon": -3.436},
    "United Kingdom": {"lat": 55.378, "lon": -3.436},
    "Ukraine": {"lat": 48.38, "lon": 31.166},
    "United Arab Emirates": {"lat": 23.424, "lon": 53.848},
    "Venezuela": {"lat": 6.424, "lon": -66.59},
    "Vietnam": {"lat": 14.058, "lon": 108.277},
    "Yemen": {"lat": 15.553, "lon": 48.516},
}


def _edge_score(label: str) -> float:
    label = label.lower()

    high_keywords = [
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
    ]
    med_keywords = [
        "restrict",
        "ban",
        "expel",
        "dispute",
        "tension",
        "pressure",
        "cyber",
        "confront",
    ]
    coop_keywords = [
        "cooperate",
        "ally",
        "partner",
        "invest",
        "aid",
        "support",
        "trade",
        "treaty",
    ]

    for k in high_keywords:
        if k in label:
            return 18.0
    for k in med_keywords:
        if k in label:
            return 9.0
    for k in coop_keywords:
        if k in label:
            return -3.0

    return 2.0


def _is_military_event(node_id: str, graph: nx.DiGraph) -> bool:
    data = graph.nodes[node_id]
    attr = data.get("attributes", {})
    if isinstance(attr, str):
        try:
            attr = eval(attr)
        except:
            attr = {}

    # Simple heuristic
    lower_id = str(node_id).lower()
    mil_words = ["war", "battle", "strike", "military", "conflict", "invasion"]
    return any(w in lower_id for w in mil_words)


def calculate_country_tensions(
    graph: nx.DiGraph, custom_thresholds: Dict[str, float] = None
) -> Dict[str, float]:
    scores = {}
    country_nodes = []

    for n, data in graph.nodes(data=True):
        if data.get("group", "").lower() == "country" or n in COUNTRY_COORDS:
            country_nodes.append(n)

    for node in country_nodes:
        score: float = 0.0

        # Outgoing hostile edges (aggressor weight)
        for _, _, data in graph.out_edges(node, data=True):
            score += float(_edge_score(data.get("label", ""))) * 1.2

        # Incoming hostile pressure (target weight)
        for _, _, data in graph.in_edges(node, data=True):
            score += _edge_score(data.get("label", "")) * 0.9

        # Military event bonus
        for _, neighbor in list(graph.out_edges(node)) + list(graph.in_edges(node)):
            if graph.nodes[neighbor].get("group", "").lower() == "event":
                if _is_military_event(neighbor, graph):
                    score += 7.0

        # Degree centrality multiplier
        node_degree = graph.degree(node)
        score *= 1.0 + 0.04 * min(node_degree, 20)

        scores[node] = score

    return normalise_tensions(scores, custom_thresholds)


def normalise_tensions(
    scores: Dict[str, float], custom_thresholds: Dict[str, float] = None
) -> Dict[str, Dict[str, Any]]:
    if not scores:
        return {}

    # Standardize to 0-100 logically.
    max_score = max(max(scores.values()), 1.0)

    normed = {}
    for node, score in scores.items():
        val = (score / max_score) * 100
        clamped = max(0.0, min(100.0, val))

        threshold = 75.0
        if custom_thresholds and node in custom_thresholds:
            threshold = custom_thresholds[node]

        alert = clamped >= threshold
        normed[node] = {"score": clamped, "alert": alert, "threshold": threshold}

    return normed


def _get_color_for_tension(tension: float) -> str:
    if tension >= 75:
        return "#ff2244"
    if tension >= 50:
        return "#ff6b35"
    if tension >= 25:
        return "#ffaa40"
    if tension >= 10:
        return "#ffe066"
    return "#00ff88"


def get_geo_data(
    graph: nx.DiGraph, custom_thresholds: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    tensions = calculate_country_tensions(graph, custom_thresholds)
    markers = []

    for node, tension in tensions.items():
        if node in COUNTRY_COORDS:
            coords = COUNTRY_COORDS[node]
            node_data = graph.nodes[node] if graph.has_node(node) else {}

            # Find top connections
            connections = []
            for u, v, d in list(graph.out_edges(node, data=True)) + list(
                graph.in_edges(node, data=True)
            ):
                if u == node:
                    connections.append(f"{d.get('label', 'connects to')} {v}")
                else:
                    connections.append(f"{u} {d.get('label', 'connects to')}")

            markers.append(
                {
                    "id": node,
                    "lat": coords["lat"],
                    "lon": coords["lon"],
                    "group": node_data.get("group", "country"),
                    "tension": round(tension["score"], 1),
                    "raw_score": 0,  # not strictly needed, UI uses tension
                    "color": _get_color_for_tension(tension["score"]),
                    "threshold": tension["threshold"],
                    "alert": tension["alert"],
                    "connections": connections[:5],
                    "degree": graph.degree(node) if graph.has_node(node) else 0,
                    "attributes": str(node_data.get("attributes", {})),
                }
            )

    return markers
