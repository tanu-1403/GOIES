"""
forecaster.py — GOIES Crisis Forecasting Engine v2
Structural analysis (hotspots, reciprocal hostility, triangle conflicts)
combined with LLM narrative to produce ranked crisis forecasts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT_SECS = 150

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


# ── Data Models ───────────────────────────────────────────────────────────────
@dataclass
class ForecastItem:
    rank: int
    title: str
    actors: List[str]
    probability: float
    severity: str
    timeframe: str
    structural_signal: str
    narrative: str
    mitigation: str


@dataclass
class ForecastResult:
    global_risk: float
    global_label: str
    structural_summary: str
    hotspot_nodes: List[str]
    forecasts: List[ForecastItem]
    model_used: str


# ── Ollama Helper ─────────────────────────────────────────────────────────────
def _call_ollama(prompt: str, model: str) -> str:
    import requests

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot reach Ollama at {OLLAMA_BASE_URL}.")
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama timed out during forecast generation.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    return resp.json().get("response", "").strip()


def _strip_json(raw: str) -> Any:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in LLM output: {raw[:300]}")
    return json.loads(match.group(0))


def _is_hostile(label: str) -> bool:
    label = label.lower()
    return any(k in label for k in HOSTILE_KEYWORDS)


def _is_cooperative(label: str) -> bool:
    label = label.lower()
    return any(k in label for k in COOPERATIVE_KEYWORDS)


# ── Structural Signal Analysis ────────────────────────────────────────────────
def _hotspot_nodes(graph: nx.DiGraph, top_n: int = 8) -> List[str]:
    """Nodes with highest hostile edge involvement."""
    scores: Dict[str, float] = {}
    for u, v, d in graph.edges(data=True):
        label = d.get("label", "")
        weight = (
            3.0 if _is_hostile(label) else (-1.0 if _is_cooperative(label) else 0.5)
        )
        scores[u] = scores.get(u, 0) + weight
        scores[v] = scores.get(v, 0) + weight * 0.6

    if not scores:
        return list(graph.nodes)[:top_n]

    # Blend with betweenness centrality for structural importance
    try:
        bet = nx.betweenness_centrality(graph)
        max_bet = max(bet.values()) if bet else 1.0
        for node in scores:
            scores[node] += (bet.get(node, 0) / max(max_bet, 0.001)) * 15.0
    except Exception:
        pass

    return [
        n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ]


def _find_reciprocal_hostility(graph: nx.DiGraph) -> List[Tuple[str, str, str, str]]:
    """Returns (A, B, label_AB, label_BA) for mutual hostile pairs."""
    pairs: List[Tuple[str, str, str, str]] = []
    checked: Set[Tuple[str, str]] = set()
    for u, v, d in graph.edges(data=True):
        if not _is_hostile(d.get("label", "")):
            continue
        key = tuple(sorted([u, v]))
        if key in checked:
            continue
        checked.add(key)
        if graph.has_edge(v, u) and _is_hostile(graph[v][u].get("label", "")):
            pairs.append((u, v, d.get("label", ""), graph[v][u].get("label", "")))
    return pairs


def _find_conflict_triangles(graph: nx.DiGraph) -> List[Tuple[str, str, str]]:
    """Returns (A, B, C) triangles where all edges are hostile."""
    triangles: List[Tuple[str, str, str]] = []
    seen: Set[frozenset] = set()
    for u, v, d in graph.edges(data=True):
        if not _is_hostile(d.get("label", "")):
            continue
        for w in graph.successors(v):
            if w == u:
                continue
            key = frozenset([u, v, w])
            if key in seen:
                continue
            if graph.has_edge(u, w) or graph.has_edge(w, u):
                d2 = graph.get_edge_data(v, w, {})
                d3 = graph.get_edge_data(u, w, {}) or graph.get_edge_data(w, u, {})
                if _is_hostile(d2.get("label", "")) and _is_hostile(
                    d3.get("label", "")
                ):
                    triangles.append((u, v, w))
                    seen.add(key)
    return triangles[:5]


def _structural_signals(graph: nx.DiGraph) -> Dict[str, Any]:
    hotspots = _hotspot_nodes(graph)
    reciprocal = _find_reciprocal_hostility(graph)
    triangles = _find_conflict_triangles(graph)

    # Global risk = weighted combination of structural factors
    n = max(graph.number_of_nodes(), 1)
    hostile_edge_ratio = sum(
        1 for _, _, d in graph.edges(data=True) if _is_hostile(d.get("label", ""))
    ) / max(graph.number_of_edges(), 1)

    risk = (
        min(len(reciprocal) * 12.0, 40.0)
        + min(len(triangles) * 8.0, 24.0)
        + hostile_edge_ratio * 36.0
    )
    risk = min(100.0, risk)

    if risk >= 75:
        label = "CRITICAL"
    elif risk >= 50:
        label = "HIGH"
    elif risk >= 25:
        label = "MEDIUM"
    else:
        label = "LOW"

    return {
        "global_risk": round(risk, 1),
        "global_label": label,
        "hotspot_nodes": hotspots,
        "reciprocal_pairs": reciprocal,
        "conflict_triangles": triangles,
        "hostile_edge_ratio": round(hostile_edge_ratio, 3),
    }


# ── LLM Forecast Generation ───────────────────────────────────────────────────
def _generate_forecasts(
    graph: nx.DiGraph, signals: Dict[str, Any], model: str, focus_query: str
) -> List[ForecastItem]:
    hotspots = signals["hotspot_nodes"]
    reciprocal = signals["reciprocal_pairs"]
    triangles = signals["conflict_triangles"]

    # Build structural context for LLM
    reciprocal_str = (
        "\n".join(
            f"  - {a} and {b} have mutual hostility: '{la}' ↔ '{lb}'"
            for a, b, la, lb in reciprocal[:5]
        )
        or "  None detected"
    )

    triangle_str = (
        "\n".join(
            f"  - Conflict triangle: {a} — {b} — {c}" for a, b, c in triangles[:3]
        )
        or "  None detected"
    )

    # Sample of high-tension edges
    hostile_edges = [
        f"{u} -[{d.get('label', '')}]-> {v}"
        for u, v, d in graph.edges(data=True)
        if _is_hostile(d.get("label", ""))
    ][:20]

    focus_clause = (
        f'\nFocus area (prioritize if relevant): "{focus_query}"'
        if focus_query.strip()
        else ""
    )

    prompt = f"""You are a senior geopolitical intelligence analyst generating crisis forecasts.

STRUCTURAL ANALYSIS RESULTS:
Global Risk Score: {signals["global_risk"]}/100 ({signals["global_label"]})
Hotspot actors: {json.dumps(hotspots[:6])}

Mutual hostility pairs:
{reciprocal_str}

Conflict triangles:
{triangle_str}

High-tension relationships in graph:
{chr(10).join(hostile_edges[:15]) or "None"}
{focus_clause}

Generate 3 ranked crisis forecasts based purely on the structural evidence above.
Return ONLY raw JSON, no markdown, no prose:
{{
  "structural_summary": "2-sentence summary of the structural patterns driving risk",
  "forecasts": [
    {{
      "rank": 1,
      "title": "Short crisis title (5-8 words)",
      "actors": ["ActorA", "ActorB"],
      "probability": 0.72,
      "severity": "CRITICAL",
      "timeframe": "near-term (0-3 months)",
      "structural_signal": "One sentence citing the specific graph structure that signals this",
      "narrative": "2-3 sentences of intelligence assessment",
      "mitigation": "One concrete de-escalation recommendation"
    }}
  ]
}}

severity must be: LOW / MEDIUM / HIGH / CRITICAL
probability must be 0.0-1.0
Generate exactly 3 forecasts ranked by probability descending."""

    data = _strip_json(_call_ollama(prompt, model))

    structural_summary = data.get("structural_summary", "Structural analysis complete.")
    raw_forecasts = data.get("forecasts", [])
    if not isinstance(raw_forecasts, list):
        raw_forecasts = []

    items: List[ForecastItem] = []
    for i, f in enumerate(raw_forecasts[:5]):
        prob = float(f.get("probability", 0.5))
        prob = max(0.0, min(1.0, prob))
        severity = f.get("severity", "MEDIUM")
        if severity not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            severity = "MEDIUM"
        items.append(
            ForecastItem(
                rank=int(f.get("rank", i + 1)),
                title=str(f.get("title", f"Crisis Scenario {i + 1}")),
                actors=f.get("actors", []) if isinstance(f.get("actors"), list) else [],
                probability=round(prob, 2),
                severity=severity,
                timeframe=str(f.get("timeframe", "unknown")),
                structural_signal=str(f.get("structural_signal", "")),
                narrative=str(f.get("narrative", "")),
                mitigation=str(f.get("mitigation", "")),
            )
        )

    return structural_summary, items


# ── Public Entry Point ────────────────────────────────────────────────────────
def run_forecast(
    graph: nx.DiGraph,
    model: str = "llama3.2",
    focus_query: str = "",
) -> ForecastResult:
    """
    Structural + LLM crisis forecast.
    Raises ConnectionError, TimeoutError, ValueError.
    """
    signals = _structural_signals(graph)
    structural_summary, forecasts = _generate_forecasts(
        graph, signals, model, focus_query
    )

    return ForecastResult(
        global_risk=signals["global_risk"],
        global_label=signals["global_label"],
        structural_summary=structural_summary,
        hotspot_nodes=signals["hotspot_nodes"],
        forecasts=forecasts,
        model_used=model,
    )
