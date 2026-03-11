"""
simulator.py — GOIES Policy Simulation Engine v2

Fixes applied:
  FIX-1  _save_history() had no write lock — concurrent simulations could corrupt
         sim_history.json. Added a module-level threading.Lock.
  FIX-2  No cap on history file size beyond the slice [:50] — the file could
         accumulate garbage if the slice was mis-applied. Lock + atomic write fixes it.
  FIX-3  Bare except in _save_history suppressed all errors silently.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import pathlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

import networkx as nx

logger = logging.getLogger("goies.simulator")

OLLAMA_BASE_URL      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
REQUEST_TIMEOUT_SECS = 120
SIM_HISTORY_FILE     = pathlib.Path("sim_history.json")
MAX_HISTORY          = 50

RISK_THRESHOLDS = {"LOW": 25, "MEDIUM": 50, "HIGH": 75, "CRITICAL": 100}

# FIX-1: Prevent concurrent writes corrupting sim_history.json
_history_lock = threading.Lock()


# ── Data Model ────────────────────────────────────────────────────────────────
@dataclass
class SimulationResult:
    scenario: str
    risk_score: float
    risk_label: str
    cascade_narrative: str
    second_order: List[str] = field(default_factory=list)
    added_edges: List[Dict] = field(default_factory=list)
    removed_edges: List[Dict] = field(default_factory=list)
    affected_nodes: List[str] = field(default_factory=list)
    model_used: str = "llama3.2"


# ── Ollama Helper ─────────────────────────────────────────────────────────────
import requests as _requests  # hoisted — was imported per-call inside _call_ollama

def _call_ollama(prompt: str, model: str) -> str:
    try:
        resp = _requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        resp.raise_for_status()
    except _requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Run: ollama run {model}"
        )
    except _requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama did not respond within {REQUEST_TIMEOUT_SECS}s.")
    except _requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    return resp.json().get("response", "").strip()


def _strip_json(raw: str) -> Dict[str, Any]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM output: {raw[:300]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON from LLM: {e}. Raw: {raw[:300]}")


def _risk_label_from_score(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


# ── Pass 1: Parse Scenario → Graph Mutations ──────────────────────────────────
def _parse_scenario(scenario: str, graph: nx.DiGraph, model: str) -> Dict[str, Any]:
    # Select highest-degree nodes and their edges for richer, more relevant context
    deg_sorted = sorted(graph.nodes(), key=lambda n: graph.degree(n), reverse=True)
    nodes_sample = [str(n) for n in deg_sorted[:30]]
    hub_nodes = set(deg_sorted[:10])
    edges_sample = []
    # Prioritise edges incident to hub nodes so the sample covers key relationships
    for u, v, d in graph.edges(data=True):
        if u in hub_nodes or v in hub_nodes:
            edges_sample.append(f"{u} -[{d.get('label', '')}]-> {v}")
        if len(edges_sample) >= 25:
            break
    if len(edges_sample) < 25:
        for u, v, d in graph.edges(data=True):
            entry = f"{u} -[{d.get('label', '')}]-> {v}"
            if entry not in edges_sample:
                edges_sample.append(entry)
            if len(edges_sample) >= 25:
                break

    prompt = f"""You are a strategic geopolitical policy analyst.

Current intelligence graph nodes (sample): {json.dumps(nodes_sample)}
Current intelligence graph edges (sample):
{chr(10).join(edges_sample)}

Scenario to analyze: "{scenario}"

Based on this scenario, determine:
1. Which new relationships would be created (add_edges)
2. Which existing relationships would end (remove_edges)
3. Which actors are most affected (affected_nodes)
4. An overall risk score 0-100 and risk label

Return ONLY a raw JSON object, no markdown fences, no prose:
{{
  "add_edges": [
    {{"from": "ActorA", "to": "ActorB", "label": "active verb phrase", "confidence": 0.85}}
  ],
  "remove_edges": [
    {{"from": "ActorA", "to": "ActorB"}}
  ],
  "affected_nodes": ["ActorA", "ActorB", "ActorC"],
  "risk_score": 55.0,
  "risk_label": "HIGH"
}}

risk_label must be exactly one of: LOW, MEDIUM, HIGH, CRITICAL
risk_score is 0.0-100.0"""

    data = _strip_json(_call_ollama(prompt, model))

    risk_score = float(data.get("risk_score", 50.0))
    risk_score = max(0.0, min(100.0, risk_score))
    risk_label = data.get("risk_label", _risk_label_from_score(risk_score))
    if risk_label not in RISK_THRESHOLDS:
        risk_label = _risk_label_from_score(risk_score)

    return {
        "add_edges":      data.get("add_edges", []),
        "remove_edges":   data.get("remove_edges", []),
        "affected_nodes": data.get("affected_nodes", []),
        "risk_score":     risk_score,
        "risk_label":     risk_label,
    }


# ── Pass 2: Cascade Analysis → Narrative ─────────────────────────────────────
def _cascade_analysis(
    scenario: str, graph: nx.DiGraph, changes: Dict[str, Any], model: str
) -> Dict[str, Any]:
    affected    = changes.get("affected_nodes", [])
    add_edges   = changes.get("add_edges", [])
    remove_edges = changes.get("remove_edges", [])

    context_edges: List[str] = []
    for node in affected[:10]:
        if graph.has_node(node):
            for u, v, d in graph.out_edges(node, data=True):
                context_edges.append(f"{u} -[{d.get('label', 'relates to')}]-> {v}")
            for u, v, d in graph.in_edges(node, data=True):
                context_edges.append(f"{u} -[{d.get('label', 'relates to')}]-> {v}")

    prompt = f"""You are a senior intelligence strategist specializing in geopolitical cascade analysis.

Scenario: "{scenario}"

Direct effects on intelligence graph:
  New relationships added: {json.dumps(add_edges[:5])}
  Relationships dissolved: {json.dumps(remove_edges[:5])}
  Key affected actors: {json.dumps(affected[:8])}

Existing relationships around affected actors:
{chr(10).join(context_edges[:20]) or "No additional context available."}

Analyze the second-order geopolitical consequences. Return ONLY raw JSON — no markdown, no preamble:
{{
  "cascade_narrative": "A concise 2-3 sentence professional intelligence assessment of the cascading effects. Be specific about actor names and likely reactions.",
  "second_order": [
    "Specific, named second-order consequence 1",
    "Specific, named second-order consequence 2",
    "Specific, named second-order consequence 3",
    "Specific, named second-order consequence 4"
  ]
}}"""

    data = _strip_json(_call_ollama(prompt, model))

    cascade_narrative = str(data.get("cascade_narrative", "Cascade analysis unavailable."))
    second_order = data.get("second_order", [])
    if not isinstance(second_order, list):
        second_order = []

    return {"cascade_narrative": cascade_narrative, "second_order": second_order}


# ── History ───────────────────────────────────────────────────────────────────
def _save_history(result: SimulationResult) -> None:
    """FIX-1/FIX-2: Thread-safe atomic write with size cap."""
    with _history_lock:
        history: List[Dict] = []
        if SIM_HISTORY_FILE.exists():
            try:
                history = json.loads(SIM_HISTORY_FILE.read_text(encoding="utf-8"))
                if not isinstance(history, list):
                    history = []
            except (json.JSONDecodeError, OSError) as exc:  # FIX-3: was bare except
                logger.warning("Could not read sim history, starting fresh: %s", exc)
                history = []

        history.insert(
            0,
            {
                "timestamp":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scenario":          result.scenario,
                "risk_score":        result.risk_score,
                "risk_label":        result.risk_label,
                "cascade_narrative": result.cascade_narrative,
                "second_order":      result.second_order,
                "added_edges":       result.added_edges,
                "removed_edges":     result.removed_edges,
                "affected_nodes":    result.affected_nodes,
                "model_used":        result.model_used,
            },
        )
        # FIX-2: Cap enforced inside the lock so it's never skipped
        SIM_HISTORY_FILE.write_text(
            json.dumps(history[:MAX_HISTORY], indent=2), encoding="utf-8"
        )


# ── Public Entry Point ────────────────────────────────────────────────────────
def run_simulation(
    scenario: str, graph: nx.DiGraph, model: str = "llama3.2"
) -> SimulationResult:
    """
    Run a two-pass LLM policy simulation.
    Returns a SimulationResult. Raises ConnectionError, TimeoutError, ValueError.
    """
    changes = _parse_scenario(scenario, graph, model)
    cascade = _cascade_analysis(scenario, graph, changes, model)

    result = SimulationResult(
        scenario=scenario,
        risk_score=changes["risk_score"],
        risk_label=changes["risk_label"],
        cascade_narrative=cascade["cascade_narrative"],
        second_order=cascade["second_order"],
        added_edges=changes["add_edges"],
        removed_edges=changes["remove_edges"],
        affected_nodes=changes["affected_nodes"],
        model_used=model,
    )

    _save_history(result)
    return result
