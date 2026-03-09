"""
extractor.py — GOIES Extraction Engine
Sends text to a local Ollama model and returns structured Extraction objects.

v2 improvements:
- Richer entity classes: Country, Person, Organization, Technology, Event, Treaty, Resource
- Confidence scores per extraction
- Chunked processing for long documents
- Dynamic model selection
- Strict JSON parsing with markdown fence stripping
"""

import json
import re
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List

from utils import chunk_text

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
REQUEST_TIMEOUT_SECS = 120

VALID_ENTITY_CLASSES = {
    "country",
    "person",
    "organization",
    "technology",
    "event",
    "treaty",
    "resource",
}
ALL_VALID_CLASSES = VALID_ENTITY_CLASSES | {"relationship"}


@dataclass
class Extraction:
    extraction_class: str
    extraction_text: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


_SYSTEM_PROMPT = """You are a precision geopolitical intelligence extraction engine.

ENTITY CLASSES — use exactly these strings:
  Country      — nation states, governments, regions, blocs (EU, NATO)
  Person       — named individuals, officials, leaders
  Organization — companies, agencies, NGOs, militaries, alliances
  Technology   — technologies, systems, platforms, weapons, chips
  Event        — incidents, conflicts, agreements, elections, sanctions
  Treaty       — formal agreements, accords, treaties, pacts
  Resource     — commodities, energy sources, minerals, currencies

RELATIONSHIP — connects two entities:
  Must include "source" and "target" keys in attributes.
  Use a short, active verb phrase as extraction_text.

CONFIDENCE — rate your certainty 0.0–1.0. Skip anything below 0.5.

OUTPUT RULES:
  1. Return ONLY a raw JSON object. No markdown fences, no prose, no explanation.
  2. extraction_text must be the EXACT phrase from the source text.
  3. Every Relationship MUST have both "source" and "target".

JSON SCHEMA:
{
  "extractions": [
    {
      "extraction_class": "Country",
      "extraction_text": "United States",
      "attributes": {"role": "instigator"},
      "confidence": 0.97
    },
    {
      "extraction_class": "Relationship",
      "extraction_text": "imposes sanctions on",
      "attributes": {"source": "United States", "target": "Iran"},
      "confidence": 0.95
    }
  ]
}"""


def _call_ollama(prompt: str, model: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT_SECS,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            f"Start with: ollama run {model}"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Ollama did not respond within {REQUEST_TIMEOUT_SECS}s.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    return response.json().get("response", "").strip()


def _parse_extractions(raw: str) -> List[Extraction]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output: {raw[:400]}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e} | Raw: {raw[:400]}")

    results: List[Extraction] = []
    for item in data.get("extractions", []):
        cls = str(item.get("extraction_class", "")).strip()
        text = str(item.get("extraction_text", "")).strip()
        attrs = item.get("attributes", {})
        conf = float(item.get("confidence", 1.0))

        if not cls or not text:
            continue
        if cls.lower() not in ALL_VALID_CLASSES:
            continue
        if conf < 0.5:
            continue
        if cls.lower() == "relationship" and (
            not attrs.get("source") or not attrs.get("target")
        ):
            continue

        results.append(
            Extraction(
                extraction_class=cls,
                extraction_text=text,
                attributes=attrs,
                confidence=conf,
            )
        )
    return results


def extract_intelligence(
    input_text: str, model: str = DEFAULT_MODEL
) -> List[Extraction]:
    """
    Main entry point. Chunks input, calls Ollama per chunk, deduplicates results.
    Raises: ConnectionError, TimeoutError, ValueError, RuntimeError
    """
    chunks = chunk_text(input_text)
    all_extractions: List[Extraction] = []
    seen: set = set()

    for chunk in chunks:
        prompt = f"{_SYSTEM_PROMPT}\n\nTEXT TO ANALYZE:\n{chunk}"
        raw = _call_ollama(prompt, model)
        for ext in _parse_extractions(raw):
            key = (ext.extraction_class.lower(), ext.extraction_text.lower())
            if key not in seen:
                seen.add(key)
                all_extractions.append(ext)

    return all_extractions


def list_available_models() -> List[str]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return models if models else [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def check_ollama_health() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"online": True, "models": models, "error": None}
    except requests.exceptions.ConnectionError:
        return {
            "online": False,
            "models": [],
            "error": "Ollama not running. Start: ollama run llama3.2",
        }
    except Exception as e:
        return {"online": False, "models": [], "error": str(e)}
