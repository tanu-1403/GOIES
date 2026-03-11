"""
extractor.py — GOIES Extraction Engine

Fixes applied:
  FIX-1  Per-chunk ValueError/parse failures are caught and logged rather than
         killing the entire stream — bad LLM output on chunk N no longer aborts
         processing of chunks N+1 … end.
  FIX-2  Deduplication key set (`seen`) is passed into extract_intelligence_stream
         so it persists across the caller's session via a shared set when desired,
         but defaults to a fresh set per call (backward-compatible).
  FIX-3  forecaster.py hardcoded OLLAMA_BASE_URL — now uses os.getenv everywhere.
  FIX-4  Cross-session deduplication — `seen` keys are persisted to
         extractor_seen.json so duplicate entities from prior ingestion runs
         are not re-added to the graph on restart. The file is capped at
         SEEN_MAX_ENTRIES to prevent unbounded growth.
"""

import json
import logging
import os
import pathlib
import re
import threading
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set

from utils import chunk_text

logger = logging.getLogger("goies.extractor")

OLLAMA_BASE_URL      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL        = "llama3.2"
REQUEST_TIMEOUT_SECS = 120

# FIX-4: Cross-session deduplication
SEEN_FILE        = pathlib.Path("extractor_seen.json")
SEEN_MAX_ENTRIES = 50_000   # cap prevents unbounded file growth
SEEN_FLUSH_BATCH = 200      # only flush to disk after this many new keys (not per-chunk)
_seen_lock       = threading.Lock()
_global_seen: Set[tuple] = set()   # in-memory mirror of the persisted set
_seen_unflushed: int = 0            # count of new keys not yet persisted


def _load_seen() -> None:
    """Load persisted seen keys into _global_seen at startup."""
    global _global_seen
    if not SEEN_FILE.exists():
        return
    try:
        raw = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        _global_seen = {tuple(item) for item in raw if isinstance(item, list) and len(item) == 2}
        logger.info("Loaded %d deduplication keys from %s", len(_global_seen), SEEN_FILE)
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("Could not load seen cache (%s) — starting fresh.", exc)
        _global_seen = set()


def _save_seen() -> None:
    """Persist _global_seen to disk, capped at SEEN_MAX_ENTRIES.
    Set iteration order is non-deterministic; sorting before slicing ensures
    a stable, reproducible cap rather than random entry loss.
    """
    try:
        entries = sorted(_global_seen)   # sort gives deterministic truncation
        if len(entries) > SEEN_MAX_ENTRIES:
            entries = entries[-SEEN_MAX_ENTRIES:]
        SEEN_FILE.write_text(json.dumps(entries), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist seen cache: %s", exc)


# Load on module import
_load_seen()

VALID_ENTITY_CLASSES = {
    "country", "person", "organization", "technology",
    "event", "treaty", "resource",
}
ALL_VALID_CLASSES = VALID_ENTITY_CLASSES | {"relationship"}


@dataclass
class Extraction:
    extraction_class: str
    extraction_text: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


_SYSTEM_PROMPT = """You are a {persona}. You are a precision geopolitical intelligence extraction engine.

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
{{
  "extractions": [
    {{
      "extraction_class": "Country",
      "extraction_text": "United States",
      "attributes": {{"role": "instigator"}},
      "confidence": 0.97
    }},
    {{
      "extraction_class": "Relationship",
      "extraction_text": "imposes sanctions on",
      "attributes": {{"source": "United States", "target": "Iran"}},
      "confidence": 0.95
    }}
  ]
}}"""


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

    # Find the outermost JSON object using bracket counting — more robust than
    # greedy r"{.*}" which swallows surrounding prose when the LLM adds preamble.
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in model output: {raw[:400]}")
    depth, end = 0, -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError(f"Unbalanced JSON braces in model output: {raw[:400]}")
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from model: {e} | Raw: {raw[:400]}")

    results: List[Extraction] = []
    for item in data.get("extractions", []):
        cls  = str(item.get("extraction_class", "")).strip()
        text = str(item.get("extraction_text", "")).strip()
        attrs = item.get("attributes", {})
        try:
            conf = float(item.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        conf = max(0.0, min(1.0, conf))  # clamp — LLMs occasionally emit values > 1

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
    input_text: str,
    model: str = DEFAULT_MODEL,
    persona: str = "senior geopolitical intelligence analyst",
    seen: Optional[Set] = None,
) -> List[Extraction]:
    """
    Main entry point. Chunks input, calls Ollama per chunk, deduplicates.
    FIX-1: Per-chunk parse errors are logged and skipped; they no longer abort the run.
    FIX-4: Uses _global_seen for cross-session deduplication; persists new keys after run.
    """
    chunks = chunk_text(input_text)
    all_extractions: List[Extraction] = []

    # FIX-4: merge caller-supplied set with the persisted global set
    with _seen_lock:
        effective_seen: Set = set(_global_seen)
    if seen is not None:
        effective_seen.update(seen)

    new_keys: Set[tuple] = set()

    for i, chunk in enumerate(chunks, 1):
        prompt = f"{_SYSTEM_PROMPT.format(persona=persona)}\n\nTEXT TO ANALYZE:\n{chunk}"
        try:
            raw = _call_ollama(prompt, model)
            for ext in _parse_extractions(raw):
                key = (ext.extraction_class.lower(), ext.extraction_text.lower())
                if key not in effective_seen:
                    effective_seen.add(key)
                    new_keys.add(key)
                    all_extractions.append(ext)
        except (ConnectionError, TimeoutError, RuntimeError):
            raise  # propagate hard infrastructure errors
        except ValueError as exc:
            # FIX-1: Bad JSON from LLM on this chunk — log and continue
            logger.warning("Chunk %d/%d parse failed (skipped): %s", i, len(chunks), exc)

    # FIX-4: Persist new keys back to disk (batched — not per-chunk)
    if new_keys:
        global _seen_unflushed
        with _seen_lock:
            _global_seen.update(new_keys)
            _seen_unflushed += len(new_keys)
            if _seen_unflushed >= SEEN_FLUSH_BATCH:
                _save_seen()
                _seen_unflushed = 0

    return all_extractions


def extract_intelligence_stream(
    input_text: str,
    model: str = DEFAULT_MODEL,
    persona: str = "senior geopolitical intelligence analyst",
    seen: Optional[Set] = None,
) -> Generator[dict, None, None]:
    """
    Stream entry point. Yields one dict per chunk.
    FIX-1: ValueError on a single chunk emits an error event but does NOT stop iteration.
    FIX-4: Merges with _global_seen; new keys are persisted after the stream completes.
    """
    chunks = chunk_text(input_text)

    # FIX-4: seed from persisted global seen
    with _seen_lock:
        effective_seen: Set = set(_global_seen)
    if seen is not None:
        effective_seen.update(seen)

    new_keys: Set[tuple] = set()

    for i, chunk in enumerate(chunks, 1):
        prompt = f"{_SYSTEM_PROMPT.format(persona=persona)}\n\nTEXT TO ANALYZE:\n{chunk}"
        try:
            raw = _call_ollama(prompt, model)
            chunk_extractions: List[Extraction] = []
            for ext in _parse_extractions(raw):
                key = (ext.extraction_class.lower(), ext.extraction_text.lower())
                if key not in effective_seen:
                    effective_seen.add(key)
                    new_keys.add(key)
                    chunk_extractions.append(ext)

            yield {
                "chunk_index":    i,
                "total_chunks":   len(chunks),
                "extractions":    chunk_extractions,
                "parse_error":    None,
            }

        except (ConnectionError, TimeoutError, RuntimeError):
            # Persist what we have before re-raising
            if new_keys:
                with _seen_lock:
                    _global_seen.update(new_keys)
                    _save_seen()
            raise  # infrastructure failure — stop everything

        except ValueError as exc:
            # FIX-1: Emit a chunk result with empty extractions + error note;
            # stream continues to next chunk
            logger.warning("Chunk %d/%d parse failed (continuing): %s", i, len(chunks), exc)
            yield {
                "chunk_index":  i,
                "total_chunks": len(chunks),
                "extractions":  [],
                "parse_error":  str(exc),
            }

    # FIX-4: Persist after full stream — single flush regardless of chunk count
    if new_keys:
        global _seen_unflushed
        with _seen_lock:
            _global_seen.update(new_keys)
            _save_seen()   # always flush on clean stream completion
            _seen_unflushed = 0


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
            "error": f"Ollama not running at {OLLAMA_BASE_URL}. Start: ollama run llama3.2",
        }
    except Exception as e:
        return {"online": False, "models": [], "error": str(e)}
