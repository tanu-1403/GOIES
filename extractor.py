"""
extractor.py  —  GOIES Intelligence Extraction Engine
======================================================
Fixes applied:
  #1  Blocking requests.post → async httpx (non-blocking)
  #2  Hardcoded 120 s timeout → configurable via OLLAMA_TIMEOUT env var (default 180 s)
  #3  Task-cancellation token passed through every await point
  #5  Sequential chunk processing → asyncio.gather (parallel)
  #6  Relationship parsing rewritten — robust JSON extraction + edge validation
  #7  Incremental graph updates pushed per-chunk via async callback
  #8  Model warm-up on startup; optional request batching via OLLAMA_BATCH env var
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
<<<<<<< HEAD
import threading
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Set
=======
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import AsyncIterator, Callable, Optional
>>>>>>> ce28496 (v3 initiate)

import httpx

<<<<<<< HEAD
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
=======
log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))  # Fix #2
OLLAMA_BATCH = int(os.getenv("OLLAMA_BATCH", "3"))  # Fix #8: parallel chunks per wave
FUZZY_THRESH = float(os.getenv("FUZZY_THRESH", "0.82"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.50"))

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

ENTITY_GROUPS = {
    "country",
    "person",
    "organization",
    "technology",
    "event",
    "treaty",
    "resource",
>>>>>>> ce28496 (v3 initiate)
}

# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class Entity:
    id: str
    group: str
    confidence: float
    attributes: dict = field(default_factory=dict)


@dataclass
class Relationship:
    from_id: str
    to_id: str
    label: str
    confidence: float


@dataclass
class ChunkResult:
    entities: list[Entity]
    relationships: list[Relationship]
    chunk_index: int
    elapsed: float


# Cancellation token (Fix #3)
class CancelToken:
    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError("Extraction cancelled by caller")


# ── Chunking ─────────────────────────────────────────────────────────────────


def _sentence_chunks(text: str) -> list[str]:
    """Split on sentence boundaries, respecting CHUNK_SIZE with CHUNK_OVERLAP."""
    import re as _re

    sentences = _re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 > CHUNK_SIZE and buf:
            chunks.append(buf.strip())
            # keep overlap from the tail
            words = buf.split()
            overlap = " ".join(words[max(0, len(words) - 40) :])
            buf = overlap + " " + sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# ── Prompt ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a geopolitical intelligence analyst. "
    "Extract entities and relationships from the text. "
    "Return ONLY valid JSON — no markdown, no explanation.\n"
    "Schema:\n"
    '{"entities":[{"id":"<name>","group":"<country|person|organization|'
    'technology|event|treaty|resource>","confidence":<0-1>,'
    '"attributes":{}}],'
    '"relationships":[{"from":"<entity_id>","to":"<entity_id>",'
    '"label":"<verb>","confidence":<0-1>}]}'
)

# ── Ollama call (Fix #1 — async httpx, Fix #2 — configurable timeout) ───────


async def _call_ollama(
    model: str,
    prompt: str,
    cancel: CancelToken,
    client: httpx.AsyncClient,
) -> str:
    cancel.raise_if_cancelled()
    payload = {
        "model": model,
        "prompt": prompt,
        "system": _SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1200},
    }
    try:
        resp = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except httpx.TimeoutException:
        log.warning("Ollama timeout after %.0f s for model %s", OLLAMA_TIMEOUT, model)
        raise
    except httpx.HTTPStatusError as exc:
        log.error(
            "Ollama HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        raise


# ── JSON extraction (Fix #6 — robust parsing) ────────────────────────────────

<<<<<<< HEAD
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
=======
_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_llm_json(raw: str) -> dict:
    """
    Robustly extract the first valid JSON object from LLM output.
    Handles: markdown fences, leading/trailing prose, partial JSON.
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the outermost {...}
    m = _JSON_RE.search(raw)
    if not m:
        return {"entities": [], "relationships": []}
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Attempt bracket-counting repair
        depth = 0
        end = 0
        for i, ch in enumerate(candidate):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            return json.loads(candidate[:end])
        except json.JSONDecodeError:
            log.debug("Could not parse LLM JSON: %s", candidate[:200])
            return {"entities": [], "relationships": []}

>>>>>>> ce28496 (v3 initiate)

# ── Fuzzy deduplication ───────────────────────────────────────────────────────


def _fuzzy_match(name: str, known: list[str]) -> Optional[str]:
    for k in known:
        if SequenceMatcher(None, name.lower(), k.lower()).ratio() >= FUZZY_THRESH:
            return k
    return None


# ── Per-chunk extraction ──────────────────────────────────────────────────────


async def _extract_chunk(
    chunk: str,
    chunk_index: int,
    model: str,
    cancel: CancelToken,
    client: httpx.AsyncClient,
) -> ChunkResult:
    cancel.raise_if_cancelled()
    t0 = time.perf_counter()
    prompt = f"Text:\n{chunk}\n\nExtract all geopolitical entities and relationships."
    try:
        raw = await _call_ollama(model, prompt, cancel, client)
    except (httpx.TimeoutException, httpx.HTTPStatusError, asyncio.CancelledError):
        raise
    except Exception as exc:
        log.warning("Chunk %d extraction failed: %s", chunk_index, exc)
        return ChunkResult([], [], chunk_index, time.perf_counter() - t0)

    data = _parse_llm_json(raw)

    entities: list[Entity] = []
    for e in data.get("entities", []):
        eid = str(e.get("id") or "").strip()
        group = str(e.get("group") or "unknown").strip().lower()
        conf = float(e.get("confidence") or 0)
        if not eid or conf < CONF_THRESHOLD:
            continue
        if group not in ENTITY_GROUPS:
            group = "unknown"
        entities.append(
            Entity(
                id=eid,
                group=group,
                confidence=conf,
                attributes=e.get("attributes") or {},
            )
        )

<<<<<<< HEAD

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
=======
    # Fix #6 — validate both 'from'/'to' fields exist and are non-empty
    relationships: list[Relationship] = []
    for r in data.get("relationships", []):
        frm = str(r.get("from") or "").strip()
        to = str(r.get("to") or "").strip()
        label = str(r.get("label") or "related").strip().lower()
        conf = float(r.get("confidence") or 0)
        if not frm or not to or conf < CONF_THRESHOLD:
            continue
        if frm == to:  # self-loops are noise
            continue
        relationships.append(
            Relationship(from_id=frm, to_id=to, label=label, confidence=conf)
        )
>>>>>>> ce28496 (v3 initiate)

    elapsed = time.perf_counter() - t0
    log.debug(
        "Chunk %d: %d entities, %d relationships in %.1f s",
        chunk_index,
        len(entities),
        len(relationships),
        elapsed,
    )
    return ChunkResult(entities, relationships, chunk_index, elapsed)


<<<<<<< HEAD
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
=======
# ── Main extraction engine ────────────────────────────────────────────────────


async def extract_text(
    text: str,
    model: str = "llama3.2",
    cancel: Optional[CancelToken] = None,
    # Fix #7: async callback for incremental graph updates
    on_chunk: Optional[Callable[[ChunkResult], None]] = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities and relationships from *text* using *model*.

    Parameters
    ----------
    text     : Raw input text.
    model    : Ollama model name.
    cancel   : CancelToken — call .cancel() from another coroutine to abort.
    on_chunk : Optional async-compatible callback invoked after each chunk
               completes, enabling incremental graph updates (Fix #7).

    Returns
    -------
    (entities, relationships) deduplicated across all chunks.
    """
    if cancel is None:
        cancel = CancelToken()

    chunks = _sentence_chunks(text)
    if not chunks:
        return [], []

    log.info(
        "Extraction started: %d chunks, model=%s, parallel=%d",
        len(chunks),
        model,
        OLLAMA_BATCH,
    )

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    known_ids: list[str] = []

    # Fix #1, #5: async client shared across all parallel chunk calls
    async with httpx.AsyncClient() as client:
        # Fix #5: process in waves of OLLAMA_BATCH parallel chunks
        for wave_start in range(0, len(chunks), OLLAMA_BATCH):
            cancel.raise_if_cancelled()
            wave = chunks[wave_start : wave_start + OLLAMA_BATCH]
            tasks = [
                _extract_chunk(chunk, wave_start + i, model, cancel, client)
                for i, chunk in enumerate(wave)
            ]
            results: list[ChunkResult] = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            for result in results:
                if isinstance(result, Exception):
                    log.warning("Chunk error (skipping): %s", result)
                    continue

                # Deduplicate entities via fuzzy matching
                for ent in result.entities:
                    canonical = _fuzzy_match(ent.id, known_ids)
                    if canonical:
                        ent.id = canonical
                    else:
                        known_ids.append(ent.id)
                        all_entities.append(ent)

                # Remap relationship IDs to canonical names
                id_map = {
                    raw: _fuzzy_match(raw, known_ids) or raw
                    for raw in {r.from_id for r in result.relationships}
                    | {r.to_id for r in result.relationships}
                }
                for rel in result.relationships:
                    rel.from_id = id_map.get(rel.from_id, rel.from_id)
                    rel.to_id = id_map.get(rel.to_id, rel.to_id)
                    all_relationships.append(rel)

                # Fix #7: incremental graph update after each chunk
                if on_chunk:
                    try:
                        on_chunk(result)
                    except Exception as cb_exc:
                        log.warning("on_chunk callback error: %s", cb_exc)

    log.info(
        "Extraction complete: %d entities, %d relationships across %d chunks",
        len(all_entities),
        len(all_relationships),
        len(chunks),
    )
    return all_entities, all_relationships
>>>>>>> ce28496 (v3 initiate)


# ── SSE streaming extraction (Fix #7 extended) ───────────────────────────────


async def extract_stream(
    text: str,
    model: str = "llama3.2",
    cancel: Optional[CancelToken] = None,
) -> AsyncIterator[dict]:
    """
    Async generator yielding SSE-ready dicts as each chunk finishes.
    Enables real-time frontend graph updates.
    """
    if cancel is None:
        cancel = CancelToken()

    chunks = _sentence_chunks(text)
    total = len(chunks)
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    known_ids: list[str] = []

    yield {"type": "start", "total_chunks": total, "model": model}

    async with httpx.AsyncClient() as client:
        for wave_start in range(0, total, OLLAMA_BATCH):
            cancel.raise_if_cancelled()
            wave = chunks[wave_start : wave_start + OLLAMA_BATCH]
            tasks = [
                _extract_chunk(chunk, wave_start + i, model, cancel, client)
                for i, chunk in enumerate(wave)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    yield {"type": "error", "message": str(result)}
                    continue

                # Deduplicate
                chunk_new_entities: list[dict] = []
                for ent in result.entities:
                    canonical = _fuzzy_match(ent.id, known_ids)
                    if canonical:
                        ent.id = canonical
                    else:
                        known_ids.append(ent.id)
                        all_entities.append(ent)
                        chunk_new_entities.append(
                            {
                                "id": ent.id,
                                "group": ent.group,
                                "confidence": ent.confidence,
                                "attributes": ent.attributes,
                            }
                        )

                id_map = {
                    raw: _fuzzy_match(raw, known_ids) or raw
                    for raw in {r.from_id for r in result.relationships}
                    | {r.to_id for r in result.relationships}
                }
                chunk_new_rels: list[dict] = []
                for rel in result.relationships:
                    rel.from_id = id_map.get(rel.from_id, rel.from_id)
                    rel.to_id = id_map.get(rel.to_id, rel.to_id)
                    all_relationships.append(rel)
                    chunk_new_rels.append(
                        {
                            "from": rel.from_id,
                            "to": rel.to_id,
                            "label": rel.label,
                            "confidence": rel.confidence,
                        }
                    )

                # Fix #7: yield incremental update per chunk
                yield {
                    "type": "chunk",
                    "chunk_index": result.chunk_index,
                    "elapsed": round(result.elapsed, 2),
                    "new_entities": chunk_new_entities,
                    "new_relationships": chunk_new_rels,
                    "totals": {
                        "entities": len(all_entities),
                        "relationships": len(all_relationships),
                        "chunks_done": result.chunk_index + 1,
                        "chunks_total": total,
                    },
                }

    yield {
        "type": "done",
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
        "total_chunks": total,
    }


# ── Model warm-up (Fix #8) ───────────────────────────────────────────────────


async def warmup_model(model: str = "llama3.2") -> bool:
    """
    Send a minimal prompt to load the model weights into GPU/RAM.
    Call once at server startup — avoids cold-start latency on first user request.
    """
    log.info("Warming up model: %s …", model)
    try:
<<<<<<< HEAD
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
=======
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
        log.info("Model %s warmed up successfully.", model)
        return True
    except Exception as exc:
        log.warning("Model warm-up failed for %s: %s", model, exc)
        return False
>>>>>>> ce28496 (v3 initiate)
