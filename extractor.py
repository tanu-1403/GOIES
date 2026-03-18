"""
extractor.py  —  GOIES Intelligence Extraction Engine
======================================================
Merge resolution: HEAD (v4.x production) + ce28496 (async rewrite)

  - async httpx, CancelToken, parallel chunks, SSE streaming, warmup  → ce28496
  - cross-session deduplication (_global_seen / SEEN_FILE)             → HEAD
  - _parse_llm_json with bracket-counting + greedy-regex fallback      → ce28496
  - check_ollama_health / list_available_models (sync)                 → HEAD
  - configurable OLLAMA_TIMEOUT / OLLAMA_BATCH env vars                → ce28496
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import AsyncIterator, Callable, Optional, Set

import httpx

logger = logging.getLogger("goies.extractor")

# ── Configuration ─────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))  # was hardcoded 120
OLLAMA_BATCH = int(os.getenv("OLLAMA_BATCH", "3"))  # parallel chunks per wave
FUZZY_THRESH = float(os.getenv("FUZZY_THRESH", "0.82"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.50"))
DEFAULT_MODEL = os.getenv("GOIES_DEFAULT_MODEL", "llama3.2")
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "500000"))

CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200

VALID_ENTITY_CLASSES = {
    "country",
    "person",
    "organization",
    "technology",
    "event",
    "treaty",
    "resource",
}

# ── Cross-session deduplication (HEAD v4.x) ───────────────────────────────────

SEEN_FILE = pathlib.Path("extractor_seen.json")
SEEN_MAX_ENTRIES = 50_000
SEEN_FLUSH_BATCH = 200
_seen_lock = threading.Lock()
_global_seen: Set[tuple] = set()
_seen_unflushed: int = 0


def _load_seen() -> None:
    global _global_seen
    if not SEEN_FILE.exists():
        return
    try:
        raw = json.loads(SEEN_FILE.read_text(encoding="utf-8"))
        _global_seen = {
            tuple(item) for item in raw if isinstance(item, list) and len(item) == 2
        }
        logger.info(
            "Loaded %d deduplication keys from %s", len(_global_seen), SEEN_FILE
        )
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("Could not load seen cache (%s) — starting fresh.", exc)
        _global_seen = set()


def _save_seen() -> None:
    try:
        entries = sorted(_global_seen)
        if len(entries) > SEEN_MAX_ENTRIES:
            entries = entries[-SEEN_MAX_ENTRIES:]
        SEEN_FILE.write_text(json.dumps(entries), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist seen cache: %s", exc)


_load_seen()

# ── Data types ────────────────────────────────────────────────────────────────


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


# ── Cancellation token ────────────────────────────────────────────────────────


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


# ── Chunking ──────────────────────────────────────────────────────────────────


def _sentence_chunks(text: str) -> list[str]:
    """Sentence-boundary-aware chunking with overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) + 1 > CHUNK_SIZE and buf:
            chunks.append(buf.strip())
            words = buf.split()
            overlap = " ".join(words[max(0, len(words) - 40) :])
            buf = overlap + " " + sent
        else:
            buf = (buf + " " + sent).strip() if buf else sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a geopolitical intelligence analyst. "
    "Extract entities and relationships from the text. "
    "Return ONLY valid JSON — no markdown, no explanation.\n"
    "Schema:\n"
    '{"entities":[{"id":"<n>","group":"<country|person|organization|'
    'technology|event|treaty|resource>","confidence":<0-1>,"attributes":{}}],'
    '"relationships":[{"from":"<entity_id>","to":"<entity_id>",'
    '"label":"<verb>","confidence":<0-1>}]}'
)

# ── Async Ollama call (non-blocking) ──────────────────────────────────────────


async def _call_ollama_async(
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
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except httpx.TimeoutException:
        logger.warning(
            "Ollama timeout after %.0f s for model %s", OLLAMA_TIMEOUT, model
        )
        raise
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Ollama HTTP error %s: %s",
            exc.response.status_code,
            exc.response.text[:200],
        )
        raise


# ── Sync Ollama call (kept for non-async callers: graph_summary, report) ─────


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    import requests as _req

    resp = _req.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# ── JSON parsing ──────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _parse_llm_json(raw: str) -> dict:
    """
    Robustly extract the first valid JSON object from LLM output.
    Handles markdown fences, prose preamble, and partially-truncated JSON.
    Strategy: bracket-counting (precise) then greedy regex fallback.
    """
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    start = raw.find("{")
    if start == -1:
        return {"entities": [], "relationships": []}
    depth, end = 0, -1
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    candidate = raw[start : end + 1] if end != -1 else ""
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    m = _JSON_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    logger.debug("Could not parse LLM JSON: %s", raw[:200])
    return {"entities": [], "relationships": []}


# ── Fuzzy deduplication ───────────────────────────────────────────────────────


def _fuzzy_match(name: str, known: list[str]) -> Optional[str]:
    for k in known:
        if SequenceMatcher(None, name.lower(), k.lower()).ratio() >= FUZZY_THRESH:
            return k
    return None


# ── Per-chunk async extraction ────────────────────────────────────────────────


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
        raw = await _call_ollama_async(model, prompt, cancel, client)
    except (httpx.TimeoutException, httpx.HTTPStatusError, asyncio.CancelledError):
        raise
    except Exception as exc:
        logger.warning("Chunk %d extraction failed: %s", chunk_index, exc)
        return ChunkResult([], [], chunk_index, time.perf_counter() - t0)

    data = _parse_llm_json(raw)

    entities: list[Entity] = []
    for e in data.get("entities", []):
        eid = str(e.get("id") or "").strip()
        group = str(e.get("group") or "unknown").strip().lower()
        conf = float(e.get("confidence") or 0)
        if not eid or conf < CONF_THRESHOLD:
            continue
        if group not in VALID_ENTITY_CLASSES:
            group = "unknown"
        entities.append(
            Entity(
                id=eid,
                group=group,
                confidence=conf,
                attributes=e.get("attributes") or {},
            )
        )

    relationships: list[Relationship] = []
    for r in data.get("relationships", []):
        frm = str(r.get("from") or "").strip()
        to = str(r.get("to") or "").strip()
        label = str(r.get("label") or "related").strip().lower()
        conf = float(r.get("confidence") or 0)
        if not frm or not to or conf < CONF_THRESHOLD:
            continue
        if frm == to:
            continue
        relationships.append(
            Relationship(from_id=frm, to_id=to, label=label, confidence=conf)
        )

    elapsed = time.perf_counter() - t0
    logger.debug(
        "Chunk %d: %d entities, %d relationships in %.1f s",
        chunk_index,
        len(entities),
        len(relationships),
        elapsed,
    )
    return ChunkResult(entities, relationships, chunk_index, elapsed)


# ── Main async extraction engine ──────────────────────────────────────────────


async def extract_text(
    text: str,
    model: str = DEFAULT_MODEL,
    cancel: Optional[CancelToken] = None,
    on_chunk: Optional[Callable[[ChunkResult], None]] = None,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities and relationships from *text* using *model*.

    Parameters
    ----------
    text     : Raw input text.
    model    : Ollama model name.
    cancel   : CancelToken — call .cancel() from another coroutine to abort.
    on_chunk : Sync callback invoked after each chunk wave (incremental graph updates).
    """
    if cancel is None:
        cancel = CancelToken()

    chunks = _sentence_chunks(text)
    if not chunks:
        return [], []

    logger.info(
        "Extraction started: %d chunks, model=%s, parallel=%d",
        len(chunks),
        model,
        OLLAMA_BATCH,
    )

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    known_ids: list[str] = []

    with _seen_lock:
        effective_seen: Set[tuple] = set(_global_seen)
    new_keys: Set[tuple] = set()

    async with httpx.AsyncClient() as client:
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
                    logger.warning("Chunk error (skipping): %s", result)
                    continue

                for ent in result.entities:
                    key = (ent.group, ent.id.lower())
                    if key in effective_seen:
                        continue
                    effective_seen.add(key)
                    new_keys.add(key)
                    canonical = _fuzzy_match(ent.id, known_ids)
                    if canonical:
                        ent.id = canonical
                    else:
                        known_ids.append(ent.id)
                        all_entities.append(ent)

                id_map = {
                    raw_id: _fuzzy_match(raw_id, known_ids) or raw_id
                    for raw_id in {r.from_id for r in result.relationships}
                    | {r.to_id for r in result.relationships}
                }
                for rel in result.relationships:
                    rel.from_id = id_map.get(rel.from_id, rel.from_id)
                    rel.to_id = id_map.get(rel.to_id, rel.to_id)
                    all_relationships.append(rel)

                if on_chunk:
                    try:
                        on_chunk(result)
                    except Exception as cb_exc:
                        logger.warning("on_chunk callback error: %s", cb_exc)

    if new_keys:
        global _seen_unflushed
        with _seen_lock:
            _global_seen.update(new_keys)
            _seen_unflushed += len(new_keys)
            if _seen_unflushed >= SEEN_FLUSH_BATCH:
                _save_seen()
                _seen_unflushed = 0

    logger.info(
        "Extraction complete: %d entities, %d relationships across %d chunks",
        len(all_entities),
        len(all_relationships),
        len(chunks),
    )
    return all_entities, all_relationships


# ── SSE streaming extraction ──────────────────────────────────────────────────


async def extract_stream(
    text: str,
    model: str = DEFAULT_MODEL,
    cancel: Optional[CancelToken] = None,
) -> AsyncIterator[dict]:
    """
    Async generator yielding SSE-ready dicts as each chunk wave finishes.
    Enables real-time frontend graph updates without waiting for the full document.
    """
    if cancel is None:
        cancel = CancelToken()

    chunks = _sentence_chunks(text)
    total = len(chunks)
    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []
    known_ids: list[str] = []

    with _seen_lock:
        effective_seen: Set[tuple] = set(_global_seen)
    new_keys: Set[tuple] = set()

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

                chunk_new_entities: list[dict] = []
                for ent in result.entities:
                    key = (ent.group, ent.id.lower())
                    if key in effective_seen:
                        continue
                    effective_seen.add(key)
                    new_keys.add(key)
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
                    raw_id: _fuzzy_match(raw_id, known_ids) or raw_id
                    for raw_id in {r.from_id for r in result.relationships}
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

    if new_keys:
        with _seen_lock:
            _global_seen.update(new_keys)
            _save_seen()

    yield {
        "type": "done",
        "total_entities": len(all_entities),
        "total_relationships": len(all_relationships),
        "total_chunks": total,
    }


# ── Model warm-up ─────────────────────────────────────────────────────────────


async def warmup_model(model: str = DEFAULT_MODEL) -> bool:
    """
    Send a 1-token prompt to pre-load model weights into GPU/RAM at startup.
    Avoids cold-start latency on the first real user request.
    """
    logger.info("Warming up model: %s …", model)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
        logger.info("Model %s warmed up successfully.", model)
        return True
    except Exception as exc:
        logger.warning("Model warm-up failed for %s: %s", model, exc)
        return False


# ── Health / model listing (sync, used at server startup) ─────────────────────


def check_ollama_health() -> dict:
    import requests as _req

    try:
        resp = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"online": True, "models": models, "error": None}
    except _req.exceptions.ConnectionError:
        return {
            "online": False,
            "models": [],
            "error": f"Ollama not running at {OLLAMA_BASE_URL}. Start: ollama run {DEFAULT_MODEL}",
        }
    except Exception as exc:
        return {"online": False, "models": [], "error": str(exc)}


def list_available_models() -> list[str]:
    health = check_ollama_health()
    return health["models"] if health["models"] else [DEFAULT_MODEL]
