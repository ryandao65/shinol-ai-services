"""
VoiceVox TTS — Japanese text-to-speech via local VOICEVOX ENGINE HTTP API.

Requires VoiceVox (or compatible engine) running, default http://127.0.0.1:50021
Set VOICEVOX_URL to override.
Long text: chunked audio_query/synthesis + WAV concat (VOICEVOX_CHUNK_MAX_CHARS, default 450).
"""
import io
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Optional

import httpx
import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from paths import get_output_root

router = APIRouter(prefix="/tts/voicevox", tags=["VoiceVox TTS"])

logger = logging.getLogger(__name__)


def _vv_log(msg: str) -> None:
    """Stdout + logger so logs show in the worker terminal (uvicorn) even if root level is WARNING."""
    line = f"[VoiceVox TTS] {msg}"
    print(line, flush=True)
    logger.info("%s", msg)


OUTPUT_DIR = get_output_root()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default engine base URL (VoiceVox / VOICEVOX ENGINE)
VOICEVOX_URL = os.environ.get("VOICEVOX_URL", "http://127.0.0.1:50021").rstrip("/")

# Long scripts: one audio_query per chunk to avoid engine 500/OOM; 0 = disable (single request).
def _voicevox_chunk_max_chars() -> int:
    raw = os.environ.get("VOICEVOX_CHUNK_MAX_CHARS", "450").strip()
    try:
        n = int(raw)
    except ValueError:
        return 450
    return max(0, min(n, 4000))


# Silence between chunks (ms) to reduce boundary clicks; 0 to disable.
def _voicevox_chunk_pause_ms() -> int:
    raw = os.environ.get("VOICEVOX_CHUNK_PAUSE_MS", "100").strip()
    try:
        n = int(raw)
    except ValueError:
        return 100
    return max(0, min(n, 2000))

# Fixed probe text/speaker for GET /tts/voicevox/ping (same path as real TTS: POST /audio_query)
_VOICEVOX_PING_TEXT = "こんにちは"
_VOICEVOX_PING_SPEAKER = 8

# Curated speaker/style IDs (typical VoiceVox core layout; engine GET /speakers is authoritative)
VOICEVOX_DEFAULT_SPEAKERS: list[dict[str, Any]] = [
    {"id": 0, "label": "四国めたん · ノーマル"},
    {"id": 1, "label": "四国めたん · あまあま"},
    {"id": 2, "label": "四国めたん · ツンツン"},
    {"id": 3, "label": "四国めたん · セクシー"},
    {"id": 8, "label": "ずんだもん · ノーマル"},
    {"id": 9, "label": "ずんだもん · あまあま"},
    {"id": 10, "label": "ずんだもん · ツンツン"},
    {"id": 11, "label": "ずんだもん · セクシー"},
    {"id": 16, "label": "春日部つむぎ · ノーマル"},
    {"id": 20, "label": "雨晴はう · ノーマル"},
    {"id": 24, "label": "九州そら · ノーマル"},
    {"id": 28, "label": "もち子 · ノーマル"},
]


def clean_text_for_japanese_tts(text: str) -> str:
    """Normalize text for Japanese TTS — keep Jp/Latin; strip URLs and markup."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    # Preserve newlines for chunk boundaries; collapse spaces within each line only.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _collect_japanese_sentence_units(text: str) -> list[str]:
    """Split into units at 。！？… and at newline (paragraph)."""
    text = text.strip()
    if not text:
        return []
    units: list[str] = []
    buf: list[str] = []
    enders = "。！？…"
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\n":
            seg = "".join(buf).strip()
            buf = []
            if seg:
                units.append(seg)
            i += 1
            while i < len(text) and text[i] == "\n":
                i += 1
            continue
        buf.append(ch)
        if ch in enders:
            seg = "".join(buf).strip()
            buf = []
            if seg:
                units.append(seg)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        units.append(tail)
    return [u for u in units if u]


def _split_oversized_unit(unit: str, max_chars: int) -> list[str]:
    u = unit.strip()
    if not u:
        return []
    if len(u) <= max_chars:
        return [u]
    return [u[i : i + max_chars] for i in range(0, len(u), max_chars)]


def chunk_text_for_voicevox(text: str, max_chars: int) -> list[str]:
    """
    Build TTS chunks <= max_chars, preferring sentence/paragraph boundaries.
    When max_chars <= 0, returns a single chunk (caller should skip chunking).
    """
    if max_chars <= 0:
        t = text.strip()
        return [t] if t else []
    t = text.strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    units: list[str] = []
    for raw in _collect_japanese_sentence_units(t):
        units.extend(_split_oversized_unit(raw, max_chars))

    merged: list[str] = []
    cur = ""
    for u in units:
        if not cur:
            cur = u
        elif len(cur) + len(u) <= max_chars:
            cur += u
        else:
            merged.append(cur)
            cur = u
    if cur:
        merged.append(cur)
    return merged


def _concat_wav_bytes(parts: list[bytes], pause_ms: int) -> bytes:
    """Concatenate WAV payloads from VoiceVox (same engine → same format)."""
    if not parts:
        return b""
    if len(parts) == 1:
        return parts[0]

    arrays: list[np.ndarray] = []
    sr: Optional[int] = None
    for b in parts:
        data, samplerate = sf.read(io.BytesIO(b), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr is None:
            sr = int(samplerate)
        elif int(samplerate) != sr:
            raise ValueError(f"Sample rate mismatch: {samplerate} vs {sr}")
        arrays.append(np.asarray(data, dtype=np.float32))

    silence = (
        np.zeros(int(sr * pause_ms / 1000.0), dtype=np.float32) if pause_ms > 0 else None
    )
    merged_list: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        merged_list.append(arr)
        if silence is not None and i < len(arrays) - 1:
            merged_list.append(silence)
    combined = np.concatenate(merged_list)
    out = io.BytesIO()
    sf.write(out, combined, sr, format="WAV", subtype="PCM_16")
    return out.getvalue()


class VoiceVoxTTSRequest(BaseModel):
    text: str
    voice: str = Field(default="8", description="Speaker/style ID as string (ずんだもん ノーマル default)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    project_code: Optional[str] = None
    episode_code: Optional[str] = None
    output_dir: Optional[str] = None  # Absolute episode dir from desktop app (preferred when provided)


class VoiceVoxPreviewRequest(BaseModel):
    """Short WAV preview for UI — no file on disk."""

    voice: str = Field(default="8", description="Speaker/style ID as string")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    text: Optional[str] = Field(
        default=None,
        description="Sample phrase; default こんにちは",
    )


def _parse_speaker_id(voice: str) -> int:
    try:
        return int(voice.strip())
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Voice must be a numeric speaker ID for VoiceVox, got: {voice!r}",
        ) from e


async def _synthesize_voicevox_with_client(
    client: httpx.AsyncClient,
    text: str,
    speaker_id: int,
    speed: float,
) -> bytes:
    """Single segment: audio_query → synthesis."""
    aq = await client.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": text, "speaker": speaker_id},
    )
    if aq.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=(
                f"VoiceVox audio_query failed ({aq.status_code}): {aq.text[:500]}. "
                f"Is the engine running at {VOICEVOX_URL}?"
            ),
        )
    query = aq.json()
    query["speedScale"] = float(speed)

    syn = await client.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        content=json.dumps(query, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    if syn.status_code >= 400:
        raise HTTPException(
            status_code=500,
            detail=f"VoiceVox synthesis failed ({syn.status_code}): {syn.text[:500]}",
        )
    return syn.content


async def _synthesize_voicevox(
    text: str,
    speaker_id: int,
    speed: float,
) -> bytes:
    """Call VOICEVOX ENGINE: audio_query → synthesis (one segment)."""
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _synthesize_voicevox_with_client(client, text, speaker_id, speed)


async def _synthesize_voicevox_chunked(
    text: str,
    speaker_id: int,
    speed: float,
) -> bytes:
    """Long text: multiple audio_query/synthesis calls, concat WAV."""
    t_total = time.perf_counter()
    max_c = _voicevox_chunk_max_chars()
    pause_ms = _voicevox_chunk_pause_ms()
    if max_c <= 0:
        _vv_log(
            f"chunking disabled (VOICEVOX_CHUNK_MAX_CHARS=0): single request, "
            f"{len(text)} chars, engine={VOICEVOX_URL}"
        )
        out = await _synthesize_voicevox(text, speaker_id, speed)
        _vv_log(
            f"single request done: {len(out)} bytes WAV in {time.perf_counter() - t_total:.1f}s"
        )
        return out

    chunks = chunk_text_for_voicevox(text, max_c)
    if not chunks:
        raise HTTPException(status_code=400, detail="Text is empty after chunking")
    if len(chunks) == 1:
        _vv_log(
            f"one segment (≤{max_c} chars): {len(chunks[0])} chars, engine={VOICEVOX_URL}"
        )
        out = await _synthesize_voicevox(chunks[0], speaker_id, speed)
        _vv_log(
            f"one segment done: {len(out)} bytes WAV in {time.perf_counter() - t_total:.1f}s"
        )
        return out

    active = [c for c in chunks if c.strip()]
    n = len(active)
    _vv_log(
        f"chunked synthesis: {n} chunks, max {max_c} chars/chunk, pause {pause_ms}ms, "
        f"{len(text)} chars total, speaker={speaker_id}, engine={VOICEVOX_URL}"
    )

    timeout = httpx.Timeout(180.0, connect=10.0)
    wav_parts: list[bytes] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for i, ch in enumerate(active, start=1):
            t_chunk = time.perf_counter()
            try:
                wav = await _synthesize_voicevox_with_client(
                    client, ch, speaker_id, speed
                )
                wav_parts.append(wav)
            except HTTPException as e:
                detail = getattr(e, "detail", str(e))
                raise HTTPException(
                    status_code=e.status_code,
                    detail=f"VoiceVox chunk {i}/{n} failed: {detail}",
                ) from e
            _vv_log(
                f"chunk {i}/{n} OK: {len(ch)} chars → {len(wav)} bytes WAV "
                f"({time.perf_counter() - t_chunk:.1f}s)"
            )

    if not wav_parts:
        raise HTTPException(status_code=400, detail="No audio produced from chunks")
    try:
        merged = _concat_wav_bytes(wav_parts, pause_ms)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"VoiceVox WAV merge failed: {e}") from e
    _vv_log(
        f"merge done: {len(merged)} bytes WAV total in {time.perf_counter() - t_total:.1f}s"
    )
    return merged


@router.get("/")
async def root():
    return {
        "message": "VoiceVox TTS (Japanese) — local engine HTTP API",
        "engine_url": VOICEVOX_URL,
        "default_voice": "8",
        "usage": {
            "ping": "GET /tts/voicevox/ping — POST engine /audio_query with probe text (こんにちは) + speaker 8",
            "generate_sync": 'POST /tts/voicevox/generate_sync with {"text": "こんにちは", "voice": "8", "speed": 1.0}',
            "preview": 'POST /tts/voicevox/preview — WAV bytes for settings “test” playback',
            "speakers": "GET /tts/voicevox/speakers — curated IDs; GET /tts/voicevox/engine_speakers — proxy engine",
        },
    }


@router.get("/speakers")
async def list_speakers():
    return {
        "engine_url": VOICEVOX_URL,
        "speakers": VOICEVOX_DEFAULT_SPEAKERS,
    }


@router.get("/ping")
async def ping_voicevox_engine():
    """
    Health check: same first step as real TTS — POST {VOICEVOX_URL}/audio_query with
    fixed Japanese text and speaker 8. Catches connect errors and audio_query 5xx (e.g. CUDA).
    """
    timeout = httpx.Timeout(60.0, connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{VOICEVOX_URL}/audio_query",
                params={
                    "text": _VOICEVOX_PING_TEXT,
                    "speaker": _VOICEVOX_PING_SPEAKER,
                },
            )
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to VoiceVox at {VOICEVOX_URL}: {e}",
        ) from e
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=503,
            detail=f"VoiceVox audio_query timed out at {VOICEVOX_URL}: {e}",
        ) from e

    if r.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=(
                f"VoiceVox audio_query failed ({r.status_code}): {r.text[:500]}. "
                f"Probe text={_VOICEVOX_PING_TEXT!r}, speaker={_VOICEVOX_PING_SPEAKER}"
            ),
        )

    return {
        "ok": True,
        "engine_url": VOICEVOX_URL,
        "message": (
            f"audio_query OK (probe: {_VOICEVOX_PING_TEXT!r}, speaker {_VOICEVOX_PING_SPEAKER})"
        ),
    }


@router.get("/engine_speakers")
async def engine_speakers():
    """Proxy GET /speakers from the VoiceVox engine (full list when engine is running)."""
    timeout = httpx.Timeout(15.0, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{VOICEVOX_URL}/speakers")
            if r.status_code >= 400:
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Engine /speakers error: {r.text[:300]}",
                )
            return r.json()
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to VoiceVox at {VOICEVOX_URL}: {e}",
        ) from e


@router.post("/preview")
async def preview_voice(request: VoiceVoxPreviewRequest):
    """Return synthesized WAV in the response body (for direct browser playback)."""
    raw = (request.text or "").strip() or _VOICEVOX_PING_TEXT
    cleaned = clean_text_for_japanese_tts(raw)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")

    speaker_id = _parse_speaker_id(request.voice)

    try:
        wav_bytes = await _synthesize_voicevox(cleaned, speaker_id, request.speed)
    except HTTPException:
        raise
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"VoiceVox engine not reachable at {VOICEVOX_URL}. ({e})",
        ) from e

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


@router.post("/generate_sync")
async def generate_sync(request: VoiceVoxTTSRequest):
    """Synchronous Japanese TTS — returns same shape as Kokoro generate_sync for the desktop job worker."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    cleaned = clean_text_for_japanese_tts(request.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")

    speaker_id = _parse_speaker_id(request.voice)
    _vv_log(
        f"POST /generate_sync: {len(cleaned)} chars (cleaned), speaker={speaker_id}, "
        f"speed={request.speed}, project={request.project_code!r}, episode={request.episode_code!r}"
    )

    try:
        wav_bytes = await _synthesize_voicevox_chunked(
            cleaned, speaker_id, request.speed
        )
    except HTTPException:
        raise
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=503,
            detail=f"VoiceVox engine not reachable at {VOICEVOX_URL}. Start VoiceVox and ensure VOICEVOX_URL is correct. ({e})",
        ) from e

    if request.output_dir and request.output_dir.strip():
        output_subdir = os.path.join(request.output_dir.strip(), "audio")
    elif request.project_code and request.episode_code:
        output_subdir = os.path.join(
            OUTPUT_DIR, request.project_code, request.episode_code, "audio"
        )
    elif request.project_code:
        output_subdir = os.path.join(OUTPUT_DIR, request.project_code, "audio")
    else:
        output_subdir = os.path.join(OUTPUT_DIR, "tts")

    os.makedirs(output_subdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"voicevox_{ts}_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(output_subdir, filename)

    with open(filepath, "wb") as f:
        f.write(wav_bytes)

    _vv_log(f"wrote {filepath} ({len(wav_bytes)} bytes)")

    if request.project_code and request.episode_code:
        audio_url_path = f"/{request.project_code}/{request.episode_code}/audio/{filename}"
    elif request.project_code:
        audio_url_path = f"/{request.project_code}/audio/{filename}"
    else:
        audio_url_path = f"/audio/{filename}"

    return {
        "success": True,
        "status": "done",
        "audio_url": audio_url_path,
        "audio_path": filepath,
        "voice": str(speaker_id),
    }
