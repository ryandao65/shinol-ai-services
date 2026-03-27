"""
Cached «Nghe thử» samples: one WAV per provider + voice + speed under
`preview_voice_samples/` in the AI services output root.
"""
from __future__ import annotations

import os
import re
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from paths import get_preview_voice_samples_dir
from controllers.kokoro_controller import KOKORO_VOICES, synthesize_kokoro_wav_bytes
from controllers.melo_controller import _tts_to_file_sync
from controllers.voicevox_controller import (
    _VOICEVOX_PING_TEXT,
    _parse_speaker_id,
    _synthesize_voicevox,
    clean_text_for_japanese_tts,
)

# Primary path avoids nested /tts/preview/... which can be shadowed by StaticFiles on "/"
# when a matching folder exists under the output root (POST → 405 on static mount).
router = APIRouter(prefix="/tts", tags=["TTS Preview Samples"])

PREVIEW_TEXT_KOKORO = "こんにちは。これはこころの音声プレビューです。"
PREVIEW_TEXT_KOKORO_EN = "Hello. This is a short Kokoro voice preview."
PREVIEW_TEXT_KOKORO_BRITISH = "Hello. This is a brief British English preview."
PREVIEW_TEXT_MELO = "こんにちは"
PREVIEW_TEXT_EDGE = "Hello. This is a short preview of your selected voice."


def _safe_segment(s: str, max_len: int = 96) -> str:
    t = (s or "").strip() or "default"
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", t)
    return t[:max_len]


def _speed_suffix(speed: float) -> str:
    return f"{float(speed):.2f}".replace(".", "_")


def _cache_file_path(provider: str, voice_key: str, speed: float) -> str:
    base = get_preview_voice_samples_dir()
    sub = os.path.join(base, provider)
    os.makedirs(sub, exist_ok=True)
    fname = f"{_safe_segment(voice_key)}_{_speed_suffix(speed)}.wav"
    return os.path.join(sub, fname)


def _cache_file_path_ext(provider: str, voice_key: str, speed: float, ext: str) -> str:
    base = get_preview_voice_samples_dir()
    sub = os.path.join(base, provider)
    os.makedirs(sub, exist_ok=True)
    e = (ext or "bin").lstrip(".")
    fname = f"{_safe_segment(voice_key)}_{_speed_suffix(speed)}.{e}"
    return os.path.join(sub, fname)


class PreviewSampleRequest(BaseModel):
    provider: Literal["voicevox", "kokoro_82m", "melotts", "edge"]
    voice: str = Field(
        ...,
        description="VoiceVox speaker id, Kokoro voice id, Melo speaker id, or Edge voice id",
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


def _read_wav_response(
    path: str, *, cached: bool, extra_headers: Optional[dict] = None
) -> Response:
    with open(path, "rb") as f:
        data = f.read()
    headers = {
        # Avoid any client/proxy treating identical POST URL as one cached body
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "X-Preview-Sample-Cached": "1" if cached else "0",
    }
    if extra_headers:
        headers.update(extra_headers)
    return Response(content=data, media_type="audio/wav", headers=headers)


def _read_binary_response(
    path: str,
    *,
    cached: bool,
    media_type: str,
    extra_headers: Optional[dict] = None,
) -> Response:
    with open(path, "rb") as f:
        data = f.read()
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "X-Preview-Sample-Cached": "1" if cached else "0",
    }
    if extra_headers:
        headers.update(extra_headers)
    return Response(content=data, media_type=media_type, headers=headers)


def _kokoro_preview_phrase(voice_key: str) -> str:
    info = KOKORO_VOICES.get(voice_key)
    if not info:
        return PREVIEW_TEXT_KOKORO
    lang = info.get("lang")
    if lang == "j":
        return PREVIEW_TEXT_KOKORO
    if lang == "b":
        return PREVIEW_TEXT_KOKORO_BRITISH
    return PREVIEW_TEXT_KOKORO_EN


@router.post("/preview_voice_sample")
@router.post("/preview/sample", include_in_schema=False)
async def preview_sample(request: PreviewSampleRequest):
    """
    Return audio for settings «Nghe thử» (WAV for VoiceVox/Kokoro/Melo, MP3 for Edge).
    Reuses on-disk cache when the file exists for this provider + voice + speed.
    """
    speed = float(request.speed)
    provider = request.provider
    voice_raw = (request.voice or "").strip()

    if provider == "voicevox":
        speaker_id = _parse_speaker_id(voice_raw or "8")
        vk = str(speaker_id)
        path = _cache_file_path("voicevox", vk, speed)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return _read_wav_response(path, cached=True)
        cleaned = clean_text_for_japanese_tts(_VOICEVOX_PING_TEXT)
        if not cleaned:
            raise HTTPException(status_code=400, detail="Preview text is empty after cleaning")
        try:
            wav_bytes = await _synthesize_voicevox(cleaned, speaker_id, speed)
        except HTTPException:
            raise
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return _read_wav_response(path, cached=False)

    if provider == "kokoro_82m":
        # Must match synthesize_kokoro_wav_bytes fallback: same key for path + bytes
        raw = (request.voice or "").strip()
        vk = raw if raw in KOKORO_VOICES else "jf_alpha"
        phrase = _kokoro_preview_phrase(vk)
        # v3: per-language preview phrase (EN/British vs Japanese)
        path = _cache_file_path("kokoro_82m_v3", vk, speed)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return _read_wav_response(
                path,
                cached=True,
                extra_headers={"X-Preview-Voice": vk},
            )
        try:
            wav_bytes = await synthesize_kokoro_wav_bytes(phrase, vk, speed)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Kokoro preview failed: {e}"
            ) from e
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return _read_wav_response(
            path, cached=False, extra_headers={"X-Preview-Voice": vk}
        )

    if provider == "edge":
        import edge_tts

        voice = voice_raw or "en-US-AvaNeural"
        vk = _safe_segment(voice)
        path = _cache_file_path_ext("edge", vk, speed, "mp3")
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return _read_binary_response(
                path, cached=True, media_type="audio/mpeg"
            )
        try:
            communicate = edge_tts.Communicate(PREVIEW_TEXT_EDGE, voice)
            await communicate.save(path)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Edge preview failed: {e}"
            ) from e
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            raise HTTPException(status_code=500, detail="Edge wrote no audio file")
        return _read_binary_response(path, cached=False, media_type="audio/mpeg")

    if provider == "melotts":
        try:
            sid = int(voice_raw) if voice_raw else 0
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Melo preview voice must be a speaker id integer, got: {voice_raw!r}",
            ) from e
        vk = str(sid)
        path = _cache_file_path("melotts", vk, speed)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return _read_wav_response(path, cached=True)
        cleaned = clean_text_for_japanese_tts(PREVIEW_TEXT_MELO)
        if not cleaned:
            raise HTTPException(status_code=400, detail="Preview text is empty after cleaning")
        try:
            await run_in_threadpool(_tts_to_file_sync, cleaned, path, speed, sid)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Melo preview failed: {e}"
            ) from e
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            raise HTTPException(status_code=500, detail="Melo wrote no audio file")
        return _read_wav_response(path, cached=False)

    raise HTTPException(status_code=400, detail="Unknown provider")
