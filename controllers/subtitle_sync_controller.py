"""
POST /generate_subtitle_sync — desktop + job_queue expect this at API root (not under /audio).

Contract matches shinol-ai-worker: body `{"audio_path": "..."}`, response
`{"status": "done"|"error", "subtitle_path": str|None, "detail": str|None}`.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from urllib.parse import urlparse

from fastapi import APIRouter
from pydantic import BaseModel

from paths import get_output_root

router = APIRouter(tags=["Subtitle"])

_model_lock = threading.Lock()
_models: dict[str, object] = {}


class SubtitleJobRequest(BaseModel):
    audio_path: str


class SubtitleJobResponse(BaseModel):
    status: str
    subtitle_path: str | None = None
    detail: str | None = None


def _format_timestamp_srt(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1_000
    millis %= 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def resolve_audio_fs_path(audio_path: str) -> Path:
    """Map worker URL or local path to a filesystem path under OUTPUT_DIR or absolute path."""
    raw = (audio_path or "").strip()
    if not raw:
        raise ValueError("audio_path is empty")
    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        rel = (parsed.path or "").lstrip("/")
        if not rel:
            raise ValueError("URL has no path component")
        return Path(get_output_root()) / rel.replace("/", os.sep)
    return Path(os.path.normpath(raw))


def _get_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    with _model_lock:
        if model_size not in _models:
            device = os.environ.get("WHISPER_DEVICE", "cpu")
            compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
            _models[model_size] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        return _models[model_size]


def _generate_subtitle_for_audio(audio_path: Path) -> None:
    model_name = os.environ.get("WHISPER_MODEL", os.environ.get("WHISPER_SUBTITLE_MODEL", "base"))
    model = _get_whisper_model(model_name)
    segments, _info = model.transcribe(
        str(audio_path),
        language=None,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )
    srt_lines: list[str] = []
    idx = 0
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        idx += 1
        start = _format_timestamp_srt(float(seg.start))
        end = _format_timestamp_srt(float(seg.end))
        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")
    if not srt_lines:
        raise RuntimeError("Whisper returned no subtitle lines")
    srt_path = audio_path.with_suffix(".srt")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")


@router.post("/generate_subtitle_sync", response_model=SubtitleJobResponse)
async def generate_subtitle_sync(req: SubtitleJobRequest) -> SubtitleJobResponse:
    try:
        audio_path = resolve_audio_fs_path(req.audio_path)
    except ValueError as e:
        return SubtitleJobResponse(status="error", subtitle_path=None, detail=str(e))

    if not audio_path.exists():
        return SubtitleJobResponse(
            status="error",
            subtitle_path=None,
            detail=f"Audio file not found at {audio_path}",
        )

    try:
        await asyncio.to_thread(_generate_subtitle_for_audio, audio_path)
    except Exception as exc:
        return SubtitleJobResponse(
            status="error",
            subtitle_path=None,
            detail=str(exc),
        )

    srt_path = audio_path.with_suffix(".srt")
    if not srt_path.exists():
        return SubtitleJobResponse(
            status="error",
            subtitle_path=None,
            detail=f"Subtitle file was not created for {audio_path}",
        )

    return SubtitleJobResponse(
        status="done",
        subtitle_path=str(srt_path.resolve()),
        detail=None,
    )
