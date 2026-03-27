"""
MeloTTS — Japanese (and other languages) text-to-speech via local MeloTTS Python package.

Requires: pip install git+https://github.com/myshell-ai/MeloTTS.git (see requirements.txt)
Japanese: run once in the same Python env as the API: `python -m unidic download` (~500MB; fixes missing mecabrc).
Set MELO_DEVICE to cpu/cuda/mps/auto (default auto).
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from paths import get_output_root
from controllers.voicevox_controller import clean_text_for_japanese_tts

router = APIRouter(prefix="/tts/melo", tags=["MeloTTS"])

logger = logging.getLogger(__name__)

OUTPUT_DIR = get_output_root()
os.makedirs(OUTPUT_DIR, exist_ok=True)

_melo_lock = threading.Lock()
_melo_model: Any = None


def _melo_log(msg: str) -> None:
    line = f"[MeloTTS] {msg}"
    print(line, flush=True)
    logger.info("%s", msg)


def _get_melo_device() -> str:
    return os.environ.get("MELO_DEVICE", "auto").strip() or "auto"


def get_melo_model():
    """Lazy-load JP MeloTTS model once per process."""
    global _melo_model
    if _melo_model is not None:
        return _melo_model
    with _melo_lock:
        if _melo_model is not None:
            return _melo_model
        try:
            from melo.api import TTS
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=(
                    "MeloTTS is not installed. Install with: "
                    "pip install git+https://github.com/myshell-ai/MeloTTS.git "
                    f"(or pip install -r requirements.txt). Import error: {e}"
                ),
            ) from e
        except RuntimeError as e:
            # Common on Windows: UniDic data not extracted (missing dicdir/mecabrc) until:
            #   python -m unidic download
            err = str(e)
            hint = (
                " Run this in the SAME Python environment as the API (not another Python): "
                "`python -m unidic download` (~500MB)."
            )
            if "mecabrc" in err or "dicdir" in err:
                hint = (
                    " Your unidic dictionary is incomplete (often missing mecabrc). "
                    "In the same venv as shinol-ai-services, run: `python -m unidic download`."
                )
            raise HTTPException(
                status_code=503,
                detail=(
                    "MeloTTS Japanese support requires MeCab + UniDic data."
                    + hint
                    + f" Original error: {e}"
                ),
            ) from e
        device = _get_melo_device()
        _melo_log(f"loading MeloTTS language=JP device={device!r} …")
        try:
            _melo_model = TTS(language="JP", device=device)
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"MeloTTS failed to initialize: {e}",
            ) from e
        _melo_log("MeloTTS JP model ready.")
        return _melo_model


def _resolve_speaker_id(model: Any, speaker_id: int) -> int:
    """Map request speaker_id to Melo internal id (0 = default JP from spk2id)."""
    spk2id = getattr(getattr(model.hps, "data", None), "spk2id", None)
    if spk2id is None:
        return speaker_id
    if isinstance(spk2id, dict):
        if speaker_id == 0 and "JP" in spk2id:
            return int(spk2id["JP"])
        return speaker_id
    # OmegaConf / other mappings
    try:
        if speaker_id == 0 and "JP" in spk2id:
            return int(spk2id["JP"])
    except Exception:
        pass
    return speaker_id


def _tts_to_file_sync(
    text: str,
    filepath: str,
    speed: float,
    speaker_id: int,
) -> None:
    model = get_melo_model()
    sid = _resolve_speaker_id(model, speaker_id)
    # Official API: tts_to_file(text, speaker_id, output_path, speed=...)
    model.tts_to_file(text, sid, filepath, speed=speed, quiet=True)


class MeloTTSRequest(BaseModel):
    text: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    speaker_id: int = Field(
        default=0,
        description="0 selects default Japanese speaker (spk2id['JP']); otherwise raw speaker index",
    )
    project_code: Optional[str] = None
    episode_code: Optional[str] = None
    output_dir: Optional[str] = None  # Absolute episode dir from desktop app (preferred when provided)


@router.get("/")
async def root():
    return {
        "message": "MeloTTS (Japanese) — local Python package",
        "device": _get_melo_device(),
        "usage": {
            "generate_sync": (
                'POST /tts/melo/generate_sync with '
                '{"text": "こんにちは", "speed": 1.0, "speaker_id": 0}'
            ),
            "speakers": "GET /tts/melo/speakers — speaker keys / ids for JP model",
        },
        "note": "JP may require: python -m unidic download (see MeloTTS install docs).",
    }


@router.get("/ping")
async def ping():
    """Load JP model once — use for health checks (may take a while on first call)."""
    try:
        await run_in_threadpool(get_melo_model)
        return {
            "ok": True,
            "available": True,
            "device": _get_melo_device(),
            "message": "MeloTTS JP model ready",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/speakers")
async def list_speakers():
    """Expose spk2id for the loaded JP model (for discovery / debugging)."""
    try:

        def _load():
            return get_melo_model()

        model = await run_in_threadpool(_load)
    except HTTPException:
        raise
    spk2id = getattr(getattr(model.hps, "data", None), "spk2id", None)
    out: dict[str, Any] = {"device": _get_melo_device(), "language": "JP"}
    if isinstance(spk2id, dict):
        out["speakers"] = [{"key": k, "id": int(v)} for k, v in spk2id.items()]
    else:
        try:
            items = dict(spk2id) if spk2id is not None else {}
            out["speakers"] = [{"key": k, "id": int(v)} for k, v in items.items()]
        except Exception:
            out["speakers"] = []
            out["raw_spk2id"] = str(spk2id)
    return out


@router.post("/generate_sync")
async def generate_sync(request: MeloTTSRequest):
    """Synchronous Japanese TTS — same response shape as VoiceVox/Kokoro generate_sync."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    cleaned = clean_text_for_japanese_tts(request.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text is empty after cleaning")

    _melo_log(
        f"POST /generate_sync: {len(cleaned)} chars (cleaned), speaker_id={request.speaker_id}, "
        f"speed={request.speed}, project={request.project_code!r}, episode={request.episode_code!r}"
    )

    if request.output_dir and request.output_dir.strip():
        output_subdir = os.path.join(request.output_dir.strip(), "audio")
    elif request.project_code and request.episode_code:
        output_subdir = os.path.join(
            OUTPUT_DIR, request.project_code, request.episode_code, "audio"
        )
    elif request.project_code:
        output_subdir = os.path.join(OUTPUT_DIR, request.project_code, "audio")
    else:
        # Match audio_url /audio/{filename} with static mount on OUTPUT_DIR
        output_subdir = os.path.join(OUTPUT_DIR, "audio")

    os.makedirs(output_subdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"melo_{ts}_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(output_subdir, filename)

    try:
        await run_in_threadpool(
            _tts_to_file_sync,
            cleaned,
            filepath,
            float(request.speed),
            int(request.speaker_id),
        )
    except HTTPException:
        raise
    except Exception as e:
        _melo_log(f"generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MeloTTS synthesis failed: {e}",
        ) from e

    try:
        size = os.path.getsize(filepath)
    except OSError:
        size = 0
    _melo_log(f"wrote {filepath} ({size} bytes)")

    if request.project_code and request.episode_code:
        audio_url_path = f"/{request.project_code}/{request.episode_code}/audio/{filename}"
    elif request.project_code:
        audio_url_path = f"/{request.project_code}/audio/{filename}"
    else:
        audio_url_path = f"/audio/{filename}"

    voice_out = str(request.speaker_id)
    return {
        "success": True,
        "status": "done",
        "audio_url": audio_url_path,
        "audio_path": filepath,
        "voice": voice_out,
    }
