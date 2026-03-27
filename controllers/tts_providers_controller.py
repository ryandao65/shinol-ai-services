"""
Aggregated health for Japanese TTS backends (VoiceVox, Kokoro 82M, MeloTTS).
Used by the desktop app before generating episodes.
"""
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/tts/providers", tags=["TTS providers"])


def _http_detail_str(exc: HTTPException) -> str:
    d = exc.detail
    if isinstance(d, str):
        return d
    return str(d)


async def _voicevox_health():
    try:
        from controllers.voicevox_controller import ping_voicevox_engine

        extra = await ping_voicevox_engine()
        return {"available": True, "detail": None, "extra": extra}
    except HTTPException as e:
        return {"available": False, "detail": _http_detail_str(e)}


async def _kokoro_health():
    try:
        from controllers.kokoro_controller import worker

        await worker.initialize()
        return {"available": True, "detail": None}
    except Exception as e:
        return {"available": False, "detail": str(e)}


async def _melotts_health():
    """True when MeloTTS JP package loads (same as GET /tts/melo/ping)."""
    try:
        from controllers.melo_controller import get_melo_model
        from starlette.concurrency import run_in_threadpool

        await run_in_threadpool(get_melo_model)
        return {"available": True, "detail": None}
    except HTTPException as e:
        return {"available": False, "detail": _http_detail_str(e)}
    except Exception as e:
        return {"available": False, "detail": str(e)}


@router.get("/health")
async def providers_health():
    """Return availability for each backend; `available` is True only when the service can synthesize."""
    voicevox = await _voicevox_health()
    kokoro = await _kokoro_health()
    melotts = await _melotts_health()
    return {"voicevox": voicevox, "kokoro": kokoro, "melotts": melotts}
