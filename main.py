"""
AI Services API - Multiple AI services including TTS and Image Generation
"""
import os

from paths import get_output_root
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from controllers.tts_controller import router as tts_router
from controllers.image_controller import router as image_router
from controllers.llm_controller import router as llm_router
from controllers.kokoro_controller import router as kokoro_router
from controllers.voicevox_controller import router as voicevox_router
from controllers.melo_controller import router as melo_router
from controllers.whisper_controller import router as whisper_router
from controllers.elevenlabs_controller import router as elevenlabs_router
from controllers.script_controller import router as script_router
from controllers.episode_controller import router as episode_router
from controllers.t2t_controller import router as t2t_router
from controllers.tts_providers_controller import router as tts_providers_router
from controllers.tts_preview_samples_controller import (
    router as tts_preview_samples_router,
)
from controllers.subtitle_sync_controller import router as subtitle_sync_router

app = FastAPI(
    title="AI Services API",
    description="Multi-service AI API including TTS and Image Generation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (preview first: avoids static mount shadowing for POST on some deployments)
app.include_router(tts_preview_samples_router)
app.include_router(tts_router)
app.include_router(image_router)
app.include_router(llm_router)
app.include_router(kokoro_router)
app.include_router(voicevox_router)
app.include_router(melo_router)
app.include_router(whisper_router)
app.include_router(elevenlabs_router)
app.include_router(script_router)
app.include_router(episode_router)
app.include_router(t2t_router)
app.include_router(tts_providers_router)
app.include_router(subtitle_sync_router)


@app.get("/")
async def root():
    return {
        "name": "AI Services API",
        "version": "1.0.0",
        "services": {
            "tts": "/tts - Text to Speech (Edge TTS)",
            "tts/elevenlabs": "/tts/elevenlabs - ElevenLabs TTS (config required)",
            "tts/kokoro": "/tts/kokoro - Kokoro TTS (Local)",
            "tts/voicevox": "/tts/voicevox - VoiceVox TTS (Japanese, local engine)",
            "tts/providers": "/tts/providers/health - Japanese TTS backends availability",
            "tts/preview_voice_sample": (
                "POST /tts/preview_voice_sample — cached «Nghe thử» WAVs (alias: /tts/preview/sample)"
            ),
            "tts/melo": "/tts/melo - MeloTTS JP (POST /tts/melo/generate_sync)",
            "image": "/image - Image Generation (Stable Diffusion XL)",
            "llm": "/llm - LLM Generation (Ollama)",
            "audio": "/audio - Whisper Transcription",
            "subtitle": "POST /generate_subtitle_sync — SRT from audio (faster-whisper)",
        },
        "docs": "/docs"
    }


# Static files MUST be registered last: mount "/" catches any path not matched above.
# If this runs before some routes, or old code lacks a route, requests fall through here → 404.
OUTPUT_DIR = get_output_root()
app.mount("/", StaticFiles(directory=OUTPUT_DIR), name="output")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
