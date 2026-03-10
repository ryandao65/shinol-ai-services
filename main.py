"""
AI Services API - Multiple AI services including TTS and Image Generation
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from controllers.tts_controller import router as tts_router
from controllers.image_controller import router as image_router
from controllers.llm_controller import router as llm_router
from controllers.kokoro_controller import router as kokoro_router
from controllers.whisper_controller import router as whisper_router
from controllers.elevenlabs_controller import router as elevenlabs_router
from controllers.script_controller import router as script_router
from controllers.episode_controller import router as episode_router
from controllers.t2t_controller import router as t2t_router

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

# Include routers
app.include_router(tts_router)
app.include_router(image_router)
app.include_router(llm_router)
app.include_router(kokoro_router)
app.include_router(whisper_router)
app.include_router(elevenlabs_router)
app.include_router(script_router)
app.include_router(episode_router)
app.include_router(t2t_router)

# Static files - Output directory for generated audio/images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
app.mount("/", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/")
async def root():
    return {
        "name": "AI Services API",
        "version": "1.0.0",
        "services": {
            "tts": "/tts - Text to Speech (Edge TTS)",
            "tts/elevenlabs": "/tts/elevenlabs - ElevenLabs TTS (config required)",
            "tts/kokoro": "/tts/kokoro - Kokoro TTS (Local)",
            "image": "/image - Image Generation (Stable Diffusion XL)",
            "llm": "/llm - LLM Generation (Ollama)",
            "audio": "/audio - Whisper Transcription",
        },
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
