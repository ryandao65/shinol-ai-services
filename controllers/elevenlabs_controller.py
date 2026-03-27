"""
ElevenLabs TTS Controller - Cloud TTS Alternative to Edge TTS
"""
import os
import uuid
import time
import requests
from enum import Enum
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from paths import get_output_root

router = APIRouter(prefix="/tts/elevenlabs", tags=["ElevenLabs TTS"])

# API Key from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Output directory
OUTPUT_DIR = os.path.join(get_output_root(), "tts_elevenlabs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class JobStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ElevenLabsTTSRequest(BaseModel):
    text: str
    voice_id: str = "pNInz6obpgDQGcFmaJgB"  # Adam voice (default)
    model: str = "eleven_monolingual_v1"  # or "eleven_multilingual_v2"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    webhook_url: Optional[str] = None


class ElevenLabsTTSResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    audio_url: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


# Popular ElevenLabs voices
ELEVENLABS_VOICES = {
    "pNInz6obpgDQGcFmaJgB": "Adam (Male)",
    "21m00Tcm4TlvDq8ikWAM": "Rachel (Female)",
    "AZnzlk1XvdzGbLyFaQEy": "Domi (Female)",
    "MFtmM23C0NxN32dDakYC": "Sarah (Female)",
    "CwhRBWXzGAHq8TQ4Fs17": "Roger (Male)",
    "bVMeCyTHy58xnoLQDWiN": "Thomas (Male)",
    "zq4A91L4wDfyh6j6D3i3": "Michael (Male)",
    "EXAVITQu4vr4xnSDxMaL": "Sarah (Female)",
}


# ============ ROUTES ============

@router.get("/")
async def root():
    if not ELEVENLABS_API_KEY:
        return {
            "message": "ElevenLabs TTS Service (Not configured)",
            "status": "missing_api_key",
            "setup": "Set ELEVENLABS_API_KEY environment variable",
            "voices": ELEVENLABS_VOICES,
        }
    
    return {
        "message": "ElevenLabs TTS Service",
        "status": "ready",
        "api_key_configured": True,
        "voices": ELEVENLABS_VOICES,
        "default_voice": "pNInz6obpgDQGcFmaJgB",
        "models": [
            "eleven_monolingual_v1",
            "eleven_multilingual_v2",
            "eleven_flash_v2_5",
        ],
    }


@router.get("/voices")
async def list_voices():
    """List available voices"""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs API key not configured. Set ELEVENLABS_API_KEY environment variable."
        )
    
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            timeout=30
        )
        response.raise_for_status()
        
        voices = response.json().get("voices", [])
        return {
            "voices": [
                {
                    "voice_id": v["voice_id"],
                    "name": v.get("name", "Unknown"),
                    "category": v.get("category", "unknown"),
                }
                for v in voices
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch voices: {str(e)}")


@router.post("/generate", response_model=ElevenLabsTTSResponse)
async def generate_speech(request: ElevenLabsTTSRequest):
    """Generate speech using ElevenLabs API"""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs API key not configured. Set ELEVENLABS_API_KEY environment variable."
        )
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    job_id = f"eleven_{uuid.uuid4().hex[:12]}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"eleven_{timestamp}_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        # Call ElevenLabs API
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": request.text,
                "model_id": request.model,
                "voice_settings": {
                    "stability": request.stability,
                    "similarity_boost": request.similarity_boost,
                    "style": request.style,
                },
            },
            timeout=120,
        )
        
        response.raise_for_status()
        
        # Save audio file
        with open(filepath, "wb") as f:
            f.write(response.content)
        
        audio_url = f"/tts/elevenlabs/audio/{filename}"
        
        # Send webhook if provided
        if request.webhook_url:
            try:
                requests.post(
                    request.webhook_url,
                    json={
                        "success": True,
                        "job_id": job_id,
                        "audio_url": audio_url,
                        "voice_id": request.voice_id,
                        "text": request.text,
                    },
                    timeout=30
                )
            except:
                pass
        
        return ElevenLabsTTSResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
        )
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"ElevenLabs API error: {e.response.status_code}"
        try:
            error_data = e.response.json()
            error_msg = error_data.get("detail", error_msg)
        except:
            pass
        
        if request.webhook_url:
            try:
                requests.post(
                    request.webhook_url,
                    json={"success": False, "job_id": job_id, "error": error_msg},
                    timeout=30
                )
            except:
                pass
        
        raise HTTPException(status_code=502, detail=error_msg)
    
    except Exception as e:
        if request.webhook_url:
            try:
                requests.post(
                    request.webhook_url,
                    json={"success": False, "job_id": job_id, "error": str(e)},
                    timeout=30
                )
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Download generated audio"""
    from fastapi.responses import FileResponse
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(
        filepath,
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
