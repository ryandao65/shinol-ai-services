"""
TTS Controller - Text to Speech using Edge TTS
Best English voices with webhook support
"""
import asyncio
import edge_tts
import os
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/tts", tags=["TTS"])

# Best English voices
BEST_VOICES = {
    "female": "en-US-AvaNeural",
    "male": "en-US-AndrewNeural",
    "british_female": "en-GB-SoniaNeural",
    "british_male": "en-GB-RyanNeural",
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
TTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tts")
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


class TTSRequest(BaseModel):
    text: str
    voice_type: Optional[str] = "female"  # female, male, british_female, british_male
    voice: Optional[str] = None  # Custom voice short name
    webhook_url: Optional[str] = None
    project_code: Optional[str] = None  # For folder structure
    episode_code: Optional[str] = None  # For folder structure


class TTSResponse(BaseModel):
    success: bool
    audio_file: str
    audio_url: str
    audio_path: Optional[str] = None  # Local file path for desktop app
    voice: str
    text: str


@router.get("/")
async def root():
    return {
        "message": "TTS Service - Edge TTS",
        "best_voices": BEST_VOICES,
        "usage": {
            "simple": "POST /tts/generate with {\"text\": \"Hello!\", \"voice_type\": \"female\"}",
            "custom": "POST /tts/generate with {\"text\": \"Hello!\", \"voice\": \"en-US-AvaNeural\"}",
        }
    }


@router.get("/voices")
async def list_voices():
    """List all available English voices"""
    voices = await edge_tts.list_voices()
    english_neural = [
        {
            "name": v['Name'],
            "short_name": v['ShortName'],
            "gender": v['Gender'],
            "locale": v['Locale']
        }
        for v in voices 
        if v['Locale'].startswith('en') and 'Neural' in v['Name']
    ]
    return {
        "count": len(english_neural),
        "voices": english_neural,
        "recommended": BEST_VOICES
    }


@router.post("/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """Generate TTS audio"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Determine voice
    if request.voice:
        voice = request.voice
    else:
        voice = BEST_VOICES.get(request.voice_type, BEST_VOICES["female"])
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}.mp3"
    
    # Build output path: output/{project_code}/{episode_code}/audio/filename.mp3
    if request.project_code and request.episode_code:
        output_subdir = os.path.join(OUTPUT_DIR, request.project_code, request.episode_code, "audio")
    elif request.project_code:
        output_subdir = os.path.join(OUTPUT_DIR, request.project_code, "audio")
    else:
        output_subdir = os.path.join(OUTPUT_DIR, "tts")
    
    os.makedirs(output_subdir, exist_ok=True)
    filepath = os.path.join(output_subdir, filename)
    
    try:
        # Create communicate object and save
        communicate = edge_tts.Communicate(request.text, voice)
        await communicate.save(filepath)
        
        # Build audio_url path: /{project_code}/{episode_code}/audio/filename.mp3
        if request.project_code and request.episode_code:
            audio_url_path = f"/{request.project_code}/{request.episode_code}/audio/{filename}"
        elif request.project_code:
            audio_url_path = f"/{request.project_code}/audio/{filename}"
        else:
            audio_url_path = f"/audio/{filename}"
        
        # Send webhook if provided
        if request.webhook_url:
            await send_webhook(request.webhook_url, {
                "success": True,
                "audio_url": audio_url_path,
                "voice": voice,
                "text": request.text
            })
        
        return TTSResponse(
            success=True,
            audio_file=filename,
            audio_url=audio_url_path,
            audio_path=filepath,  # Local file path for desktop app
            voice=voice,
            text=request.text
        )
        
    except Exception as e:
        # Send error webhook
        if request.webhook_url:
            await send_webhook(request.webhook_url, {
                "success": False,
                "error": str(e),
                "text": request.text
            })
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@router.get("/audio/{path:path}")
async def get_audio(path: str):
    """Serve the generated audio file"""
    from fastapi.responses import FileResponse
    
    # Try multiple possible locations
    # 1. output/tts/filename.mp3 (old structure)
    # 2. output/{project_code}/audio/filename.mp3
    # 3. output/{project_code}/{episode_code}/audio/filename.mp3
    
    possible_paths = [
        os.path.join(OUTPUT_DIR, "tts", path.split("/")[-1]),  # output/tts/filename
        os.path.join(OUTPUT_DIR, path),  # output/project/episode/audio/filename
    ]
    
    # Try to find the file
    filepath = None
    for p in possible_paths:
        if os.path.exists(p):
            filepath = p
            break
    
    if not filepath:
        # Try direct path
        filepath = os.path.join(OUTPUT_DIR, "tts", path)
        if not os.path.exists(filepath):
            filepath = os.path.join(OUTPUT_DIR, path)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        filepath, 
        media_type="audio/mpeg",
        headers={"Content-Disposition": f"attachment; filename={path.split('/')[-1]}"}
    )


@router.delete("/audio/{filename}")
async def delete_audio(filename: str):
    """Delete an audio file"""
    filepath = os.path.join(TTS_OUTPUT_DIR, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"success": True, "message": f"Deleted {filename}"}
    
    raise HTTPException(status_code=404, detail="File not found")


async def send_webhook(url: str, data: dict):
    """Send webhook notification"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=data, timeout=10.0)
    except Exception as e:
        print(f"Webhook failed: {e}")
