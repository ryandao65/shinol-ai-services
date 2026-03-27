"""
TTS Controller - Text to Speech using Edge TTS
Best English voices with webhook support
"""
import asyncio
import edge_tts
import os
import uuid
import re
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from paths import get_output_root

router = APIRouter(prefix="/tts", tags=["TTS"])

# Best English voices
BEST_VOICES = {
    "female": "en-US-AvaNeural",
    "male": "en-US-AndrewNeural",
    "british_female": "en-GB-SoniaNeural",
    "british_male": "en-GB-RyanNeural",
}

# Output directory
OUTPUT_DIR = get_output_root()
TTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tts")
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)


def _resolve_episode_root(output_dir: Optional[str], project_code: Optional[str], episode_code: Optional[str]) -> str:
    """
    Resolve on-disk episode root.
    - If `output_dir` is provided by desktop app, use it directly.
    - Else fallback to OUTPUT_DIR/{project_code}/{episode_code} (or OUTPUT_DIR for generic requests).
    """
    if output_dir and output_dir.strip():
        return output_dir.strip()
    if project_code and episode_code:
        return os.path.join(OUTPUT_DIR, project_code, episode_code)
    if project_code:
        return os.path.join(OUTPUT_DIR, project_code)
    return OUTPUT_DIR


def clean_text_for_tts(text: str) -> str:
    """Clean text for TTS - remove special characters that can cause issues"""
    # Remove SSML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Replace multiple newlines/spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


class TTSRequest(BaseModel):
    text: str
    voice_type: Optional[str] = "female"  # female, male, british_female, british_male
    voice: Optional[str] = None  # Custom voice short name
    webhook_url: Optional[str] = None
    project_code: Optional[str] = None  # For folder structure
    episode_code: Optional[str] = None  # For folder structure
    filename: Optional[str] = None  # Custom filename for both audio and text
    output_dir: Optional[str] = None  # Absolute episode dir from desktop app (preferred when provided)


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
    print(f"[TTS] Received request - text length: {len(request.text) if request.text else 0}, voice: {request.voice}")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Clean text before generating audio - remove special characters that can cause Edge TTS to fail
    cleaned_text = clean_text_for_tts(request.text)
    
    # Check if text is empty after cleaning
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="Text is empty after removing special characters (URLs, SSML tags, etc.)")
    
    print(f"[TTS] Cleaned text length: {len(cleaned_text)} chars")
    
    # Determine voice
    if request.voice:
        voice = request.voice
    else:
        voice = BEST_VOICES.get(request.voice_type, BEST_VOICES["female"])
    
    # Generate filename - use custom name if provided
    if request.filename:
        audio_filename = f"{request.filename}.mp3"
        txt_filename = f"{request.filename}.txt"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"tts_{timestamp}_{uuid.uuid4().hex[:8]}.mp3"
        txt_filename = None
    
    # Build output path
    if request.project_code or request.episode_code or (request.output_dir and request.output_dir.strip()):
        episode_root = _resolve_episode_root(
            request.output_dir, request.project_code, request.episode_code
        )
        output_subdir = os.path.join(episode_root, "audio")
    else:
        output_subdir = os.path.join(OUTPUT_DIR, "tts")
    
    os.makedirs(output_subdir, exist_ok=True)
    audio_filepath = os.path.join(output_subdir, audio_filename)
    
    try:
        # Create communicate object and save - use cleaned_text instead of request.text
        print(f"[TTS] Generating audio with voice: {voice}")
        
        # Try with requested voice first
        communicate = edge_tts.Communicate(cleaned_text, voice)
        
        # Use asyncio.wait_for to add timeout - 180s for long text (up to ~15 min audio)
        try:
            await asyncio.wait_for(communicate.save(audio_filepath), timeout=180.0)
        except asyncio.TimeoutError:
            raise Exception("TTS generation timed out after 180 seconds - text too long")
        except Exception as te:
            # If the specific voice fails, try with a default voice
            error_msg = str(te)
            if "No audio was received" in error_msg or "voice" in error_msg.lower():
                print(f"[TTS] Voice {voice} failed, trying default voice en-US-AvaNeural")
                default_voice = "en-US-AvaNeural"
                communicate = edge_tts.Communicate(cleaned_text, default_voice)
                await asyncio.wait_for(communicate.save(audio_filepath), timeout=180.0)
                voice = default_voice  # Update voice to the one that worked
                print(f"[TTS] Successfully generated with fallback voice: {default_voice}")
            else:
                raise Exception(f"TTS communication error: {error_msg}")
        
        print(f"[TTS] Audio saved to: {audio_filepath}")
        
        # Check file size
        file_size = os.path.getsize(audio_filepath) if os.path.exists(audio_filepath) else 0
        print(f"[TTS] Audio file size: {file_size} bytes")
        
        # Save text file if filename is provided
        txt_filepath = None
        if txt_filename and (request.project_code or request.episode_code or (request.output_dir and request.output_dir.strip())):
            episode_root = _resolve_episode_root(
                request.output_dir, request.project_code, request.episode_code
            )
            txt_dir = os.path.join(episode_root, "scripts")
            os.makedirs(txt_dir, exist_ok=True)
            txt_filepath = os.path.join(txt_dir, txt_filename)
            with open(txt_filepath, 'w', encoding='utf-8') as f:
                f.write(request.text)
        
        # Build audio_url path: /{project_code}/{episode_code}/audio/filename.mp3
        if request.project_code and request.episode_code:
            audio_url_path = f"/{request.project_code}/{request.episode_code}/audio/{audio_filename}"
        elif request.project_code:
            audio_url_path = f"/{request.project_code}/audio/{audio_filename}"
        else:
            audio_url_path = f"/audio/{audio_filename}"
        
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
            audio_file=audio_filename,
            audio_url=audio_url_path,
            audio_path=audio_filepath,
            voice=voice,
            text=request.text
        )
        
    except Exception as e:
        print(f"[TTS] ERROR: {str(e)}")
        
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


class SaveScriptRequest(BaseModel):
    text: str
    project_code: str
    episode_code: str
    script_name: str = "full_script"
    output_dir: Optional[str] = None  # Absolute episode dir from desktop app (preferred when provided)


class SaveScriptResponse(BaseModel):
    success: bool
    file_path: str
    file_url: str


@router.post("/save_script", response_model=SaveScriptResponse)
async def save_script(request: SaveScriptRequest):
    """Save script text to file"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Build output path
    episode_root = _resolve_episode_root(
        request.output_dir, request.project_code, request.episode_code
    )
    output_dir = os.path.join(episode_root, "scripts")
    os.makedirs(output_dir, exist_ok=True)
    
    txt_filename = f"{request.script_name}.txt"
    filepath = os.path.join(output_dir, txt_filename)
    
    # Write text file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(request.text)
    
    file_url = f"/{request.project_code}/{request.episode_code}/scripts/{txt_filename}"
    
    return SaveScriptResponse(
        success=True,
        file_path=filepath,
        file_url=file_url
    )


@router.get("/get_script")
async def get_script(
    project_code: str,
    episode_code: str,
    script_name: str = "full_script"
):
    """Get saved script content from file"""
    output_dir = os.path.join(OUTPUT_DIR, project_code, episode_code, "scripts")
    txt_filename = f"{script_name}.txt"
    filepath = os.path.join(output_dir, txt_filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Script '{script_name}' not found")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        "success": True,
        "content": content,
        "file_path": filepath
    }


async def send_webhook(url: str, data: dict):
    """Send webhook notification"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            await client.post(url, json=data, timeout=10.0)
    except Exception as e:
        print(f"Webhook failed: {e}")
