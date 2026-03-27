"""
Episode Enhancement API - Add intro/outro/comments to existing episodes
"""
import os
import uuid
import asyncio
import httpx
from datetime import datetime
from typing import Optional, List, Dict, Any
from paths import get_output_root
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/episode", tags=["Episode Enhancement"])

# Output directory
OUTPUT_DIR = get_output_root()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM endpoint
LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


class AddIntroRequest(BaseModel):
    episode_id: str
    body_content: str
    theme: str = "American Family Drama for Elderly"
    voice: str = "en-US-AndrewNeural"


class AddOutroRequest(BaseModel):
    episode_id: str
    body_content: str
    intro_content: Optional[str] = None
    theme: str = "American Family Drama for Elderly"
    voice: str = "en-US-AndrewNeural"


class AddCommentsRequest(BaseModel):
    episode_id: str
    body_content: str
    intro_content: Optional[str] = None
    outro_content: Optional[str] = None
    target_duration_minutes: int = 30
    theme: str = "American Family Drama for Elderly"
    host_voice: str = "en-US-AndrewNeural"
    storyteller_voice: str = "en-US-JennyNeural"


class MergeAudioRequest(BaseModel):
    episode_id: str
    segment_paths: List[str]  # List of audio file paths in order
    background_music: Optional[str] = None  # Path to background music file


async def generate_with_llm(prompt: str) -> str:
    """Generate text using Ollama or fallback"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                LLM_URL,
                json={
                    "model": DEFAULT_MODEL,
                    "prompt": prompt,
                    "stream": False
                }
            )
            result = response.json()
            return result.get("response", "")
    except Exception as e:
        print(f"LLM call failed: {e}. Using fallback response.")
        # Fallback: return a simple generated text
        if "intro" in prompt.lower():
            return "Welcome to another episode of our family drama series. Today, we explore a story about life transitions and finding purpose in later years. Get ready for an emotional journey."
        elif "outro" in prompt.lower():
            return "Thank you for joining us for this heartfelt story. Remember to subscribe for more family dramas and share your thoughts in the comments below."
        elif "comment" in prompt.lower():
            return "This moment really highlights the emotional depth of the story. It's a powerful reminder of the challenges many face."
        else:
            return "Generated content based on the provided story."


@router.post("/add_intro")
async def add_intro(request: AddIntroRequest):
    """Generate intro for an existing episode with body content"""
    
    prompt = f"""You are a YouTube host creating an INTRO for an episode about: {request.theme}

EPISODE BODY CONTENT:
{request.body_content}

Requirements:
- Hook the viewer in the first 10 seconds
- Introduce the theme and what they'll hear
- Build excitement and curiosity
- Keep it under 150 words
- Conversational, energetic tone
- End with a smooth transition to the main story

Write only the intro script:"""
    
    try:
        intro_content = await generate_with_llm(prompt)
        
        # Generate TTS audio
        tts_response = await generate_tts(intro_content, request.voice, request.episode_id, "intro")
        
        return {
            "success": True,
            "episode_id": request.episode_id,
            "intro_content": intro_content.strip(),
            "intro_audio": tts_response["audio_file"],
            "intro_url": tts_response["audio_url"],
            "voice": request.voice
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Intro generation failed: {str(e)}")


@router.post("/add_outro")
async def add_outro(request: AddOutroRequest):
    """Generate outro for an existing episode"""
    
    prompt = f"""You are a YouTube host creating an OUTRO for an episode about: {request.theme}

EPISODE BODY CONTENT:
{request.body_content}

{"EPISODE INTRO CONTENT (if available):" + request.intro_content if request.intro_content else ""}

Requirements:
- Summarize key takeaways from the story
- Thank the viewer for watching
- Call to action (like, subscribe, comment)
- Mention related topics or next episodes
- Keep it under 150 words
- Warm, closing tone

Write only the outro script:"""
    
    try:
        outro_content = await generate_with_llm(prompt)
        
        # Generate TTS audio
        tts_response = await generate_tts(outro_content, request.voice, request.episode_id, "outro")
        
        return {
            "success": True,
            "episode_id": request.episode_id,
            "outro_content": outro_content.strip(),
            "outro_audio": tts_response["audio_file"],
            "outro_url": tts_response["audio_url"],
            "voice": request.voice
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outro generation failed: {str(e)}")


@router.post("/add_comments")
async def add_comments(request: AddCommentsRequest):
    """Generate host comments at strategic points in the body content"""
    
    # Split body content into paragraphs
    paragraphs = [p.strip() for p in request.body_content.split('\n\n') if p.strip()]
    
    # Determine where to insert comments (every 3-5 paragraphs)
    comment_positions = []
    if len(paragraphs) >= 3:
        # Insert comments at strategic points
        step = max(3, len(paragraphs) // 3)  # 3 comments max
        for i in range(step, len(paragraphs), step):
            if i < len(paragraphs):  # Ensure we don't go out of bounds
                comment_positions.append(i)
    
    if not comment_positions:
        # If body is short, add one comment in the middle
        middle = len(paragraphs) // 2
        if middle > 0:
            comment_positions.append(middle)
    
    comments = []
    for pos in comment_positions:
        print(f"Generating comment at position {pos}")
        context_paragraph = paragraphs[pos] if pos < len(paragraphs) else paragraphs[-1]
        
        prompt = f"""You are a YouTube host adding a COMMENTARY to a story about: {request.theme}

STORY PARAGRAPH (what the storyteller just said):
{context_paragraph}

Requirements:
- Add insightful commentary or reaction
- Connect to broader theme: {request.theme}
- Keep it brief (50-100 words)
- Natural, conversational tone
- Transition smoothly back to the story

Write only the host comment:"""
        
        try:
            comment_content = await generate_with_llm(prompt)
            print(f"Generated comment: {comment_content[:50]}...")
            
            # Generate TTS audio
            tts_response = await generate_tts(
                comment_content, 
                request.host_voice, 
                request.episode_id, 
                f"comment_{pos}"
            )
            
            comments.append({
                "position": pos,
                "position_percentage": int((pos / len(paragraphs)) * 100),
                "content": comment_content.strip(),
                "audio": tts_response["audio_file"],
                "audio_url": tts_response["audio_url"],
                "voice": request.host_voice
            })
            
        except Exception as e:
            print(f"Failed to generate comment at position {pos}: {e}")
    
    print(f"Total comments generated: {len(comments)}")
    
    return {
        "success": True,
        "episode_id": request.episode_id,
        "comments": comments,
        "total_comments": len(comments),
        "host_voice": request.host_voice
    }


async def generate_tts(text: str, voice: str, episode_id: str, segment_type: str):
    """Generate TTS audio using ASA's TTS endpoint"""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "http://localhost:8000/tts/generate",
            json={
                "text": text,
                "voice": voice,
                "output_dir": os.path.join(OUTPUT_DIR, episode_id)
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"TTS generation failed: {response.text()}"
            )
        
        return response.json()


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify the module is working"""
    return {
        "message": "Episode Enhancement API is working",
        "endpoints": [
            "POST /episode/add_intro - Generate intro for existing episode",
            "POST /episode/add_outro - Generate outro",
            "POST /episode/add_comments - Generate host comments",
            "POST /episode/merge_audio - Combine audio segments"
        ]
    }