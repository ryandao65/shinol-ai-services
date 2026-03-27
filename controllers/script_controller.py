"""
YouTube Script Generator - Creates scripts with intro, body, outro
Different voices for intro/outro vs body storytelling
"""
import os
import uuid
import asyncio
import httpx
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from paths import get_output_root

router = APIRouter(prefix="/script", tags=["Script"])

# Output directory for scripts
OUTPUT_DIR = get_output_root()
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ScriptRequest(BaseModel):
    topic: str
    target_duration_minutes: int = 30  # 30-60 minutes
    style: str = "storytelling"  # storytelling, educational, entertaining
    language: str = "en"


class ScriptSection(BaseModel):
    type: str  # intro, body, outro
    content: str
    voice: str
    duration_estimate: str


class ScriptResponse(BaseModel):
    success: bool
    script: List[ScriptSection]
    total_duration_estimate: str
    word_count: int


# Voice mapping
VOICES = {
    "intro_outro": {
        "male": "en-US-AndrewNeural",
        "female": "en-US-AvaNeural",
    },
    "body": {
        "male": "en-US-GuyNeural",
        "female": "en-US-JennyNeural",
    }
}

# LLM endpoint
LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


async def generate_with_llm(prompt: str) -> str:
    """Generate text using Ollama"""
    async with httpx.AsyncClient(timeout=300.0) as client:
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


def estimate_duration(word_count: int, words_per_minute: int = 150) -> str:
    """Estimate audio duration from word count"""
    minutes = word_count / words_per_minute
    mins = int(minutes)
    secs = int((minutes - mins) * 60)
    return f"{mins}m {secs}s"


@router.get("/")
async def root():
    return {
        "message": "YouTube Script Generator",
        "usage": "POST /script/generate with {\"topic\": \"your topic\", \"target_duration_minutes\": 30}"
    }


@router.post("/generate", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    """Generate a YouTube script with intro, body, outro"""
    
    # Calculate word count target (150 words per minute average)
    target_words = request.target_duration_minutes * 150
    
    # Estimate section distribution
    intro_outro_words = int(target_words * 0.1)  # 10% intro + 10% outro
    body_words = target_words - (intro_outro_words * 2)
    
    # Build prompts for each section
    intro_prompt = f"""You are a YouTube script writer. Create an engaging INTRO for a {request.target_duration_minutes}-minute video about: {request.topic}

Requirements:
- Hook the viewer in the first 10 seconds
- Introduce the topic and why it matters
- Set expectations for what they'll learn
- Keep it under {intro_outro_words} words
- Conversational, energetic tone
- NO timestamps or filler

Write only the intro script:"""

    body_prompt = f"""You are a YouTube script writer. Create the MAIN BODY of a YouTube video about: {request.topic}

Requirements:
- Storytelling format with clear narrative flow
- Cover the topic in depth suitable for {request.target_duration_minutes} minutes
- Use stories, examples, and explanations
- Build tension and interest
- Approximately {body_words} words
- Make it engaging and educational
- NO timestamps

Write only the body content:"""

    outro_prompt = f"""You are a YouTube script writer. Create an engaging OUTRO for a YouTube video about: {request.topic}

Requirements:
- Summarize key takeaways
- Call to action (like, subscribe, comment)
- Mention related topics
- Thank the viewer
- Keep it under {intro_outro_words} words
- Warm, closing tone
- NO timestamps

Write only the outro script:"""
    
    try:
        # Generate all sections
        intro_content = await generate_with_llm(intro_prompt)
        body_content = await generate_with_llm(body_prompt)
        outro_content = await generate_with_llm(outro_prompt)
        
        # Calculate word counts
        intro_words = len(intro_content.split())
        body_words_actual = len(body_content.split())
        outro_words = len(outro_content.split())
        total_words = intro_words + body_words_actual + outro_words
        
        # Build response
        script = [
            ScriptSection(
                type="intro",
                content=intro_content.strip(),
                voice=VOICES["intro_outro"]["male"],
                duration_estimate=estimate_duration(intro_words)
            ),
            ScriptSection(
                type="body",
                content=body_content.strip(),
                voice=VOICES["body"]["female"],
                duration_estimate=estimate_duration(body_words_actual)
            ),
            ScriptSection(
                type="outro",
                content=outro_content.strip(),
                voice=VOICES["intro_outro"]["male"],
                duration_estimate=estimate_duration(outro_words)
            )
        ]
        
        # Save script to file
        script_id = f"script_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        script_file = os.path.join(OUTPUT_DIR, f"{script_id}.txt")
        
        with open(script_file, "w") as f:
            f.write(f"TOPIC: {request.topic}\n")
            f.write(f"TARGET DURATION: {request.target_duration_minutes} minutes\n\n")
            for section in script:
                f.write(f"\n=== {section.type.upper()} ===\n")
                f.write(f"Voice: {section.voice}\n")
                f.write(f"Duration: {section.duration_estimate}\n\n")
                f.write(section.content + "\n")
        
        return ScriptResponse(
            success=True,
            script=script,
            total_duration_estimate=estimate_duration(total_words),
            word_count=total_words
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Script generation failed: {str(e)}")
