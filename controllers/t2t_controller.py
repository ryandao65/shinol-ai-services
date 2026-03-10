"""
Text-to-Prompt Controller - Generate image prompts from content
"""
import os
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/t2t", tags=["Text-to-Prompt"])

# LLM endpoint
LLM_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:3b"


class T2TRequest(BaseModel):
    content: str
    instruction: str = "Tạo ảnh từ nội dung"
    split_ideas: bool = False
    ideas_count: int = 10


class T2TResponse(BaseModel):
    success: bool
    prompt: str
    ideas: Optional[List[str]] = None


async def generate_with_llm(prompt: str) -> str:
    """Generate text using Ollama"""
    import httpx
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


@router.get("/")
async def root():
    return {
        "message": "Text-to-Prompt Service",
        "usage": {
            "generate": "POST /t2t/generate with {\"content\": \"your content\", \"instruction\": \"Tạo ảnh\"}",
            "split": "POST /t2t/generate with {\"content\": \"...\", \"split_ideas\": true, \"ideas_count\": 10}"
        }
    }


@router.post("/generate", response_model=T2TResponse)
async def generate_prompts(request: T2TRequest):
    """Generate image prompts from content"""
    
    if not request.content or len(request.content.strip()) == 0:
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    
    prompt = f"""You are an expert at creating detailed image prompts for AI image generation (Stable Diffusion, Midjourney, etc.).

CONTENT TO VISUALIZE:
{request.content}

INSTRUCTION:
{request.instruction}

Requirements:
- Create vivid, detailed visual descriptions
- Include lighting, composition, style details
- Keep each prompt self-contained (1-2 sentences)
- Use comma-separated keywords
- For portraits: include age, ethnicity, clothing, expression, mood
- For scenes: include setting, time of day, weather, atmosphere
- DO NOT include text, logos, or watermarks

{"Generate exactly " + str(request.ideas_count) + " distinct prompts, one per line, starting with a dash (-)" if request.split_ideas else "Generate ONE detailed prompt for this content:"}

Your response:"""
    
    try:
        result = await generate_with_llm(prompt)
        
        if request.split_ideas:
            # Split into lines
            ideas = [
                line.strip().trim_start_matches("- ").trim_start_matches("* ").strip()
                for line in result.split("\n")
                if line.strip()
            ][:request.ideas_count]
            
            return T2TResponse(
                success=True,
                prompt=result,
                ideas=ideas
            )
        else:
            return T2TResponse(
                success=True,
                prompt=result.strip()
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {str(e)}")


@router.post("/generate_sync")
async def generate_prompts_sync(request: T2TRequest):
    """Generate prompts synchronously (same as /generate)"""
    return await generate_prompts(request)
