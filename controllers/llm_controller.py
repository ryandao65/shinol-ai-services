"""
LLM Controller - Local LLM using Ollama
"""
import os
import uuid
import asyncio
import threading
import time
from enum import Enum
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

router = APIRouter(prefix="/llm", tags=["LLM"])

# Job management
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    model: str = "qwen2.5-coder:14b-instruct-q5_K_M"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    webhook_url: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "qwen2.5-coder:14b-instruct-q5_K_M"
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    webhook_url: Optional[str] = None


class GenerationResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    response: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    size: str
    modified: str


# ============ OLLAMA CLIENT ============

def call_ollama(prompt: str, model: str, temperature: float, max_tokens: int, system_prompt: str = None) -> str:
    """Call Ollama API for generation"""
    import requests
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    
    result = response.json()
    return result.get("response", "")


def call_ollama_chat(messages: List[dict], model: str, temperature: float, max_tokens: int) -> str:
    """Call Ollama API for chat"""
    import requests
    
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    
    result = response.json()
    return result.get("message", {}).get("content", "")


# ============ ROUTES ============

@router.get("/")
async def root():
    return {
        "message": "LLM Service - Ollama",
        "usage": {
            "generate": "POST /llm/generate with {\"prompt\": \"your prompt\"}",
            "chat": "POST /llm/chat with {\"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}",
            "models": "GET /llm/models",
            "status": "GET /llm/status/{job_id}",
        }
    }


@router.get("/models")
async def list_models():
    """List available Ollama models"""
    import requests
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        
        return {
            "models": [
                {
                    "name": m["name"],
                    "size": f"{m['size'] / (1024**3):.1f} GB",
                    "modified": m.get("modified_at", "")
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate text from prompt"""
    job_id = f"llm_{uuid.uuid4().hex[:12]}"
    
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.QUEUED,
            "request": request.dict(),
            "response": None,
            "error": None,
            "created_at": time.time(),
            "completed_at": None,
        }
    
    # Process in background
    threading.Thread(
        target=_process_generation,
        args=(job_id,),
        daemon=True
    ).start()
    
    return GenerationResponse(
        job_id=job_id,
        status=JobStatus.QUEUED
    )


@router.post("/chat", response_model=GenerationResponse)
async def chat(request: ChatRequest):
    """Chat completion"""
    job_id = f"chat_{uuid.uuid4().hex[:12]}"
    
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.QUEUED,
            "request": request.dict(),
            "response": None,
            "error": None,
            "created_at": time.time(),
            "completed_at": None,
        }
    
    # Process in background
    threading.Thread(
        target=_process_chat,
        args=(job_id,),
        daemon=True
    ).start()
    
    return GenerationResponse(
        job_id=job_id,
        status=JobStatus.QUEUED
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Get job status"""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        return JobStatusResponse(
            job_id=job["id"],
            status=job["status"],
            response=job.get("response"),
            error=job.get("error"),
            created_at=job["created_at"],
            completed_at=job.get("completed_at")
        )


@router.post("/generate/stream/{job_id}")
async def generate_stream(job_id: str):
    """Stream generation response"""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
    
    from fastapi.responses import StreamingResponse
    
    async def event_generator():
        import json
        
        # Wait for completion
        while job["status"] == JobStatus.QUEUED or job["status"] == JobStatus.PROCESSING:
            yield f"data: {json.dumps({'status': job['status']})}\n\n"
            time.sleep(0.5)
        
        # Send final result
        if job["status"] == JobStatus.COMPLETED:
            yield f"data: {json.dumps({'status': 'completed', 'response': job['response']})}\n\n"
        else:
            yield f"data: {json.dumps({'status': 'failed', 'error': job.get('error')})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============ BACKGROUND PROCESSING ============

def _process_generation(job_id: str):
    """Process generation in background"""
    with jobs_lock:
        job = jobs[job_id]
        job["status"] = JobStatus.PROCESSING
    
    try:
        request = job["request"]
        
        response = call_ollama(
            prompt=request["prompt"],
            model=request["model"],
            temperature=request["temperature"],
            max_tokens=request["max_tokens"],
            system_prompt=request.get("system_prompt")
        )
        
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["response"] = response
            jobs[job_id]["completed_at"] = time.time()
        
        # Send webhook
        if request.get("webhook_url"):
            _send_webhook(request["webhook_url"], {
                "success": True,
                "job_id": job_id,
                "response": response,
                "prompt": request["prompt"]
            })
    
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = time.time()
        
        # Send error webhook
        request = job["request"]
        if request.get("webhook_url"):
            _send_webhook(request["webhook_url"], {
                "success": False,
                "job_id": job_id,
                "error": str(e),
                "prompt": request.get("prompt")
            })


def _process_chat(job_id: str):
    """Process chat in background"""
    with jobs_lock:
        job = jobs[job_id]
        job["status"] = JobStatus.PROCESSING
    
    try:
        request = job["request"]
        
        messages = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
        
        response = call_ollama_chat(
            messages=messages,
            model=request["model"],
            temperature=request["temperature"],
            max_tokens=request["max_tokens"]
        )
        
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["response"] = response
            jobs[job_id]["completed_at"] = time.time()
        
        # Send webhook
        if request.get("webhook_url"):
            _send_webhook(request["webhook_url"], {
                "success": True,
                "job_id": job_id,
                "response": response,
            })
    
    except Exception as e:
        with jobs_lock:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = time.time()


def _send_webhook(url: str, data: dict):
    """Send webhook notification"""
    try:
        import requests
        requests.post(url, json=data, timeout=30)
    except Exception as e:
        print(f"Webhook failed: {e}")
