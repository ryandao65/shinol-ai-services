"""
LLM Controller - Local LLM using Ollama.
Synchronous chat-turn endpoints for workflow template 2 (Gemini / Ollama) live here so the desktop app only calls this API.
"""
import os
import uuid
import asyncio
import threading
import time
from enum import Enum
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import httpx

router = APIRouter(prefix="/llm", tags=["LLM"])

# Long-running story chapters (workflow template 2 multi-turn)
_CHAT_TURN_HTTP_TIMEOUT = 600.0

# Ollama defaults live here (desktop does not embed localhost:11434 or num_predict rules)
_DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
_OLLAMA_CHAT_TURN_NUM_PREDICT_FLOOR = 8192


def _resolve_ollama_chat_turn_base_url(override: Optional[str]) -> str:
    if override and str(override).strip():
        return str(override).rstrip("/")
    return _DEFAULT_OLLAMA_BASE_URL


def _ollama_chat_turn_num_predict(max_tokens: int) -> int:
    return max(int(max_tokens), _OLLAMA_CHAT_TURN_NUM_PREDICT_FLOOR)

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


# ============ CHAT TURN (workflow template 2: desktop → Google Gemini / Ollama) ============


class ChatTurnGeminiRequest(BaseModel):
    """One multi-turn step for スカッと朗読 body generation (Gemini generateContent v1beta)."""

    api_key: str
    model: str
    system_instruction: str
    contents: List[Dict[str, Any]] = Field(default_factory=list)
    user_text: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_output_tokens: int = 8192


class ChatTurnGeminiResponse(BaseModel):
    text: str
    contents: List[Dict[str, Any]]


class ChatTurnOllamaRequest(BaseModel):
    """One multi-turn step for Japanese workflow body (Ollama /api/chat).

    Defaults for Ollama host and output length are applied server-side; desktop sends DB fields only.
    """

    model: str
    system_instruction: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    user_text: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    ollama_base_url: Optional[str] = None


class ChatTurnOllamaResponse(BaseModel):
    text: str
    messages: List[Dict[str, Any]]


@router.post("/chat/turn/gemini", response_model=ChatTurnGeminiResponse)
async def chat_turn_gemini(req: ChatTurnGeminiRequest):
    """
    Append user turn, call Google Gemini v1beta, append model turn.
    Keeps provider keys and HTTP details out of the Rust/Tauri app.
    """
    contents: List[Dict[str, Any]] = [dict(x) for x in req.contents]
    contents.append({"role": "user", "parts": [{"text": req.user_text}]})
    body = {
        "systemInstruction": {"parts": [{"text": req.system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_output_tokens,
            "topP": req.top_p,
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{req.model}:generateContent?key={req.api_key}"
    )
    try:
        async with httpx.AsyncClient(timeout=_CHAT_TURN_HTTP_TIMEOUT) as client:
            r = await client.post(url, json=body, headers={"Content-Type": "application/json"})
    except httpx.ConnectError as e:
        raise HTTPException(status_code=502, detail=f"Gemini request failed (network): {e}") from e
    text_body = r.text
    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Google Gemini API error {r.status_code}: {text_body[:2000]}",
        )
    try:
        result = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Gemini: {e}") from e

    try:
        text = (
            result["candidates"][0]["content"]["parts"][0]["text"]
        )
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no text in candidates (check API key, model, or safety blocking).",
        )

    contents.append({"role": "model", "parts": [{"text": text}]})
    return ChatTurnGeminiResponse(text=text, contents=contents)


@router.post("/chat/turn/ollama", response_model=ChatTurnOllamaResponse)
async def chat_turn_ollama(req: ChatTurnOllamaRequest):
    """Append user message, call Ollama /api/chat, append assistant message."""
    messages: List[Dict[str, Any]] = [dict(x) for x in req.messages]
    if not messages:
        messages.append({"role": "system", "content": req.system_instruction})
    messages.append({"role": "user", "content": req.user_text})

    base = _resolve_ollama_chat_turn_base_url(req.ollama_base_url)
    chat_url = f"{base}/api/chat"
    num_predict = _ollama_chat_turn_num_predict(req.max_tokens)
    payload = {
        "model": req.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "num_predict": num_predict,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=_CHAT_TURN_HTTP_TIMEOUT) as client:
            r = await client.post(chat_url, json=payload)
    except httpx.ConnectError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to Ollama at {base}. Is Ollama running? ({e})",
        ) from e

    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama chat error {r.status_code}: {r.text[:2000]}",
        )
    try:
        data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from Ollama: {e}") from e

    try:
        text = data["message"]["content"]
    except (KeyError, TypeError):
        raise HTTPException(status_code=502, detail="Ollama response missing message.content")

    messages.append({"role": "assistant", "content": text})
    return ChatTurnOllamaResponse(text=text, messages=messages)


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
            "chat_turn_gemini": "POST /llm/chat/turn/gemini (workflow template 2 → Google Gemini multi-turn)",
            "chat_turn_ollama": "POST /llm/chat/turn/ollama (workflow template 2 → Ollama /api/chat multi-turn)",
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
