"""
Whisper Controller - Audio Transcription using faster-whisper
"""
import os
import uuid
import time
import threading
from enum import Enum
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import httpx

router = APIRouter(prefix="/audio", tags=["Audio Transcription"])

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output", "transcriptions")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model cache directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "whisper")
os.makedirs(MODEL_DIR, exist_ok=True)


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionRequest(BaseModel):
    language: Optional[str] = None  # Auto-detect if None
    model: str = "base"  # tiny, base, small, medium, large
    task: str = "transcribe"  # transcribe or translate
    webhook_url: Optional[str] = None


class TranscriptionResponse(BaseModel):
    job_id: str
    status: str
    position: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    text: Optional[str] = None
    language: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


# Available models
WHISPER_MODELS = {
    "tiny": {"size": "75 MB", "vram": "1 GB", "speed": "32x"},
    "base": {"size": "150 MB", "vram": "1 GB", "speed": "16x"},
    "small": {"size": "500 MB", "vram": "2 GB", "speed": "6x"},
    "medium": {"size": "1.5 GB", "vram": "5 GB", "speed": "2x"},
    "large": {"size": "3 GB", "vram": "8 GB", "speed": "1x"},
}


class WhisperWorker:
    """Worker that manages Whisper transcription"""
    
    def __init__(self):
        self.jobs = {}
        self.queue = []
        self.queue_lock = threading.Lock()
        self.is_processing = False
        self.models = {}  # Cache for different model sizes
        
    def add_job(self, job_id: str, audio_path: str, request: TranscriptionRequest) -> int:
        with self.queue_lock:
            position = len(self.queue) + 1
            self.jobs[job_id] = {
                "id": job_id,
                "audio_path": audio_path,
                "request": request,
                "status": JobStatus.QUEUED,
                "progress": 0,
                "text": None,
                "language": None,
                "error": None,
                "created_at": time.time(),
                "completed_at": None,
            }
            self.queue.append(job_id)
            
            if not self.is_processing:
                self.start_worker()
            
            return position
    
    def get_status(self, job_id: str) -> Dict:
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return self.jobs[job_id]
    
    def start_worker(self):
        self.is_processing = True
        threading.Thread(target=self._process_queue, daemon=True).start()
    
    def _process_queue(self):
        while True:
            try:
                with self.queue_lock:
                    if not self.queue:
                        break
                    job_id = self.queue.pop(0)
                
                job = self.jobs.get(job_id)
                if job:
                    self._transcribe(job_id, job["audio_path"], job["request"])
            except Exception as e:
                print(f"Queue error: {e}")
                break
        
        self.is_processing = False
    
    def _transcribe(self, job_id: str, audio_path: str, request: TranscriptionRequest):
        job = self.jobs[job_id]
        
        try:
            job["status"] = JobStatus.LOADING
            job["progress"] = 10
            
            # Import and load model
            from faster_whisper import WhisperModel
            
            model_size = request.model
            
            # Load model (cache it)
            if model_size not in self.models:
                print(f"Loading Whisper {model_size} model...")
                self.models[model_size] = WhisperModel(
                    model_size,
                    device="cpu",
                    compute_type="int8",
                )
                print(f"Whisper {model_size} model loaded!")
            
            model = self.models[model_size]
            
            job["progress"] = 30
            job["status"] = JobStatus.TRANSCRIBING
            
            # Transcribe
            print(f"Transcribing audio for job {job_id}...")
            
            segments, info = model.transcribe(
                audio_path,
                language=request.language,  # None = auto-detect
                task=request.task,
                beam_size=5,
                vad_filter=True,  # Voice activity detection
            )
            
            # Collect all segments
            full_text = []
            total_duration = info.duration or 0
            
            for i, segment in enumerate(segments):
                text = segment.text.strip()
                full_text.append(text)
                
                # Update progress
                if total_duration > 0:
                    segment_end = segment.end or 0
                    job["progress"] = 30 + int((segment_end / total_duration) * 60)
            
            # Join all segments
            result_text = " ".join(full_text)
            
            job["status"] = JobStatus.COMPLETED
            job["progress"] = 100
            job["text"] = result_text
            job["language"] = info.language or "unknown"
            job["completed_at"] = time.time()
            
            print(f"Transcription completed: {len(result_text)} chars in {job['language']}")
            
            # Send webhook
            if request.webhook_url:
                self._send_webhook(request.webhook_url, {
                    "success": True,
                    "job_id": job_id,
                    "text": result_text,
                    "language": info.language,
                    "duration": total_duration
                })
                
        except Exception as e:
            job["status"] = JobStatus.FAILED
            job["error"] = str(e)
            job["completed_at"] = time.time()
            print(f"Transcription failed: {e}")
            
            if request.webhook_url:
                self._send_webhook(request.webhook_url, {
                    "success": False,
                    "job_id": job_id,
                    "error": str(e)
                })
        
        finally:
            # Clean up audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
    
    def _send_webhook(self, url: str, data: dict):
        try:
            import requests
            requests.post(url, json=data, timeout=30)
        except Exception as e:
            print(f"Webhook failed: {e}")


# Initialize worker
worker = WhisperWorker()


# ============ ROUTES ============

@router.get("/")
async def root():
    return {
        "message": "Audio Transcription Service - Faster Whisper",
        "models": WHISPER_MODELS,
        "default_model": "base",
        "usage": {
            "transcribe": "POST /audio/transcribe with audio file",
            "status": "GET /audio/status/{job_id}",
        }
    }


@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": WHISPER_MODELS
    }


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    model: str = "base",
    task: str = "transcribe",
    webhook_url: Optional[str] = None
):
    """Transcribe audio file"""
    # Validate model
    if model not in WHISPER_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Available: {list(WHISPER_MODELS.keys())}")
    
    # Validate task
    if task not in ["transcribe", "translate"]:
        raise HTTPException(status_code=400, detail="Task must be 'transcribe' or 'translate'")
    
    # Save uploaded file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"audio_{timestamp}_{uuid.uuid4().hex[:8]}.{file.filename.split('.')[-1]}"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Ensure directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Write file
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)
    
    # Create request object
    request = TranscriptionRequest(
        language=language,
        model=model,
        task=task,
        webhook_url=webhook_url
    )
    
    # Add to queue
    job_id = f"whisper_{uuid.uuid4().hex[:12]}"
    position = worker.add_job(job_id, filepath, request)
    
    return TranscriptionResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        position=position
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Get job status"""
    status = worker.get_status(job_id)
    
    return JobStatusResponse(
        job_id=status["id"],
        status=status["status"],
        progress=status.get("progress"),
        text=status.get("text"),
        language=status.get("language"),
        error=status.get("error"),
        created_at=status["created_at"],
        completed_at=status.get("completed_at")
    )
