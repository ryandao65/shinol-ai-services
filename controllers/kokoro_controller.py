"""
Kokoro TTS Controller - Local High-Quality Text to Speech
Using kokoro package (same as worker)
"""
import os
import re
import uuid
import time
import asyncio
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np

router = APIRouter(prefix="/tts/kokoro", tags=["Kokoro TTS"])

# Output directory - use same structure as tts_controller
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class KokoroTTSRequest(BaseModel):
    text: str
    voice: str = "af_sarah"  # Default voice
    speed: float = 1.0
    project_code: Optional[str] = None
    episode_code: Optional[str] = None
    webhook_url: Optional[str] = None


class KokoroTTSResponse(BaseModel):
    job_id: str
    status: str
    position: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    audio_url: Optional[str] = None
    audio_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


# Available voices - Kokoro voice IDs
KOKORO_VOICES = {
    # American Female
    "af_sarah": {"lang": "a", "name": "American Female - Sarah"},
    "af_nicole": {"lang": "a", "name": "American Female - Nicole"},
    "af_serena": {"lang": "a", "name": "American Female - Serena"},
    "af_heart": {"lang": "a", "name": "American Female - Heart"},
    "af_emma": {"lang": "a", "name": "American Female - Emma"},
    "af_isabella": {"lang": "a", "name": "American Female - Isabella"},
    "af_jennifer": {"lang": "a", "name": "American Female - Jennifer"},
    # American Male
    "af_adam": {"lang": "a", "name": "American Male - Adam"},
    "af_michael": {"lang": "a", "name": "American Male - Michael"},
    # British Female
    "bf_emma": {"lang": "b", "name": "British Female - Emma"},
    "bf_isabella": {"lang": "b", "name": "British Female - Isabella"},
    "bf_alice": {"lang": "b", "name": "British Female - Alice"},
    # British Male
    "bf_daniel": {"lang": "b", "name": "British Male - Daniel"},
}


def clean_text_for_tts(text: str) -> str:
    """
    Clean text to prevent special characters from being spoken.
    Removes/replaces:
    - SSML tags <...>
    - URLs
    - Email addresses
    - Special symbols that might be spoken
    """
    # Remove SSML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Replace multiple newlines/spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove remaining special characters except basic punctuation
    # Keep: letters, numbers, basic punctuation (.,!?,-)
    text = re.sub(r'[^\w\s.,!?\-\']+', '', text)
    
    return text.strip()


class KokoroWorker:
    """Worker that manages Kokoro TTS using kokoro package"""
    
    def __init__(self):
        self.pipeline = None
        self.lang_code = None
        self.jobs: Dict[str, Dict] = {}
        self.queue: list = []
        self.queue_lock = asyncio.Lock()
        self.is_processing = False
        
    async def initialize(self):
        """Initialize Kokoro pipeline"""
        if self.pipeline is None:
            print("Loading Kokoro model...")
            from kokoro import KPipeline
            # Default to American English
            self.lang_code = "a"  # 'a' = American, 'b' = British
            self.pipeline = KPipeline(lang_code=self.lang_code)
            print("Kokoro model loaded!")
    
    async def add_job(self, job_id: str, request: KokoroTTSRequest) -> int:
        async with self.queue_lock:
            position = len(self.queue) + 1
            self.jobs[job_id] = {
                "id": job_id,
                "request": request,
                "status": JobStatus.QUEUED,
                "progress": 0,
                "audio_url": None,
                "audio_path": None,
                "error": None,
                "created_at": time.time(),
                "completed_at": None,
            }
            self.queue.append(job_id)
            
            if not self.is_processing:
                self.is_processing = True
                asyncio.create_task(self._process_queue())
            
            return position
    
    async def get_status(self, job_id: str) -> Dict:
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        status = self.jobs[job_id].copy()
        status["job_id"] = status.pop("id")
        return status
    
    async def _process_queue(self):
        while True:
            try:
                async with self.queue_lock:
                    if not self.queue:
                        break
                    job_id = self.queue.pop(0)
                
                job = self.jobs.get(job_id)
                if job:
                    await self._generate_speech(job_id, job["request"])
            except Exception as e:
                print(f"Queue error: {e}")
                break
        
        self.is_processing = False
    
    async def _generate_speech(self, job_id: str, request: KokoroTTSRequest):
        job = self.jobs[job_id]
        
        try:
            job["status"] = JobStatus.LOADING
            job["progress"] = 10
            
            # Initialize pipeline if needed
            await self.initialize()
            
            # Clean text - remove special characters that shouldn't be spoken
            clean_text = clean_text_for_tts(request.text)
            if not clean_text:
                raise ValueError("Text is empty after cleaning")
            
            job["progress"] = 30
            job["status"] = JobStatus.GENERATING
            
            # Get voice info
            voice_info = KOKORO_VOICES.get(request.voice, KOKORO_VOICES["af_sarah"])
            voice = request.voice
            lang = voice_info["lang"]
            
            print(f"Generating speech for job {job_id}...")
            
            # Run synthesis in thread to avoid blocking
            audio, sample_rate = await asyncio.to_thread(
                self._synthesize_sync,
                clean_text,
                voice,
                request.speed,
                lang
            )
            
            job["progress"] = 80
            
            # Build output path
            if request.project_code and request.episode_code:
                output_subdir = os.path.join(OUTPUT_DIR, request.project_code, request.episode_code, "audio")
            elif request.project_code:
                output_subdir = os.path.join(OUTPUT_DIR, request.project_code, "audio")
            else:
                output_subdir = OUTPUT_DIR
            
            os.makedirs(output_subdir, exist_ok=True)
            
            # Save audio as WAV (MP3 requires pydub)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"kokoro_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
            filepath = os.path.join(output_subdir, filename)
            
            # Convert to MP3 using pydub
            await asyncio.to_thread(self._save_as_mp3, audio, sample_rate, filepath)
            
            # Build URL path
            if request.project_code and request.episode_code:
                audio_url_path = f"/{request.project_code}/{request.episode_code}/audio/{filename}"
            elif request.project_code:
                audio_url_path = f"/{request.project_code}/audio/{filename}"
            else:
                audio_url_path = f"/audio/{filename}"
            
            job["status"] = JobStatus.COMPLETED
            job["progress"] = 100
            job["audio_url"] = audio_url_path
            job["audio_path"] = filepath
            job["completed_at"] = time.time()
            
            print(f"Audio generated: {filename}")
            
            # Send webhook
            if request.webhook_url:
                await self._send_webhook(request.webhook_url, {
                    "success": True,
                    "job_id": job_id,
                    "audio_url": audio_url_path,
                    "audio_path": filepath,
                    "voice": voice,
                    "text": clean_text
                })
                
        except Exception as e:
            job["status"] = JobStatus.FAILED
            job["error"] = str(e)
            job["completed_at"] = time.time()
            print(f"Kokoro generation failed: {e}")
            
            if request.webhook_url:
                await self._send_webhook(request.webhook_url, {
                    "success": False,
                    "job_id": job_id,
                    "error": str(e),
                    "text": request.text
                })
    
    def _synthesize_sync(self, text: str, voice: str, speed: float, lang: str) -> Tuple[np.ndarray, int]:
        """Synchronous synthesis using kokoro"""
        # Create pipeline with correct lang if needed
        if lang != self.lang_code:
            from kokoro import KPipeline
            pipeline = KPipeline(lang_code=lang)
        else:
            pipeline = self.pipeline
        
        generator = pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=r"\n+"
        )
        
        chunks = [audio for *_ignored, audio in generator]
        if not chunks:
            raise RuntimeError("Kokoro returned no audio chunks")
        
        audio = np.concatenate(chunks)
        sample_rate = 24000  # Kokoro default
        return audio, sample_rate
    
    def _save_as_mp3(self, audio: np.ndarray, sample_rate: int, filepath: str):
        """Save audio as MP3"""
        # Try using soundfile first (simpler), fall back to manual MP3 encoding
        try:
            import soundfile as sf
            
            # Save as WAV first (soundfile doesn't support MP3 directly)
            wav_path = filepath.replace('.mp3', '.wav')
            sf.write(wav_path, audio, sample_rate)
            
            # Convert to MP3 using pydub if available
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_wav(wav_path)
                audio_segment.export(filepath, format="mp3", bitrate="128k")
                # Remove temp WAV
                os.remove(wav_path)
            except ImportError:
                # pydub not available, return WAV path instead
                if filepath.endswith('.mp3'):
                    filepath = wav_path
        except ImportError:
            # soundfile not available, use scipy
            import scipy.io.wavfile as wavfile
            wav_path = filepath.replace('.mp3', '.wav')
            wavfile.write(wav_path, sample_rate, (audio * 32767).astype(np.int16))
            if not filepath.endswith('.wav'):
                # Rename to wav if needed
                os.rename(wav_path, filepath)
    
    async def _send_webhook(self, url: str, data: dict):
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(url, json=data, timeout=30)
        except Exception as e:
            print(f"Webhook failed: {e}")


# Initialize worker
worker = KokoroWorker()


# ============ ROUTES ============

@router.get("/")
async def root():
    return {
        "message": "Kokoro TTS Service - Local High-Quality Voice",
        "voices": {k: v["name"] for k, v in KOKORO_VOICES.items()},
        "default_voice": "af_sarah",
        "usage": {
            "generate": "POST /tts/kokoro/generate with {\"text\": \"Hello!\", \"voice\": \"af_sarah\"}",
            "generate_sync": "POST /tts/kokoro/generate_sync with {\"text\": \"Hello!\"}",
            "status": "GET /tts/kokoro/status/{job_id}",
        }
    }


@router.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": [
            {"id": k, "name": v["name"], "lang": v["lang"]}
            for k, v in KOKORO_VOICES.items()
        ]
    }


@router.post("/generate", response_model=KokoroTTSResponse)
async def generate_speech(request: KokoroTTSRequest):
    """Queue TTS generation (async)"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    job_id = f"kokoro_{uuid.uuid4().hex[:12]}"
    position = await worker.add_job(job_id, request)
    
    return KokoroTTSResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        position=position
    )


@router.post("/generate_sync")
async def generate_speech_sync(request: KokoroTTSRequest):
    """Generate TTS synchronously (waits for completion)"""
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    job_id = f"kokoro_{uuid.uuid4().hex[:12]}"
    await worker.add_job(job_id, request)
    
    # Poll for completion
    max_wait = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        await asyncio.sleep(2)
        try:
            status = await worker.get_status(job_id)
            if status["status"] == JobStatus.COMPLETED:
                return {
                    "success": True,
                    "status": "done",
                    "audio_url": status["audio_url"],
                    "audio_path": status["audio_path"],
                    "job_id": job_id,
                }
            elif status["status"] == JobStatus.FAILED:
                raise HTTPException(
                    status_code=500,
                    detail=f"Kokoro TTS failed: {status.get('error')}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=504, detail="Kokoro TTS generation timed out")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str):
    """Get job status"""
    status = await worker.get_status(job_id)
    return JobStatusResponse(**status)
