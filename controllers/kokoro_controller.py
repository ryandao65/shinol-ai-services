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

from paths import get_output_root

router = APIRouter(prefix="/tts/kokoro", tags=["Kokoro TTS"])

# Output directory - use same structure as tts_controller
OUTPUT_DIR = get_output_root()
os.makedirs(OUTPUT_DIR, exist_ok=True)


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class KokoroTTSRequest(BaseModel):
    text: str
    voice: str = "jf_alpha"  # Default: Japanese (Kokoro JP voices use jf_* / jm_*)
    speed: float = 1.0
    project_code: Optional[str] = None
    episode_code: Optional[str] = None
    webhook_url: Optional[str] = None
    output_dir: Optional[str] = None  # Absolute episode dir from desktop app (preferred when provided)
    filename: Optional[str] = None  # Optional custom basename from desktop app (e.g. body_part_1)


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


# Available voices — lang matches KPipeline(lang_code=...): j=Japanese, a=American, b=British
KOKORO_VOICES = {
    # Japanese (use for 日本語 content; requires misaki[ja] / JP pipeline deps)
    "jf_alpha": {"lang": "j", "name": "Japanese Female - alpha"},
    "jf_gongitsune": {"lang": "j", "name": "Japanese Female - gongitsune"},
    "jf_nezumi": {"lang": "j", "name": "Japanese Female - nezumi"},
    "jf_tebukuro": {"lang": "j", "name": "Japanese Female - tebukuro"},
    "jm_kumo": {"lang": "j", "name": "Japanese Male - kumo"},
    # American English (legacy)
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
    
    # Keep line boundaries for long-form chunking; collapse spaces/tabs per line only.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join([ln for ln in lines if ln])

    # Remove control chars (except newline) without stripping Japanese punctuation.
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    return text.strip()


def get_kokoro_split_pattern() -> str:
    """Regex used by Kokoro pipeline to split long Japanese text into stable chunks."""
    return r"[。！？!?]+|\n+"


def sanitize_filename_stem(name: str) -> str:
    """Keep predictable, filesystem-safe basename for saved audio."""
    stem = re.sub(r'\.[A-Za-z0-9]+$', '', (name or '').strip())
    stem = re.sub(r'[^A-Za-z0-9._-]+', '_', stem)
    stem = stem.strip('._-')
    return stem[:120] if stem else ""


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
            voice_info = KOKORO_VOICES.get(request.voice, KOKORO_VOICES["jf_alpha"])
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
            
            # Build output path (must match audio_url: /audio/... → OUTPUT_DIR/audio/...)
            if request.output_dir and request.output_dir.strip():
                output_subdir = os.path.join(request.output_dir.strip(), "audio")
            elif request.project_code and request.episode_code:
                output_subdir = os.path.join(OUTPUT_DIR, request.project_code, request.episode_code, "audio")
            elif request.project_code:
                output_subdir = os.path.join(OUTPUT_DIR, request.project_code, "audio")
            else:
                output_subdir = os.path.join(OUTPUT_DIR, "audio")
            
            os.makedirs(output_subdir, exist_ok=True)
            
            # Save audio with optional deterministic filename from desktop app.
            custom_stem = sanitize_filename_stem(request.filename or "")
            if custom_stem:
                filename = f"{custom_stem}.wav"
            else:
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
            # Split on Japanese sentence boundaries and newlines for stable long synthesis.
            split_pattern=get_kokoro_split_pattern(),
        )

        chunk_count = 0
        chunk_text_lens: list[int] = []
        chunk_item_types: list[str] = []
        chunks = []
        for item in generator:
            chunk_count += 1
            chunk_item_types.append(type(item).__name__)
            try:
                # Kokoro yields iterable Result-like entries, commonly (graphemes, phonemes, audio).
                *prefix, audio = item
                graphemes = prefix[0] if len(prefix) > 0 else ""
                if isinstance(graphemes, str):
                    chunk_text_lens.append(len(graphemes))
                chunks.append(np.asarray(audio))
            except Exception as e:
                raise RuntimeError(f"Invalid Kokoro generator item: {e}") from e

        if not chunks:
            raise RuntimeError("Kokoro returned no audio chunks")

        try:
            audio = np.concatenate(chunks)
        except Exception as e:
            raise
        sample_rate = 24000  # Kokoro default
        return audio, sample_rate
    
    def _save_as_mp3(self, audio: np.ndarray, sample_rate: int, filepath: str):
        """Write WAV or MP3. Final path ending in .wav is written as PCM WAV (no temp delete bug)."""
        import tempfile

        lower = filepath.lower()
        if lower.endswith(".wav"):
            try:
                import soundfile as sf

                sf.write(filepath, audio, sample_rate)
            except ImportError:
                import scipy.io.wavfile as wavfile

                wavfile.write(filepath, sample_rate, (audio * 32767).astype(np.int16))
            return

        # MP3: must use a temp WAV path distinct from filepath (never filepath.replace for .wav targets)
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="kokoro_")
        os.close(fd)
        try:
            try:
                import soundfile as sf

                sf.write(tmp_wav, audio, sample_rate)
            except ImportError:
                import scipy.io.wavfile as wavfile

                wavfile.write(tmp_wav, sample_rate, (audio * 32767).astype(np.int16))
            try:
                from pydub import AudioSegment

                seg = AudioSegment.from_wav(tmp_wav)
                seg.export(filepath, format="mp3", bitrate="128k")
            except ImportError as e:
                raise RuntimeError(
                    "MP3 output requires pydub; install pydub or use a .wav filename."
                ) from e
        finally:
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
    
    async def _send_webhook(self, url: str, data: dict):
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                await client.post(url, json=data, timeout=30)
        except Exception as e:
            print(f"Webhook failed: {e}")


# Initialize worker
worker = KokoroWorker()


async def synthesize_kokoro_wav_bytes(text: str, voice: str, speed: float) -> bytes:
    """WAV bytes for preview / cache (no job queue)."""
    await worker.initialize()
    clean_text = clean_text_for_tts(text)
    if not clean_text:
        raise ValueError("Text is empty after cleaning")
    voice_key = voice if voice in KOKORO_VOICES else "jf_alpha"
    voice_info = KOKORO_VOICES[voice_key]
    lang = voice_info["lang"]
    audio, sample_rate = await asyncio.to_thread(
        worker._synthesize_sync, clean_text, voice_key, speed, lang
    )
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ============ ROUTES ============

@router.get("/")
async def root():
    return {
        "message": "Kokoro TTS Service - Local High-Quality Voice",
        "voices": {k: v["name"] for k, v in KOKORO_VOICES.items()},
        "default_voice": "jf_alpha",
        "usage": {
            "generate": "POST /tts/kokoro/generate with {\"text\": \"…\", \"voice\": \"jf_alpha\"}",
            "generate_sync": "POST /tts/kokoro/generate_sync with {\"text\": \"…\"}",
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


@router.get("/ping")
async def ping_kokoro():
    """Lightweight check: load Kokoro pipeline (may take time on first run)."""
    try:
        await worker.initialize()
        return {"ok": True, "message": "Kokoro pipeline ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


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
    
    # Poll for completion (align with desktop SHINOL_LONG_TTS_MAX_WAIT_SEC; default 4h)
    def _sync_max_wait_sec() -> float:
        for key in ("KOKORO_SYNC_MAX_WAIT_SEC", "SHINOL_LONG_TTS_MAX_WAIT_SEC"):
            raw = os.environ.get(key)
            if raw is not None and str(raw).strip():
                try:
                    v = float(raw)
                    if v > 0:
                        return v
                except ValueError:
                    pass
        return 14400.0

    max_wait = _sync_max_wait_sec()
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
