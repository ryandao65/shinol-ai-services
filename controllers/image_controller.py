"""
Image Generation Controller - Stable Diffusion XL with Queue System
Prevents VRAM overflow by processing one request at a time
"""
import os
import uuid
import time
import asyncio
import threading
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import httpx

from paths import get_output_root, package_root

router = APIRouter(prefix="/image", tags=["Image Generation"])

# Output directory
OUTPUT_DIR = os.path.join(get_output_root(), "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model cache directory
MODEL_DIR = os.path.join(package_root(), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING = "loading"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1024
    height: int = 576  # 16:9 horizontal for YouTube
    num_inference_steps: int = 16  # SDXL base - 16 steps for good quality
    guidance_scale: float = 7.5  # Standard guidance for base model
    seed: Optional[int] = None
    webhook_url: Optional[str] = None
    model: str = "base"  # "base" = SDXL fp16 (default)
    project_code: Optional[str] = None  # For folder structure
    episode_code: Optional[str] = None  # For folder structure


class ImageGenerationResponse(BaseModel):
    job_id: str
    status: str
    position: int  # Position in queue


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None  # 0-100
    image_url: Optional[str] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


# ============ QUEUE SYSTEM ============

class ImageGenerationWorker:
    """Worker that manages SDXL model and generation queue"""
    
    def __init__(self, max_queue_size: int = 100):
        self.queue = []
        self.queue_lock = threading.Lock()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.is_processing = False
        self.model = None
        self.pipe = None
        self.worker_thread = None
        
    def add_job(self, job_id: str, request: ImageGenerationRequest) -> int:
        """Add job to queue, return position"""
        with self.queue_lock:
            position = len(self.queue) + 1
            self.jobs[job_id] = {
                "id": job_id,
                "request": request,
                "status": JobStatus.QUEUED,
                "progress": 0,
                "image_url": None,
                "error": None,
                "created_at": time.time(),
                "completed_at": None,
            }
            self.queue.append(job_id)
        
        # Start worker if not running
        if not self.is_processing:
            self.start_worker()
        
        return position
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        job = self.jobs[job_id]
        return {
            "job_id": job["id"],
            "status": job["status"],
            "progress": job["progress"],
            "image_url": job.get("image_url"),
            "error": job.get("error"),
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),
        }
    
    def start_worker(self):
        """Start worker thread"""
        self.is_processing = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def _process_queue(self):
        """Process jobs from queue one by one"""
        print("Worker thread started, waiting for jobs...")
        while True:
            try:
                # Get next job from queue
                with self.queue_lock:
                    if not self.queue:
                        print("Queue empty, worker sleeping...")
                        break
                    job_id = self.queue.pop(0)
                    print(f"Processing job: {job_id}")
                
                job = self.jobs.get(job_id)
                if not job:
                    continue
                
                # Process the job
                self._generate_image(job_id, job["request"])
                
            except Exception as e:
                print(f"Queue processing error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("Worker thread exiting...")
        self.is_processing = False
    
    def _generate_image(self, job_id: str, request: ImageGenerationRequest):
        """Generate image using SDXL (Base)"""
        job = self.jobs[job_id]
        
        try:
            # Update status to loading
            job["status"] = JobStatus.LOADING
            job["progress"] = 5
            
            # Import and load model
            import torch
            from diffusers import StableDiffusionXLPipeline
            
            # Use a single cached pipe
            if not hasattr(self, 'pipe') or self.pipe is None:
                print(f"Loading SDXL model...")
                
                # SDXL Base - fp16 for speed
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    cache_dir=MODEL_DIR,
                )
                
                # Move to GPU
                self.pipe = self.pipe.to("cuda")
                # Optimize for low VRAM
                self.pipe.enable_vae_slicing()
                self.pipe.enable_vae_tiling()
                
                print(f"SDXL model loaded!")
            
            job["progress"] = 20
            
            # Update status to generating
            job["status"] = JobStatus.GENERATING
            
            # Prepare generation kwargs
            generator = None
            if request.seed is not None:
                generator = torch.Generator(device="cuda").manual_seed(request.seed)
            
            # Generate
            print(f"Generating image for job {job_id}...")
            result = self.pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            
            job["progress"] = 90
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}_{uuid.uuid4().hex[:8]}.png"
            
            # Build output path: output/{project_code}/{episode_code}/image/filename.png
            if request.project_code and request.episode_code:
                output_subdir = os.path.join(OUTPUT_DIR, request.project_code, request.episode_code, "image")
            elif request.project_code:
                output_subdir = os.path.join(OUTPUT_DIR, request.project_code, "image")
            else:
                output_subdir = OUTPUT_DIR
            
            os.makedirs(output_subdir, exist_ok=True)
            filepath = os.path.join(output_subdir, filename)
            
            result.images[0].save(filepath)
            
            # Build image_url path: /{project_code}/{episode_code}/image/filename.png
            if request.project_code and request.episode_code:
                image_url_path = f"/{request.project_code}/{request.episode_code}/image/{filename}"
            elif request.project_code:
                image_url_path = f"/{request.project_code}/image/{filename}"
            else:
                image_url_path = f"/image/{filename}"
            
            # Update job status
            job["status"] = JobStatus.COMPLETED
            job["progress"] = 100
            job["image_url"] = image_url_path
            job["completed_at"] = time.time()
            
            print(f"Image generated: {filename}")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Send webhook
            if request.webhook_url:
                self._send_webhook(request.webhook_url, {
                    "success": True,
                    "job_id": job_id,
                    "image_url": job["image_url"],
                    "prompt": request.prompt,
                })
                
        except Exception as e:
            job["status"] = JobStatus.FAILED
            job["error"] = str(e)
            job["completed_at"] = time.time()
            print(f"Image generation failed: {e}")
            
            # Clear CUDA cache on error
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
            
            # Send error webhook
            if request.webhook_url:
                self._send_webhook(request.webhook_url, {
                    "success": False,
                    "job_id": job_id,
                    "error": str(e),
                    "prompt": request.prompt,
                })
    
    def _send_webhook(self, url: str, data: dict):
        """Send webhook notification"""
        try:
            import requests
            requests.post(url, json=data, timeout=30)
        except Exception as e:
            print(f"Webhook failed: {e}")


# Initialize worker
worker = ImageGenerationWorker()


# ============ ROUTES ============

@router.get("/")
async def root():
    return {
        "message": "Image Generation Service - Stable Diffusion XL",
        "models": {
            "base_fp16": "stabilityai/stable-diffusion-xl-base-1.0 (fp16) - ~12s cached"
        },
        "default_size": "1024x576 (16:9 YouTube format)",
        "default_steps": 16,
        "usage": {
            "generate": "POST /image/generate with {\"prompt\": \"your prompt\"}",
            "status": "GET /image/status/{job_id}",
            "download": "GET /image/download/{filename}",
        }
    }


@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """Queue image generation job (async)"""
    job_id = f"img_{uuid.uuid4().hex[:12]}"
    
    # Validate dimensions
    if request.width > 1920 or request.height > 1920:
        raise HTTPException(
            status_code=400,
            detail="Maximum dimension is 1920px"
        )
    
    # Add to queue
    position = worker.add_job(job_id, request)
    
    return ImageGenerationResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        position=position
    )


@router.post("/generate_sync")
async def generate_image_sync(request: ImageGenerationRequest):
    """Generate image synchronously (waits for completion)"""
    import asyncio
    
    # Validate dimensions
    if request.width > 1920 or request.height > 1920:
        raise HTTPException(
            status_code=400,
            detail="Maximum dimension is 1920px"
        )
    
    job_id = f"img_{uuid.uuid4().hex[:12]}"
    
    # Add job to queue
    worker.add_job(job_id, request)
    
    # Poll for completion
    max_wait = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        await asyncio.sleep(2)
        try:
            status = worker.get_job_status(job_id)
            if status["status"] == JobStatus.COMPLETED:
                return {
                    "success": True,
                    "status": "done",
                    "image_path": status["image_url"],
                    "job_id": job_id,
                }
            elif status["status"] == JobStatus.FAILED:
                raise HTTPException(
                    status_code=500,
                    detail=f"Image generation failed: {status.get('error')}"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=504, detail="Image generation timed out")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status"""
    return worker.get_job_status(job_id)


@router.get("/download/{filename}")
async def download_image(filename: str):
    """Download generated image"""
    from fastapi.responses import FileResponse
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        filepath,
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.delete("/cleanup")
async def cleanup_old_images(days: int = 7):
    """Cleanup old generated images"""
    import time
    
    cutoff = time.time() - (days * 86400)
    deleted = 0
    
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                deleted += 1
    
    return {"deleted": deleted, "days": days}
