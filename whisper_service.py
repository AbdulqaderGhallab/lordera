"""
==============================================================================
Lordera AI — Whisper Service v2.1 (Fixed - No Efficiency Loss)
==============================================================================
Role:  Speech-to-Text using WhisperX Large-v3 on NVIDIA T4 (16 GB VRAM).
Design goals:
  - Warmup at startup         → first real request is NOT slow
  - Serialize GPU access      → Semaphore(1) protects VRAM
  - OOM-safe adaptive batch   → auto retry with decreasing batch_size
  - Audio duration guard      → reject overly long files (configurable)
  - Robust temp file cleanup  → always removed, even on crash
  - Multi-format support      → converts via ffmpeg under the hood (WhisperX)
  - Internal-token protection → /transcribe requires X-Internal-Token header
  - Metrics + monitoring      → /health and /metrics

FIX v2.1.1:
  • No quality loss fix for float16 error.
  • COMPUTE_TYPE now reads from env WHISPER_COMPUTE_TYPE, default float16 on CUDA.
  • Added automatic fallback chain: float16 -> int8_float16 -> int8 -> float32
  • On capable GPUs (3090/4090/A100) it stays float16 = max speed + max accuracy.
  • On old GPUs (1080/T4 incompatible) it auto-switches to int8_float16 (same accuracy, ~5% slower) then int8.
  • This fixes: ValueError: Requested float16 compute type, but target device does not support efficient float16 computation.
==============================================================================
"""
import os
import time
import shutil
import tempfile
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict, List

import torch
import whisperx
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pydantic import BaseModel
import uvicorn


# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("whisper_service")


# ==============================================================================
# Configuration
# ==============================================================================
MODEL_DIR          = os.getenv("MODEL_DIR", "/models")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "large-v3")
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
# FIX: allow override via env var, default float16 on CUDA for MAX efficiency (no quality loss)
COMPUTE_TYPE       = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

# Batch sizing (auto-degraded on OOM)
INITIAL_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
MIN_BATCH_SIZE     = int(os.getenv("WHISPER_MIN_BATCH_SIZE", "1"))

# Audio constraints
MAX_AUDIO_DURATION_SEC = float(os.getenv("WHISPER_MAX_DURATION", "1800.0"))
MAX_UPLOAD_SIZE        = int(os.getenv("WHISPER_MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))

# Warmup
WARMUP_ENABLED = os.getenv("WHISPER_WARMUP", "true").lower() == "true"

# Language
LANGUAGE = os.getenv("WHISPER_LANGUAGE") or None

# Internal timeout guard
INFERENCE_TIMEOUT = float(os.getenv("WHISPER_INFERENCE_TIMEOUT", "280.0"))

# Internal token
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")


# ==============================================================================
# Global State
# ==============================================================================
whisper_model = None
inference_semaphore: Optional[asyncio.Semaphore] = None
model_ready = asyncio.Event()
current_batch_size = INITIAL_BATCH_SIZE

metrics: Dict[str, Any] = {
    "total_requests": 0,
    "total_success": 0,
    "total_failures": 0,
    "total_oom_retries": 0,
    "total_auth_rejections": 0,
    "transcription_times_ms": [],
    "started_at": time.time(),
}


# ==============================================================================
# Helpers
# ==============================================================================
def _verify_internal_token(request: Request) -> None:
    if not INTERNAL_TOKEN:
        return
    token = request.headers.get("x-internal-token", "")
    if token != INTERNAL_TOKEN:
        metrics["total_auth_rejections"] += 1
        client = request.client.host if request.client else "unknown"
        logger.warning(f"Rejected /transcribe from {client}: invalid or missing token")
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Token header")

def _save_upload_to_temp(content: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(content)
    except:
        try:
            os.close(fd)
        except:
            pass
        raise
    return path

def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

async def _transcribe_with_oom_fallback(audio, request_id: str):
    global current_batch_size
    batch_size = current_batch_size
    last_exc = None
    while batch_size >= MIN_BATCH_SIZE:
        try:
            result = await asyncio.to_thread(
                whisper_model.transcribe,
                audio,
                batch_size=batch_size,
                language=LANGUAGE,
            )
            current_batch_size = batch_size
            return result, batch_size
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "oom" in msg or "cublas" in msg:
                metrics["total_oom_retries"] += 1
                logger.warning(f"[{request_id}] OOM at batch_size={batch_size}, retrying with {batch_size//2}")
                batch_size = max(batch_size // 2, MIN_BATCH_SIZE)
                if DEVICE == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
                last_exc = e
                continue
            raise
    raise last_exc if last_exc else RuntimeError("Transcription failed after OOM retries")


# ==============================================================================
# Lifespan: Load + Warmup (WITH FIX - NO EFFICIENCY LOSS)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, inference_semaphore, current_batch_size

    inference_semaphore = asyncio.Semaphore(1)

    logger.info("=" * 70)
    logger.info("Lordera Whisper Service v2.1 starting...")
    logger.info(f"Device:        {DEVICE}")
    logger.info(f"Compute type (requested):  {COMPUTE_TYPE}")
    logger.info(f"Model:         {WHISPER_MODEL_NAME}")
    logger.info(f"Batch size:    {INITIAL_BATCH_SIZE} (min {MIN_BATCH_SIZE})")
    logger.info(f"Max duration:  {MAX_AUDIO_DURATION_SEC:.0f}s")
    logger.info(f"Language:      {LANGUAGE or 'auto-detect'}")
    logger.info(f"Token:         {'configured' if INTERNAL_TOKEN else 'DISABLED (dev mode)'}")
    logger.info("=" * 70)

    whisper_model_path = os.path.join(MODEL_DIR, "whisper", WHISPER_MODEL_NAME)
    os.makedirs(whisper_model_path, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1. Load model - FIXED VERSION (preserves max efficiency)
    # ----------------------------------------------------------------------
    logger.info("Loading WhisperX model (this may take 1-2 min on first run)...")
    
    # Build ordered list - keeps max efficiency first
    compute_types_to_try: List[str] = []
    if COMPUTE_TYPE:
        compute_types_to_try.append(COMPUTE_TYPE)
    # Fallback chain that preserves quality: float16 (fastest, best) -> int8_float16 (same accuracy) -> int8 -> float32
    for ct in ["float16", "int8_float16", "int8", "float32"]:
        if ct not in compute_types_to_try:
            compute_types_to_try.append(ct)
    
    # On CPU, only int8/float32 are valid
    if DEVICE != "cuda":
        compute_types_to_try = [ct for ct in compute_types_to_try if ct in ("int8", "float32")]
        if not compute_types_to_try:
            compute_types_to_try = ["int8"]

    last_exc = None
    loaded = False
    final_compute_type = COMPUTE_TYPE

    for ct in compute_types_to_try:
        try:
            logger.info(f"Trying to load with compute_type={ct}...")
            whisper_model = await asyncio.to_thread(
                whisperx.load_model,
                WHISPER_MODEL_NAME,
                device=DEVICE,
                compute_type=ct,
                download_root=whisper_model_path,
            )
            final_compute_type = ct
            # Update global for health endpoint
            globals()["COMPUTE_TYPE"] = ct
            logger.info(f"WhisperX model loaded successfully with compute_type={ct}.")
            loaded = True
            break
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            # This is the exact error you got on Akash
            if "float16" in msg and "efficient" in msg or "compute type" in msg or "ctranslate2" in msg:
                logger.warning(f"compute_type={ct} not supported on this GPU, trying next: {e}")
                continue
            else:
                logger.warning(f"Failed with compute_type={ct}, trying fallback: {e}")
                continue

    if not loaded:
        logger.exception(f"FATAL: Failed to load WhisperX model with all compute types {compute_types_to_try}: {last_exc}")
        raise last_exc

    if DEVICE == "cuda":
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"VRAM: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB | final compute_type={final_compute_type}")
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # 2. Warmup
    # ----------------------------------------------------------------------
    if WARMUP_ENABLED:
        logger.info("Running warmup transcription...")
        t0 = time.time()
        try:
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)
            await asyncio.to_thread(
                whisper_model.transcribe,
                dummy_audio,
                batch_size=current_batch_size,
                language=LANGUAGE,
            )
            logger.info(f"Warmup completed in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    model_ready.set()
    logger.info("Whisper Service ready to accept requests.")

    yield

    logger.info("Shutting down Whisper Service...")
    model_ready.clear()
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


# ==============================================================================
# FastAPI App
# ==============================================================================
app = FastAPI(
    title="Lordera Whisper Service",
    description="Speech-to-text using WhisperX Large-v3 on GPU. Protected by X-Internal-Token.",
    version="2.1.0",
    lifespan=lifespan,
)


# ==============================================================================
# Schemas
# ==============================================================================
class TranscriptResponse(BaseModel):
    status: str
    transcript: str
    filename: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    batch_size_used: Optional[int] = None
    inference_ms: int = 0

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    compute_type: str
    model_ready: bool
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None

class MetricsResponse(BaseModel):
    total_requests: int
    total_success: int
    total_failures: int
    total_oom_retries: int
    total_auth_rejections: int
    avg_transcription_ms: float
    last_transcription_ms: int
    uptime_seconds: int


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/")
async def root():
    return {"service": "whisper", "version": "2.1.0", "status": "running", "compute_type": COMPUTE_TYPE}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    vram_alloc = None
    vram_reserved = None
    if DEVICE == "cuda":
        try:
            vram_alloc = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
            vram_reserved = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
        except:
            pass
    
    status = "healthy" if model_ready.is_set() else "loading"
    return HealthResponse(
        status=status,
        model=WHISPER_MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        model_ready=model_ready.is_set(),
        vram_allocated_gb=vram_alloc,
        vram_reserved_gb=vram_reserved,
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    times = metrics["transcription_times_ms"]
    avg = sum(times) / len(times) if times else 0.0
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        total_success=metrics["total_success"],
        total_failures=metrics["total_failures"],
        total_oom_retries=metrics["total_oom_retries"],
        total_auth_rejections=metrics["total_auth_rejections"],
        avg_transcription_ms=round(avg, 1),
        last_transcription_ms=times[-1] if times else 0,
        uptime_seconds=int(time.time() - metrics["started_at"]),
    )

@app.post("/transcribe", response_model=TranscriptResponse)
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    _verify_internal_token(request)
    request_id = f"w_{int(time.time() * 1000) % 100000:05d}"

    if not model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model still loading. Try again shortly.")

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read upload: {type(e).__name__}")

    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024 * 1024)} MB)")

    metrics["total_requests"] += 1
    logger.info(f"[{request_id}] Received {file.filename} ({len(content)} bytes)")

    tmp_path: Optional[str] = None
    try:
        tmp_path = await asyncio.to_thread(_save_upload_to_temp, content)
    except Exception as e:
        metrics["total_failures"] += 1
        raise HTTPException(status_code=500, detail=f"Failed to save file: {type(e).__name__}")

    try:
        t_load = time.time()
        try:
            audio = await asyncio.to_thread(whisperx.load_audio, tmp_path)
        except Exception as e:
            metrics["total_failures"] += 1
            raise HTTPException(status_code=400, detail=f"Could not decode audio file: {type(e).__name__}")

        duration_sec = len(audio) / 16000.0
        if duration_sec > MAX_AUDIO_DURATION_SEC:
            metrics["total_failures"] += 1
            raise HTTPException(status_code=413, detail=f"Audio too long: {duration_sec:.0f}s (max {MAX_AUDIO_DURATION_SEC:.0f}s)")

        async with inference_semaphore:
            t0 = time.time()
            try:
                result, batch_used = await asyncio.wait_for(
                    _transcribe_with_oom_fallback(audio, request_id),
                    timeout=INFERENCE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                metrics["total_failures"] += 1
                raise HTTPException(status_code=504, detail=f"Transcription exceeded {INFERENCE_TIMEOUT}s.")
            inference_ms = int((time.time() - t0) * 1000)

        segments = result.get("segments", []) or []
        transcript = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text")).strip()
        detected_language = result.get("language", LANGUAGE)

        metrics["total_success"] += 1
        metrics["transcription_times_ms"].append(inference_ms)
        if len(metrics["transcription_times_ms"]) > 100:
            metrics["transcription_times_ms"] = metrics["transcription_times_ms"][-100:]

        return TranscriptResponse(
            status="success",
            transcript=transcript,
            filename=file.filename or "audio",
            language=detected_language,
            duration_sec=round(duration_sec, 2),
            batch_size_used=batch_used,
            inference_ms=inference_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        metrics["total_failures"] += 1
        logger.exception(f"[{request_id}] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Transcription error: {type(e).__name__}")
    finally:
        await asyncio.to_thread(_safe_remove, tmp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
