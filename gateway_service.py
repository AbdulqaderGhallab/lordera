"""
==============================================================================
Lordera AI — Gateway Service v2.3.1
==============================================================================
CHANGELOG v2.3 → v2.3.1:
  • Added CORSMiddleware to allow the local frontend (HTML/JS) to connect.
  • This is strictly necessary for browser-based testing.

CHANGELOG v2.2 → v2.3:
  • Added support for direct text input via POST /jobs/text.
  • Added `job_type` and `input_text` to the Job dataclass.
  • `process_job` now conditionally calls Whisper (if audio) and always calls Qwen.
==============================================================================
"""
import os
import time
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Deque
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gateway")


# ==============================================================================
# Configuration
# ==============================================================================
ADMIN_TOKEN = os.getenv("GATEWAY_ADMIN_TOKEN", "")
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

_downstream: Dict[str, str] = {
    "whisper": os.getenv("WHISPER_SERVICE_URL", "http://localhost:8001"),
    "qwen":    os.getenv("QWEN_SERVICE_URL",    "http://localhost:8002"),
}

WHISPER_TIMEOUT = float(os.getenv("WHISPER_TIMEOUT", "300.0"))
QWEN_TIMEOUT    = float(os.getenv("QWEN_TIMEOUT",    "120.0"))
HEALTH_TIMEOUT  = float(os.getenv("HEALTH_TIMEOUT",  "5.0"))

MAX_RETRIES    = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF  = float(os.getenv("RETRY_BACKOFF", "2.0"))

MAX_QUEUE_SIZE   = int(os.getenv("MAX_QUEUE_SIZE", "50"))
REJECT_THRESHOLD = float(os.getenv("REJECT_THRESHOLD", "0.9"))

JOB_TTL_SECONDS      = int(os.getenv("JOB_TTL_SECONDS", "1800"))
JOB_CLEANUP_INTERVAL = int(os.getenv("JOB_CLEANUP_INTERVAL", "60"))
MAX_JOBS_IN_STORE    = int(os.getenv("MAX_JOBS_IN_STORE", "1000"))

RATE_LIMIT_WINDOW = float(os.getenv("RATE_LIMIT_WINDOW", "60.0"))
RATE_LIMIT_MAX    = int(os.getenv("RATE_LIMIT_MAX", "10"))
MAX_UPLOAD_SIZE   = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))

LEGACY_WAIT_TIMEOUT = float(os.getenv("LEGACY_WAIT_TIMEOUT", "600.0"))
MAX_TEXT_LENGTH     = int(os.getenv("MAX_TEXT_LENGTH", "10000"))


def get_whisper_url() -> str:
    return _downstream["whisper"]

def get_qwen_url() -> str:
    return _downstream["qwen"]

def internal_headers() -> Dict[str, str]:
    if INTERNAL_TOKEN:
        return {"X-Internal-Token": INTERNAL_TOKEN}
    return {}


# ==============================================================================
# Enums & Data Structures
# ==============================================================================
class JobState(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


@dataclass
class Job:
    id: str
    client_ip: str
    filename: str
    content_type: str
    file_content: bytes
    job_type: str = "audio"  # "audio" or "text"
    input_text: Optional[str] = None
    state: JobState = JobState.QUEUED
    position: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    future: Optional[asyncio.Future] = None


# ==============================================================================
# Job Store
# ==============================================================================
class JobStore:
    def __init__(self, max_size: int, ttl: int):
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl = ttl

    async def add(self, job: Job) -> None:
        async with self._lock:
            if len(self._jobs) >= self._max_size:
                finished = [
                    (jid, j) for jid, j in self._jobs.items()
                    if j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
                ]
                finished.sort(key=lambda x: x[1].completed_at or x[1].created_at)
                for jid, _ in finished[: max(1, len(finished) // 10)]:
                    self._jobs.pop(jid, None)
            self._jobs[job.id] = job

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **kwargs) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)

    async def recompute_positions(self) -> None:
        async with self._lock:
            queued = [
                (jid, j) for jid, j in self._jobs.items()
                if j.state == JobState.QUEUED
            ]
            queued.sort(key=lambda x: x[1].created_at)
            for idx, (_, j) in enumerate(queued, start=1):
                j.position = idx

    async def cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(JOB_CLEANUP_INTERVAL)
                cutoff = time.time() - self._ttl
                async with self._lock:
                    stale = [
                        jid for jid, j in self._jobs.items()
                        if j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
                        and (j.completed_at or j.created_at) < cutoff
                    ]
                    for jid in stale:
                        self._jobs.pop(jid, None)
                    if stale:
                        logger.info(f"JobStore: cleaned {len(stale)} stale jobs. Total now: {len(self._jobs)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"JobStore cleanup error: {e}")

    def size(self) -> int:
        return len(self._jobs)


# ==============================================================================
# Rate Limiter
# ==============================================================================
class RateLimiter:
    def __init__(self, window: float, max_requests: int):
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._window = window
        self._max = max_requests

    async def check(self, ip: str) -> bool:
        async with self._lock:
            now = time.time()
            bucket = self._buckets[ip]
            while bucket and bucket[0] < now - self._window:
                bucket.popleft()
            if len(bucket) >= self._max:
                return False
            bucket.append(now)
            return True

    async def cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(300)
                cutoff = time.time() - self._window * 2
                async with self._lock:
                    dead = [ip for ip, b in self._buckets.items() if not b or b[-1] < cutoff]
                    for ip in dead:
                        self._buckets.pop(ip, None)
            except asyncio.CancelledError:
                break


# ==============================================================================
# Circuit Breaker
# ==============================================================================
class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.name = name
        self.failures = 0
        self.threshold = failure_threshold
        self.recovery = recovery_timeout
        self.opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self.opened_at is None:
                return True
            if time.time() - self.opened_at > self.recovery:
                logger.info(f"CircuitBreaker[{self.name}]: half-open retry")
                self.opened_at = None
                self.failures = 0
                return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self.failures = 0
            self.opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self.failures += 1
            if self.failures >= self.threshold and self.opened_at is None:
                self.opened_at = time.time()
                logger.warning(f"CircuitBreaker[{self.name}]: OPENED after {self.failures} failures")

    def is_open(self) -> bool:
        return self.opened_at is not None


# ==============================================================================
# Global State
# ==============================================================================
http_client: Optional[httpx.AsyncClient] = None
job_queue:   Optional[asyncio.Queue] = None
job_store:   Optional[JobStore] = None
rate_limiter: Optional[RateLimiter] = None
whisper_cb:  Optional[CircuitBreaker] = None
qwen_cb:     Optional[CircuitBreaker] = None

worker_task:       Optional[asyncio.Task] = None
cleanup_task:      Optional[asyncio.Task] = None
rate_cleanup_task: Optional[asyncio.Task] = None

metrics = {
    "total_submitted": 0,
    "total_completed": 0,
    "total_failed": 0,
    "total_rejected": 0,
    "total_cancelled": 0,
    "processing_times": deque(maxlen=50),
    "started_at": time.time(),
}


# ==============================================================================
# Retry with Exponential Backoff + Circuit Breaker
# ==============================================================================
async def post_with_retry(
    url: str,
    *,
    files: Optional[Dict] = None,
    json_body: Optional[Dict] = None,
    timeout: float = 60.0,
    request_id: str = "",
    breaker: Optional[CircuitBreaker] = None,
) -> httpx.Response:
    if breaker and not await breaker.allow():
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable (circuit open: {breaker.name})",
        )

    last_exception: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await http_client.post(
                url,
                files=files,
                json=json_body,
                headers=internal_headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            if breaker:
                await breaker.record_success()
            return response

        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                raise
            last_exception = e
            if breaker:
                await breaker.record_failure()

        except (httpx.ConnectError, httpx.ReadTimeout,
                httpx.RemoteProtocolError, httpx.PoolTimeout) as e:
            last_exception = e
            if breaker:
                await breaker.record_failure()

        if attempt < MAX_RETRIES:
            wait_time = RETRY_BACKOFF ** attempt
            logger.warning(
                f"[{request_id}] Attempt {attempt}/{MAX_RETRIES} failed for {url}: "
                f"{last_exception}. Retrying in {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)

    logger.error(f"[{request_id}] All {MAX_RETRIES} attempts failed for {url}.")
    raise last_exception if last_exception else Exception("Unknown downstream error")


# ==============================================================================
# Worker: Sequential Job Processor
# ==============================================================================
async def process_job(job: Job) -> Dict[str, Any]:
    request_id = job.id
    t_start = time.time()

    await job_store.update(request_id, state=JobState.PROCESSING, started_at=t_start)
    logger.info(f"[{request_id}] Processing started ({job.job_type}). Queue left: {job_queue.qsize()}")

    transcript = ""

    # ---- Step 1: Transcription (Only if Audio) ----
    if job.job_type == "audio":
        try:
            files = {"file": (job.filename, job.file_content, job.content_type)}
            whisper_resp = await post_with_retry(
                f"{get_whisper_url()}/transcribe",
                files=files,
                timeout=WHISPER_TIMEOUT,
                request_id=request_id,
                breaker=whisper_cb,
            )
            wdata = whisper_resp.json()
            transcript = (wdata.get("transcript") or "").strip()
            if not transcript:
                raise HTTPException(status_code=502, detail="Whisper returned empty transcript")
            logger.info(f"[{request_id}] Transcription OK ({len(transcript)} chars)")
        except HTTPException:
            raise
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Whisper error: {e.response.text[:200]}")
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Whisper unreachable: {type(e).__name__}")
    
    elif job.job_type == "text":
        transcript = (job.input_text or "").strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="Input text is empty")
        logger.info(f"[{request_id}] Text input OK ({len(transcript)} chars)")

    # ---- Step 2: Qwen Analysis (Always) ----
    try:
        qwen_resp = await post_with_retry(
            f"{get_qwen_url()}/analyze",
            json_body={"transcript": transcript},
            timeout=QWEN_TIMEOUT,
            request_id=request_id,
            breaker=qwen_cb,
        )
        qdata = qwen_resp.json()
        analysis = qdata.get("analysis", {})
        logger.info(f"[{request_id}] Analysis OK")
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Qwen error: {e.response.text[:200]}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Qwen unreachable: {type(e).__name__}")

    duration = time.time() - t_start
    metrics["processing_times"].append(duration)
    metrics["total_completed"] += 1

    logger.info(f"[{request_id}] DONE in {duration:.1f}s")
    return {"transcript": transcript, "analysis": analysis}


async def worker_loop() -> None:
    logger.info("Worker loop started.")
    while True:
        try:
            job: Job = await job_queue.get()

            await job_store.recompute_positions()

            if job.state == JobState.CANCELLED:
                logger.info(f"[{job.id}] Skipping cancelled job.")
                job_queue.task_done()
                continue

            try:
                result = await process_job(job)
                await job_store.update(
                    job.id,
                    state=JobState.COMPLETED,
                    result=result,
                    completed_at=time.time(),
                )
                if job.future and not job.future.done():
                    job.future.set_result(result)

            except HTTPException as e:
                metrics["total_failed"] += 1
                await job_store.update(
                    job.id,
                    state=JobState.FAILED,
                    error=str(e.detail),
                    completed_at=time.time(),
                )
                if job.future and not job.future.done():
                    job.future.set_exception(e)

            except Exception as e:
                metrics["total_failed"] += 1
                logger.exception(f"[{job.id}] Unexpected error")
                await job_store.update(
                    job.id,
                    state=JobState.FAILED,
                    error=f"Internal error: {type(e).__name__}",
                    completed_at=time.time(),
                )
                if job.future and not job.future.done():
                    job.future.set_exception(e)

            job_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker cancelled, shutting down.")
            break
        except Exception as e:
            logger.exception(f"Worker loop error: {e}")
            await asyncio.sleep(1)


# ==============================================================================
# Helpers
# ==============================================================================
def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def estimate_wait_seconds() -> float:
    if not metrics["processing_times"]:
        avg = 60.0
    else:
        avg = sum(metrics["processing_times"]) / len(metrics["processing_times"])
    return avg * max(1, job_queue.qsize() + 1)


# ==============================================================================
# Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, job_queue, job_store, rate_limiter
    global whisper_cb, qwen_cb
    global worker_task, cleanup_task, rate_cleanup_task

    logger.info("=" * 70)
    logger.info("Lordera Gateway v2.3.1 starting...")
    logger.info(f"Whisper:      {get_whisper_url()}")
    logger.info(f"Qwen:         {get_qwen_url()}")
    logger.info(f"Queue:        max={MAX_QUEUE_SIZE}, reject@{REJECT_THRESHOLD*100:.0f}%")
    logger.info(f"Rate:         {RATE_LIMIT_MAX} req / {RATE_LIMIT_WINDOW}s per IP")
    logger.info(f"Admin token:  {'configured' if ADMIN_TOKEN else 'DISABLED'}")
    logger.info(f"Internal tok: {'configured' if INTERNAL_TOKEN else 'DISABLED (dev mode)'}")
    logger.info("=" * 70)

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(WHISPER_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100),
    )
    job_queue    = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    job_store    = JobStore(max_size=MAX_JOBS_IN_STORE, ttl=JOB_TTL_SECONDS)
    rate_limiter = RateLimiter(window=RATE_LIMIT_WINDOW, max_requests=RATE_LIMIT_MAX)
    whisper_cb   = CircuitBreaker("whisper")
    qwen_cb      = CircuitBreaker("qwen")

    worker_task       = asyncio.create_task(worker_loop())
    cleanup_task      = asyncio.create_task(job_store.cleanup_loop())
    rate_cleanup_task = asyncio.create_task(rate_limiter.cleanup_loop())

    yield

    logger.info("Shutting down Gateway...")
    for t in (worker_task, cleanup_task, rate_cleanup_task):
        if t:
            t.cancel()
    await asyncio.gather(
        worker_task, cleanup_task, rate_cleanup_task, return_exceptions=True
    )
    if http_client:
        await http_client.aclose()
    logger.info("Shutdown complete.")


# ==============================================================================
# FastAPI App
# ==============================================================================
app = FastAPI(
    title="Lordera AI Gateway",
    description="Async job-based orchestrator for Whisper + Qwen.",
    version="2.3.1",
    lifespan=lifespan,
)

# --- CORS MIDDLEWARE (Required for Frontend connection) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------


# ==============================================================================
# Response Models
# ==============================================================================
class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    position: int
    estimated_wait_seconds: float
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    position: int
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_wait_seconds: Optional[float] = None
    transcript: Optional[str] = None
    analysis: Optional[dict] = None
    error: Optional[str] = None
    filename: str


class HealthResponse(BaseModel):
    status: str
    whisper_service: str
    qwen_service: str
    whisper_url: str
    qwen_url: str
    whisper_circuit_open: bool
    qwen_circuit_open: bool
    queue_size: int
    queue_max: int
    worker_running: bool
    jobs_in_store: int
    internal_token_set: bool


class DownstreamUpdateRequest(BaseModel):
    whisper_url: Optional[str] = None
    qwen_url: Optional[str] = None
    token: str


class DownstreamUpdateResponse(BaseModel):
    status: str
    whisper_url: str
    qwen_url: str


class TextJobSubmitRequest(BaseModel):
    text: str


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/")
async def root():
    return {
        "service": "Lordera AI Gateway",
        "version": "2.3.1",
        "endpoints": {
            "submit_audio": "POST /jobs",
            "submit_text":  "POST /jobs/text",
            "poll":         "GET  /jobs/{job_id}",
            "cancel":       "DELETE /jobs/{job_id}",
            "health":       "GET  /health",
            "queue":        "GET  /queue-status",
            "metrics":      "GET  /metrics",
            "legacy":       "POST /process-audio",
            "admin":        "POST /admin/downstream",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    whisper_status = "unknown"
    qwen_status = "unknown"

    try:
        r = await http_client.get(f"{get_whisper_url()}/health", headers=internal_headers(), timeout=HEALTH_TIMEOUT)
        whisper_status = "healthy" if r.status_code == 200 else f"unhealthy({r.status_code})"
    except Exception as e:
        whisper_status = f"unreachable({type(e).__name__})"

    try:
        r = await http_client.get(f"{get_qwen_url()}/health", headers=internal_headers(), timeout=HEALTH_TIMEOUT)
        qwen_status = "healthy" if r.status_code == 200 else f"unhealthy({r.status_code})"
    except Exception as e:
        qwen_status = f"unreachable({type(e).__name__})"

    overall = "healthy" if (whisper_status == "healthy" and qwen_status == "healthy") else "degraded"

    return HealthResponse(
        status=overall,
        whisper_service=whisper_status,
        qwen_service=qwen_status,
        whisper_url=get_whisper_url(),
        qwen_url=get_qwen_url(),
        whisper_circuit_open=whisper_cb.is_open(),
        qwen_circuit_open=qwen_cb.is_open(),
        queue_size=job_queue.qsize(),
        queue_max=MAX_QUEUE_SIZE,
        worker_running=worker_task is not None and not worker_task.done(),
        jobs_in_store=job_store.size(),
        internal_token_set=bool(INTERNAL_TOKEN),
    )


@app.get("/queue-status")
async def queue_status():
    avg = (
        sum(metrics["processing_times"]) / len(metrics["processing_times"])
        if metrics["processing_times"] else 0.0
    )
    return {
        "queue_size": job_queue.qsize(),
        "queue_max": MAX_QUEUE_SIZE,
        "queue_fill_pct": round(job_queue.qsize() / MAX_QUEUE_SIZE * 100, 1),
        "worker_running": worker_task is not None and not worker_task.done(),
        "jobs_in_store": job_store.size(),
        "avg_processing_seconds": round(avg, 1),
        "total_submitted":  metrics["total_submitted"],
        "total_completed":  metrics["total_completed"],
        "total_failed":     metrics["total_failed"],
        "total_rejected":   metrics["total_rejected"],
        "total_cancelled":  metrics["total_cancelled"],
        "whisper_circuit_open": whisper_cb.is_open(),
        "qwen_circuit_open":    qwen_cb.is_open(),
        "whisper_url": get_whisper_url(),
        "qwen_url": get_qwen_url(),
    }


@app.get("/metrics")
async def get_metrics():
    times = list(metrics["processing_times"])
    avg = sum(times) / len(times) if times else 0.0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 5 else avg
    return {
        "uptime_seconds": int(time.time() - metrics["started_at"]),
        "total_submitted":  metrics["total_submitted"],
        "total_completed":  metrics["total_completed"],
        "total_failed":     metrics["total_failed"],
        "total_rejected":   metrics["total_rejected"],
        "total_cancelled":  metrics["total_cancelled"],
        "avg_processing_seconds": round(avg, 2),
        "p95_processing_seconds": round(p95, 2),
        "queue_depth": job_queue.qsize(),
        "jobs_in_store": job_store.size(),
    }


# --------------- Async Audio Job Submission -----------------
@app.post("/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_audio_job(request: Request, file: UploadFile = File(...)):
    client_ip = get_client_ip(request)

    # ---- Rate limit ----
    if not await rate_limiter.check(client_ip):
        metrics["total_rejected"] += 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_MAX}/{int(RATE_LIMIT_WINDOW)}s). Slow down.",
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW))},
        )

    # ---- Backpressure ----
    if job_queue.qsize() >= int(MAX_QUEUE_SIZE * REJECT_THRESHOLD):
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=503, detail="Server busy. Please retry shortly.", headers={"Retry-After": "30"})
    if job_queue.full():
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=503, detail="Server busy (queue full).", headers={"Retry-After": "60"})

    # ---- Validate ----
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)} MB)")

    # ---- Create job ----
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        client_ip=client_ip,
        filename=file.filename or "audio",
        content_type=file.content_type,
        file_content=content,
        job_type="audio",
        position=job_queue.qsize() + 1,
    )
    await job_store.add(job)
    metrics["total_submitted"] += 1

    try:
        job_queue.put_nowait(job)
    except asyncio.QueueFull:
        metrics["total_rejected"] += 1
        await job_store.update(job_id, state=JobState.FAILED, error="queue full")
        raise HTTPException(status_code=503, detail="Server busy.", headers={"Retry-After": "30"})

    await job_store.recompute_positions()
    logger.info(f"[{job_id}] Audio submitted by {client_ip}. Queue: {job_queue.qsize()}/{MAX_QUEUE_SIZE}")

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        position=job.position,
        estimated_wait_seconds=round(estimate_wait_seconds(), 1),
        poll_url=f"/jobs/{job_id}",
    )


# --------------- Async Text Job Submission -----------------
@app.post("/jobs/text", response_model=JobSubmitResponse, status_code=202)
async def submit_text_job(request: Request, body: TextJobSubmitRequest):
    client_ip = get_client_ip(request)

    # ---- Rate limit ----
    if not await rate_limiter.check(client_ip):
        metrics["total_rejected"] += 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_MAX}/{int(RATE_LIMIT_WINDOW)}s). Slow down.",
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW))},
        )

    # ---- Backpressure ----
    if job_queue.qsize() >= int(MAX_QUEUE_SIZE * REJECT_THRESHOLD):
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=503, detail="Server busy. Please retry shortly.", headers={"Retry-After": "30"})
    if job_queue.full():
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=503, detail="Server busy (queue full).", headers={"Retry-After": "60"})

    # ---- Validate ----
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=413, detail=f"Text too long (max {MAX_TEXT_LENGTH} chars)")

    # ---- Create job ----
    job_id = uuid.uuid4().hex[:12]
    job = Job(
        id=job_id,
        client_ip=client_ip,
        filename="text_input.txt",
        content_type="text/plain",
        file_content=b"",
        job_type="text",
        input_text=text,
        position=job_queue.qsize() + 1,
    )
    await job_store.add(job)
    metrics["total_submitted"] += 1

    try:
        job_queue.put_nowait(job)
    except asyncio.QueueFull:
        metrics["total_rejected"] += 1
        await job_store.update(job_id, state=JobState.FAILED, error="queue full")
        raise HTTPException(status_code=503, detail="Server busy.", headers={"Retry-After": "30"})

    await job_store.recompute_positions()
    logger.info(f"[{job_id}] Text submitted by {client_ip}. Queue: {job_queue.qsize()}/{MAX_QUEUE_SIZE}")

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        position=job.position,
        estimated_wait_seconds=round(estimate_wait_seconds(), 1),
        poll_url=f"/jobs/{job_id}",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    estimated = None
    if job.state == JobState.QUEUED:
        estimated = round(estimate_wait_seconds(), 1)

    return JobStatusResponse(
        job_id=job.id,
        status=job.state.value,
        position=job.position if job.state == JobState.QUEUED else 0,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        estimated_wait_seconds=estimated,
        transcript=(job.result or {}).get("transcript") if job.state == JobState.COMPLETED else None,
        analysis=(job.result or {}).get("analysis") if job.state == JobState.COMPLETED else None,
        error=job.error,
        filename=job.filename,
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state == JobState.PROCESSING:
        raise HTTPException(status_code=409, detail="Job already processing; cannot cancel")
    if job.state in (JobState.COMPLETED, JobState.FAILED):
        raise HTTPException(status_code=409, detail=f"Job already {job.state.value}")

    await job_store.update(job_id, state=JobState.CANCELLED, completed_at=time.time())
    metrics["total_cancelled"] += 1
    logger.info(f"[{job_id}] Cancelled by client")
    return {"status": "cancelled", "job_id": job_id}


# --------------- Admin: Dynamic Downstream Update -----------------
@app.post("/admin/downstream", response_model=DownstreamUpdateResponse)
async def update_downstream(req: DownstreamUpdateRequest):
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Admin token not configured on this Gateway")
    if req.token != ADMIN_TOKEN:
        logger.warning("Rejected admin/downstream with invalid token")
        raise HTTPException(status_code=403, detail="Invalid admin token")

    updated: Dict[str, str] = {}
    if req.whisper_url:
        _downstream["whisper"] = req.whisper_url.rstrip("/")
        updated["whisper"] = _downstream["whisper"]
    if req.qwen_url:
        _downstream["qwen"] = req.qwen_url.rstrip("/")
        updated["qwen"] = _downstream["qwen"]

    if updated:
        logger.info(f"Downstream URLs updated: {updated}")
    else:
        logger.info("Downstream update called with no changes")

    return DownstreamUpdateResponse(
        status="updated" if updated else "no_change",
        whisper_url=_downstream["whisper"],
        qwen_url=_downstream["qwen"],
    )


# --------------- Legacy Blocking Endpoint (compat only) ---------------
@app.post("/process-audio")
async def process_audio_legacy(request: Request, file: UploadFile = File(...)):
    client_ip = get_client_ip(request)

    if not await rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    if job_queue.full():
        raise HTTPException(status_code=503, detail="Server busy", headers={"Retry-After": "60"})

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    job_id = uuid.uuid4().hex[:12]
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    job = Job(
        id=job_id,
        client_ip=client_ip,
        filename=file.filename or "audio",
        content_type=file.content_type,
        file_content=content,
        job_type="audio",
        position=job_queue.qsize() + 1,
        future=future,
    )
    await job_store.add(job)
    metrics["total_submitted"] += 1
    job_queue.put_nowait(job)

    try:
        result = await asyncio.wait_for(future, timeout=LEGACY_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        future.cancel()
        raise HTTPException(status_code=504, detail=f"Timeout after {LEGACY_WAIT_TIMEOUT}s. Use /jobs for async pattern.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success",
        "job_id": job_id,
        "transcript": result["transcript"],
        "analysis": result["analysis"],
        "filename": file.filename,
    }


# ==============================================================================
# Run (for local development)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)