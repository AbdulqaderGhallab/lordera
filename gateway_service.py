"""
==============================================================================
Lordera AI — Gateway Service v3.5.0
Smart Chat Orchestrator + Edge-TTS (Free Arabic Voices)
==============================================================================
CHANGELOG v3.4.0 → v3.5.0:
  • MAJOR: Replaced ElevenLabs with Microsoft Edge-TTS (free, no API key).
  • NEW: Arabic voices (Hamed, Zariyah, Salma, Shakir) available by default.
  • FIX: Output is standard MP3 (audio/mpeg) — universally supported.
  • REMOVED: All ElevenLabs configurations and dependencies.
  • Version bumped to 3.5.0.
==============================================================================
"""
import os
import json
import time
import uuid
import re
import base64
import logging
import asyncio
import urllib.parse
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Deque, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

import httpx
import edge_tts
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
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
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_API_BASE      = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
GROQ_QWEN_MODEL    = os.getenv("GROQ_QWEN_MODEL", "qwen/qwen3.8-27b")

POLLINATIONS_API_KEY     = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_IMAGE_BASE  = "https://image.pollinations.ai"
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")

WHISPER_TIMEOUT = float(os.getenv("WHISPER_TIMEOUT", "120.0"))
QWEN_TIMEOUT    = float(os.getenv("QWEN_TIMEOUT",    "60.0"))
IMAGE_TIMEOUT   = float(os.getenv("IMAGE_TIMEOUT",   "120.0"))

MAX_RETRIES   = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "2.0"))

MAX_QUEUE_SIZE   = int(os.getenv("MAX_QUEUE_SIZE", "50"))
REJECT_THRESHOLD = float(os.getenv("REJECT_THRESHOLD", "0.9"))

JOB_TTL_SECONDS      = int(os.getenv("JOB_TTL_SECONDS", "1800"))
JOB_CLEANUP_INTERVAL = int(os.getenv("JOB_CLEANUP_INTERVAL", "60"))
MAX_JOBS_IN_STORE    = int(os.getenv("MAX_JOBS_IN_STORE", "1000"))

RATE_LIMIT_WINDOW = float(os.getenv("RATE_LIMIT_WINDOW", "60.0"))
RATE_LIMIT_MAX    = int(os.getenv("RATE_LIMIT_MAX", "10"))
MAX_UPLOAD_SIZE   = int(os.getenv("MAX_UPLOAD_SIZE", str(25 * 1024 * 1024)))
LEGACY_WAIT_TIMEOUT = float(os.getenv("LEGACY_WAIT_TIMEOUT", "120.0"))
MAX_TEXT_LENGTH     = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
CHAT_MAX_HISTORY    = int(os.getenv("CHAT_MAX_HISTORY", "20"))
CHAT_MAX_TOKENS     = int(os.getenv("CHAT_MAX_TOKENS", "1024"))

# ── Edge TTS Voices ──
EDGE_VOICES = {
    "hamed": "ar-SA-HamedNeural",      # ذكر سعودي (افتراضي)
    "zariyah": "ar-SA-ZariyahNeural",  # أنثى سعودية
    "salma": "ar-EG-SalmaNeural",      # أنثى مصرية
    "shakir": "ar-EG-ShakirNeural",    # ذكر مصري
}

def _normalize_tts_voice(v: Optional[str]) -> str:
    if not v:
        return EDGE_VOICES["hamed"]
    v_lower = v.lower()
    if v_lower in EDGE_VOICES:
        return EDGE_VOICES[v_lower]
    # If it's already a full Microsoft voice name (e.g., ar-SA-HamedNeural), keep it
    if "-Neural" in v:
        return v
    return EDGE_VOICES["hamed"]


# ==============================================================================
# Prompts
# ==============================================================================
ANALYZER_SYSTEM_PROMPT = (
    "You are a precise semantic analyzer. "
    "You MUST respond with a single valid JSON object and nothing else."
)

ANALYZER_USER_PROMPT = """Analyze the following transcript and return a JSON object with exactly these keys:
- "summary": a concise 1-2 sentence summary in the SAME LANGUAGE as the transcript.
- "sentiment": one of "positive", "negative", or "neutral".
- "key_topics": an array of 1-5 short topic strings in the SAME LANGUAGE as the transcript.
- "intent": one of "request", "complaint", "inquiry", "informational", "feedback", or "other".

Rules:
- Output ONLY the JSON. No explanations, no markdown fences.
- Keep the summary under 200 characters.
- If the transcript is empty or unclear, use empty strings/arrays.

Transcript:
\"\"\"
{transcript}
\"\"\"

JSON:"""

CHAT_SYSTEM_PROMPT = """أنت Lordera — مساعد ذكاء اصطناعي عربي ذكي وودود ومتعدد القدرات.

مهمتك: اقرأ المحادثة، قرر الإجراء المناسب، واكتب رداً عربياً طبيعياً.

الإجراءات المتاحة:
- "chat": محادثة عادية (افتراضي). استخدمها لأي كلام عادي، أسئلة، دردشة.
- "image": المستخدم يطلب توليد صورة. توفير "prompt_en" (وصف إنجليزي غني جداً).
- "tts": تحويل نص إلى كلام. توفير "text" (النص المطلوب نطقه بالعربية الفصحى أو حسب السياق). 
  *ملاحظة:* محرك الصوت لا يدعم وسوم المشاعر، لذا اكتب النص نظيفاً وواضحاً.
- "analyze": تحليل نص. توفير "text".

قواعد صارمة:
1. أجب فقط بكائن JSON واحد بهذا الشكل:
   {
     "action": "chat" | "image" | "tts" | "analyze",
     "reply": "ردك العربي الطبيعي الذي سيظهر للمستخدم",
     "params": { ... }
   }
2. "reply" دائماً بالعربية، طبيعي، دافئ، مختصر (سطر إلى ثلاثة).
3. للصور: اجعل prompt_en مفصلاً جداً بالإنجليزية.
4. للصوت: استخرج النص المطلوب نطقه فقط، بدون "قل:" أو "انطق:".
"""


# ==============================================================================
# Enums & Data
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
    job_type: str = "audio"
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
# JobStore, RateLimiter, CircuitBreaker
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
                finished = [(jid, j) for jid, j in self._jobs.items()
                            if j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)]
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
            if not job: return
            for k, v in kwargs.items():
                setattr(job, k, v)

    async def recompute_positions(self) -> None:
        async with self._lock:
            queued = [(jid, j) for jid, j in self._jobs.items() if j.state == JobState.QUEUED]
            queued.sort(key=lambda x: x[1].created_at)
            for idx, (_, j) in enumerate(queued, start=1):
                j.position = idx

    async def cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(JOB_CLEANUP_INTERVAL)
                cutoff = time.time() - self._ttl
                async with self._lock:
                    stale = [jid for jid, j in self._jobs.items()
                             if j.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
                             and (j.completed_at or j.created_at) < cutoff]
                    for jid in stale:
                        self._jobs.pop(jid, None)
            except asyncio.CancelledError:
                break

    def size(self) -> int:
        return len(self._jobs)


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
            if self.opened_at is None: return True
            if time.time() - self.opened_at > self.recovery:
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

    def is_open(self) -> bool:
        return self.opened_at is not None


# ==============================================================================
# Global State
# ==============================================================================
http_client: Optional[httpx.AsyncClient] = None
job_queue:    Optional[asyncio.Queue] = None
job_store:    Optional[JobStore] = None
rate_limiter: Optional[RateLimiter] = None
whisper_cb:   Optional[CircuitBreaker] = None
qwen_cb:      Optional[CircuitBreaker] = None
tts_cb:       Optional[CircuitBreaker] = None

worker_task:       Optional[asyncio.Task] = None
cleanup_task:      Optional[asyncio.Task] = None
rate_cleanup_task: Optional[asyncio.Task] = None

metrics = {
    "total_submitted": 0, "total_completed": 0, "total_failed": 0,
    "total_rejected": 0, "total_cancelled": 0, "total_chat": 0,
    "total_images": 0, "total_tts": 0,
    "processing_times": deque(maxlen=50), "started_at": time.time(),
}


# ==============================================================================
# JSON extraction & Analysis Helpers
# ==============================================================================
def _extract_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try: return json.loads(fenced)
    except json.JSONDecodeError: pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[start:i+1])
                    except: break
    return {"raw_analysis": text[:2000]}


def _normalize_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    if "raw_analysis" in data: return data
    summary = str(data.get("summary", "")).strip()[:500]
    sentiment = str(data.get("sentiment", "neutral")).strip().lower()
    if sentiment not in ("positive", "negative", "neutral"): sentiment = "neutral"
    topics = data.get("key_topics", [])
    if isinstance(topics, str): topics = [topics]
    if not isinstance(topics, list): topics = []
    topics = [str(t).strip()[:80] for t in topics if str(t).strip()][:5]
    intent = str(data.get("intent", "other")).strip().lower()
    allowed_intents = {"request", "complaint", "inquiry", "informational", "feedback", "other"}
    if intent not in allowed_intents: intent = "other"
    return {"summary": summary, "sentiment": sentiment, "key_topics": topics, "intent": intent}


# ==============================================================================
# API Clients
# ==============================================================================
async def _groq_request(method: str, path: str, **kwargs) -> httpx.Response:
    if not GROQ_API_KEY: raise HTTPException(503, "GROQ_API_KEY missing")
    breaker = kwargs.pop("breaker", None)
    if breaker and not await breaker.allow(): raise HTTPException(503, "Circuit open")

    url = f"{GROQ_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = await http_client.request(method, url, headers=headers, **kwargs)
            res.raise_for_status()
            if breaker: await breaker.record_success()
            return res
        except httpx.HTTPStatusError as e:
            if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                if breaker: await breaker.record_failure()
                raise HTTPException(502, f"Groq error: {e.response.text[:200]}")
            if breaker: await breaker.record_failure()
        except Exception:
            if breaker: await breaker.record_failure()
        if attempt < MAX_RETRIES: await asyncio.sleep(RETRY_BACKOFF ** attempt)

    raise HTTPException(502, "Groq unreachable")


async def transcribe_audio(audio_bytes: bytes, filename: str, content_type: str) -> str:
    files = {"file": (filename, audio_bytes, content_type or "audio/mpeg")}
    data = {"model": GROQ_WHISPER_MODEL, "response_format": "json"}
    res = await _groq_request("POST", "/audio/transcriptions", files=files, data=data, timeout=WHISPER_TIMEOUT, breaker=whisper_cb)
    return (res.json().get("text") or "").strip()


async def analyze_transcript(transcript: str) -> Dict[str, Any]:
    json_body = {
        "model": GROQ_QWEN_MODEL,
        "messages": [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": ANALYZER_USER_PROMPT.format(transcript=transcript)}
        ],
        "temperature": 0, "max_tokens": 512
    }
    res = await _groq_request("POST", "/chat/completions", json_body=json_body, timeout=QWEN_TIMEOUT, breaker=qwen_cb)
    return _normalize_analysis(_extract_json(res.json()["choices"][0]["message"]["content"]))


async def _generate_image_bytes(prompt_en: str) -> Tuple[bytes, str]:
    encoded = urllib.parse.quote(prompt_en)
    url = f"{POLLINATIONS_IMAGE_BASE}/prompt/{encoded}?model={POLLINATIONS_IMAGE_MODEL}&width=1024&height=1024&nologo=true&enhance=true"
    if POLLINATIONS_API_KEY: url += f"&key={POLLINATIONS_API_KEY}"
    headers = {"Authorization": f"Bearer {POLLINATIONS_API_KEY}"} if POLLINATIONS_API_KEY else {}
    res = await http_client.get(url, headers=headers, timeout=IMAGE_TIMEOUT)
    res.raise_for_status()
    return res.content, res.headers.get("content-type", "image/jpeg")


async def _generate_tts_bytes(text: str, voice: str) -> bytes:
    if tts_cb and not await tts_cb.allow():
        raise HTTPException(status_code=503, detail="TTS circuit open")
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]
        
        if not audio_bytes:
            raise HTTPException(status_code=502, detail="Edge TTS returned empty audio")
        
        if tts_cb: await tts_cb.record_success()
        return audio_bytes
    except Exception as e:
        if tts_cb: await tts_cb.record_failure()
        logger.error(f"Edge TTS failed: {e}")
        raise HTTPException(status_code=502, detail=f"Edge TTS failed: {type(e).__name__}")


# ==============================================================================
# Chat Orchestrator
# ==============================================================================
async def orchestrate_chat(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    trimmed = messages[-CHAT_MAX_HISTORY:] if len(messages) > CHAT_MAX_HISTORY else messages
    qwen_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for m in trimmed:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            qwen_messages.append({"role": m["role"], "content": m["content"]})

    json_body = {"model": GROQ_QWEN_MODEL, "messages": qwen_messages, "temperature": 0.4, "max_tokens": CHAT_MAX_TOKENS}
    res = await _groq_request("POST", "/chat/completions", json_body=json_body, timeout=QWEN_TIMEOUT, breaker=qwen_cb)
    decision = _extract_json(res.json()["choices"][0]["message"]["content"])

    action = decision.get("action")
    reply = decision.get("reply", "حسناً.")
    params = decision.get("params") or {}

    if action not in ("chat", "image", "tts", "analyze"):
        return {"reply": decision.get("raw_analysis", reply), "action": "chat"}

    if action == "image":
        prompt_en = params.get("prompt_en", "").strip()
        if not prompt_en: return {"reply": reply + "\n\n⚠️ لم أستطع تحديد وصف الصورة.", "action": "chat"}
        try:
            img_bytes, ct = await _generate_image_bytes(prompt_en)
            metrics["total_images"] += 1
            return {"reply": reply, "action": "image", "media_base64": f"data:{ct};base64,{base64.b64encode(img_bytes).decode()}", "media_content_type": ct}
        except Exception as e:
            return {"reply": reply + f"\n\n⚠️ تعذر توليد الصورة ({type(e).__name__}).", "action": "chat"}

    if action == "tts":
        text = params.get("text", "").strip()
        voice = _normalize_tts_voice(params.get("voice"))
        if not text: return {"reply": reply + "\n\n⚠️ لم أستطع تحديد النص.", "action": "chat"}
        try:
            audio_bytes = await _generate_tts_bytes(text, voice)
            metrics["total_tts"] += 1
            return {"reply": reply, "action": "tts", "media_base64": f"data:audio/mpeg;base64,{base64.b64encode(audio_bytes).decode()}", "media_content_type": "audio/mpeg"}
        except Exception as e:
            return {"reply": reply + f"\n\n⚠️ تعذر توليد الصوت ({type(e).__name__}).", "action": "chat"}

    if action == "analyze":
        text = params.get("text", "").strip()
        if not text: return {"reply": reply + "\n\n⚠️ لم أستطع تحديد النص.", "action": "chat"}
        try:
            analysis = await analyze_transcript(text)
            return {"reply": reply, "action": "analyze", "analysis": analysis}
        except Exception as e:
            return {"reply": reply + f"\n\n⚠️ تعذر التحليل ({type(e).__name__}).", "action": "chat"}

    return {"reply": reply, "action": "chat"}


# ==============================================================================
# Worker
# ==============================================================================
async def process_job(job: Job) -> Dict[str, Any]:
    await job_store.update(job.id, state=JobState.PROCESSING, started_at=time.time())
    transcript = ""
    if job.job_type == "audio":
        transcript = await transcribe_audio(job.file_content, job.filename, job.content_type)
    elif job.job_type == "text":
        transcript = (job.input_text or "").strip()

    analysis = await analyze_transcript(transcript)
    metrics["total_completed"] += 1
    return {"transcript": transcript, "analysis": analysis}


async def worker_loop() -> None:
    while True:
        try:
            if job_queue.empty(): await asyncio.sleep(1); continue
            job = await job_queue.get()
            await job_store.recompute_positions()
            if job.state == JobState.CANCELLED: job_queue.task_done(); continue
            try:
                result = await process_job(job)
                await job_store.update(job.id, state=JobState.COMPLETED, result=result, completed_at=time.time())
                if job.future and not job.future.done(): job.future.set_result(result)
            except Exception as e:
                metrics["total_failed"] += 1
                await job_store.update(job.id, state=JobState.FAILED, error=str(e), completed_at=time.time())
                if job.future and not job.future.done(): job.future.set_exception(e)
            job_queue.task_done()
        except asyncio.CancelledError: break
        except Exception as e: logger.exception(f"Worker error: {e}"); await asyncio.sleep(1)


# ==============================================================================
# Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, job_queue, job_store, rate_limiter, whisper_cb, qwen_cb, tts_cb
    global worker_task, cleanup_task, rate_cleanup_task

    http_client = httpx.AsyncClient(timeout=httpx.Timeout(WHISPER_TIMEOUT, connect=10.0))
    job_queue    = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    job_store    = JobStore(max_size=MAX_JOBS_IN_STORE, ttl=JOB_TTL_SECONDS)
    rate_limiter = RateLimiter(window=RATE_LIMIT_WINDOW, max_requests=RATE_LIMIT_MAX)
    whisper_cb   = CircuitBreaker("whisper")
    qwen_cb      = CircuitBreaker("qwen")
    tts_cb       = CircuitBreaker("edge-tts")

    worker_task       = asyncio.create_task(worker_loop())
    cleanup_task      = asyncio.create_task(job_store.cleanup_loop())
    rate_cleanup_task = asyncio.create_task(rate_limiter.cleanup_loop())

    yield

    for t in (worker_task, cleanup_task, rate_cleanup_task): t.cancel()
    await asyncio.gather(worker_task, cleanup_task, rate_cleanup_task, return_exceptions=True)
    await http_client.aclose()


# ==============================================================================
# FastAPI App & Endpoints
# ==============================================================================
app = FastAPI(title="Lordera AI Gateway", version="3.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ChatMessage(BaseModel): role: str; content: str
class ChatRequest(BaseModel): messages: List[ChatMessage]
class TTSRequest(BaseModel): text: str; voice: Optional[str] = None
class ImageRequest(BaseModel): prompt: str

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if GROQ_API_KEY else "degraded",
        "version": "3.5.0",
        "tts_engine": "Edge-TTS (Free)",
        "groq_configured": bool(GROQ_API_KEY)
    }

@app.post("/chat")
async def chat_endpoint(request: Request, body: ChatRequest):
    if not await rate_limiter.check(request.client.host): raise HTTPException(429, "Rate limited")
    try:
        return await orchestrate_chat([{"role": m.role, "content": m.content} for m in body.messages])
    except Exception as e:
        logger.exception("Chat failed")
        return {"reply": f"⚠️ خطأ: {type(e).__name__}", "action": "chat"}

@app.post("/transcribe")
async def transcribe_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    return {"transcript": await transcribe_audio(content, file.filename, file.content_type)}

@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    voice = _normalize_tts_voice(req.voice)
    audio = await _generate_tts_bytes(req.text, voice)
    return Response(content=audio, media_type="audio/mpeg")

@app.post("/image")
async def image_endpoint(req: ImageRequest):
    img, ct = await _generate_image_bytes(req.prompt)
    return Response(content=img, media_type=ct)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)