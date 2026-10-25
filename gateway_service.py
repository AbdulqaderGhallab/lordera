"""
==============================================================================
Lordera AI — Gateway Service v3.3.0
Smart Chat Orchestrator + Groq TTS Arabic + Pollinations Images
==============================================================================
CHANGELOG v3.2.0 → v3.3.0:
  • NEW: /chat endpoint — Qwen acts as a smart orchestrator that decides
    between normal chat / image generation / TTS / analysis.
  • NEW: Arabic → English prompt translation for images (Qwen handles it).
  • NEW: /transcribe endpoint for fast audio → text.
  • FIX: TTS voice validation (only Orpheus Arabic voices accepted).
  • FIX: Media returned as base64 inside /chat response.
  • Chat history is trimmed to last 20 messages to fit Qwen context.
  • Version bumped to 3.3.0.
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
GROQ_TTS_MODEL     = os.getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-arabic-saudi")
GROQ_TTS_VOICE     = os.getenv("GROQ_TTS_VOICE", "abdullah")

POLLINATIONS_API_KEY     = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_IMAGE_BASE  = "https://image.pollinations.ai"
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux")

WHISPER_TIMEOUT = float(os.getenv("WHISPER_TIMEOUT", "120.0"))
QWEN_TIMEOUT    = float(os.getenv("QWEN_TIMEOUT",    "60.0"))
TTS_TIMEOUT     = float(os.getenv("TTS_TIMEOUT",     "60.0"))
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

# Valid Orpheus Arabic voices on Groq
VALID_ARABIC_VOICES = {"abdullah", "fahad", "sultan", "lulwa", "noura", "aisha"}


def _normalize_tts_voice(v: Optional[str]) -> str:
    if not v:
        return GROQ_TTS_VOICE
    v = v.strip().lower()
    return v if v in VALID_ARABIC_VOICES else GROQ_TTS_VOICE


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


CHAT_SYSTEM_PROMPT = """أنت Lordera — مساعد ذكاء اصطناعي عربي ذكي وودود ومتعدد القدرات، تشبه ChatGPT وClaude في طبيعته.

مهمتك: اقرأ المحادثة، قرر الإجراء المناسب، واكتب رداً عربياً طبيعياً.

الإجراءات المتاحة:
- "chat": محادثة عادية (افتراضي). استخدمها لأي كلام عادي، أسئلة، دردشة، تحية، سؤال عنك...
- "image": المستخدم يطلب توليد صورة/رسم/تخيّل. يجب توفير "prompt_en" — وصف إنجليزي غني جداً (الموضوع، الأسلوب، الألوان، الإضاءة، الكاميرا). حتى لو كتب المستخدم بالعربية، الوصف يجب أن يكون إنجليزياً لأن نموذج الصور لا يفهم العربية.
- "tts": المستخدم يطلب تحويل نص إلى كلام. توفير "text" (النص المطلوب نطقه كما هو) و"voice" (اختياري: abdullah, fahad, sultan, lulwa, noura, aisha — الافتراضي abdullah).
- "analyze": المستخدم يطلب تحليل نص (ملخص/مشاعر/مواضيع/نية). توفير "text".

قواعد صارمة:
1. أجب فقط بكائن JSON واحد بهذا الشكل:
   {
     "action": "chat" | "image" | "tts" | "analyze",
     "reply": "ردك العربي الطبيعي الذي سيظهر للمستخدم",
     "params": { ... }
   }
2. "reply" دائماً بالعربية، طبيعي، دافئ، مختصر (سطر إلى ثلاثة).
3. لا تذكر أبداً JSON أو أدوات أو نماذج أو خادم في "reply".
4. للصور: اجعل prompt_en مفصلاً جداً بالإنجليزية (فاعل + أسلوب + إضاءة + جودة).
5. للصوت: استخرج النص المطلوب نطقه فقط، بدون "قل:" أو "انطق:".
6. للتحليل: استخدم النص الذي أعطاه المستخدم حرفياً في params.text.

أمثلة:
- "كيف حالك؟" → {"action":"chat","reply":"الحمد لله، بخير! كيف أقدر أساعدك اليوم؟","params":{}}
- "ارسم لي قطة على كتاب في مكتبة قديمة" → {"action":"image","reply":"سأرسم لك هذا المشهد الآن","params":{"prompt_en":"A cute cat sitting on an old open book in a vintage wooden library, warm golden lighting, dust particles in air, cinematic photography, high detail, 4k"}}
- "حوّل هذا النص إلى صوت: مرحباً بكم" → {"action":"tts","reply":"حسناً، سأنطق النص بصوت abdullah","params":{"text":"مرحباً بكم","voice":"abdullah"}}
- "حلّل: المنتج ممتاز لكن الشحن بطيء" → {"action":"analyze","reply":"سأحلل لك النص الآن","params":{"text":"المنتج ممتاز لكن الشحن بطيء"}}
- "ما هي عاصمة فرنسا؟" → {"action":"chat","reply":"عاصمة فرنسا هي باريس. هل تود معرفة شيء آخر عنها؟","params":{}}
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
# JobStore
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
            if not job:
                return
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
                    if stale:
                        logger.info(f"JobStore cleaned {len(stale)} jobs. Total: {len(self._jobs)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"JobStore cleanup error: {e}")

    def size(self) -> int:
        return len(self._jobs)


# ==============================================================================
# RateLimiter
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
# CircuitBreaker
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
                logger.info(f"CircuitBreaker[{self.name}] half-open")
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
                logger.warning(f"CircuitBreaker[{self.name}] OPENED after {self.failures} failures")

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
    "total_submitted": 0,
    "total_completed": 0,
    "total_failed": 0,
    "total_rejected": 0,
    "total_cancelled": 0,
    "total_chat": 0,
    "total_images": 0,
    "total_tts": 0,
    "processing_times": deque(maxlen=50),
    "started_at": time.time(),
}


# ==============================================================================
# JSON extraction
# ==============================================================================
def _extract_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return {"raw_analysis": text[:2000]}


def _normalize_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    if "raw_analysis" in data:
        return data
    summary = str(data.get("summary", "")).strip()[:500]
    sentiment = str(data.get("sentiment", "neutral")).strip().lower()
    if sentiment not in ("positive", "negative", "neutral"):
        sentiment = "neutral"
    topics = data.get("key_topics", [])
    if isinstance(topics, str):
        topics = [topics]
    if not isinstance(topics, list):
        topics = []
    topics = [str(t).strip()[:80] for t in topics if str(t).strip()][:5]
    intent = str(data.get("intent", "other")).strip().lower()
    allowed_intents = {"request", "complaint", "inquiry", "informational", "feedback", "other"}
    if intent not in allowed_intents:
        intent = "other"
    return {"summary": summary, "sentiment": sentiment, "key_topics": topics, "intent": intent}


# ==============================================================================
# Groq API client
# ==============================================================================
async def _groq_request(
    method: str,
    path: str,
    *,
    json_body: Optional[Dict] = None,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None,
    timeout: float = 60.0,
    request_id: str = "",
    breaker: Optional[CircuitBreaker] = None,
) -> httpx.Response:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the gateway")
    if breaker and not await breaker.allow():
        raise HTTPException(status_code=503, detail=f"Service temporarily unavailable (circuit open: {breaker.name})")

    url = f"{GROQ_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    last_exception: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await http_client.request(
                method, url, headers=headers,
                json=json_body, files=files, data=data, timeout=timeout,
            )
            if response.status_code == 429:
                retry_after = float(response.headers.get("retry-after", "5") or 5)
                logger.warning(f"[{request_id}] Groq 429. Retry in {retry_after}s")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(min(retry_after, 30))
                    continue
            response.raise_for_status()
            if breaker:
                await breaker.record_success()
            return response

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            # 4xx (except 429) is a hard error - don't retry
            if 400 <= status < 500 and status != 429:
                if breaker:
                    await breaker.record_failure()
                raise HTTPException(
                    status_code=502,
                    detail=f"Groq API {status}: {e.response.text[:300]}",
                )
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
            logger.warning(f"[{request_id}] Groq attempt {attempt}/{MAX_RETRIES} failed. Retry in {wait_time:.1f}s")
            await asyncio.sleep(wait_time)

    if breaker:
        await breaker.record_failure()
    raise HTTPException(
        status_code=502,
        detail=f"Groq API unreachable after {MAX_RETRIES} attempts: {type(last_exception).__name__}",
    )


async def transcribe_audio(audio_bytes: bytes, filename: str, content_type: str, request_id: str) -> str:
    files = {"file": (filename, audio_bytes, content_type or "audio/mpeg")}
    data = {"model": GROQ_WHISPER_MODEL, "response_format": "json"}
    response = await _groq_request(
        "POST", "/audio/transcriptions",
        files=files, data=data,
        timeout=WHISPER_TIMEOUT, request_id=request_id, breaker=whisper_cb,
    )
    payload = response.json()
    return (payload.get("text") or "").strip()


async def analyze_transcript(transcript: str, request_id: str) -> Dict[str, Any]:
    prompt = ANALYZER_USER_PROMPT.format(transcript=transcript)
    json_body = {
        "model": GROQ_QWEN_MODEL,
        "messages": [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    response = await _groq_request(
        "POST", "/chat/completions",
        json_body=json_body,
        timeout=QWEN_TIMEOUT, request_id=request_id, breaker=qwen_cb,
    )
    payload = response.json()
    try:
        raw = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Groq response shape: {e}")
    return _normalize_analysis(_extract_json(raw))


async def _generate_image_bytes(prompt_en: str) -> Tuple[bytes, str]:
    encoded = urllib.parse.quote(prompt_en)
    url = (f"{POLLINATIONS_IMAGE_BASE}/prompt/{encoded}"
           f"?model={POLLINATIONS_IMAGE_MODEL}&width=1024&height=1024"
           f"&nologo=true&enhance=true")
    if POLLINATIONS_API_KEY:
        url += f"&key={POLLINATIONS_API_KEY}"
    headers = {}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
    resp = await http_client.get(url, headers=headers, timeout=IMAGE_TIMEOUT)
    resp.raise_for_status()
    ct = resp.headers.get("content-type", "image/jpeg")
    if "application/json" in ct:
        raise RuntimeError(f"Pollinations error: {resp.text[:200]}")
    return resp.content, ct


async def _generate_tts_bytes(text: str, voice: str) -> bytes:
    json_body = {
        "model": GROQ_TTS_MODEL,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
    }
    resp = await _groq_request(
        "POST", "/audio/speech",
        json_body=json_body,
        timeout=TTS_TIMEOUT, request_id="tts", breaker=tts_cb,
    )
    return resp.content


# ==============================================================================
# Chat Orchestrator
# ==============================================================================
async def orchestrate_chat(messages: List[Dict[str, str]], request_id: str) -> Dict[str, Any]:
    """
    Sends the conversation to Qwen, gets a JSON decision, executes the action,
    returns {reply, action, media_base64?, media_content_type?, analysis?}.
    """
    # Trim history
    trimmed = messages[-CHAT_MAX_HISTORY:] if len(messages) > CHAT_MAX_HISTORY else messages

    qwen_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for m in trimmed:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            qwen_messages.append({"role": role, "content": content})

    json_body = {
        "model": GROQ_QWEN_MODEL,
        "messages": qwen_messages,
        "temperature": 0.4,
        "max_tokens": CHAT_MAX_TOKENS,
    }

    response = await _groq_request(
        "POST", "/chat/completions",
        json_body=json_body,
        timeout=QWEN_TIMEOUT, request_id=request_id, breaker=qwen_cb,
    )
    payload = response.json()
    try:
        raw = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise HTTPException(status_code=502, detail=f"Unexpected Groq response shape: {e}")

    decision = _extract_json(raw)

    action = decision.get("action")
    reply = decision.get("reply")
    params = decision.get("params") or {}

    # If Qwen returned plain text (no JSON), treat as chat reply
    if action not in ("chat", "image", "tts", "analyze"):
        fallback = decision.get("raw_analysis") or reply or str(raw)[:2000]
        return {"reply": fallback.strip() or "عذراً، لم أفهم طلبك.", "action": "chat"}

    if not reply:
        reply = "حسناً."

    # ---- IMAGE ----
    if action == "image":
        prompt_en = (params.get("prompt_en") or "").strip()
        if not prompt_en:
            # Fallback: use raw user text as English prompt
            prompt_en = trimmed[-1].get("content", "") if trimmed else ""
        if not prompt_en:
            return {"reply": reply + "\n\n⚠️ لم أستطع تحديد وصف الصورة.", "action": "chat"}
        try:
            image_bytes, content_type = await _generate_image_bytes(prompt_en)
            metrics["total_images"] += 1
            b64 = base64.b64encode(image_bytes).decode()
            return {
                "reply": reply,
                "action": "image",
                "media_base64": f"data:{content_type};base64,{b64}",
                "media_content_type": content_type,
            }
        except Exception as e:
            logger.error(f"[{request_id}] Chat image failed: {e}")
            return {
                "reply": reply + f"\n\n⚠️ تعذر توليد الصورة ({type(e).__name__}). حاول مجدداً.",
                "action": "chat",
            }

    # ---- TTS ----
    if action == "tts":
        text = (params.get("text") or "").strip()
        voice = _normalize_tts_voice(params.get("voice"))
        if not text:
            return {"reply": reply + "\n\n⚠️ لم أستطع تحديد النص المطلوب نطقه.", "action": "chat"}
        try:
            audio_bytes = await _generate_tts_bytes(text, voice)
            metrics["total_tts"] += 1
            b64 = base64.b64encode(audio_bytes).decode()
            return {
                "reply": reply,
                "action": "tts",
                "media_base64": f"data:audio/mpeg;base64,{b64}",
                "media_content_type": "audio/mpeg",
            }
        except Exception as e:
            logger.error(f"[{request_id}] Chat TTS failed: {e}")
            return {
                "reply": reply + f"\n\n⚠️ تعذر توليد الصوت ({type(e).__name__}).",
                "action": "chat",
            }

    # ---- ANALYZE ----
    if action == "analyze":
        text = (params.get("text") or "").strip()
        if not text:
            return {"reply": reply + "\n\n⚠️ لم أستطع تحديد النص المطلوب تحليله.", "action": "chat"}
        try:
            analysis = await analyze_transcript(text, request_id)
            return {"reply": reply, "action": "analyze", "analysis": analysis}
        except Exception as e:
            logger.error(f"[{request_id}] Chat analyze failed: {e}")
            return {
                "reply": reply + f"\n\n⚠️ تعذر التحليل ({type(e).__name__}).",
                "action": "chat",
            }

    # ---- CHAT ----
    return {"reply": reply, "action": "chat"}


# ==============================================================================
# Worker
# ==============================================================================
async def process_job(job: Job) -> Dict[str, Any]:
    request_id = job.id
    t_start = time.time()

    await job_store.update(request_id, state=JobState.PROCESSING, started_at=t_start)
    logger.info(f"[{request_id}] Processing ({job.job_type}). Queue left: {job_queue.qsize()}")

    transcript = ""
    if job.job_type == "audio":
        try:
            transcript = await transcribe_audio(job.file_content, job.filename, job.content_type, request_id)
            if not transcript:
                raise HTTPException(status_code=502, detail="Whisper returned empty transcript")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Transcription failed: {type(e).__name__}")
    elif job.job_type == "text":
        transcript = (job.input_text or "").strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="Input text is empty")

    try:
        analysis = await analyze_transcript(transcript, request_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Analysis failed: {type(e).__name__}")

    duration = time.time() - t_start
    metrics["processing_times"].append(duration)
    metrics["total_completed"] += 1
    logger.info(f"[{request_id}] DONE in {duration:.1f}s")
    return {"transcript": transcript, "analysis": analysis}


async def worker_loop() -> None:
    logger.info("Worker loop started.")
    while True:
        try:
            if job_queue.empty():
                await asyncio.sleep(1)
                continue
            job: Job = await job_queue.get()
            await job_store.recompute_positions()

            if job.state == JobState.CANCELLED:
                job_queue.task_done()
                continue

            try:
                result = await process_job(job)
                await job_store.update(job.id, state=JobState.COMPLETED,
                                       result=result, completed_at=time.time())
                if job.future and not job.future.done():
                    job.future.set_result(result)
            except HTTPException as e:
                metrics["total_failed"] += 1
                await job_store.update(job.id, state=JobState.FAILED,
                                       error=str(e.detail), completed_at=time.time())
                if job.future and not job.future.done():
                    job.future.set_exception(e)
            except Exception as e:
                metrics["total_failed"] += 1
                logger.exception(f"[{job.id}] Unexpected error")
                await job_store.update(job.id, state=JobState.FAILED,
                                       error=f"Internal error: {type(e).__name__}",
                                       completed_at=time.time())
                if job.future and not job.future.done():
                    job.future.set_exception(e)
            job_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Worker cancelled.")
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
        avg = 30.0
    else:
        avg = sum(metrics["processing_times"]) / len(metrics["processing_times"])
    return avg * max(1, job_queue.qsize() + 1)


# ==============================================================================
# Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, job_queue, job_store, rate_limiter
    global whisper_cb, qwen_cb, tts_cb
    global worker_task, cleanup_task, rate_cleanup_task

    logger.info("=" * 70)
    logger.info("Lordera Gateway v3.3.0 starting (Smart Chat + Groq TTS + Pollinations)")
    logger.info(f"Groq base:        {GROQ_API_BASE}")
    logger.info(f"Whisper:          {GROQ_WHISPER_MODEL}")
    logger.info(f"Qwen (chat):      {GROQ_QWEN_MODEL}")
    logger.info(f"TTS:              {GROQ_TTS_MODEL} / {GROQ_TTS_VOICE}")
    logger.info(f"Image:            {POLLINATIONS_IMAGE_MODEL}")
    logger.info(f"Groq key:         {'✓' if GROQ_API_KEY else '✗ MISSING'}")
    logger.info(f"Pollinations key: {'✓' if POLLINATIONS_API_KEY else '✗ not set'}")
    logger.info(f"Queue:            max={MAX_QUEUE_SIZE}, reject@{REJECT_THRESHOLD*100:.0f}%")
    logger.info(f"Rate limit:       {RATE_LIMIT_MAX} req / {RATE_LIMIT_WINDOW}s per IP")
    logger.info(f"Chat history max: {CHAT_MAX_HISTORY} messages")
    logger.info("=" * 70)

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(WHISPER_TIMEOUT, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=30, max_connections=100),
    )
    job_queue    = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    job_store    = JobStore(max_size=MAX_JOBS_IN_STORE, ttl=JOB_TTL_SECONDS)
    rate_limiter = RateLimiter(window=RATE_LIMIT_WINDOW, max_requests=RATE_LIMIT_MAX)
    whisper_cb   = CircuitBreaker("groq-whisper")
    qwen_cb      = CircuitBreaker("groq-qwen")
    tts_cb       = CircuitBreaker("groq-tts")

    worker_task       = asyncio.create_task(worker_loop())
    cleanup_task      = asyncio.create_task(job_store.cleanup_loop())
    rate_cleanup_task = asyncio.create_task(rate_limiter.cleanup_loop())

    yield

    logger.info("Shutting down...")
    for t in (worker_task, cleanup_task, rate_cleanup_task):
        if t:
            t.cancel()
    await asyncio.gather(worker_task, cleanup_task, rate_cleanup_task, return_exceptions=True)
    if http_client:
        await http_client.aclose()
    logger.info("Shutdown complete.")


# ==============================================================================
# FastAPI App
# ==============================================================================
app = FastAPI(
    title="Lordera AI Gateway",
    description="Smart chat orchestrator using Groq (Qwen/Whisper/TTS) + Pollinations.",
    version="3.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ==============================================================================
# Models
# ==============================================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str
    action: str = "chat"
    media_base64: Optional[str] = None
    media_content_type: Optional[str] = None
    analysis: Optional[dict] = None

class TranscribeResponse(BaseModel):
    transcript: str
    filename: str

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str

class TextJobSubmitRequest(BaseModel):
    text: str

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
    version: str
    groq_api_configured: bool
    pollinations_api_configured: bool
    groq_whisper_model: str
    groq_qwen_model: str
    groq_tts_model: str
    groq_tts_voice: str
    image_model: str
    whisper_circuit_open: bool
    qwen_circuit_open: bool
    tts_circuit_open: bool
    queue_size: int
    queue_max: int
    worker_running: bool
    jobs_in_store: int


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/")
async def root():
    return {
        "service": "Lordera AI Gateway",
        "version": "3.3.0",
        "backend": "Groq + Pollinations",
        "endpoints": {
            "chat":        "POST /chat",
            "transcribe":  "POST /transcribe",
            "submit_audio":"POST /jobs",
            "submit_text": "POST /jobs/text",
            "poll":        "GET  /jobs/{job_id}",
            "cancel":      "DELETE /jobs/{job_id}",
            "tts":         "POST /tts",
            "image":       "POST /image",
            "health":      "GET  /health",
            "queue":       "GET  /queue-status",
            "metrics":     "GET  /metrics",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if GROQ_API_KEY else "degraded",
        version="3.3.0",
        groq_api_configured=bool(GROQ_API_KEY),
        pollinations_api_configured=bool(POLLINATIONS_API_KEY),
        groq_whisper_model=GROQ_WHISPER_MODEL,
        groq_qwen_model=GROQ_QWEN_MODEL,
        groq_tts_model=GROQ_TTS_MODEL,
        groq_tts_voice=GROQ_TTS_VOICE,
        image_model=POLLINATIONS_IMAGE_MODEL,
        whisper_circuit_open=whisper_cb.is_open(),
        qwen_circuit_open=qwen_cb.is_open(),
        tts_circuit_open=tts_cb.is_open(),
        queue_size=job_queue.qsize(),
        queue_max=MAX_QUEUE_SIZE,
        worker_running=worker_task is not None and not worker_task.done(),
        jobs_in_store=job_store.size(),
    )


# ------------------- MAIN: Smart Chat -------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: Request, body: ChatRequest):
    client_ip = get_client_ip(request)

    if not await rate_limiter.check(client_ip):
        raise HTTPException(status_code=429,
                            detail=f"Rate limit ({RATE_LIMIT_MAX}/{int(RATE_LIMIT_WINDOW)}s). Slow down.",
                            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW))})

    if not body.messages:
        raise HTTPException(status_code=400, detail="messages array is empty")

    # Validate roles
    for m in body.messages:
        if m.role not in ("user", "assistant", "system"):
            raise HTTPException(status_code=400, detail=f"Invalid role: {m.role}")

    request_id = uuid.uuid4().hex[:12]
    metrics["total_chat"] += 1
    logger.info(f"[{request_id}] Chat ({len(body.messages)} msgs from {client_ip})")

    try:
        result = await orchestrate_chat(
            [{"role": m.role, "content": m.content} for m in body.messages],
            request_id,
        )
        return ChatResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Chat orchestration failed")
        return ChatResponse(
            reply=f"⚠️ عذراً، حدث خطأ تقني ({type(e).__name__}). حاول مجدداً.",
            action="chat",
        )


# ------------------- Fast Transcribe -------------------
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(request: Request, file: UploadFile = File(...)):
    client_ip = get_client_ip(request)
    if not await rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded",
                            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW))})

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_SIZE // (1024*1024)} MB)")

    request_id = uuid.uuid4().hex[:12]
    try:
        transcript = await transcribe_audio(content, file.filename or "audio",
                                             file.content_type, request_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {type(e).__name__}")

    return TranscribeResponse(transcript=transcript, filename=file.filename or "audio")


# ------------------- Direct TTS -------------------
@app.post("/tts")
async def tts_endpoint(req: TTSRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 4000:
        raise HTTPException(status_code=413, detail="text too long (max 4000 chars)")

    voice = _normalize_tts_voice(req.voice)
    try:
        audio = await _generate_tts_bytes(text, voice)
        metrics["total_tts"] += 1
        return Response(content=audio, media_type="audio/mpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=502, detail=f"TTS failed: {type(e).__name__}")


# ------------------- Direct Image -------------------
@app.post("/image")
async def image_endpoint(req: ImageRequest):
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        image, ct = await _generate_image_bytes(prompt)
        metrics["total_images"] += 1
        return Response(content=image, media_type=ct)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Image failed: {e.response.text[:200]}")
    except Exception as e:
        logger.exception("Image error")
        raise HTTPException(status_code=502, detail=f"Image error: {type(e).__name__}")


# ------------------- Async Jobs (legacy compat) -------------------
@app.post("/jobs", response_model=JobSubmitResponse, status_code=202)
async def submit_audio_job(request: Request, file: UploadFile = File(...)):
    client_ip = get_client_ip(request)
    if not await rate_limiter.check(client_ip):
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=429, detail="Rate limit exceeded",
                            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW))})
    if job_queue.qsize() >= int(MAX_QUEUE_SIZE * REJECT_THRESHOLD):
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=503, detail="Server busy", headers={"Retry-After": "30"})
    if job_queue.full():
        metrics["total_rejected"] += 1
        raise HTTPException(status_code=503, detail="Queue full", headers={"Retry-After": "60"})

    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, client_ip=client_ip,
              filename=file.filename or "audio", content_type=file.content_type,
              file_content=content, job_type="audio", position=job_queue.qsize() + 1)
    await job_store.add(job)
    metrics["total_submitted"] += 1

    try:
        job_queue.put_nowait(job)
    except asyncio.QueueFull:
        metrics["total_rejected"] += 1
        await job_store.update(job_id, state=JobState.FAILED, error="queue full")
        raise HTTPException(status_code=503, detail="Server busy")

    await job_store.recompute_positions()
    return JobSubmitResponse(
        job_id=job_id, status="queued", position=job.position,
        estimated_wait_seconds=round(estimate_wait_seconds(), 1),
        poll_url=f"/jobs/{job_id}",
    )


@app.post("/jobs/text", response_model=JobSubmitResponse, status_code=202)
async def submit_text_job(request: Request, body: TextJobSubmitRequest):
    client_ip = get_client_ip(request)
    if not await rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if job_queue.full():
        raise HTTPException(status_code=503, detail="Server busy")

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text empty")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=413, detail="Text too long")

    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, client_ip=client_ip,
              filename="text_input.txt", content_type="text/plain",
              file_content=b"", job_type="text", input_text=text,
              position=job_queue.qsize() + 1)
    await job_store.add(job)
    metrics["total_submitted"] += 1
    job_queue.put_nowait(job)
    await job_store.recompute_positions()

    return JobSubmitResponse(
        job_id=job_id, status="queued", position=job.position,
        estimated_wait_seconds=round(estimate_wait_seconds(), 1),
        poll_url=f"/jobs/{job_id}",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    estimated = round(estimate_wait_seconds(), 1) if job.state == JobState.QUEUED else None
    return JobStatusResponse(
        job_id=job.id, status=job.state.value,
        position=job.position if job.state == JobState.QUEUED else 0,
        created_at=job.created_at, started_at=job.started_at,
        completed_at=job.completed_at, estimated_wait_seconds=estimated,
        transcript=(job.result or {}).get("transcript") if job.state == JobState.COMPLETED else None,
        analysis=(job.result or {}).get("analysis") if job.state == JobState.COMPLETED else None,
        error=job.error, filename=job.filename,
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = await job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.state == JobState.PROCESSING:
        raise HTTPException(status_code=409, detail="Already processing")
    if job.state in (JobState.COMPLETED, JobState.FAILED):
        raise HTTPException(status_code=409, detail=f"Job already {job.state.value}")
    await job_store.update(job_id, state=JobState.CANCELLED, completed_at=time.time())
    metrics["total_cancelled"] += 1
    return {"status": "cancelled", "job_id": job_id}


@app.get("/queue-status")
async def queue_status():
    avg = (sum(metrics["processing_times"]) / len(metrics["processing_times"])
           if metrics["processing_times"] else 0.0)
    return {
        "queue_size": job_queue.qsize(), "queue_max": MAX_QUEUE_SIZE,
        "queue_fill_pct": round(job_queue.qsize() / MAX_QUEUE_SIZE * 100, 1),
        "worker_running": worker_task is not None and not worker_task.done(),
        "jobs_in_store": job_store.size(),
        "avg_processing_seconds": round(avg, 1),
        **{k: metrics[k] for k in ("total_submitted", "total_completed", "total_failed",
                                    "total_rejected", "total_cancelled", "total_chat",
                                    "total_images", "total_tts")},
        "whisper_circuit_open": whisper_cb.is_open(),
        "qwen_circuit_open": qwen_cb.is_open(),
        "tts_circuit_open": tts_cb.is_open(),
    }


@app.get("/metrics")
async def get_metrics():
    times = list(metrics["processing_times"])
    avg = sum(times) / len(times) if times else 0.0
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) >= 5 else avg
    return {
        "uptime_seconds": int(time.time() - metrics["started_at"]),
        **{k: metrics[k] for k in ("total_submitted", "total_completed", "total_failed",
                                    "total_rejected", "total_cancelled", "total_chat",
                                    "total_images", "total_tts")},
        "avg_processing_seconds": round(avg, 2),
        "p95_processing_seconds": round(p95, 2),
        "queue_depth": job_queue.qsize(),
        "jobs_in_store": job_store.size(),
    }


# ------------------- Legacy blocking endpoint -------------------
@app.post("/process-audio")
async def process_audio_legacy(request: Request, file: UploadFile = File(...)):
    client_ip = get_client_ip(request)
    if not await rate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    if job_queue.full():
        raise HTTPException(status_code=503, detail="Server busy")
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be audio/*")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    job_id = uuid.uuid4().hex[:12]
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    job = Job(id=job_id, client_ip=client_ip,
              filename=file.filename or "audio", content_type=file.content_type,
              file_content=content, job_type="audio",
              position=job_queue.qsize() + 1, future=future)
    await job_store.add(job)
    metrics["total_submitted"] += 1
    job_queue.put_nowait(job)

    try:
        result = await asyncio.wait_for(future, timeout=LEGACY_WAIT_TIMEOUT)
    except asyncio.TimeoutError:
        future.cancel()
        raise HTTPException(status_code=504, detail=f"Timeout after {LEGACY_WAIT_TIMEOUT}s")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "status": "success", "job_id": job_id,
        "transcript": result["transcript"], "analysis": result["analysis"],
        "filename": file.filename,
    }


# ==============================================================================
# Run
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)