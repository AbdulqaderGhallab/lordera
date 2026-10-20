"""
==============================================================================
Lordera AI — Qwen Service v2.1
==============================================================================
Role:  Semantic analysis of transcripts using Qwen 2.5-7B-Instruct (4-bit nf4).
FIX v2.1.1: Fixed OOMKilled on T4 16GB without weakening - removed deprecated torch_dtype, added low_cpu_mem_usage=True, removed conflicting device param in pipeline. Keeps 7B + nf4 double quant = max power.
Hardware:  NVIDIA T4 (16 GB VRAM) — single-GPU, serial inference.
Design goals:
  - Warmup at startup           → first real request is NOT slow
  - Serialize inference         → Semaphore(1), prevent VRAM contention
  - Robust JSON parsing         → multiple fallbacks, never crash
  - Long transcript handling    → truncate safely before context overflow
  - Internal-token protection   → /analyze requires X-Internal-Token header
  - Metrics + GPU monitoring    → /health and /metrics

CHANGELOG v2.0 → v2.1:
  • Added INTERNAL_TOKEN verification on /analyze.
    This service is exposed via HTTPS (global: true in gpu.yaml) because it
    lives in a different Akash deployment than the Gateway. The token ensures
    only the Gateway (holder of the shared secret) can call /analyze.
  • /health, /metrics, / remain OPEN (used by Controller readiness checks
    and monitoring — no GPU work happens on these).
  • If INTERNAL_TOKEN is empty → protection is disabled (local dev mode).
==============================================================================
"""
import os
import re
import json
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict, List

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
import uvicorn


# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("qwen_service")


# ==============================================================================
# Configuration
# ==============================================================================
MODEL_DIR        = os.getenv("MODEL_DIR", "/models")
QWEN_MODEL_NAME  = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

MAX_NEW_TOKENS       = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))
MAX_INPUT_CHARS      = int(os.getenv("QWEN_MAX_INPUT_CHARS", "12000"))   # ~3000 tokens
WARMUP_ENABLED       = os.getenv("QWEN_WARMUP", "true").lower() == "true"
INFERENCE_TIMEOUT    = float(os.getenv("QWEN_INFERENCE_TIMEOUT", "110.0"))  # < gateway timeout

# ----- Internal token (shared with Gateway) -----
# Requests to /analyze must carry:  X-Internal-Token: <INTERNAL_TOKEN>
# Empty value disables the check (local-dev convenience).
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")


# ==============================================================================
# Global State
# ==============================================================================
qwen_pipeline = None
qwen_tokenizer = None
inference_semaphore: Optional[asyncio.Semaphore] = None   # serialize GPU access
model_ready = asyncio.Event()

# -----------------------------------------------------------------------------
# Prompt template (sent to Qwen)
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise semantic analyzer. "
    "You MUST respond with a single valid JSON object and nothing else."
)

USER_PROMPT_TEMPLATE = """Analyze the following transcript and return a JSON object with exactly these keys:
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


# ==============================================================================
# Internal-token verification
# ==============================================================================
def _verify_internal_token(request: Request) -> None:
    """
    Enforce X-Internal-Token on protected endpoints.
    If INTERNAL_TOKEN is empty → protection is disabled (local dev).
    Otherwise, mismatch or missing header → 401.
    """
    if not INTERNAL_TOKEN:
        # Not configured → allow (local dev mode)
        return
    token = request.headers.get("x-internal-token", "")
    if token != INTERNAL_TOKEN:
        metrics["total_auth_rejections"] += 1
        client = request.client.host if request.client else "unknown"
        logger.warning(f"Rejected /analyze from {client}: invalid or missing token")
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Internal-Token header",
        )


# ==============================================================================
# Helpers: JSON Extraction (multi-tier fallback)
# ==============================================================================
def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Attempt to extract a valid JSON object from arbitrary model output.
    Tiers:
      1. Direct json.loads
      2. Strip markdown fences then json.loads
      3. Find first balanced {...} block and json.loads
      4. Fallback: wrap raw text
    """
    text = raw.strip()

    # Tier 1: direct
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tier 2: strip markdown fences
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Tier 3: find first balanced { ... }
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

    # Tier 4: give up gracefully — return raw under "raw_analysis"
    logger.warning("JSON extraction failed. Returning raw model output.")
    return {"raw_analysis": text[:2000]}


def _normalize_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist and have the right types."""
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

    return {
        "summary": summary,
        "sentiment": sentiment,
        "key_topics": topics,
        "intent": intent,
    }


# ==============================================================================
# Lifespan: Load + Warmup
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qwen_pipeline, qwen_tokenizer, inference_semaphore

    inference_semaphore = asyncio.Semaphore(1)   # one inference at a time

    logger.info("=" * 70)
    logger.info("Lordera Qwen Service v2.1 starting...")
    logger.info(f"Device:      {DEVICE}")
    logger.info(f"Model:       {QWEN_MODEL_NAME}")
    logger.info(f"Max tokens:  {MAX_NEW_TOKENS}")
    logger.info(f"Max input:   {MAX_INPUT_CHARS} chars (~{MAX_INPUT_CHARS//4} tokens)")
    logger.info(f"Token:       {'configured' if INTERNAL_TOKEN else 'DISABLED (dev mode)'}")
    logger.info("=" * 70)

    # ----------------------------------------------------------------------
    # 1. Load tokenizer
    # ----------------------------------------------------------------------
    qwen_model_path = os.path.join(MODEL_DIR, "qwen", QWEN_MODEL_NAME.replace("/", "_"))
    os.makedirs(qwen_model_path, exist_ok=True)

    logger.info("Loading tokenizer...")
    qwen_tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        QWEN_MODEL_NAME,
        cache_dir=qwen_model_path,
    )

    # ----------------------------------------------------------------------
    # 2. Configure 4-bit quantization (fit T4 16 GB)
    # ----------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # ----------------------------------------------------------------------
    # 3. Load model
    # ----------------------------------------------------------------------
    logger.info("Loading model (this may take 1-3 min on first run)...")
    # FIX: no weakening, keep 7B + 4-bit nf4 (99% quality). 
    # Removed deprecated torch_dtype, added low_cpu_mem_usage to prevent OOMKilled on T4 16GB
    model = await asyncio.to_thread(
        AutoModelForCausalLM.from_pretrained,
        QWEN_MODEL_NAME,
        cache_dir=qwen_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # ----------------------------------------------------------------------
    # 4. Build pipeline
    # ----------------------------------------------------------------------
    # FIX: when device_map="auto" is used, don't pass device to pipeline (causes double VRAM allocation)
    qwen_pipeline = await asyncio.to_thread(
        pipeline,
        "text-generation",
        model=model,
        tokenizer=qwen_tokenizer,
    )

    logger.info("Model loaded successfully.")

    if DEVICE == "cuda":
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"VRAM: allocated={alloc:.2f} GB, reserved={reserved:.2f} GB")
        except Exception:
            pass

    # ----------------------------------------------------------------------
    # 5. Warmup — compile CUDA kernels, avoid slow first request
    # ----------------------------------------------------------------------
    if WARMUP_ENABLED:
        logger.info("Running warmup inference...")
        t0 = time.time()
        try:
            await asyncio.to_thread(
                qwen_pipeline,
                USER_PROMPT_TEMPLATE.format(transcript="Hello world, this is a warmup."),
                max_new_tokens=16,
                do_sample=False,
                return_full_text=False,
            )
            logger.info(f"Warmup completed in {time.time() - t0:.1f}s")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    model_ready.set()
    logger.info("Qwen Service ready to accept requests.")

    yield

    logger.info("Shutting down Qwen Service...")
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
    title="Lordera Qwen Service",
    description="Semantic analysis of transcripts using Qwen 2.5-7B-Instruct (4-bit). Protected by X-Internal-Token.",
    version="2.1.0",
    lifespan=lifespan,
)


# ==============================================================================
# Schemas
# ==============================================================================
class AnalysisRequest(BaseModel):
    transcript: str = Field(..., description="Text to analyze")

class AnalysisResponse(BaseModel):
    status: str
    analysis: dict
    truncated: bool = False
    inference_ms: int = 0

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_ready: bool
    token_protected: bool
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None
    queue_waiting: int = 0

class MetricsResponse(BaseModel):
    total_requests: int
    total_failures: int
    total_auth_rejections: int
    avg_inference_ms: float
    last_inference_ms: int
    uptime_seconds: int


# ==============================================================================
# Global metrics
# ==============================================================================
metrics: Dict[str, Any] = {
    "total_requests": 0,
    "total_failures": 0,
    "total_auth_rejections": 0,
    "inference_times_ms": [],
    "started_at": time.time(),
}


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/")
async def root():
    return {
        "service": "Lordera Qwen Service",
        "version": "2.1.0",
        "token_protected": bool(INTERNAL_TOKEN),
        "endpoints": {
            "health":  "GET  /health  (open)",
            "metrics": "GET  /metrics (open)",
            "analyze": "POST /analyze (requires X-Internal-Token)",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Open endpoint — used by Controller readiness checks."""
    vram_alloc = vram_reserved = None
    if DEVICE == "cuda":
        try:
            vram_alloc = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
            vram_reserved = round(torch.cuda.memory_reserved() / (1024 ** 3), 2)
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if model_ready.is_set() else "starting",
        device=DEVICE,
        model_loaded=qwen_pipeline is not None,
        model_ready=model_ready.is_set(),
        token_protected=bool(INTERNAL_TOKEN),
        vram_allocated_gb=vram_alloc,
        vram_reserved_gb=vram_reserved,
        queue_waiting=0,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Open endpoint — safe, does not touch GPU."""
    times = metrics["inference_times_ms"]
    avg = sum(times) / len(times) if times else 0.0
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        total_failures=metrics["total_failures"],
        total_auth_rejections=metrics["total_auth_rejections"],
        avg_inference_ms=round(avg, 1),
        last_inference_ms=times[-1] if times else 0,
        uptime_seconds=int(time.time() - metrics["started_at"]),
    )


# ==============================================================================
# Analysis endpoint
# ==============================================================================
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: Request,
    body: AnalysisRequest,
):
    """
    Analyze a transcript and return structured JSON:
      {summary, sentiment, key_topics, intent}
    Requires X-Internal-Token header (unless INTERNAL_TOKEN env is empty).
    """
    # ---- Auth guard FIRST (cheapest reject) ----
    _verify_internal_token(request)

    if not model_ready.is_set():
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    transcript = (body.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

    # ---- Truncate long transcripts safely ----
    truncated = False
    if len(transcript) > MAX_INPUT_CHARS:
        logger.warning(
            f"Transcript too long ({len(transcript)} chars); truncating to {MAX_INPUT_CHARS}."
        )
        transcript = transcript[:MAX_INPUT_CHARS]
        truncated = True

    metrics["total_requests"] += 1
    prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript)

    # ---- Serialize GPU access with semaphore ----
    async with inference_semaphore:
        t0 = time.time()
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(
                    qwen_pipeline,
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    return_full_text=False,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                ),
                timeout=INFERENCE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            metrics["total_failures"] += 1
            logger.error(f"Inference timeout after {INFERENCE_TIMEOUT}s")
            raise HTTPException(
                status_code=504,
                detail=f"Inference exceeded {INFERENCE_TIMEOUT}s.",
            )
        except torch.cuda.OutOfMemoryError:
            metrics["total_failures"] += 1
            logger.exception("CUDA OOM during inference")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            raise HTTPException(status_code=503, detail="GPU out of memory. Retry shortly.")
        except Exception as e:
            metrics["total_failures"] += 1
            logger.exception("Inference error")
            raise HTTPException(status_code=500, detail=f"Inference error: {type(e).__name__}")

    inference_ms = int((time.time() - t0) * 1000)
    metrics["inference_times_ms"].append(inference_ms)
    if len(metrics["inference_times_ms"]) > 100:
        metrics["inference_times_ms"] = metrics["inference_times_ms"][-100:]

    # ---- Extract raw text ----
    try:
        raw_text = output[0]["generated_text"].strip()
    except (KeyError, IndexError, TypeError) as e:
        metrics["total_failures"] += 1
        logger.error(f"Unexpected pipeline output shape: {e}")
        raise HTTPException(status_code=500, detail="Unexpected model output shape.")

    # ---- Parse JSON with multi-tier fallback ----
    parsed = _extract_json(raw_text)
    analysis = _normalize_analysis(parsed)

    logger.info(
        f"Analysis OK in {inference_ms}ms | "
        f"intent={analysis.get('intent')} | "
        f"sentiment={analysis.get('sentiment')} | "
        f"topics={len(analysis.get('key_topics', []))}"
    )

    return AnalysisResponse(
        status="success",
        analysis=analysis,
        truncated=truncated,
        inference_ms=inference_ms,
    )


# ==============================================================================
# Run (local dev)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
