"""
==============================================================================
Lordera AI — Controller Service v1.3.1
==============================================================================
ROLE:
  Monitor the Gateway queue and manage the lifecycle of a GPU deployment
  (Whisper + Qwen together) on Akash via the Console Managed Wallet API.
  This is the brain that reduces monthly cost from ~$108 to ~$8-20.

CHANGELOG v1.3 → v1.3.1:
  • GPU_SDL_PATH default changed: /app/sdl/gpu.yaml → /app/gpu.yaml.
    Reason: Dockerfile v2.0-restored preserves the pip cache by removing
    the explicit `COPY gpu.yaml /app/sdl/gpu.yaml` layer. Instead, gpu.yaml
    is copied implicitly by `COPY . .` and lands at /app/gpu.yaml.

CHANGELOG v1.2 → v1.3:
  • Fixed scheme: http:// → https:// in _extract_service_uris().
    Akash Console exposes public URIs via HTTPS (TLS termination is automatic).
    Using http:// caused ConnectError on every downstream call.

CHANGELOG v1.1 → v1.2:
  • Dynamic downstream URL resolution: after lease creation, poll
    GET /v1/deployments/{dseq} until URIs appear, then push them to Gateway
    via POST /admin/downstream.
  • RUNTIME_LIMIT_HOURS hard cap: safety net if controller fails to stop.
  • extract_service_uris_from_deployment(): parse URIs from deployment body.
  • push_downstream_urls(): send new URLs to Gateway.
  • check_services_ready() now uses the dynamically resolved URLs.

STATE MACHINE:

    ┌────────┐  queue>0   ┌──────────┐  ready   ┌─────────┐
    │  IDLE  │ ─────────► │ STARTING │ ───────► │ RUNNING │
    └────────┘            └──────────┘          └─────────┘
        ▲                                           │
        │                                           │ idle>10m
        │              ┌──────────┐                 │
        └──────────────│ STOPPING │◄────────────────┘
                       └──────────┘

AKASH CONSOLE API FACTS (verified from official docs):
  • Base URL:      https://console-api.akash.network
  • Auth:          x-api-key: <KEY>   (NOT Bearer!)
  • Envelope:      Every response is wrapped in {"data": {...}}
  • Create:        POST /v1/deployments  {"data": {"sdl": "<yaml>"}}
  • Bids:          GET  /v1/bids?dseq=<dseq>   (wait ~30-60s)
  • Lease:         POST /v1/leas  {"leases": [{dseq,gseq,oseq,provider}]}
  • Get:           GET  /v1/deployments/{dseq}
  • Close:         DELETE /v1/deployments/{dseq}
  • No deposit needed — Console funds automatically from credits.

DESIGN PRINCIPLES:
  1. NEVER crash        → every external call wrapped in try/except
  2. NEVER hang         → hard timeouts on every HTTP call
  3. NEVER flip-flop    → cooldown periods between START and STOP
  4. RECOVERABLE        → persist DSEQs to disk, restore on restart
  5. TESTABLE           → DRY_RUN mode prints actions without calling Akash
  6. IDEMPOTENT         → safe to call START/STOP repeatedly
==============================================================================
"""
import os
import json
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("controller")


# ==============================================================================
# Configuration
# ==============================================================================
GATEWAY_URL          = os.getenv("GATEWAY_URL", "http://gateway:8000")
GATEWAY_ADMIN_TOKEN  = os.getenv("GATEWAY_ADMIN_TOKEN", "")

AKASH_BASE_URL       = os.getenv("AKASH_BASE_URL", "https://console-api.akash.network")
AKASH_API_KEY        = os.getenv("AKASH_API_KEY", "")
AKASH_TIMEOUT        = float(os.getenv("AKASH_TIMEOUT", "30.0"))
AKASH_BIDS_TIMEOUT   = float(os.getenv("AKASH_BIDS_TIMEOUT", "90.0"))
AKASH_BIDS_INTERVAL  = float(os.getenv("AKASH_BIDS_INTERVAL", "5.0"))
AKASH_RUNTIME_LIMIT  = os.getenv("AKASH_RUNTIME_LIMIT_HOURS", "4")   # hard cap (safety net)

# SDL path — default matches Dockerfile v2.0-restored layout (/app/gpu.yaml).
# gpu.yaml is copied implicitly via `COPY . .` (not a dedicated COPY layer).
GPU_SDL_PATH         = os.getenv("GPU_SDL_PATH", "/app/gpu.yaml")

POLL_INTERVAL        = float(os.getenv("CONTROLLER_POLL_INTERVAL", "30.0"))
IDLE_TIMEOUT         = float(os.getenv("CONTROLLER_IDLE_TIMEOUT", "600.0"))
START_TIMEOUT        = float(os.getenv("CONTROLLER_START_TIMEOUT", "300.0"))
STOP_COOLDOWN        = float(os.getenv("CONTROLLER_STOP_COOLDOWN", "120.0"))
START_COOLDOWN       = float(os.getenv("CONTROLLER_START_COOLDOWN", "60.0"))

# How long to wait for URIs to appear after lease
URI_WAIT_TIMEOUT     = float(os.getenv("CONTROLLER_URI_WAIT_TIMEOUT", "180.0"))
URI_POLL_INTERVAL    = float(os.getenv("CONTROLLER_URI_POLL_INTERVAL", "10.0"))

DRY_RUN              = os.getenv("CONTROLLER_DRY_RUN", "false").lower() == "true"
STATE_FILE           = os.getenv("CONTROLLER_STATE_FILE", "/tmp/controller_state.json")


# ==============================================================================
# State Machine
# ==============================================================================
class ControllerState(str, Enum):
    IDLE     = "idle"
    STARTING = "starting"
    RUNNING  = "running"
    STOPPING = "stopping"
    ERROR    = "error"


@dataclass
class ControllerContext:
    state: ControllerState = ControllerState.IDLE
    last_activity_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    gpu_dseq: str = ""
    gpu_provider: str = ""
    gpu_gseq: int = 0
    gpu_oseq: int = 0
    whisper_url: str = ""
    qwen_url: str = ""
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    total_starts: int = 0
    total_stops: int = 0
    total_cycles: int = 0
    last_gateway_status: Optional[Dict[str, Any]] = None
    last_gateway_check_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


# ==============================================================================
# Global instances
# ==============================================================================
ctx = ControllerContext()
http_client: Optional[httpx.AsyncClient] = None
controller_task: Optional[asyncio.Task] = None
stop_event: Optional[asyncio.Event] = None


# ==============================================================================
# Persistence
# ==============================================================================
def _persist_state() -> None:
    try:
        Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({
                "gpu_dseq":     ctx.gpu_dseq,
                "gpu_provider": ctx.gpu_provider,
                "gpu_gseq":     ctx.gpu_gseq,
                "gpu_oseq":     ctx.gpu_oseq,
                "whisper_url":  ctx.whisper_url,
                "qwen_url":     ctx.qwen_url,
                "total_starts": ctx.total_starts,
                "total_stops":  ctx.total_stops,
                "state":        ctx.state.value,
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"persist_state failed: {e}")


def _restore_state() -> None:
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        ctx.gpu_dseq     = data.get("gpu_dseq", "")
        ctx.gpu_provider = data.get("gpu_provider", "")
        ctx.gpu_gseq     = int(data.get("gpu_gseq", 0) or 0)
        ctx.gpu_oseq     = int(data.get("gpu_oseq", 0) or 0)
        ctx.whisper_url  = data.get("whisper_url", "")
        ctx.qwen_url     = data.get("qwen_url", "")
        ctx.total_starts = data.get("total_starts", 0)
        ctx.total_stops  = data.get("total_stops", 0)
        logger.info(
            f"Restored: dseq={ctx.gpu_dseq or 'none'} "
            f"whisper={ctx.whisper_url or 'none'} qwen={ctx.qwen_url or 'none'}"
        )
    except Exception as e:
        logger.warning(f"restore_state failed: {e}")


# ==============================================================================
# Gateway Helpers
# ==============================================================================
async def fetch_gateway_status() -> Optional[Dict[str, Any]]:
    try:
        r = await http_client.get(f"{GATEWAY_URL}/queue-status", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"gateway status failed: {type(e).__name__}")
        return None


async def push_downstream_urls(whisper_url: str, qwen_url: str) -> bool:
    """
    Tell the Gateway the current URLs of Whisper and Qwen.
    Protected by GATEWAY_ADMIN_TOKEN.
    """
    if not GATEWAY_ADMIN_TOKEN:
        logger.warning("GATEWAY_ADMIN_TOKEN not set — cannot push URLs to Gateway")
        return False
    try:
        r = await http_client.post(
            f"{GATEWAY_URL}/admin/downstream",
            json={
                "whisper_url": whisper_url,
                "qwen_url": qwen_url,
                "token": GATEWAY_ADMIN_TOKEN,
            },
            timeout=10.0,
        )
        if r.status_code == 200:
            logger.info(f"Pushed URLs to Gateway: whisper={whisper_url} qwen={qwen_url}")
            return True
        logger.error(f"push_downstream HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"push_downstream exception: {type(e).__name__}: {e}")
    return False


# ==============================================================================
# Akash Console API Client
# ==============================================================================
class AkashAPI:
    """Wrapper over the Akash Console Managed Wallet API. Never raises."""

    def __init__(self, base_url: str, api_key: str, dry_run: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dry_run = dry_run

    def _headers(self) -> Dict[str, str]:
        # CORRECT AUTH: x-api-key (not Bearer)
        return {"x-api-key": self.api_key, "Content-Type": "application/json"}

    def _unwrap(self, r: httpx.Response) -> Optional[Any]:
        """Extract 'data' from the response envelope."""
        try:
            body = r.json()
        except Exception:
            logger.error(f"Non-JSON response: {r.text[:200]}")
            return None
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body

    async def _request(self, method: str, path: str, **kwargs) -> Optional[httpx.Response]:
        url = f"{self.base_url}{path}"
        try:
            return await http_client.request(
                method, url,
                headers=self._headers(),
                timeout=AKASH_TIMEOUT,
                **kwargs,
            )
        except httpx.TimeoutException:
            logger.error(f"Akash {method} {path}: timeout")
        except httpx.HTTPError as e:
            logger.error(f"Akash {method} {path}: {type(e).__name__}: {e}")
        except Exception:
            logger.exception(f"Akash {method} {path}: unexpected error")
        return None

    # ---------------------------------------------------------------- create
    async def create_deployment(self, sdl_yaml: str) -> Optional[str]:
        """POST /v1/deployments  {"data": {"sdl": "<yaml>"}}  → returns dseq."""
        if self.dry_run:
            fake = f"DRYRUN-{int(time.time())}"
            logger.info(f"[DRY-RUN] Would create deployment → DSEQ {fake}")
            return fake

        payload: Dict[str, Any] = {"data": {"sdl": sdl_yaml}}
        if AKASH_RUNTIME_LIMIT:
            try:
                payload["data"]["runtimeLimitHours"] = float(AKASH_RUNTIME_LIMIT)
            except ValueError:
                logger.warning(f"Invalid AKASH_RUNTIME_LIMIT_HOURS={AKASH_RUNTIME_LIMIT}; ignored")

        r = await self._request("POST", "/v1/deployments", json=payload)
        if r is None:
            return None
        if r.status_code not in (200, 201):
            logger.error(f"create_deployment HTTP {r.status_code}: {r.text[:300]}")
            return None

        data = self._unwrap(r)
        if not isinstance(data, dict):
            logger.error(f"create_deployment: unexpected body {r.text[:200]}")
            return None

        dseq = str(data.get("dseq") or "")
        if not dseq:
            logger.error(f"create_deployment: no dseq in {data}")
            return None
        logger.info(f"Created deployment → DSEQ {dseq}")
        return dseq

    # ---------------------------------------------------------------- bids
    async def fetch_bids(self, dseq: str) -> List[Dict[str, Any]]:
        """GET /v1/bids?dseq=<dseq>. Returns list of bid objects."""
        if self.dry_run:
            return [{"bid": {"id": {
                "owner": "akash1dryrun", "dseq": dseq,
                "gseq": 1, "oseq": 1,
                "provider": "akash1dryprovider", "bseq": 1,
            }, "state": "open",
               "price": {"denom": "uact", "amount": "1000"}}}]

        r = await self._request("GET", "/v1/bids", params={"dseq": dseq})
        if r is None or r.status_code != 200:
            if r is not None:
                logger.warning(f"fetch_bids HTTP {r.status_code}: {r.text[:200]}")
            return []

        data = self._unwrap(r)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("bids", []) or []
        return []

    # ---------------------------------------------------------------- lease
    async def create_lease(self, dseq: str, gseq: int, oseq: int, provider: str) -> bool:
        """POST /v1/leases  {"leases": [{dseq,gseq,oseq,provider}]}"""
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would create lease dseq={dseq} provider={provider}")
            return True

        payload = {"leases": [{
            "dseq": dseq, "gseq": gseq, "oseq": oseq, "provider": provider,
        }]}
        r = await self._request("POST", "/v1/leases", json=payload)
        if r is None:
            return False
        if r.status_code == 200:
            logger.info(f"Lease created dseq={dseq} provider={provider}")
            return True
        if r.status_code == 502:
            logger.error(f"Lease HTTP 502 (provider unreachable): {r.text[:200]}")
            return False
        logger.error(f"create_lease HTTP {r.status_code}: {r.text[:300]}")
        return False

    # ---------------------------------------------------------------- get
    async def get_deployment(self, dseq: str) -> Optional[Dict[str, Any]]:
        """GET /v1/deployments/{dseq}"""
        if self.dry_run:
            return {"leases": [{"status": {"services": {
                "whisper": {"uris": ["dry-whisper.example.com"]},
                "qwen":    {"uris": ["dry-qwen.example.com"]},
            }}}]}
        r = await self._request("GET", f"/v1/deployments/{dseq}")
        if r is None or r.status_code != 200:
            return None
        return self._unwrap(r)

    # ---------------------------------------------------------------- close
    async def close_deployment(self, dseq: str) -> bool:
        """DELETE /v1/deployments/{dseq}"""
        if not dseq:
            return True
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would close DSEQ {dseq}")
            return True

        r = await self._request("DELETE", f"/v1/deployments/{dseq}")
        if r is None:
            return False
        if r.status_code in (200, 202, 204):
            logger.info(f"Closed DSEQ {dseq}")
            return True
        if r.status_code == 404:
            logger.warning(f"DSEQ {dseq} not found — assuming already closed")
            return True
        logger.error(f"close_deployment HTTP {r.status_code}: {r.text[:200]}")
        return False


akash = AkashAPI(AKASH_BASE_URL, AKASH_API_KEY, dry_run=DRY_RUN)


# ==============================================================================
# SDL loading
# ==============================================================================
def _load_sdl(path: str) -> Optional[str]:
    try:
        if not os.path.exists(path):
            logger.error(f"SDL file not found: {path}")
            return None
        with open(path) as f:
            return f.read()
    except Exception:
        logger.exception(f"Failed to read SDL {path}")
        return None


# ==============================================================================
# URI extraction (defensive parser) — FIXED: https:// (was http://)
# ==============================================================================
def _extract_service_uris(deployment: Dict[str, Any]) -> Dict[str, str]:
    """
    Walk the deployment JSON and extract public URIs for 'whisper' and 'qwen'.
    The exact shape varies by Console version — this parser is defensive.

    IMPORTANT: Akash exposes public URIs via HTTPS (automatic TLS termination).
    We prepend https:// if the URI has no scheme.
    """
    out: Dict[str, str] = {}
    try:
        leases = deployment.get("leases") or []
        for lease in leases:
            status = lease.get("status") or {}
            services = status.get("services") or lease.get("services") or {}
            if isinstance(services, dict):
                for svc_name, svc in services.items():
                    uris = svc.get("uris") if isinstance(svc, dict) else None
                    if uris and isinstance(uris, list):
                        u = str(uris[0]).strip()
                        if u and not u.startswith("http"):
                            # Akash public URIs are HTTPS (port 443, TLS terminated)
                            u = "https://" + u
                        out[svc_name] = u
    except Exception as e:
        logger.warning(f"URI extraction error: {e}")
    return out


async def resolve_and_push_uris(dseq: str) -> bool:
    """
    Poll the deployment until URIs for whisper and qwen appear, then push them
    to the Gateway. Returns True when both URLs were pushed successfully.
    """
    deadline = time.time() + URI_WAIT_TIMEOUT
    while time.time() < deadline:
        dep = await akash.get_deployment(dseq)
        if dep:
            uris = _extract_service_uris(dep)
            whisper_url = uris.get("whisper", "")
            qwen_url    = uris.get("qwen", "")

            if whisper_url and qwen_url:
                ctx.whisper_url = whisper_url
                ctx.qwen_url = qwen_url
                _persist_state()
                if await push_downstream_urls(whisper_url, qwen_url):
                    return True
                logger.warning("URIs found but push failed — will retry")
            else:
                logger.info(
                    f"Waiting for URIs... whisper={whisper_url or '-'} qwen={qwen_url or '-'}"
                )
        await asyncio.sleep(URI_POLL_INTERVAL)
    logger.error(f"URIs did not appear within {URI_WAIT_TIMEOUT}s")
    return False


# ==============================================================================
# Wait for bids + create lease
# ==============================================================================
async def _wait_and_lease(dseq: str) -> bool:
    if DRY_RUN:
        ok = await akash.create_lease(dseq, 1, 1, "akash1dryprovider")
        if ok:
            ctx.gpu_gseq = 1
            ctx.gpu_oseq = 1
            ctx.gpu_provider = "akash1dryprovider"
        return ok

    deadline = time.time() + AKASH_BIDS_TIMEOUT
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        bids = await akash.fetch_bids(dseq)
        if bids:
            def _price(b):
                try: return float(b["bid"]["price"]["amount"])
                except Exception: return float("inf")
            bids.sort(key=_price)
            for b in bids:
                bid_id = b["bid"]["id"]
                gseq = int(bid_id.get("gseq", 1))
                oseq = int(bid_id.get("oseq", 1))
                prov = bid_id.get("provider", "")
                logger.info(f"Attempting lease with provider={prov} price={_price(b)}")
                if await akash.create_lease(dseq, gseq, oseq, prov):
                    ctx.gpu_gseq = gseq
                    ctx.gpu_oseq = oseq
                    ctx.gpu_provider = prov
                    return True
            logger.warning("All bids failed — retrying in a moment")
        logger.info(f"No bids yet (attempt {attempt})")
        await asyncio.sleep(AKASH_BIDS_INTERVAL)
    logger.error(f"No bid accepted within {AKASH_BIDS_TIMEOUT}s")
    return False


# ==============================================================================
# Core actions: START / STOP
# ==============================================================================
async def action_start() -> bool:
    """Create GPU deployment, wait for bids, create lease, push URLs."""
    ctx.state = ControllerState.STARTING
    ctx.started_at = time.time()
    logger.info("=" * 60)
    logger.info("ACTION: START")

    if ctx.gpu_dseq:
        logger.info(f"GPU deployment already exists (dseq={ctx.gpu_dseq}); skipping create")
        return True

    sdl = _load_sdl(GPU_SDL_PATH)
    if not sdl:
        ctx.last_error = f"SDL missing: {GPU_SDL_PATH}"
        ctx.state = ControllerState.ERROR
        return False

    # 1. Create deployment
    dseq = await akash.create_deployment(sdl)
    if not dseq:
        ctx.last_error = "create_deployment failed"
        ctx.state = ControllerState.ERROR
        return False
    ctx.gpu_dseq = dseq
    _persist_state()

    # 2. Wait for bids + create lease
    if not await _wait_and_lease(dseq):
        ctx.last_error = "lease creation failed"
        logger.warning(f"Lease failed → closing orphan deployment {dseq}")
        await akash.close_deployment(dseq)
        ctx.gpu_dseq = ""
        ctx.gpu_provider = ""
        ctx.state = ControllerState.ERROR
        _persist_state()
        return False

    # 3. Resolve URIs and push to Gateway (fire and forget — non-blocking)
    asyncio.create_task(resolve_and_push_uris(dseq))

    ctx.total_starts += 1
    ctx.last_error = None
    _persist_state()
    logger.info("START complete — waiting for services to become ready")
    return True


async def action_stop() -> bool:
    """Close the GPU deployment."""
    ctx.state = ControllerState.STOPPING
    logger.info("=" * 60)
    logger.info("ACTION: STOP")

    if not ctx.gpu_dseq:
        ctx.state = ControllerState.IDLE
        ctx.stopped_at = time.time()
        logger.info("Nothing to stop (no active deployment)")
        return True

    ok = await akash.close_deployment(ctx.gpu_dseq)
    if ok:
        ctx.gpu_dseq = ""
        ctx.gpu_provider = ""
        ctx.gpu_gseq = 0
        ctx.gpu_oseq = 0
        ctx.whisper_url = ""
        ctx.qwen_url = ""
        ctx.total_stops += 1
        ctx.stopped_at = time.time()
        ctx.state = ControllerState.IDLE
        ctx.last_error = None
        logger.info("STOP complete — IDLE")
    else:
        ctx.last_error = "close_deployment failed"
        ctx.state = ControllerState.ERROR
        logger.warning("STOP failed")

    _persist_state()
    return ok


# ==============================================================================
# Readiness check (uses dynamic URLs when available)
# ==============================================================================
async def check_services_ready() -> bool:
    """
    If we have URIs → probe /health on both Whisper and Qwen.
    Otherwise fall back to a time-based readiness check.
    """
    if ctx.whisper_url and ctx.qwen_url:
        try:
            results = await asyncio.gather(
                http_client.get(f"{ctx.whisper_url}/health", timeout=5.0),
                http_client.get(f"{ctx.qwen_url}/health", timeout=5.0),
                return_exceptions=True,
            )
            return all(
                not isinstance(r, Exception) and r.status_code == 200
                for r in results
            )
        except Exception:
            return False

    if ctx.started_at is None:
        return False
    return (time.time() - ctx.started_at) >= START_COOLDOWN


# ==============================================================================
# Main control loop
# ==============================================================================
async def control_loop() -> None:
    logger.info("=" * 70)
    logger.info("Controller control loop started")
    logger.info(f"  Mode:          {'DRY-RUN' if DRY_RUN else 'LIVE'}")
    logger.info(f"  Gateway:       {GATEWAY_URL}")
    logger.info(f"  Akash API:     {AKASH_BASE_URL}")
    logger.info(f"  API key set:   {bool(AKASH_API_KEY)}")
    logger.info(f"  Runtime cap:   {AKASH_RUNTIME_LIMIT}h")
    logger.info(f"  SDL path:      {GPU_SDL_PATH}")
    logger.info(f"  Poll every:    {POLL_INTERVAL}s")
    logger.info(f"  Idle timeout:  {IDLE_TIMEOUT}s")
    logger.info(f"  Bids wait:     {AKASH_BIDS_TIMEOUT}s")
    logger.info("=" * 70)

    while not stop_event.is_set():
        try:
            ctx.total_cycles += 1

            gs = await fetch_gateway_status()
            if gs is None:
                ctx.consecutive_errors += 1
                logger.warning(
                    f"Cycle {ctx.total_cycles}: Gateway unreachable "
                    f"({ctx.consecutive_errors} consecutive errors)"
                )
                await asyncio.sleep(POLL_INTERVAL)
                continue

            ctx.last_gateway_status = gs
            ctx.last_gateway_check_at = time.time()
            ctx.consecutive_errors = 0

            queue_size     = int(gs.get("queue_size", 0))
            worker_running = bool(gs.get("worker_running", False))

            has_activity = (queue_size > 0) or worker_running
            if has_activity:
                ctx.last_activity_at = time.time()

            logger.info(
                f"Cycle {ctx.total_cycles} | state={ctx.state.value} | "
                f"queue={queue_size} | worker={worker_running} | "
                f"idle={int(time.time()-ctx.last_activity_at)}s | "
                f"dseq={ctx.gpu_dseq or '-'}"
            )

            # -------------------- State transitions --------------------
            if ctx.state == ControllerState.IDLE:
                if ctx.stopped_at and (time.time() - ctx.stopped_at) < STOP_COOLDOWN:
                    pass  # cooldown
                elif has_activity:
                    logger.info("Activity detected → START")
                    await action_start()

            elif ctx.state == ControllerState.STARTING:
                ready = await check_services_ready()
                if ready:
                    ctx.state = ControllerState.RUNNING
                    ctx.last_activity_at = time.time()
                    logger.info("Services ready → RUNNING")
                elif ctx.started_at and (time.time() - ctx.started_at) > START_TIMEOUT:
                    logger.error(f"Not ready within {START_TIMEOUT}s — still waiting")

            elif ctx.state == ControllerState.RUNNING:
                # Ensure URIs are still pushed (in case push failed earlier)
                if ctx.whisper_url and ctx.qwen_url and GATEWAY_ADMIN_TOKEN:
                    await push_downstream_urls(ctx.whisper_url, ctx.qwen_url)

                idle_for = time.time() - ctx.last_activity_at
                if idle_for >= IDLE_TIMEOUT:
                    logger.info(f"Idle {idle_for:.0f}s → STOP")
                    await action_stop()

            elif ctx.state == ControllerState.STOPPING:
                logger.warning("Still STOPPING — retrying")
                await action_stop()

            elif ctx.state == ControllerState.ERROR:
                logger.warning("Recovering from ERROR → IDLE")
                ctx.state = ControllerState.IDLE
                ctx.last_error = None

        except asyncio.CancelledError:
            logger.info("Control loop cancelled")
            break
        except Exception:
            logger.exception("Unexpected error in control loop")
            ctx.consecutive_errors += 1
            await asyncio.sleep(min(POLL_INTERVAL, 10))

        # Interruptible sleep
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=POLL_INTERVAL)
        except asyncio.TimeoutError:
            pass


# ==============================================================================
# Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, controller_task, stop_event

    logger.info("=" * 70)
    logger.info("Lordera Controller Service v1.3.1 starting...")
    logger.info(f"  API key set:        {bool(AKASH_API_KEY)}")
    logger.info(f"  Gateway admin set:  {bool(GATEWAY_ADMIN_TOKEN)}")
    logger.info(f"  SDL path:           {GPU_SDL_PATH}")
    logger.info(f"  Dry-run:            {DRY_RUN}")
    logger.info("=" * 70)

    if not AKASH_API_KEY and not DRY_RUN:
        logger.warning(
            "AKASH_API_KEY is not set and DRY_RUN=false. "
            "All Akash API calls will fail."
        )
    if not GATEWAY_ADMIN_TOKEN and not DRY_RUN:
        logger.warning(
            "GATEWAY_ADMIN_TOKEN not set — Controller cannot push URLs to Gateway."
        )

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
    )

    _restore_state()
    stop_event = asyncio.Event()
    controller_task = asyncio.create_task(control_loop())

    yield

    logger.info("Shutting down Controller...")
    if stop_event:
        stop_event.set()
    if controller_task:
        controller_task.cancel()
        try:
            await controller_task
        except asyncio.CancelledError:
            pass
    if http_client:
        await http_client.aclose()
    _persist_state()
    logger.info("Shutdown complete.")


# ==============================================================================
# FastAPI App
# ==============================================================================
app = FastAPI(
    title="Lordera Controller Service",
    description="Monitors Gateway queue and manages on-demand Akash GPU deployments.",
    version="1.3.1",
    lifespan=lifespan,
)


# ==============================================================================
# Schemas
# ==============================================================================
class HealthResponse(BaseModel):
    status: str
    state: str
    gpu_dseq: str
    gpu_provider: str
    whisper_url: str
    qwen_url: str
    consecutive_errors: int
    dry_run: bool
    akash_key_set: bool
    uptime_seconds: int


class StatusResponse(BaseModel):
    state: str
    gpu_dseq: str
    gpu_provider: str
    gpu_gseq: int
    gpu_oseq: int
    whisper_url: str
    qwen_url: str
    last_activity_at: float
    idle_seconds: float
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    total_starts: int
    total_stops: int
    total_cycles: int
    last_error: Optional[str] = None
    last_gateway_status: Optional[Dict[str, Any]] = None
    config: Dict[str, Any]


# ==============================================================================
# Endpoints
# ==============================================================================
@app.get("/")
async def root():
    return {
        "service": "Lordera Controller",
        "version": "1.3.1",
        "endpoints": {
            "health":      "GET  /health",
            "status":      "GET  /status",
            "force_start": "POST /force-start",
            "force_stop":  "POST /force-stop",
            "reset":       "POST /reset",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        state=ctx.state.value,
        gpu_dseq=ctx.gpu_dseq,
        gpu_provider=ctx.gpu_provider,
        whisper_url=ctx.whisper_url,
        qwen_url=ctx.qwen_url,
        consecutive_errors=ctx.consecutive_errors,
        dry_run=DRY_RUN,
        akash_key_set=bool(AKASH_API_KEY),
        uptime_seconds=int(time.time() - (ctx.last_gateway_check_at or time.time())),
    )


@app.get("/status", response_model=StatusResponse)
async def get_status():
    idle_seconds = time.time() - ctx.last_activity_at
    return StatusResponse(
        state=ctx.state.value,
        gpu_dseq=ctx.gpu_dseq,
        gpu_provider=ctx.gpu_provider,
        gpu_gseq=ctx.gpu_gseq,
        gpu_oseq=ctx.gpu_oseq,
        whisper_url=ctx.whisper_url,
        qwen_url=ctx.qwen_url,
        last_activity_at=ctx.last_activity_at,
        idle_seconds=round(idle_seconds, 1),
        started_at=ctx.started_at,
        stopped_at=ctx.stopped_at,
        total_starts=ctx.total_starts,
        total_stops=ctx.total_stops,
        total_cycles=ctx.total_cycles,
        last_error=ctx.last_error,
        last_gateway_status=ctx.last_gateway_status,
        config={
            "poll_interval": POLL_INTERVAL,
            "idle_timeout": IDLE_TIMEOUT,
            "start_cooldown": START_COOLDOWN,
            "stop_cooldown": STOP_COOLDOWN,
            "start_timeout": START_TIMEOUT,
            "bids_timeout": AKASH_BIDS_TIMEOUT,
            "bids_interval": AKASH_BIDS_INTERVAL,
            "runtime_limit_hours": float(AKASH_RUNTIME_LIMIT) if AKASH_RUNTIME_LIMIT else None,
            "uri_wait_timeout": URI_WAIT_TIMEOUT,
            "uri_poll_interval": URI_POLL_INTERVAL,
            "gpu_sdl_path": GPU_SDL_PATH,
            "dry_run": DRY_RUN,
        },
    )


@app.post("/force-start")
async def force_start():
    if ctx.state in (ControllerState.STARTING, ControllerState.RUNNING):
        raise HTTPException(status_code=409, detail=f"Already in state {ctx.state.value}")
    ok = await action_start()
    if not ok:
        raise HTTPException(status_code=502, detail=ctx.last_error or "Start failed")
    return {"status": "started", "state": ctx.state.value, "dseq": ctx.gpu_dseq}


@app.post("/force-stop")
async def force_stop():
    if ctx.state == ControllerState.STOPPING:
        raise HTTPException(status_code=409, detail="Already stopping")
    ok = await action_stop()
    if not ok:
        raise HTTPException(status_code=502, detail=ctx.last_error or "Stop failed")
    return {"status": "stopped", "state": ctx.state.value}


@app.post("/reset")
async def reset_state():
    ctx.gpu_dseq = ""
    ctx.gpu_provider = ""
    ctx.gpu_gseq = 0
    ctx.gpu_oseq = 0
    ctx.whisper_url = ""
    ctx.qwen_url = ""
    ctx.state = ControllerState.IDLE
    ctx.last_error = None
    _persist_state()
    return {"status": "reset", "state": ctx.state.value}


# ==============================================================================
# Run (local dev)
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)