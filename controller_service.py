"""
==============================================================================
Lordera AI — Controller Service v1.4.0 (Local Air)
==============================================================================
ROLE:
  Monitor Gateway queue and manage GPU deployment lifecycle on Akash
  via direct blockchain access (mnemonic, no Console API key).
  Cost saver: ~$108 -> ~$8-20/month.

CHANGE v1.3.1 -> v1.4.0:
  • Removed Console API (x-api-key)
  • Added direct chain client using akash SDK + AKASH_MNEMONIC
  • Full control: create = start, close = full stop/delete
  • Works with local Akash Air, no central console needed

STATE MACHINE:
    ┌────────┐  queue>0   ┌──────────┐  ready   ┌─────────┐
    │  IDLE  │ ─────────► │ STARTING │ ───────► │ RUNNING │
    └────────┘            └──────────┘          └─────────┘
        ▲                                           │
        │              ┌──────────┐                 │
        └──────────────│ STOPPING │◄────────────────┘
                       └──────────┘

DESIGN:
  1. Never crash   -> try/except on external calls
  2. Never hang    -> hard timeouts
  3. No flip-flop  -> cooldowns
  4. Recoverable   -> persist dseq to disk
  5. Testable      -> DRY_RUN mode
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
from dotenv import load_dotenv

load_dotenv()

# Try Akash SDK - official PyPI: pip install akash
try:
    from akash import AkashClient
    try:
        from akash import SDL
    except ImportError:
        SDL = None
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("controller")


# ==============================================================================
# Config
# ==============================================================================
GATEWAY_URL          = os.getenv("GATEWAY_URL", "http://gateway:8000")
GATEWAY_ADMIN_TOKEN  = os.getenv("GATEWAY_ADMIN_TOKEN", "")

# New - direct chain access (local Air)
AKASH_MNEMONIC       = os.getenv("AKASH_MNEMONIC", "")
AKASH_NODE           = os.getenv("AKASH_NODE", "https://rpc.akash.network:443")
AKASH_CHAIN_ID       = os.getenv("AKASH_CHAIN_ID", "akashnet-2")
AKASH_TIMEOUT        = float(os.getenv("AKASH_TIMEOUT", "30.0"))
AKASH_BIDS_TIMEOUT   = float(os.getenv("AKASH_BIDS_TIMEOUT", "90.0"))
AKASH_BIDS_INTERVAL  = float(os.getenv("AKASH_BIDS_INTERVAL", "5.0"))
AKASH_RUNTIME_LIMIT  = os.getenv("AKASH_RUNTIME_LIMIT_HOURS", "4")

GPU_SDL_PATH         = os.getenv("GPU_SDL_PATH", "/app/gpu.yaml")

POLL_INTERVAL        = float(os.getenv("CONTROLLER_POLL_INTERVAL", "30.0"))
IDLE_TIMEOUT         = float(os.getenv("CONTROLLER_IDLE_TIMEOUT", "600.0"))
START_TIMEOUT        = float(os.getenv("CONTROLLER_START_TIMEOUT", "300.0"))
STOP_COOLDOWN        = float(os.getenv("CONTROLLER_STOP_COOLDOWN", "120.0"))
START_COOLDOWN       = float(os.getenv("CONTROLLER_START_COOLDOWN", "60.0"))

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
    if not GATEWAY_ADMIN_TOKEN:
        logger.warning("GATEWAY_ADMIN_TOKEN not set — cannot push URLs")
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
# Akash Direct Client (Local Air - no API key)
# ==============================================================================
class AkashAPI:
    """Direct blockchain client using mnemonic. Full start/stop control."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.mnemonic = os.getenv("AKASH_MNEMONIC", "") or AKASH_MNEMONIC
        self.node = AKASH_NODE
        self.chain_id = AKASH_CHAIN_ID
        self.client = None

        if self.dry_run:
            logger.info("DRY_RUN enabled - no real chain calls")
            return

        if not SDK_AVAILABLE:
            logger.error("akash SDK not installed. Run: pip install akash")
            return

        if not self.mnemonic:
            logger.warning("AKASH_MNEMONIC not set - chain calls will fail")
            return

        try:
            # SDK init - sync
            self.client = AkashClient.from_mnemonic(
                mnemonic=self.mnemonic,
                node=self.node,
                chain_id=self.chain_id,
            )
            logger.info(f"Connected to Akash chain: {self.chain_id} via {self.node}")
        except Exception as e:
            logger.error(f"Failed to init AkashClient: {e}")

    # ---------------------------------------------------------------- create
    async def create_deployment(self, sdl_yaml: str) -> Optional[str]:
        if self.dry_run:
            fake = f"DRYRUN-{int(time.time())}"
            logger.info(f"[DRY-RUN] Would create deployment -> DSEQ {fake}")
            return fake

        if not self.client:
            logger.error("Akash client not initialized")
            return None

        try:
            sdl = SDL.from_yaml(sdl_yaml)
            deployment = await asyncio.to_thread(self.client.create_deployment, sdl)
            
            # Robust DSEQ extraction
            dseq = ""
            if hasattr(deployment, 'id') and hasattr(deployment.id, 'dseq'):
                dseq = str(deployment.id.dseq)
            elif isinstance(deployment, dict):
                dseq = str(deployment.get('dseq') or deployment.get('data', {}).get('dseq') or '')
            else:
                # Fallback: try fetching recent deployments
                deps = await asyncio.to_thread(self.client.fetch_deployments)
                if deps:
                    dseq = str(getattr(deps[0], 'dseq', ''))
            
            if not dseq:
                logger.error(f"Could not extract DSEQ from response: {deployment}")
                return None

            logger.info(f"Created deployment -> DSEQ {dseq}")
            return dseq
        except Exception as e:
            logger.error(f"create_deployment failed: {type(e).__name__}: {e}")
            return None

    # ---------------------------------------------------------------- bids
    async def fetch_bids(self, dseq: str) -> List[Dict[str, Any]]:
        if self.dry_run:
            return [{"bid": {"id": {
                "owner": "akash1dryrun", "dseq": dseq,
                "gseq": 1, "oseq": 1,
                "provider": "akash1dryprovider", "bseq": 1,
            }, "state": "open",
               "price": {"denom": "uakt", "amount": "1000"}}}]

        if not self.client:
            return []

        try:
            bids = await asyncio.to_thread(self.client.fetch_bids, dseq)
            result = []
            for b in bids:
                # Safely parse SDK objects to dicts
                bid_obj = getattr(b, 'bid', b)
                bid_id = getattr(bid_obj, 'bid_id', getattr(b, 'bid_id', None))
                
                if bid_id:
                    gseq = getattr(bid_id, 'gseq', 1)
                    oseq = getattr(bid_id, 'oseq', 1)
                    provider = getattr(bid_id, 'provider', '')
                else:
                    gseq, oseq, provider = 1, 1, ""
                
                price_obj = getattr(bid_obj, 'price', getattr(b, 'price', None))
                amount = getattr(price_obj, 'amount', '1000') if price_obj else '1000'

                result.append({
                    "bid": {
                        "id": {"gseq": gseq, "oseq": oseq, "provider": provider},
                        "price": {"amount": str(amount)}
                    }
                })
            return result
        except Exception as e:
            logger.warning(f"fetch_bids failed: {e}")
            return []

    # ---------------------------------------------------------------- lease
    async def create_lease(self, dseq: str, gseq: int, oseq: int, provider: str) -> bool:
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would create lease dseq={dseq} provider={provider}")
            return True

        if not self.client:
            return False

        try:
            await asyncio.to_thread(self.client.create_lease, dseq, gseq, oseq, provider)
            logger.info(f"Lease created dseq={dseq} provider={provider}")
            return True
        except Exception as e:
            logger.error(f"create_lease failed: {e}")
            return False

    # ---------------------------------------------------------------- get (Lease Status)
    async def get_deployment(self, dseq: str) -> Optional[Dict[str, Any]]:
        if self.dry_run:
            return {"leases": [{"status": {"services": {
                "whisper": {"uris": ["dry-whisper.example.com"]},
                "qwen":    {"uris": ["dry-qwen.example.com"]},
            }}}]}

        if not self.client:
            return None

        try:
            # Fetch leases for the deployment
            leases = await asyncio.to_thread(self.client.fetch_leases, dseq)
            if not leases:
                return None
            
            lease = leases[0]
            # Extract lease identifiers safely
            gseq = getattr(lease, 'gseq', 1)
            oseq = getattr(lease, 'oseq', 1)
            provider = getattr(lease, 'provider', '')
            
            # Fetch lease status from provider
            status = await asyncio.to_thread(self.client.fetch_lease_status, dseq, gseq, oseq, provider)
            
            services_dict = {}
            if hasattr(status, 'services'):
                for svc_name, svc_obj in status.services.items():
                    uris = getattr(svc_obj, 'uris', [])
                    services_dict[svc_name] = {"uris": uris}
            elif isinstance(status, dict):
                for svc_name, svc_obj in status.get('services', {}).items():
                    uris = svc_obj.get('uris', []) if isinstance(svc_obj, dict) else getattr(svc_obj, 'uris', [])
                    services_dict[svc_name] = {"uris": uris}

            return {"leases": [{"status": {"services": services_dict}}]}
        except Exception as e:
            logger.warning(f"get_deployment(lease_status) failed: {e}")
            return None

    # ---------------------------------------------------------------- close
    async def close_deployment(self, dseq: str) -> bool:
        """Full stop - closes deployment on chain."""
        if not dseq:
            return True
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would close DSEQ {dseq}")
            return True

        if not self.client:
            logger.error("Akash client not initialized")
            return False

        try:
            await asyncio.to_thread(self.client.close_deployment, dseq)
            logger.info(f"Closed DSEQ {dseq}")
            return True
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                logger.warning(f"DSEQ {dseq} not found - assuming already closed")
                return True
            logger.error(f"close_deployment failed: {e}")
            return False


akash = AkashAPI(dry_run=DRY_RUN)


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
# URI extraction
# ==============================================================================
def _extract_service_uris(deployment: Dict[str, Any]) -> Dict[str, str]:
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
                            u = "https://" + u
                        out[svc_name] = u
    except Exception as e:
        logger.warning(f"URI extraction error: {e}")
    return out


async def resolve_and_push_uris(dseq: str) -> bool:
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
                logger.warning("URIs found but push failed - will retry")
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
            logger.warning("All bids failed - retrying")
        logger.info(f"No bids yet (attempt {attempt})")
        await asyncio.sleep(AKASH_BIDS_INTERVAL)
    logger.error(f"No bid accepted within {AKASH_BIDS_TIMEOUT}s")
    return False


# ==============================================================================
# Core actions: START / STOP
# ==============================================================================
async def action_start() -> bool:
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

    dseq = await akash.create_deployment(sdl)
    if not dseq:
        ctx.last_error = "create_deployment failed"
        ctx.state = ControllerState.ERROR
        return False

    ctx.gpu_dseq = dseq
    _persist_state()

    if not await _wait_and_lease(dseq):
        ctx.last_error = "bids/lease failed"
        ctx.state = ControllerState.ERROR
        await akash.close_deployment(dseq)
        ctx.gpu_dseq = ""
        return False

    if not await resolve_and_push_uris(dseq):
        ctx.last_error = "URI resolution failed"
        ctx.state = ControllerState.ERROR
        return False

    ctx.state = ControllerState.RUNNING
    ctx.total_starts += 1
    ctx.last_activity_at = time.time()
    _persist_state()
    logger.info(f"START complete dseq={dseq}")
    return True


async def action_stop() -> bool:
    ctx.state = ControllerState.STOPPING
    logger.info("=" * 60)
    logger.info(f"ACTION: STOP dseq={ctx.gpu_dseq}")

    dseq = ctx.gpu_dseq
    if not dseq:
        ctx.state = ControllerState.IDLE
        return True

    ok = await akash.close_deployment(dseq)
    if not ok:
        ctx.last_error = f"close_deployment failed for {dseq}"
        ctx.state = ControllerState.ERROR
        return False

    ctx.gpu_dseq = ""
    ctx.gpu_provider = ""
    ctx.gpu_gseq = 0
    ctx.gpu_oseq = 0
    ctx.whisper_url = ""
    ctx.qwen_url = ""
    ctx.stopped_at = time.time()
    ctx.state = ControllerState.IDLE
    ctx.total_stops += 1
    ctx.total_cycles += 1
    _persist_state()

    # Clear Gateway downstream
    await push_downstream_urls("", "")
    logger.info("STOP complete")
    return True


# ==============================================================================
# Control loop
# ==============================================================================
async def control_loop():
    logger.info("Control loop started")
    while True:
        try:
            if stop_event and stop_event.is_set():
                break

            status = await fetch_gateway_status()
            if status:
                ctx.last_gateway_status = status
                ctx.last_gateway_check_at = time.time()
                
                # FIXED: Gateway uses "queue_size" not "queue_depth"
                queue_depth = int(status.get("queue_size", 0) or 0)
                if queue_depth > 0:
                    ctx.last_activity_at = time.time()

            now = time.time()
            idle_for = now - ctx.last_activity_at

            # Runtime limit safety
            if ctx.state == ControllerState.RUNNING and ctx.started_at:
                try:
                    limit_h = float(AKASH_RUNTIME_LIMIT) if AKASH_RUNTIME_LIMIT else 0
                except ValueError:
                    limit_h = 0
                if limit_h > 0 and (now - ctx.started_at) > limit_h * 3600:
                    logger.warning(f"Runtime limit {limit_h}h reached - stopping")
                    await action_stop()
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

            if ctx.state == ControllerState.IDLE:
                if status and int(status.get("queue_size", 0) or 0) > 0:
                    if ctx.stopped_at and (now - ctx.stopped_at) < START_COOLDOWN:
                        logger.info(f"Start cooldown {START_COOLDOWN}s")
                    else:
                        await action_start()

            elif ctx.state == ControllerState.RUNNING:
                if idle_for > IDLE_TIMEOUT:
                    if ctx.started_at and (now - ctx.started_at) < STOP_COOLDOWN:
                        logger.info(f"Stop cooldown {STOP_COOLDOWN}s")
                    else:
                        logger.info(f"Idle {idle_for:.0f}s > {IDLE_TIMEOUT}s - stopping")
                        await action_stop()

            elif ctx.state == ControllerState.ERROR:
                logger.info(f"In ERROR state: {ctx.last_error} - will retry in {POLL_INTERVAL}s")
                if ctx.gpu_dseq:
                    await akash.close_deployment(ctx.gpu_dseq)
                    ctx.gpu_dseq = ""
                ctx.state = ControllerState.IDLE
                ctx.consecutive_errors += 1

            await asyncio.sleep(POLL_INTERVAL)

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("control_loop error")
            ctx.consecutive_errors += 1
            await asyncio.sleep(POLL_INTERVAL)


# ==============================================================================
# Lifespan
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, controller_task, stop_event

    logger.info("=" * 70)
    logger.info("Lordera Controller v1.4.0 starting (Local Air - direct chain)")
    logger.info(f"Config: poll={POLL_INTERVAL}s idle={IDLE_TIMEOUT}s dry_run={DRY_RUN}")
    logger.info(f"Akash: node={AKASH_NODE} chain={AKASH_CHAIN_ID} sdk={SDK_AVAILABLE}")
    logger.info("=" * 70)

    if not AKASH_MNEMONIC and not DRY_RUN:
        logger.warning("AKASH_MNEMONIC not set and DRY_RUN=false - chain calls will fail")
    if not GATEWAY_ADMIN_TOKEN and not DRY_RUN:
        logger.warning("GATEWAY_ADMIN_TOKEN not set - cannot push URLs to Gateway")

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
    description="Monitors Gateway queue and manages on-demand Akash GPU deployments via direct chain access.",
    version="1.4.0",
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
        "version": "1.4.0",
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
        akash_key_set=bool(AKASH_MNEMONIC),
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