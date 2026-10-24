"""
==============================================================================
Lordera AI — Controller Service v1.6.13 (REST API for bid queries)
==============================================================================
CHANGELOG v1.6.12 -> v1.6.13:
  • CRITICAL FIX: The `akash query market bid list` CLI inside the container
    always returns {"bids":[]} even when bids exist on-chain (verified by
    comparing the same query from the host, which returns bids). The CLI is
    unreliable for bid listing in our container.
  • fetch_bids now queries Akash REST API endpoints directly via httpx
    (4 fallback URLs, v1beta4 then v1beta3). This bypasses the broken CLI.
  • Falls back to the CLI only if all REST endpoints fail.
  • Filters bids: only `open`/`active` states are kept; `closed` are skipped.
  • Extensive logging so future issues are diagnosable in one glance.
==============================================================================
"""
import os
import json
import time
import logging
import asyncio
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

AKASH_MNEMONIC       = os.getenv("AKASH_MNEMONIC", "")
AKASH_NODE           = os.getenv("AKASH_NODE", "https://rpc.akashnet.net:443")
AKASH_CHAIN_ID       = os.getenv("AKASH_CHAIN_ID", "akashnet-2")
AKASH_TIMEOUT        = float(os.getenv("AKASH_TIMEOUT", "60.0"))
AKASH_BIDS_TIMEOUT   = float(os.getenv("AKASH_BIDS_TIMEOUT", "360.0"))
AKASH_BIDS_INTERVAL  = float(os.getenv("AKASH_BIDS_INTERVAL", "5.0"))
AKASH_RUNTIME_LIMIT  = os.getenv("AKASH_RUNTIME_LIMIT_HOURS", "4")

WHISPER_SDL_PATH     = os.getenv("WHISPER_SDL_PATH", "/app/whisper.yaml")
QWEN_SDL_PATH        = os.getenv("QWEN_SDL_PATH", "/app/qwen.yaml")

POLL_INTERVAL        = float(os.getenv("CONTROLLER_POLL_INTERVAL", "30.0"))
IDLE_TIMEOUT         = float(os.getenv("CONTROLLER_IDLE_TIMEOUT", "600.0"))
START_TIMEOUT        = float(os.getenv("CONTROLLER_START_TIMEOUT", "300.0"))
STOP_COOLDOWN        = float(os.getenv("CONTROLLER_STOP_COOLDOWN", "120.0"))
START_COOLDOWN       = float(os.getenv("CONTROLLER_START_COOLDOWN", "60.0"))

URI_WAIT_TIMEOUT     = float(os.getenv("CONTROLLER_URI_WAIT_TIMEOUT", "180.0"))
URI_POLL_INTERVAL    = float(os.getenv("CONTROLLER_URI_POLL_INTERVAL", "10.0"))

DRY_RUN              = os.getenv("CONTROLLER_DRY_RUN", "false").lower() == "true"
STATE_FILE           = os.getenv("CONTROLLER_STATE_FILE", "/tmp/controller_state.json")

AKASH_AVAILABLE = shutil.which("akash") is not None
PROVIDER_SERVICES_AVAILABLE = shutil.which("provider-services") is not None


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
    whisper_dseq: str = ""
    whisper_provider: str = ""
    whisper_gseq: int = 0
    whisper_oseq: int = 0
    qwen_dseq: str = ""
    qwen_provider: str = ""
    qwen_gseq: int = 0
    qwen_oseq: int = 0
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
                "whisper_dseq":     ctx.whisper_dseq,
                "whisper_provider": ctx.whisper_provider,
                "whisper_gseq":     ctx.whisper_gseq,
                "whisper_oseq":     ctx.whisper_oseq,
                "qwen_dseq":        ctx.qwen_dseq,
                "qwen_provider":    ctx.qwen_provider,
                "qwen_gseq":        ctx.qwen_gseq,
                "qwen_oseq":        ctx.qwen_oseq,
                "whisper_url":      ctx.whisper_url,
                "qwen_url":         ctx.qwen_url,
                "total_starts":     ctx.total_starts,
                "total_stops":      ctx.total_stops,
                "state":            ctx.state.value,
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"persist_state failed: {e}")


def _restore_state() -> None:
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        ctx.whisper_dseq     = data.get("whisper_dseq", "")
        ctx.whisper_provider = data.get("whisper_provider", "")
        ctx.whisper_gseq     = int(data.get("whisper_gseq", 0) or 0)
        ctx.whisper_oseq     = int(data.get("whisper_oseq", 0) or 0)
        ctx.qwen_dseq        = data.get("qwen_dseq", "")
        ctx.qwen_provider    = data.get("qwen_provider", "")
        ctx.qwen_gseq        = int(data.get("qwen_gseq", 0) or 0)
        ctx.qwen_oseq        = int(data.get("qwen_oseq", 0) or 0)
        ctx.whisper_url      = data.get("whisper_url", "")
        ctx.qwen_url         = data.get("qwen_url", "")
        ctx.total_starts     = data.get("total_starts", 0)
        ctx.total_stops      = data.get("total_stops", 0)
        logger.info(f"Restored: whisper_dseq={ctx.whisper_dseq or 'none'} qwen_dseq={ctx.qwen_dseq or 'none'}")
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
# Akash CLI Client
# ==============================================================================
class AkashCLI:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.mnemonic = os.getenv("AKASH_MNEMONIC", "") or AKASH_MNEMONIC
        self.node = AKASH_NODE
        self.chain_id = AKASH_CHAIN_ID
        self.wallet_name = "controller-wallet"
        self.wallet_address = ""
        self._initialized = False

    async def _run_cmd(self, cmd: List[str], input_data: str = None) -> Tuple[int, str, str]:
        logger.debug(f"Running: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if input_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate(input_data.encode() if input_data else None)
        return proc.returncode, stdout.decode(), stderr.decode()

    async def _ensure_initialized(self) -> bool:
        if self.dry_run:
            self._initialized = True
            self.wallet_address = "akash1dryrun"
            return True

        if self._initialized:
            return True

        if not AKASH_AVAILABLE or not PROVIDER_SERVICES_AVAILABLE:
            logger.error("CLI tools missing!")
            return False

        if not self.mnemonic:
            logger.error("AKASH_MNEMONIC is not set.")
            return False

        # ----------------------------------------------------------------------
        # 1. Import Key (if not already present)
        # ----------------------------------------------------------------------
        rc, out, err = await self._run_cmd(
            ["akash", "keys", "show", self.wallet_name, "-a", "--keyring-backend", "test"]
        )
        if rc != 0:
            logger.info(f"Importing wallet {self.wallet_name} from mnemonic...")
            rc, out, err = await self._run_cmd(
                ["akash", "keys", "add", self.wallet_name, "--recover", "--keyring-backend", "test"],
                input_data=self.mnemonic + "\n"
            )
            if rc != 0:
                logger.error(f"Failed to import wallet: {err}")
                return False
            rc, out, err = await self._run_cmd(
                ["akash", "keys", "show", self.wallet_name, "-a", "--keyring-backend", "test"]
            )

        self.wallet_address = out.strip()
        logger.info(f"Wallet address: {self.wallet_address}")

        # ----------------------------------------------------------------------
        # 2. Ensure ~/.akash exists, then check LOCAL PEM
        # ----------------------------------------------------------------------
        akash_home = os.path.expanduser("~/.akash")
        os.makedirs(akash_home, exist_ok=True)
        cert_path = f"{akash_home}/{self.wallet_address}.pem"
        cert_exists_locally = os.path.exists(cert_path)

        # ----------------------------------------------------------------------
        # 3. Check ON-CHAIN validity
        # ----------------------------------------------------------------------
        cert_valid_on_chain = False
        rc, out, err = await self._run_cmd([
            "provider-services", "query", "cert", "list",
            "--owner", self.wallet_address,
            "--node", self.node,
            "--state", "valid",
            "-o", "json",
        ])
        if rc == 0:
            try:
                data = json.loads(out)
                certs = data.get("certificates", [])
                cert_valid_on_chain = len(certs) > 0
            except Exception as e:
                logger.warning(f"Failed to parse on-chain cert query: {e} (raw={out[:200]})")
        else:
            logger.warning(f"Cert query failed (rc={rc}): {err.strip()[:200]}")

        logger.info(
            f"Certificate status: local_pem={cert_exists_locally}, "
            f"on_chain_valid={cert_valid_on_chain}, path={cert_path}"
        )

        # ----------------------------------------------------------------------
        # 4. Regenerate if EITHER local PEM is missing OR on-chain is invalid
        # ----------------------------------------------------------------------
        if not cert_exists_locally or not cert_valid_on_chain:
            logger.warning(
                "Certificate missing locally or invalid on-chain - "
                "regenerating (revoke-then-generate)..."
            )

            # Step 1: Best-effort revoke of any existing on-chain cert (WITH --fees)
            logger.info("Revoking any existing certificate (best-effort)...")
            rc_rev, out_rev, err_rev = await self._run_cmd([
                "provider-services", "tx", "cert", "revoke",
                "--from", self.wallet_name,
                "--node", self.node,
                "--chain-id", self.chain_id,
                "--keyring-backend", "test",
                "--gas", "500000",
                "--fees", "20000uakt",
                "-y",
            ])
            if rc_rev != 0:
                logger.warning(f"Cert revoke failed (may not exist): {err_rev.strip()[:200]}")
            else:
                logger.info("Old certificate revoked (or none existed). Waiting 3s...")
                await asyncio.sleep(3)

            # Step 2: Generate new cert (NO --rie; local only, no gas/fees needed)
            logger.info("Generating new certificate...")
            rc, out, err = await self._run_cmd([
                "provider-services", "tx", "cert", "generate", "client",
                "--from", self.wallet_name,
                "--node", self.node,
                "--chain-id", self.chain_id,
                "--keyring-backend", "test",
                "-y",
            ])
            if rc != 0:
                logger.error(f"Failed to generate certificate: {err.strip()[:400]}")
                return False

            # Verify PEM was actually created
            if not os.path.exists(cert_path):
                logger.error(f"PEM file NOT created at {cert_path} after generate.")
                rc2, out2, _ = await self._run_cmd(["ls", "-la", akash_home])
                logger.error(f"Contents of {akash_home}:\n{out2}")
                return False
            logger.info(f"Certificate PEM created at {cert_path}")

            # Step 3: Publish cert to blockchain (WITH --gas AND --fees)
            logger.info("Publishing certificate to blockchain...")
            rc, out, err = await self._run_cmd([
                "provider-services", "tx", "cert", "publish", "client",
                "--from", self.wallet_name,
                "--node", self.node,
                "--chain-id", self.chain_id,
                "--keyring-backend", "test",
                "--gas", "500000",
                "--fees", "20000uakt",
                "-y",
            ])
            if rc != 0:
                logger.error(f"Failed to publish certificate: {err.strip()[:400]}")
                return False

            logger.info("Certificate published. Waiting 5s for chain confirmation...")
            await asyncio.sleep(5)

            # Final verification
            rc, out, err = await self._run_cmd([
                "provider-services", "query", "cert", "list",
                "--owner", self.wallet_address,
                "--node", self.node,
                "--state", "valid",
                "-o", "json",
            ])
            if rc == 0:
                try:
                    data = json.loads(out)
                    certs = data.get("certificates", [])
                    if certs and len(certs) > 0:
                        logger.info("Certificate verified on-chain after publishing.")
                    else:
                        logger.warning("Certificate published but not yet visible on-chain (may need a few blocks).")
                except Exception as e:
                    logger.warning(f"Post-publish cert verify failed: {e}")
        else:
            logger.info("Certificate valid both locally and on-chain. Skipping regeneration.")

        self._initialized = True
        logger.info(f"Wallet {self.wallet_name} initialized with valid certificate: {self.wallet_address}")
        return True

    async def create_deployment(self, sdl_yaml: str, label: str) -> Optional[str]:
        if self.dry_run:
            fake = f"DRYRUN-{label}-{int(time.time())}"
            logger.info(f"[DRY-RUN] Would create {label} deployment -> DSEQ {fake}")
            return fake

        if not await self._ensure_initialized():
            return None

        sdl_path = f"/tmp/{label}_sdl.yaml"
        with open(sdl_path, "w") as f:
            f.write(sdl_yaml)

        cmd = [
            "akash", "tx", "deployment", "create", sdl_path,
            "--from", self.wallet_name,
            "--node", self.node,
            "--chain-id", self.chain_id,
            "--gas", "500000",
            "--fees", "20000uakt",
            "-y",
            "--keyring-backend", "test",
            "-o", "json"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.error(f"create_deployment ({label}) failed: {err}")
            return None

        await asyncio.sleep(5)

        cmd = [
            "akash", "query", "deployment", "list",
            "--owner", self.wallet_address,
            "--node", self.node,
            "--state", "active",
            "-o", "json"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.error(f"Query deployments failed: {err}")
            return None

        try:
            data = json.loads(out)
            deployments = data.get("deployments", [])
            if not deployments:
                logger.error(f"No active deployments found after creating {label}.")
                return None

            def get_dseq(dep):
                d = dep.get("deployment", {})
                id_obj = d.get("id") or d.get("deployment_id") or {}
                return int(id_obj.get("dseq", 0))

            deployments.sort(key=get_dseq, reverse=True)

            best_dep = deployments[0].get("deployment", {})
            best_id = best_dep.get("id") or best_dep.get("deployment_id") or {}
            dseq = str(best_id.get("dseq", ""))

            if not dseq:
                logger.error(f"Could not extract dseq from {deployments[0]}")
                return None

            logger.info(f"Created {label} deployment -> DSEQ {dseq}")
            return dseq
        except Exception as e:
            logger.error(f"Failed to parse deployment list: {e}")
            return None

    async def fetch_bids(self, dseq: str) -> List[Dict[str, Any]]:
        if self.dry_run:
            return [{"bid": {"id": {"gseq": 1, "oseq": 1, "provider": "akash1dryprovider"}, "price": {"amount": "1000"}}}]

        # ------------------------------------------------------------------
        # Strategy 1: REST API (primary — bypasses the broken CLI bid query)
        # ------------------------------------------------------------------
        rest_endpoints = [
            "https://akash-api.polkachu.com",
            "https://akash-rest.publicnode.com",
            "https://akash-api.lavenderfive.com",
            "https://api.akashnet.net",
        ]

        for base_url in rest_endpoints:
            for version in ("v1beta4", "v1beta3"):
                try:
                    url = f"{base_url}/akash/market/{version}/bids/list"
                    params = {
                        "filters.owner": self.wallet_address,
                        "filters.dseq": str(dseq),
                        "pagination.limit": "50",
                    }
                    r = await http_client.get(url, params=params, timeout=15.0)
                    if r.status_code != 200:
                        logger.debug(f"REST {base_url}/{version} -> HTTP {r.status_code}")
                        continue
                    data = r.json()
                    raw_bids = data.get("bids", []) or []
                    if not raw_bids:
                        logger.debug(f"REST {base_url}/{version} -> 0 bids")
                        continue

                    result: List[Dict[str, Any]] = []
                    for b in raw_bids:
                        bid_obj = b.get("bid", {})
                        state = str(bid_obj.get("state", "")).lower()
                        # Skip closed/lost bids; keep open or active
                        if state and state not in ("open", "active"):
                            continue
                        bid_id = bid_obj.get("id", {}) or bid_obj.get("bid_id", {})
                        price = bid_obj.get("price", {})
                        result.append({
                            "bid": {
                                "id": {
                                    "gseq": int(bid_id.get("gseq", 1)),
                                    "oseq": int(bid_id.get("oseq", 1)),
                                    "provider": bid_id.get("provider", "")
                                },
                                "price": {
                                    "amount": price.get("amount", "10000")
                                }
                            }
                        })

                    if result:
                        logger.info(
                            f"fetch_bids via REST ({base_url}/{version}) -> "
                            f"{len(result)} open bids for dseq={dseq}"
                        )
                        return result
                    else:
                        logger.info(
                            f"REST {base_url}/{version} returned {len(raw_bids)} bids, "
                            f"none in open/active state"
                        )
                except Exception as e:
                    logger.debug(f"REST {base_url}/{version} exception: {type(e).__name__}: {e}")

        # ------------------------------------------------------------------
        # Strategy 2: CLI fallback (in case REST is unreachable)
        # ------------------------------------------------------------------
        logger.info(f"REST bid lookup failed for dseq={dseq}; falling back to CLI")
        if not await self._ensure_initialized():
            return []

        cmd = [
            "akash", "query", "market", "bid", "list",
            "--owner", self.wallet_address,
            "--dseq", str(dseq),
            "--node", self.node,
            "-o", "json"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.warning(f"fetch_bids CLI failed: {err.strip()[:300]}")
            return []

        try:
            data = json.loads(out)
            bids = data.get("bids", [])
            result = []
            for b in bids:
                bid_obj = b.get("bid", {})
                state = str(bid_obj.get("state", "")).lower()
                if state and state not in ("open", "active"):
                    continue
                bid_id = bid_obj.get("id", {}) or bid_obj.get("bid_id", {})
                price = bid_obj.get("price", {})
                result.append({
                    "bid": {
                        "id": {
                            "gseq": int(bid_id.get("gseq", 1)),
                            "oseq": int(bid_id.get("oseq", 1)),
                            "provider": bid_id.get("provider", "")
                        },
                        "price": {
                            "amount": price.get("amount", "10000")
                        }
                    }
                })
            return result
        except Exception as e:
            logger.warning(f"Failed to parse bids: {e}")
            return []

    async def create_lease(self, dseq: str, gseq: int, oseq: int, provider: str, sdl_path: str) -> bool:
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would create lease dseq={dseq} provider={provider}")
            return True

        if not await self._ensure_initialized():
            return False

        cmd = [
            "akash", "tx", "market", "lease", "create",
            "--dseq", str(dseq),
            "--gseq", str(gseq),
            "--oseq", str(oseq),
            "--provider", provider,
            "--from", self.wallet_name,
            "--node", self.node,
            "--chain-id", self.chain_id,
            "--gas", "500000",
            "--fees", "20000uakt",
            "-y",
            "--keyring-backend", "test"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.error(f"create_lease failed (rc={rc}) for {provider}: stderr={err.strip()[:400]} stdout={out.strip()[:400]}")
            return False

        logger.info(f"Lease created dseq={dseq} provider={provider}")

        cmd = [
            "provider-services", "send-manifest", sdl_path,
            "--dseq", str(dseq),
            "--provider", provider,
            "--from", self.wallet_name,
            "--node", self.node,
            "--home", os.path.expanduser("~/.akash"),
            "--keyring-backend", "test"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.error(f"send-manifest failed for {provider}: {err.strip()[:400]}")
            return False

        logger.info(f"Manifest sent successfully for dseq={dseq} provider={provider}")
        return True

    async def get_deployment(self, dseq: str) -> Optional[Dict[str, Any]]:
        if self.dry_run:
            return {"leases": [{"status": {"services": {"whisper": {"uris": ["dry-whisper.example.com"]}, "qwen": {"uris": ["dry-qwen.example.com"]}}}}]}

        if not await self._ensure_initialized():
            return None

        cmd = [
            "akash", "query", "market", "lease", "list",
            "--owner", self.wallet_address,
            "--dseq", str(dseq),
            "--node", self.node,
            "-o", "json"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.warning(f"fetch leases failed: {err}")
            return None

        try:
            data = json.loads(out)
            leases = data.get("leases", [])
            if not leases:
                return None
            lease = leases[0]
            lease_id = lease.get("lease", {}).get("lease_id", {})
            provider = lease_id.get("provider", "")
        except Exception as e:
            logger.warning(f"Failed to parse leases: {e}")
            return None

        cmd = [
            "provider-services", "lease-status",
            "--dseq", str(dseq),
            "--provider", provider,
            "--from", self.wallet_name,
            "--node", self.node,
            "--home", os.path.expanduser("~/.akash"),
            "--keyring-backend", "test",
            "-o", "json"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.warning(f"lease-status failed: {err}")
            return None

        try:
            status = json.loads(out)
            services_dict = {}
            for svc_name, svc_obj in status.get("services", {}).items():
                uris = svc_obj.get("uris", [])
                services_dict[svc_name] = {"uris": uris}
            return {"leases": [{"status": {"services": services_dict}}]}
        except Exception as e:
            logger.warning(f"Failed to parse lease status: {e}")
            return None

    async def close_deployment(self, dseq: str) -> bool:
        if not dseq:
            return True
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would close DSEQ {dseq}")
            return True

        if not await self._ensure_initialized():
            return False

        cmd = [
            "akash", "tx", "deployment", "close",
            "--dseq", str(dseq),
            "--from", self.wallet_name,
            "--node", self.node,
            "--chain-id", self.chain_id,
            "--gas", "500000",
            "--fees", "20000uakt",
            "-y",
            "--keyring-backend", "test"
        ]
        rc, out, err = await self._run_cmd(cmd)
        if rc != 0:
            logger.error(f"close_deployment failed: {err}")
            return False

        logger.info(f"Closed DSEQ {dseq}")
        return True


akash = AkashCLI(dry_run=DRY_RUN)


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


async def resolve_and_push_uris(whisper_dseq: str, qwen_dseq: str) -> bool:
    deadline = time.time() + URI_WAIT_TIMEOUT
    while time.time() < deadline:
        whisper_dep = await akash.get_deployment(whisper_dseq) if whisper_dseq else None
        qwen_dep = await akash.get_deployment(qwen_dseq) if qwen_dseq else None

        whisper_uris = _extract_service_uris(whisper_dep) if whisper_dep else {}
        qwen_uris = _extract_service_uris(qwen_dep) if qwen_dep else {}

        whisper_url = whisper_uris.get("whisper", "")
        qwen_url = qwen_uris.get("qwen", "")

        if whisper_url and qwen_url:
            ctx.whisper_url = whisper_url
            ctx.qwen_url = qwen_url
            _persist_state()
            if await push_downstream_urls(whisper_url, qwen_url):
                return True
            logger.warning("URIs found but push failed - will retry")
        else:
            logger.info(f"Waiting for URIs... whisper={whisper_url or '-'} qwen={qwen_url or '-'}")
        await asyncio.sleep(URI_POLL_INTERVAL)
    logger.error(f"URIs did not appear within {URI_WAIT_TIMEOUT}s")
    return False


# ==============================================================================
# Wait for bids + create lease for a single deployment
# ==============================================================================
async def _wait_and_lease(dseq: str, label: str, sdl_path: str) -> bool:
    if DRY_RUN:
        ok = await akash.create_lease(dseq, 1, 1, "akash1dryprovider", sdl_path)
        if ok:
            if label == "whisper":
                ctx.whisper_gseq = 1
                ctx.whisper_oseq = 1
                ctx.whisper_provider = "akash1dryprovider"
            else:
                ctx.qwen_gseq = 1
                ctx.qwen_oseq = 1
                ctx.qwen_provider = "akash1dryprovider"
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
                logger.info(f"Attempting lease ({label}) with provider={prov} price={_price(b)}")
                if await akash.create_lease(dseq, gseq, oseq, prov, sdl_path):
                    if label == "whisper":
                        ctx.whisper_gseq = gseq
                        ctx.whisper_oseq = oseq
                        ctx.whisper_provider = prov
                    else:
                        ctx.qwen_gseq = gseq
                        ctx.qwen_oseq = oseq
                        ctx.qwen_provider = prov
                    return True
                await asyncio.sleep(6)
            logger.warning(f"All bids failed for {label} - retrying")
        logger.info(f"No bids yet for {label} (attempt {attempt})")
        await asyncio.sleep(AKASH_BIDS_INTERVAL)
    logger.error(f"No bid accepted for {label} within {AKASH_BIDS_TIMEOUT}s")
    return False


# ==============================================================================
# Core actions: START / STOP
# ==============================================================================
async def action_start() -> bool:
    ctx.state = ControllerState.STARTING
    ctx.started_at = time.time()
    logger.info("=" * 60)
    logger.info("ACTION: START")

    if ctx.whisper_dseq and ctx.qwen_dseq:
        logger.info(f"Both GPU deployments already exist (whisper={ctx.whisper_dseq}, qwen={ctx.qwen_dseq}); skipping create")
        return True

    whisper_sdl = _load_sdl(WHISPER_SDL_PATH)
    qwen_sdl = _load_sdl(QWEN_SDL_PATH)

    if not whisper_sdl:
        ctx.last_error = f"Whisper SDL missing: {WHISPER_SDL_PATH}"
        ctx.state = ControllerState.ERROR
        return False
    if not qwen_sdl:
        ctx.last_error = f"Qwen SDL missing: {QWEN_SDL_PATH}"
        ctx.state = ControllerState.ERROR
        return False

    whisper_dseq = await akash.create_deployment(whisper_sdl, "whisper")
    if not whisper_dseq:
        ctx.last_error = "create whisper deployment failed"
        ctx.state = ControllerState.ERROR
        return False

    qwen_dseq = await akash.create_deployment(qwen_sdl, "qwen")
    if not qwen_dseq:
        ctx.last_error = "create qwen deployment failed"
        ctx.state = ControllerState.ERROR
        await akash.close_deployment(whisper_dseq)
        return False

    ctx.whisper_dseq = whisper_dseq
    ctx.qwen_dseq = qwen_dseq
    _persist_state()

    whisper_lease_ok = await _wait_and_lease(whisper_dseq, "whisper", f"/tmp/whisper_sdl.yaml")
    qwen_lease_ok = await _wait_and_lease(qwen_dseq, "qwen", f"/tmp/qwen_sdl.yaml")

    if not whisper_lease_ok or not qwen_lease_ok:
        ctx.last_error = "failed to secure lease for one or both GPU deployments"
        ctx.state = ControllerState.ERROR
        await akash.close_deployment(whisper_dseq)
        await akash.close_deployment(qwen_dseq)
        ctx.whisper_dseq = ""
        ctx.qwen_dseq = ""
        return False

    if not await resolve_and_push_uris(whisper_dseq, qwen_dseq):
        ctx.last_error = "URI resolution failed"
        ctx.state = ControllerState.ERROR
        return False

    ctx.state = ControllerState.RUNNING
    ctx.total_starts += 1
    ctx.last_activity_at = time.time()
    _persist_state()
    logger.info(f"START complete: whisper_dseq={whisper_dseq}, qwen_dseq={qwen_dseq}")
    return True


async def action_stop() -> bool:
    ctx.state = ControllerState.STOPPING
    logger.info("=" * 60)
    logger.info(f"ACTION: STOP whisper_dseq={ctx.whisper_dseq} qwen_dseq={ctx.qwen_dseq}")

    ok = True
    if ctx.whisper_dseq:
        ok = await akash.close_deployment(ctx.whisper_dseq) and ok
    if ctx.qwen_dseq:
        ok = await akash.close_deployment(ctx.qwen_dseq) and ok

    if not ok:
        ctx.last_error = "failed to close one or both deployments"
        ctx.state = ControllerState.ERROR
        return False

    ctx.whisper_dseq = ""
    ctx.whisper_provider = ""
    ctx.whisper_gseq = 0
    ctx.whisper_oseq = 0
    ctx.qwen_dseq = ""
    ctx.qwen_provider = ""
    ctx.qwen_gseq = 0
    ctx.qwen_oseq = 0
    ctx.whisper_url = ""
    ctx.qwen_url = ""
    ctx.stopped_at = time.time()
    ctx.state = ControllerState.IDLE
    ctx.total_stops += 1
    ctx.total_cycles += 1
    _persist_state()

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

                queue_depth = int(status.get("queue_size", 0) or 0)
                if queue_depth > 0:
                    ctx.last_activity_at = time.time()

            now = time.time()
            idle_for = now - ctx.last_activity_at

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
                await action_stop()
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
    logger.info("Lordera Controller v1.6.13 starting (REST API for bid queries)")
    logger.info(f"Config: poll={POLL_INTERVAL}s idle={IDLE_TIMEOUT}s dry_run={DRY_RUN}")
    logger.info(f"Akash: node={AKASH_NODE} chain={AKASH_CHAIN_ID}")
    logger.info(f"Binaries: akash={AKASH_AVAILABLE}, provider-services={PROVIDER_SERVICES_AVAILABLE}")
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
    description="Monitors Gateway queue and manages on-demand Akash GPU deployments via CLI.",
    version="1.6.13",
    lifespan=lifespan,
)


# ==============================================================================
# Schemas
# ==============================================================================
class HealthResponse(BaseModel):
    status: str
    state: str
    whisper_dseq: str
    whisper_provider: str
    qwen_dseq: str
    qwen_provider: str
    whisper_url: str
    qwen_url: str
    consecutive_errors: int
    dry_run: bool
    akash_key_set: bool
    uptime_seconds: int


class StatusResponse(BaseModel):
    state: str
    whisper_dseq: str
    whisper_provider: str
    whisper_gseq: int
    whisper_oseq: int
    qwen_dseq: str
    qwen_provider: str
    qwen_gseq: int
    qwen_oseq: int
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
        "version": "1.6.13",
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
        whisper_dseq=ctx.whisper_dseq,
        whisper_provider=ctx.whisper_provider,
        qwen_dseq=ctx.qwen_dseq,
        qwen_provider=ctx.qwen_provider,
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
        whisper_dseq=ctx.whisper_dseq,
        whisper_provider=ctx.whisper_provider,
        whisper_gseq=ctx.whisper_gseq,
        whisper_oseq=ctx.whisper_oseq,
        qwen_dseq=ctx.qwen_dseq,
        qwen_provider=ctx.qwen_provider,
        qwen_gseq=ctx.qwen_gseq,
        qwen_oseq=ctx.qwen_oseq,
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
            "whisper_sdl_path": WHISPER_SDL_PATH,
            "qwen_sdl_path": QWEN_SDL_PATH,
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
    return {"status": "started", "state": ctx.state.value,
            "whisper_dseq": ctx.whisper_dseq, "qwen_dseq": ctx.qwen_dseq}


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
    ctx.whisper_dseq = ""
    ctx.whisper_provider = ""
    ctx.whisper_gseq = 0
    ctx.whisper_oseq = 0
    ctx.qwen_dseq = ""
    ctx.qwen_provider = ""
    ctx.qwen_gseq = 0
    ctx.qwen_oseq = 0
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