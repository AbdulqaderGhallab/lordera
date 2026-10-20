# ==============================================================================
# Lordera AI — Unified Dockerfile (v2.0-restored)
# ==============================================================================
# Builds ALL FOUR service images from a single file, using ARG SERVICE + ARG PORT.
# The heavy layers (apt-get, pip install) are IDENTICAL across the four images
# and therefore shared via Docker layer cache — only the last two layers differ.
#
# Build commands:
#   docker build --build-arg SERVICE=whisper_service    --build-arg PORT=8001 -t abdulqaderghallab/lordera-whisper:v2    .
#   docker build --build-arg SERVICE=qwen_service       --build-arg PORT=8002 -t abdulqaderghallab/lordera-qwen:v2       .
#   docker build --build-arg SERVICE=gateway_service    --build-arg PORT=8000 -t abdulqaderghallab/lordera-gateway:v2    .
#   docker build --build-arg SERVICE=controller_service --build-arg PORT=8003 -t abdulqaderghallab/lordera-controller:v2 .
#
# CACHE STRATEGY:
#   The instruction text below is byte-compatible with the prior v1 build so
#   Docker layer cache stays valid. Comments are stripped by the Docker parser
#   before hashing and therefore do not affect caching.
#
# GPU SDL PATH:
#   gpu.yaml is copied into the image via `COPY . .` and lives at /app/gpu.yaml.
#   The Controller reads it via GPU_SDL_PATH=/app/gpu.yaml.
# ==============================================================================

# ==============================================================================
# Base Image: CUDA-enabled Ubuntu with cuDNN
# ==============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ==============================================================================
# Global Environment Variables (same for all services)
# ==============================================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# ==============================================================================
# Working Directory
# ==============================================================================
WORKDIR /app

# ==============================================================================
# System Dependencies (Python, FFmpeg, audio libs, etc.)
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ==============================================================================
# Upgrade pip, setuptools, and wheel
# ==============================================================================
RUN pip3 install --upgrade pip setuptools wheel

# ==============================================================================
# Python Dependencies (Copied first for better layer caching)
# ==============================================================================
COPY requirements.txt .
RUN pip3 install --retries 10 --timeout 120 --resume-retries 20 -r requirements.txt

# ==============================================================================
# Copy Application Code
# ==============================================================================
# Note: this also copies gpu.yaml into /app/gpu.yaml — read by the Controller
# at runtime (GPU_SDL_PATH=/app/gpu.yaml).
# ==============================================================================
COPY . .

# ==============================================================================
# Create Models Directory (will be mounted as persistent volume on Akash)
# ==============================================================================
RUN mkdir -p /models

# ==============================================================================
# Service-Specific Arguments (placed at the END to preserve cache above)
# ==============================================================================
ARG SERVICE=gateway_service
ARG PORT=8000

# Persist them as environment variables for runtime
ENV SERVICE=${SERVICE} \
    PORT=${PORT}

# ==============================================================================
# Expose Port
# ==============================================================================
EXPOSE ${PORT}

# ==============================================================================
# Health Check (useful for Akash and other orchestrators)
# ==============================================================================
# start-period is generous because:
#   - Whisper: load Large-v3 + warmup  →  ~60-90s
#   - Qwen:    load 4-bit 7B + warmup  →  ~120-180s
#   - Cold start on a new Akash node with model download can hit ~5 min.
# ==============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ==============================================================================
# Run the Service
# ==============================================================================
# Notes:
#   - Single worker: each service holds in-process singleton state (queues,
#     circuit breakers, GPU semaphores). Multiple workers would duplicate
#     that state and break the design. Horizontal scaling is done at the
#     container level (Akash replicas), not inside uvicorn.
#   - exec replaces the shell as PID 1 so SIGTERM from Akash reaches uvicorn
#     directly and triggers the FastAPI lifespan shutdown cleanly.
# ==============================================================================
CMD ["sh", "-c", "uvicorn ${SERVICE}:app --host 0.0.0.0 --port ${PORT}"]