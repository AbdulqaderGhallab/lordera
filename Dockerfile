# ==============================================================================
# Lordera AI — Unified Dockerfile (v3.0 - FIXED)
# ==============================================================================
# Builds ALL FOUR service images from a single file
#   docker build --build-arg SERVICE=whisper_service    --build-arg PORT=8001 -t abdulqaderghallab/lordera-whisper:v3    .
#   docker build --build-arg SERVICE=qwen_service       --build-arg PORT=8002 -t abdulqaderghallab/lordera-qwen:v3       .
#   docker build --build-arg SERVICE=gateway_service    --build-arg PORT=8000 -t abdulqaderghallab/lordera-gateway:v3    .
#   docker build --build-arg SERVICE=controller_service --build-arg PORT=8003 -t abdulqaderghallab/lordera-controller:v3 .
# ==============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# 1- System deps + Akash binary (هذا هو الإصلاح الأساسي للـ controller)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSfL https://raw.githubusercontent.com/akash-network/node/main/install.sh | sh \
    && mv ./bin/akash /usr/local/bin/akash \
    && chmod +x /usr/local/bin/akash \
    && rm -rf ./bin

RUN pip3 install --upgrade pip setuptools wheel

# 2- Python deps (انسخ كل ملفات المتطلبات للاستفادة من cache)
COPY requirements.txt .
COPY requirements-controller.txt ./requirements-controller.txt

# تثبيت المتطلبات المشتركة
RUN pip3 install --retries 10 --timeout 120 --resume-retries 20 -r requirements.txt

# تثبيت متطلبات الـ controller الإضافية (ecdsa, cosmospy, grpcio...)
RUN if [ -f requirements-controller.txt ]; then pip3 install --retries 10 --timeout 120 -r requirements-controller.txt; fi

# 3- Copy code (يشمل gpu.yaml)
COPY . .

RUN mkdir -p /models /state

ARG SERVICE=gateway_service
ARG PORT=8000
ENV SERVICE=${SERVICE} \
    PORT=${PORT}

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# exec لوصول SIGTERM مباشرة لـ uvicorn
CMD ["sh", "-c", "exec uvicorn ${SERVICE}:app --host 0.0.0.0 --port ${PORT} --workers 1"]
