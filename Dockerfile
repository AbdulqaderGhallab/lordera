# ==============================================================================
# Lordera AI — Unified Dockerfile (v3.2 - FIXED)
# ==============================================================================
# Builds ALL FOUR service images from a single file
#   docker build --build-arg SERVICE=whisper_service    --build-arg PORT=8001 -t abdulqaderghallab/lordera-whisper:v3    .
#   docker build --build-arg SERVICE=qwen_service       --build-arg PORT=8002 -t abdulqaderghallab/lordera-qwen:v3       .
#   docker build --build-arg SERVICE=gateway_service    --build-arg PORT=8000 -t abdulqaderghallab/lordera-gateway:v3    .
#   docker build --build-arg SERVICE=controller_service --build-arg PORT=8003 -t abdulqaderghallab/lordera-controller:v3 .
# ==============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# استخدام bash بدلاً من sh لضمان توافق سكربتات التثبيت
SHELL ["/bin/bash", "-c"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# 1- تثبيت الحزم الأساسية أولاً (بما فيها curl و ca-certificates)
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
    jq \
    && rm -rf /var/lib/apt/lists/*

# 2- تحميل وتثبيت Akash & Provider-Services (بعد التأكد من وجود curl)
RUN curl -sSfL https://raw.githubusercontent.com/akash-network/node/main/install.sh | bash \
    && curl -sSfL https://raw.githubusercontent.com/akash-network/provider/main/install.sh | bash \
    && mv ./bin/akash /usr/local/bin/akash \
    && mv ./bin/provider-services /usr/local/bin/provider-services \
    && chmod +x /usr/local/bin/akash /usr/local/bin/provider-services \
    && rm -rf ./bin

RUN pip3 install --upgrade pip setuptools wheel

# 3- تثبيت متطلبات بايثون
COPY requirements.txt .
RUN pip3 install --retries 10 --timeout 120 --resume-retries 20 -r requirements.txt

# 4- نسخ الكود
COPY . .

RUN mkdir -p /models /state

ARG SERVICE=gateway_service
ARG PORT=8000
ENV SERVICE=${SERVICE} \
    PORT=${PORT}

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "exec uvicorn ${SERVICE}:app --host 0.0.0.0 --port ${PORT} --workers 1"]