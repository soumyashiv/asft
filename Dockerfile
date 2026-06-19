# ASFT Production Dockerfile
# Multi-stage build for optimal image size and caching

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install hatchling (build backend)
RUN pip install --no-cache-dir hatchling

# Copy project files
COPY pyproject.toml README.md ./
COPY asft/ asft/

# Build wheel
RUN hatch build -t wheel

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# Labels
LABEL maintainer="ASFT Team"
LABEL description="Adaptive Sparse Fine-Tuning Framework"
LABEL version="0.2.0"

WORKDIR /app

# Install runtime dependencies (e.g., git for HuggingFace)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash asft_user

# Create directories for persistent data
RUN mkdir -p /app/asft_data/checkpoints /app/asft_data/datasets \
    && chown -R asft_user:asft_user /app/asft_data

# Copy built wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install the wheel and production dependencies
# Note: we explicitly install PyTorch with CPU/CUDA depending on the deployment.
# This Dockerfile defaults to CPU. For GPU, use a nvidia/cuda base image.
RUN pip install --no-cache-dir /tmp/*.whl \
    && pip install --no-cache-dir uvicorn[standard] aiosqlite structlog

# Switch to non-root user
USER asft_user

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["uvicorn", "asft.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--proxy-headers"]
