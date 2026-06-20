# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Production
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for FAISS and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code
COPY asft/ ./asft/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY README.md .

# Create non-root user
RUN useradd -m asftuser && \
    mkdir -p /app/asft_data && \
    chown -R asftuser:asftuser /app

USER asftuser

# Expose API port
EXPOSE 8000

# Start command
CMD ["uvicorn", "asft.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
