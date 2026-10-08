# ========================================================
# 🌐 Public Forensic Web UI Dockerfile (Port 5000 Only)
# ========================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, PIL, and native tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for layer caching
COPY requirements-docker.txt ./requirements.txt

# Install CPU-optimized PyTorch and web dependencies in a single pass to eliminate redundant downloads
RUN pip install --no-cache-dir --root-user-action=ignore --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy application source code
COPY . /app

# Create directory for uploads, cache and benchmark runs
RUN mkdir -p /app/uploads /root/.cache/huggingface /app/runs/benchmarks

# Expose Web UI Port ONLY (5000)
EXPOSE 5000

# Healthcheck to verify Flask Web Service status
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start Flask web server with SocketIO eventlet
CMD ["python", "app.py"]
