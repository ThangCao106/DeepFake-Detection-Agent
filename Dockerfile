# Use official lightweight Python image
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
COPY requirements.txt .

# Install CPU-optimized PyTorch and dependencies to minimize image size (~1.5GB vs 5GB+ CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Create directory for uploads and cache
RUN mkdir -p /app/uploads /root/.cache/huggingface

# Expose Web Application Port
EXPOSE 5000

# Healthcheck to verify Flask Web Service status
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/api/status || exit 1

# Start Flask web server with SocketIO eventlet
CMD ["python", "app.py"]
