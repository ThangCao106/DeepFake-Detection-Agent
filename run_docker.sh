#!/usr/bin/env bash
# ========================================================
# 🚀 Build and Run Deepfake Benchmark UI on Port 5010 (Linux/Mac/WSL)
# ========================================================
set -e

echo "========================================================"
echo "🚀 Building and Running Deepfake Benchmark UI on Port 5010"
echo "========================================================"

# Stop old container if exists
docker rm -f deepfake_detection_web 2>/dev/null || true

# Build image
echo "[1/2] Building Docker Image (deepfake-benchmark-ui:latest)..."
docker build -t deepfake-benchmark-ui:latest .

# Run container
echo "[2/2] Starting Container on http://localhost:5010..."
docker run -d \
  --name deepfake_detection_web \
  -p 5010:5010 \
  -e PORT=5010 \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/runs/benchmarks:/app/runs/benchmarks" \
  --restart unless-stopped \
  deepfake-benchmark-ui:latest

echo "========================================================"
echo "✅ Container started successfully!"
echo "🌐 Open Web Browser at: http://localhost:5010"
echo "📋 To view logs: docker logs -f deepfake_detection_web"
echo "========================================================"
