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

# Run container with dual ports: 5000 (Forensic Web UI) and 5010 (Benchmark Hub)
echo "[2/2] Starting Container on Ports 5000 & 5010..."
docker run -d \
  --name deepfake_detection_web \
  -p 5000:5000 \
  -p 5010:5000 \
  -e PORT=5000 \
  -v "$(pwd)/uploads:/app/uploads" \
  -v "$(pwd)/runs/benchmarks:/app/runs/benchmarks" \
  --restart unless-stopped \
  deepfake-benchmark-ui:latest

echo "========================================================"
echo "✅ Container started successfully!"
echo "🌐 Port 5000 (Public Forensic Web UI): http://localhost:5000"
echo "⚡ Port 5010 (Benchmark Hub Gate Login): http://localhost:5010"
echo "📋 To view logs: docker logs -f deepfake_detection_web"
echo "========================================================"
