#!/usr/bin/env bash
# ========================================================
# 🚀 Build and Run Deepfake Multi-Zone Services via Docker Compose
#   - Port 5000: Public Forensic Studio (/)
#   - Port 5010: Benchmark Hub Nginx Proxy (/benchmark/)
#   - Port 8787: Forensic SRE Cockpit (/dfagent/)
# ========================================================
set -euo pipefail

echo "========================================================"
echo "🚀 Starting Deepfake Multi-Zone Architecture (Ports 5000, 5010, 8787)"
echo "========================================================"

# Stop legacy duplicate containers if present
docker rm -f deepfake_detection_web deepfake_benchmark_hub deepfake_observability_agent 2>/dev/null || true

docker compose up -d --build --remove-orphans

echo "========================================================"
echo "✅ All Docker Compose services started cleanly!"
echo "🌐 Public Forensic Studio : http://localhost:5000/"
echo "🔬 Benchmark Hub Proxy    : http://localhost:5010/benchmark/"
echo "🛡️ Forensic SRE Cockpit   : http://localhost:8787/dfagent/"
echo "📋 Logs (Web UI / SRE)    : docker compose logs -f deepfake-web-ui observability-agent"
echo "========================================================"
