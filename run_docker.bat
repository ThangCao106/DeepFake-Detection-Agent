@echo off
REM ========================================================
REM 🚀 Build and Run 2 Separate Docker Containers (Ports 5000 & 5010)
REM ========================================================

echo ========================================================
echo 🚀 Building and Running Deepfake Services on Ports 5000 & 5010
echo ========================================================

REM Stop old containers if exist
docker rm -f deepfake_detection_web deepfake_benchmark_hub 2>nul

REM 1. Build and Run Public Forensic Web UI (Port 5000 Only)
echo [1/2] Building and Starting Container 1: Forensic Web UI (Port 5000)...
docker build -f Dockerfile -t deepfake-web-ui:latest .
docker run -d ^
  --name deepfake_detection_web ^
  -p 5000:5000 ^
  -e PORT=5000 ^
  -v "%cd%\uploads:/app/uploads" ^
  -v "%cd%\runs\benchmarks:/app/runs/benchmarks" ^
  --restart unless-stopped ^
  deepfake-web-ui:latest

REM 2. Build and Run Dedicated Benchmark Hub (Port 5010 Only)
echo [2/2] Building and Starting Container 2: Benchmark Hub (Port 5010)...
docker build -f Dockerfile.benchmark -t deepfake-benchmark-hub:latest .
docker run -d ^
  --name deepfake_benchmark_hub ^
  -p 5010:5010 ^
  -e PORT=5010 ^
  -v "%cd%\uploads:/app/uploads" ^
  -v "%cd%\runs\benchmarks:/app/runs/benchmarks" ^
  --restart unless-stopped ^
  deepfake-benchmark-hub:latest

echo ========================================================
echo ✅ Both Docker containers started successfully and independently!
echo 🌐 Container 1 (Public Forensic Web UI):  http://localhost:5000
echo ⚡ Container 2 (Benchmark Hub Subsystem): http://localhost:5010
echo 📋 To view logs Web UI:       docker logs -f deepfake_detection_web
echo 📋 To view logs Benchmark:    docker logs -f deepfake_benchmark_hub
echo ========================================================
pause
