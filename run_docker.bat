@echo off
REM ========================================================
REM 🚀 Build and Run 2 Separate Docker Containers (Ports 5000 & 5010)
REM ========================================================

echo ========================================================
echo 🚀 Building and Running Deepfake Services on Ports 5000 & 5010
echo ========================================================

REM Stop old containers if exist
docker rm -f deepfake_detection_web deepfake_benchmark_hub 2>nul

REM Build shared image
echo [1/3] Building Docker Image (deepfake-detection-app:latest)...
docker build -t deepfake-detection-app:latest .

REM 1. Run Public Forensic Web UI on Port 5000
echo [2/3] Starting Container 1: Forensic Web UI on Port 5000...
docker run -d ^
  --name deepfake_detection_web ^
  -p 5000:5000 ^
  -e PORT=5000 ^
  -v "%cd%\uploads:/app/uploads" ^
  -v "%cd%\runs\benchmarks:/app/runs/benchmarks" ^
  --restart unless-stopped ^
  deepfake-detection-app:latest

REM 2. Run Dedicated Benchmark Hub on Port 5010
echo [3/3] Starting Container 2: Benchmark Hub on Port 5010...
docker run -d ^
  --name deepfake_benchmark_hub ^
  -p 5010:5010 ^
  -e PORT=5010 ^
  -v "%cd%\uploads:/app/uploads" ^
  -v "%cd%\runs\benchmarks:/app/runs/benchmarks" ^
  --restart unless-stopped ^
  deepfake-detection-app:latest

echo ========================================================
echo ✅ Both Docker containers started successfully and independently!
echo 🌐 Container 1 (Public Forensic Web UI):  http://localhost:5000
echo ⚡ Container 2 (Benchmark Hub Subsystem): http://localhost:5010
echo 📋 To view logs Web UI:       docker logs -f deepfake_detection_web
echo 📋 To view logs Benchmark:    docker logs -f deepfake_benchmark_hub
echo ========================================================
pause
