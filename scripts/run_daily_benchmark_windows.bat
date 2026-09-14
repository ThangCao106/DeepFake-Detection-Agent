@echo off
REM =========================================================
REM Windows Task Scheduler Script for Daily Benchmark Run
REM =========================================================
cd /d "d:\DATA\TC\Py_Ex\NCS\DeepFake-Detection-Agent"
echo [%DATE% %TIME%] Starting daily benchmark run... >> benchmark.log
py -3 benchmark/daily_benchmark_search.py >> benchmark.log 2>&1
git add report_colab/daily_benchmark/ >> benchmark.log 2>&1
git commit -m "Auto-generated daily benchmark report [windows task scheduler]" >> benchmark.log 2>&1
git pull --rebase origin main >> benchmark.log 2>&1
git push origin main >> benchmark.log 2>&1
echo [%DATE% %TIME%] Daily benchmark completed successfully. >> benchmark.log
