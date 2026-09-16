"""
Evaluation Benchmark Script for General Deepfake Detector Model (General AI Media, CIFAKE, Omni-Fake).
"""
import os
import sys
import json
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BENCHMARK_DIR))

from eval_benchmark import run_benchmark

def run_general_benchmark():
    print("=" * 70)
    print("📊 Executing General AI Deepfake Benchmark Evaluation")
    print("=" * 70)
    run_benchmark()

if __name__ == "__main__":
    run_general_benchmark()
