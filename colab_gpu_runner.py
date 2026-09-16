"""
Google Colab / GPU Worker Execution Pipeline for DeepFake Pentest
Orchestrates:
  1. Reading Raw Crawled Media from GDrive ('01_raw_originals')
  2. Running Open-Weight FaceSwap Engine ('02_manipulated_faceswap')
  3. Running Open-Weight Generative AI Engine ('03_manipulated_openweight')
  4. Executing Pentest Evaluation & ECE Calibration Audit
"""
import os
import sys
import time
import asyncio
from pathlib import Path

from crawler.gdrive_sync import GDriveSyncAgent
from crawler.threads_crawler import crawl_threads_media
from generators.faceswap_engine import FaceSwapEngine
from generators.openweight_gen_engine import OpenWeightGenerativeEngine

async def run_colab_gpu_pipeline(batch_images=5000, batch_videos=5000):
    timestamp_str = time.strftime("%Y-%m-%d_%Hh")
    date_str = time.strftime("%Y-%m-%d")

    print("=" * 70)
    print(f"🚀 Starting Colab GPU DeepFake Generation & Pentest Pipeline [{timestamp_str}]")
    print("=" * 70)

    # Step 1: Crawl / Sync Raw Media to GDrive (01_raw_originals)
    print("\n[Step 1/4] Crawling Threads Media & Syncing to GDrive 01_raw_originals...")
    raw_reg_file, num_img, num_vid = await crawl_threads_media(
        batch_size_images=batch_images, batch_size_videos=batch_videos
    )

    # Step 2: Run Open-Weight FaceSwap Engine (02_manipulated_faceswap)
    print("\n[Step 2/4] Executing InsightFace / SimSwap FaceSwap Engine...")
    faceswap_engine = FaceSwapEngine(model_name="InsightFace-InSwapper128")
    fs_reg_file, fs_img, fs_vid = faceswap_engine.process_batch(
        raw_registry_path=str(raw_reg_file), date_str=date_str
    )

    # Step 3: Run Open-Weight Generative AI Engine (03_manipulated_openweight)
    print("\n[Step 3/4] Executing Flux.1-dev & Wan2.1 Open-Weight Generative AI Engine...")
    openweight_engine = OpenWeightGenerativeEngine(image_model="Flux.1-dev", video_model="Wan2.1-14B-VideoGen")
    ow_reg_file, ow_img, ow_vid = openweight_engine.process_batch(
        raw_registry_path=str(raw_reg_file), date_str=date_str
    )

    # Step 4: Run Model Pentest Evaluation & ECE Audit
    print("\n[Step 4/4] Executing Model Pentest Evaluation across Raw vs. FaceSwap vs. OpenWeight...")
    import subprocess
    cmd = [sys.executable, "pentest_runner.py", "--batch-tag", timestamp_str]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[Warning/Err]:", res.stderr)

    print("\n" + "=" * 70)
    print(f"✅ Colab GPU Pipeline Completed Successfully!")
    print(f"📊 Summary:")
    print(f"   - Raw Media (01_raw_originals): {num_img} Images, {num_vid} Videos")
    print(f"   - FaceSwap (02_manipulated_faceswap): {fs_img} Images, {fs_vid} Videos")
    print(f"   - Open-Weight Gen (03_manipulated_openweight): {ow_img} Images, {ow_vid} Videos")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_colab_gpu_pipeline(batch_images=5000, batch_videos=5000))
