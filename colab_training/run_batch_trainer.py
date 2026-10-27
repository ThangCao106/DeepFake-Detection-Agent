"""
Option B Dedicated Batch Trainer for Google Colab.
Decoupled Architecture:
  - Job 1 (Harvester): Crawls Threads & Wikimedia Commons, creates paired counterfactuals,
    and uploads 100% of media assets to Google Drive every 4h.
  - Job 2 (This Trainer): Reads accumulated media batches from Google Drive, stages data
    to fast local NVMe SSD (/content/train_cache), trains Model 1 (ViT Vision) and
    Model 2 (Human Biometrics), and commits checkpoints back to Google Drive.

Usage:
  # Train on all available dates in Google Drive:
  !python colab_training/run_batch_trainer.py --date all

  # Train on a specific date (including all its batch_* subfolders):
  !python colab_training/run_batch_trainer.py --date 2026-09-29 --epochs 3

  # Train only Model 1 or Model 2:
  !python colab_training/run_batch_trainer.py --date 2026-09-29 --model1-only
  !python colab_training/run_batch_trainer.py --date 2026-09-29 --model2-only
"""

import os
import sys
import time
import shutil
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from colab_training.train_threads_continual import run_threads_continual_training, find_all_manifests
from colab_training.train_human import train_human_model


def resolve_drive_root(custom_path: str = None) -> Path:
    """Finds the active Google Drive Thread_Data directory."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([
        Path("/content/drive/MyDrive/Thread_Data"),
        Path("/content/drive/MyDrive/DeepFake-Detection-Benchmark"),
        Path("/content/drive/Shareddrives/Thread_Data"),
        PROJECT_ROOT / "report_colab" / "gdrive_storage",
        PROJECT_ROOT / "report_colab" / "pentest"
    ])
    for p in candidates:
        if p.exists():
            return p
    # Fallback to local
    fallback = PROJECT_ROOT / "report_colab" / "gdrive_storage"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def stage_drive_data_to_local(
    drive_root: Path,
    target_dates: list[str],
    local_cache_dir: Path,
    max_workers: int = 16
) -> tuple[Path, int]:
    """
    Copies images and manifests from Google Drive FUSE into fast local NVMe SSD
    to eliminate I/O latency and avoid 403 Drive rate-limiting during training.
    """
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 75)
    print(f"⚡ [Option B Staging] Pre-caching dataset from Google Drive to local SSD:")
    print(f"   Source Drive: {drive_root}")
    print(f"   Local Cache:  {local_cache_dir}")
    print("=" * 75)

    # Collect source subdirectories
    subdirs_to_copy = []
    if "all" in target_dates:
        for child in sorted(drive_root.iterdir()):
            if child.is_dir() and not child.name.startswith((".", "checkpoints")):
                subdirs_to_copy.append(child)
    else:
        for d_str in target_dates:
            target_p = drive_root / d_str
            if target_p.exists() and target_p.is_dir():
                subdirs_to_copy.append(target_p)
                
    # Always include Option B compact biometric features cache if present
    features_p = drive_root / "features"
    if features_p.exists() and features_p.is_dir() and features_p not in subdirs_to_copy:
        subdirs_to_copy.append(features_p)

    if not subdirs_to_copy:
        print("⚠️ No matching date directories found on Google Drive. Training will search Drive directly.")
        return drive_root, 0

    files_to_copy = []
    for sdir in subdirs_to_copy:
        for fpath in sdir.glob("**/*.*"):
            if fpath.is_file() and not fpath.name.startswith("."):
                rel = fpath.relative_to(drive_root)
                dest = local_cache_dir / rel
                files_to_copy.append((fpath, dest))

    print(f"   Identified {len(files_to_copy)} files across {len(subdirs_to_copy)} directories to stage...")

    def _copy_worker(pair):
        src, dst = pair
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(str(src), str(dst))

    start_c = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_copy_worker, files_to_copy))
        
    duration = time.time() - start_c
    print(f"✅ [Staging Complete] Cached {len(files_to_copy)} files in {duration:.1f}s ({len(files_to_copy)/(max(duration, 0.1)):.0f} files/sec)!")
    return local_cache_dir, len(files_to_copy)


def main():
    parser = argparse.ArgumentParser(description="Option B Dedicated Batch Trainer for Threads-DFBench")
    parser.add_argument("--drive-root", "--drive_root", type=str, default=None, help="Path to Google Drive Thread_Data root")
    parser.add_argument("--date", type=str, default="2026-09-29", help="Target date(s) (e.g. 2026-09-29, or 'all', default: 2026-09-29)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--model1-only", action="store_true", help="Train only Model 1 (General Vision ViT)")
    parser.add_argument("--model2-only", action="store_true", help="Train only Model 2 (Human Biometrics)")
    parser.add_argument("--no-cache", action="store_true", help="Disable local SSD staging cache and read directly from Google Drive")
    args = parser.parse_args()

    drive_root = resolve_drive_root(args.drive_root)
    target_dates = [d.strip() for d in args.date.split(",") if d.strip()]

    print("=" * 80)
    print("🚀 OPTION B: DEDICATED BATCH TRAINER (GOOGLE COLAB)")
    print(f"📂 Google Drive Root: {drive_root}")
    print(f"📅 Target Date Filter: {', '.join(target_dates)}")
    print(f"⚙️ Config: Epochs={args.epochs} | Batch Size={args.batch_size} | LR={args.lr}")
    print("=" * 80)

    # 1. Staging Data to Local NVMe SSD (Avoid FUSE Bottleneck)
    local_cache_dir = Path("/content/train_cache") if Path("/content").exists() else PROJECT_ROOT / "report_colab" / "train_cache"
    data_source_root = drive_root
    staged_count = 0
    if not args.no_cache:
        data_source_root, staged_count = stage_drive_data_to_local(drive_root, target_dates, local_cache_dir)

    checkpoint_dir = drive_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    do_train_model1 = not args.model2_only
    do_train_model2 = not args.model1_only

    model1_res = None
    model2_res = None

    # 2. Train Model 1 (General Vision & Signal Detector)
    if do_train_model1:
        print("\n" + "=" * 80)
        print("[Step 1/2] Training Model 1: General Vision ViT Backbone...")
        print("=" * 80)
        try:
            model1_res = run_threads_continual_training(
                drive_root=str(data_source_root),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                output_checkpoint_dir=checkpoint_dir
            )
            print(f"🎯 Model 1 Training Completed: Acc={model1_res.get('test_acc')}%, AUC={model1_res.get('auc')}")
        except Exception as e1:
            print(f"❌ Model 1 Training Failed: {e1}")

    # 3. Train Model 2 (Human Biometric Multi-Modal Detector)
    if do_train_model2:
        print("\n" + "=" * 80)
        print("[Step 2/2] Training Model 2: Human Biometric Multi-Modal Detector...")
        print("=" * 80)
        try:
            out_human_dir = PROJECT_ROOT / "human_vit_model"
            model2_res = train_human_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                drive_root=str(data_source_root),
                output_dir=str(out_human_dir)
            )
            print(f"🧬 Model 2 Training Completed: Acc={model2_res.get('accuracy')}%, AUC={model2_res.get('auc')}, EER={model2_res.get('eer')}%")
        except Exception as e2:
            print(f"❌ Model 2 Training Failed: {e2}")

    # 4. FUSE Flush & Safety Release Gate
    try:
        if hasattr(os, "sync"):
            print("\n[FUSE Final Flush] Committing all weights and reports to Google Drive...")
            os.sync()
            time.sleep(2)
            os.sync()
    except Exception:
        pass

    # 5. Cleanup Local Cache
    if not args.no_cache and local_cache_dir.exists():
        try:
            shutil.rmtree(str(local_cache_dir))
            print(f"🧹 [Cleanup] Freed temporary local NVMe cache at: {local_cache_dir}")
        except Exception:
            pass

    print("\n" + "=" * 80)
    print("🏁 [Option B Training Complete] All models successfully fine-tuned!")
    print(f"📁 Checkpoints Location: {checkpoint_dir}")
    if model1_res:
        print(f"   - Model 1 Checkpoint: {checkpoint_dir / 'model_continual_latest.pt'} (Acc: {model1_res.get('test_acc')}%)")
    if model2_res:
        print(f"   - Model 2 Checkpoint: {checkpoint_dir / 'weights_human.pt'} (Acc: {model2_res.get('accuracy')}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
