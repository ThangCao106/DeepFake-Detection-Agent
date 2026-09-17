"""
Option A Google Colab Direct Mount & GDrive Synchronizer
Synchronizes sample media & pentest data directly to Google Drive WITHOUT requiring GCP Service Account or API keys.

Usage on Google Colab:
  1. Mount Drive:
     from google.colab import drive
     drive.mount('/content/drive')
  2. Run Sync:
     !python colab_sync_to_gdrive.py --samples 20
"""

import os
import sys
import shutil
import time
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


from crawler.media_generator import generate_and_upload_pentest_sample_images
from crawler.threads_crawler import crawl_threads_media

DEFAULT_LOCAL_STORAGE = PROJECT_ROOT / "report_colab" / "gdrive_storage"

def find_or_create_colab_drive_path(folder_id="1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet", base_folder_name="Thread_Data"):
    """
    Locates mounted Google Drive path on Colab.
    Checks for user's exact shared folder shortcut Thread_Data or default MyDrive directory.
    """
    colab_mydrive = Path("/content/drive/MyDrive")
    if not colab_mydrive.exists():
        alt_mount = Path("/content/drive/Shareddrives")
        if alt_mount.exists():
            colab_mydrive = alt_mount
        else:
            print("[Colab GDrive Sync] Notice: /content/drive/MyDrive not detected.")
            print("  If on Google Colab, please run first:")
            print("    from google.colab import drive")
            print("    drive.mount('/content/drive')")
            return None

    # Search for known shortcut names, prioritizing user's exact folder Thread_Data
    candidate_names = [
        "Thread_Data",
        "Deepfake Detection",
        "DeepFake_Pentest_Data",
        "Deepfake_Pentest_Data",
        folder_id
    ]

    target_path = None
    for name in candidate_names:
        cand = colab_mydrive / name
        if cand.exists() and cand.is_dir():
            target_path = cand
            print(f"[Colab GDrive Sync] [MATCH] Found existing Google Drive folder: {target_path}")
            break

    if not target_path:
        # Create Thread_Data directly in MyDrive
        target_path = colab_mydrive / base_folder_name
        target_path.mkdir(parents=True, exist_ok=True)
        print(f"[Colab GDrive Sync] [CREATED] Initialized Google Drive folder: {target_path}")

    return target_path


def sync_local_samples_to_drive(dest_drive_root, date_str=None):
    """Copies all local generated media from report_colab/gdrive_storage to mounted Google Drive."""
    if not date_str:
        date_str = time.strftime("%Y-%m-%d")
        
    local_daily_dir = DEFAULT_LOCAL_STORAGE / date_str
    drive_daily_dir = dest_drive_root / date_str
    
    subfolders = ["01_raw_originals", "02_manipulated_faceswap", "03_manipulated_openweight"]
    copied_count = 0
    
    print(f"\n[Colab GDrive Sync] Synchronizing media from {local_daily_dir} -> {drive_daily_dir}...")
    
    for sub in subfolders:
        local_sub = local_daily_dir / sub
        drive_sub = drive_daily_dir / sub
        drive_sub.mkdir(parents=True, exist_ok=True)
        
        if local_sub.exists():
            for f in local_sub.iterdir():
                if f.is_file():
                    dest_file = drive_sub / f.name
                    if not dest_file.exists() or dest_file.stat().st_size != f.stat().st_size:
                        shutil.copy2(str(f), str(dest_file))
                        copied_count += 1
                        
    # Also sync benchmark manifest and summary files to date root
    extra_sources = [local_daily_dir, Path("report_colab") / "temp_media_samples"]
    for src_dir in extra_sources:
        if src_dir.exists():
            for meta_name in ["dataset_manifest.jsonl", "dataset_summary.md"]:
                meta_src = src_dir / meta_name
                if meta_src.exists():
                    dest_meta = drive_daily_dir / meta_name
                    shutil.copy2(str(meta_src), str(dest_meta))
                    copied_count += 1
                    print(f"  📄 Synced benchmark metadata: {meta_name} -> {drive_daily_dir}")

    print(f"[Colab GDrive Sync] Completed! Successfully synced {copied_count} files to Google Drive:")
    print(f"  📂 Destination: {drive_daily_dir}")
    return copied_count

def run_option_a_pipeline(
    num_images=1000, 
    num_videos=1000, 
    num_audios=1000, 
    num_samples=20,
    do_train=False,
    epochs=3,
    batch_size=16,
    lr=1e-5
):
    """
    Executes end-to-end Option A & Option C pipeline:
      1. Generates inspection images and crawls media
      2. Direct-syncs to mounted Google Drive
      3. (Optional) Runs Active Continual Training on GPU & saves checkpoints to Google Drive
    """
    date_str = time.strftime("%Y-%m-%d")
    timestamp_tag = time.strftime("%Y-%m-%d_%Hh")
    
    print("=" * 75)
    print(f"🚀 Option A + Option C: Google Colab GDrive Direct Sync & Continual Training ({timestamp_tag})")
    print(f"⚙️ Config: Images={num_images}, Videos={num_videos}, Audios={num_audios} | Visual Samples={num_samples}")
    if do_train:
        print(f"🧠 Active Continual Training: ENABLED (Epochs={epochs}, BatchSize={batch_size}, LR={lr})")
    print("=" * 75)

    # 1. Check mounted Google Drive
    drive_root = find_or_create_colab_drive_path()
    
    # 2. Generate high-quality visual inspection sample images (.jpg)
    print(f"\n[Step 1] Generating {num_samples} visual inspection samples per bucket...")
    generate_and_upload_pentest_sample_images(num_samples=num_samples, date_str=date_str)
    
    # 3. Synchronize to mounted Google Drive (if mounted)
    if drive_root:
        print(f"\n[Step 2] Transferring media directly to Google Drive...")
        total_copied = sync_local_samples_to_drive(drive_root, date_str=date_str)
        print(f"✅ Success: {total_copied} media assets now live on your Google Drive workspace!")
    else:
        print(f"\n[Step 2 Notice] Running in standalone/local mode. Files stored at:")
        print(f"   {DEFAULT_LOCAL_STORAGE / date_str}")
        print("   To push to Google Drive, run this script inside Google Colab after drive.mount('/content/drive').")

    # 4. Optional Active Continual Training (Option C)
    if do_train:
        print(f"\n[Step 3] Initiating Active Continual Fine-Tuning on GPU (Epochs={epochs}, LR={lr})...")
        from colab_training.train_threads_continual import run_threads_continual_training
        ckpt_dir = (drive_root / "checkpoints") if drive_root else None
        res = run_threads_continual_training(
            drive_root=str(drive_root) if drive_root else None,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_checkpoint_dir=ckpt_dir
        )
        print(f"\n🎯 Continual Training Complete: OOD Acc={res.get('test_acc')}%, AUC={res.get('auc')}")
        if drive_root:
            print(f"   📁 Checkpoint is safely stored in Google Drive: {drive_root / 'checkpoints'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Option A + C Google Colab GDrive Sync & Continual Trainer")
    parser.add_argument("--images", type=int, default=1000, help="Number of human images to crawl")
    parser.add_argument("--videos", type=int, default=1000, help="Number of human videos to crawl")
    parser.add_argument("--audios", type=int, default=1000, help="Number of human audio clips to crawl")
    parser.add_argument("--samples", type=int, default=20, help="Number of sample inspection images per bucket")
    parser.add_argument("--train", action="store_true", help="Trigger GPU active continual fine-tuning after sync")
    parser.add_argument("--epochs", type=int, default=3, help="Number of continual fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    args = parser.parse_args()
    
    run_option_a_pipeline(
        num_images=args.images,
        num_videos=args.videos,
        num_audios=args.audios,
        num_samples=args.samples,
        do_train=args.train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
