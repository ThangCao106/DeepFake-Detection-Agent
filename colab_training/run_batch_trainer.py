"""
Option B Dedicated Batch Trainer for Google Colab (Burst-GPU Closed-Loop Ready).
Decoupled Architecture:
  - Job 1 (Harvester): Crawls Threads & Wikimedia Commons, creates paired counterfactuals,
    and uploads media assets to Google Drive (`Thread_Data/YYYY-MM-DD/batch_*`).
  - Job 2 (This Trainer): Stages Drive batches + approved Human Review Queue disputes to fast local NVMe SSD,
    trains Vision ViT (`model.safetensors`), Human Biometrics (`weights_human.pt`), and Audio (`audio_fusionnet.pt`),
    enforces the OOD Safety Gate, uploads to Hugging Face Hub, and triggers hot-reload on `https://deeptrace.io.vn`.
"""

import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COLAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COLAB_DIR.parent
for _p in (str(COLAB_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from colab_training.colab_auto_runner import (
    _build_callback_payload,
    maybe_auto_disconnect_colab,
    resolve_credentials,
)
from colab_training.colab_dataset_pipeline import ColabAutoSyncManager
from colab_training.train_human import train_human_model
from colab_training.train_threads_continual import run_threads_continual_training

DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def resolve_drive_root(custom_path: str = None) -> Path:
    """Find the active Google Drive Thread_Data directory."""
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([
        Path("/content/drive/MyDrive/Thread_Data"),
        Path("/content/drive/MyDrive/DeepFake-Detection-Benchmark"),
        Path("/content/drive/Shareddrives/Thread_Data"),
        PROJECT_ROOT / "report_colab" / "gdrive_storage",
        PROJECT_ROOT / "report_colab" / "pentest",
    ])
    for p in candidates:
        if p.exists():
            return p
    fallback = PROJECT_ROOT / "report_colab" / "gdrive_storage"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _resolve_target_subdirs(drive_root: Path, target_dates: list[str]) -> list[Path]:
    """Resolve `latest`, `all`, or explicit `YYYY-MM-DD` folders under `drive_root`."""
    all_dirs = [
        c for c in sorted(drive_root.iterdir())
        if c.is_dir() and not c.name.startswith((".", "checkpoints", "features"))
    ] if drive_root.exists() else []
    dated_dirs = [c for c in all_dirs if DATE_DIR_RE.match(c.name)]

    if "all" in target_dates:
        return all_dirs
    if "latest" in target_dates:
        if dated_dirs:
            return [dated_dirs[-1]]
        return all_dirs[-1:] if all_dirs else []

    matched = []
    for d_str in target_dates:
        p = drive_root / d_str
        if p.exists() and p.is_dir():
            matched.append(p)
    if not matched and dated_dirs:
        print(f"ℹ️ Requested date(s) {target_dates} not found on Drive; auto-selecting latest batch '{dated_dirs[-1].name}'.")
        return [dated_dirs[-1]]
    return matched


def stage_drive_data_to_local(
    drive_root: Path,
    target_dates: list[str],
    local_cache_dir: Path,
    max_workers: int = 16,
) -> tuple[Path, int]:
    """Copy images and manifests from Google Drive FUSE into fast local NVMe SSD."""
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    subdirs_to_copy = _resolve_target_subdirs(drive_root, target_dates)

    features_p = drive_root / "features"
    if features_p.exists() and features_p.is_dir() and features_p not in subdirs_to_copy:
        subdirs_to_copy.append(features_p)

    if not subdirs_to_copy:
        print("⚠️ No date subdirectories found on Google Drive; trainer will scan verified roots directly.")
        return drive_root, 0

    files_to_copy = []
    for sdir in subdirs_to_copy:
        for fpath in sdir.glob("**/*.*"):
            if fpath.is_file() and not fpath.name.startswith("."):
                rel = fpath.relative_to(drive_root)
                files_to_copy.append((fpath, local_cache_dir / rel))

    print(f"⚡ [Option B Staging] Pre-caching {len(files_to_copy)} files across {[d.name for d in subdirs_to_copy]} -> {local_cache_dir}...")

    def _copy_worker(pair):
        src, dst = pair
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(str(src), str(dst))

    start_c = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_copy_worker, files_to_copy))

    duration = time.time() - start_c
    print(f"✅ [Staging Complete] Cached {len(files_to_copy)} files in {duration:.1f}s!")
    return local_cache_dir, len(files_to_copy)


def main():
    parser = argparse.ArgumentParser(description="Option B Dedicated Batch Trainer for Threads-DFBench")
    parser.add_argument("--drive-root", "--drive_root", type=str, default=None, help="Path to Google Drive Thread_Data root")
    parser.add_argument("--date", type=str, default="latest", help="Target date ('latest', 'all', or 'YYYY-MM-DD')")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument("--model1-only", action="store_true", help="Train only Model 1 (General Vision ViT)")
    parser.add_argument("--model2-only", action="store_true", help="Train only Model 2 (Human Biometrics)")
    parser.add_argument("--train-audio", action="store_true", default=True, help="Also train Stream 3 Audio FusionNet")
    parser.add_argument("--no-audio", action="store_true", help="Skip Stream 3 Audio FusionNet")
    parser.add_argument("--no-cache", action="store_true", help="Disable local SSD staging cache")
    parser.add_argument("--upload-hf", action="store_true", default=True, help="Upload trained weights to Hugging Face Hub if HF_TOKEN is present")
    parser.add_argument("--no-upload-hf", action="store_true", help="Disable Hugging Face Hub upload")
    parser.add_argument("--callback-url", type=str, default="https://deeptrace.io.vn", help="Production server URL for /api/colab/callback")
    parser.add_argument("--no-notify", action="store_true", help="Disable POST /api/colab/callback notification")
    parser.add_argument("--sync-human-review", action="store_true", default=True, help="Ingest approved Human Review Queue disputes")
    parser.add_argument("--min-auc", type=float, default=0.70, help="OOD Safety Gate minimum ROC-AUC")
    parser.add_argument("--max-ece", type=float, default=0.15, help="OOD Safety Gate maximum ECE")
    parser.add_argument("--auto-disconnect", action="store_true", help="Unassign Colab GPU runtime immediately after completion")
    args = parser.parse_args()

    hf_token, gh_token, cb_secret = resolve_credentials()
    sync_manager = ColabAutoSyncManager()
    drive_root = resolve_drive_root(args.drive_root)
    target_dates = [d.strip() for d in args.date.split(",") if d.strip()]

    print("=" * 80)
    print("🚀 OPTION B: DEDICATED BATCH TRAINER (CLOSED-LOOP BURST GPU)")
    print(f"📂 Google Drive Root: {drive_root} | Target Date: {', '.join(target_dates)}")
    print(f"⚙️ Config: Epochs={args.epochs} | Batch Size={args.batch_size} | LR={args.lr}")
    print("=" * 80)

    extra_manifests = []
    if args.sync_human_review and args.callback_url:
        hr_manifest, _ = sync_manager.pull_human_review_feedback(api_base=args.callback_url)
        if hr_manifest:
            extra_manifests.append(hr_manifest)

    local_cache_dir = Path("/content/train_cache") if Path("/content").exists() else PROJECT_ROOT / "report_colab" / "train_cache"
    data_source_root = drive_root
    if not args.no_cache:
        data_source_root, _ = stage_drive_data_to_local(drive_root, target_dates, local_cache_dir)

    checkpoint_dir = drive_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    do_train_model1 = not args.model2_only
    do_train_model2 = not args.model1_only
    do_train_audio = args.train_audio and not args.no_audio and not args.model1_only and not args.model2_only

    model1_res = None
    model2_res = None
    audio_meta = None

    if do_train_model1:
        print("\n[Step 1/3] Training Model 1: Continual Vision ViT + Spectral FCB...")
        try:
            model1_res = run_threads_continual_training(
                drive_root=str(data_source_root),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                output_checkpoint_dir=checkpoint_dir,
                extra_manifests=extra_manifests,
                min_ood_auc=args.min_auc,
                max_ood_ece=args.max_ece,
            )
            from colab_training.train_general import train_general_model
            train_general_model(epochs=max(3, args.epochs * 2), batch_size=args.batch_size, output_dir=str(PROJECT_ROOT / "custom_vit_model"))
        except Exception as e1:
            print(f"❌ Model 1 Training Failed: {e1}")

    if do_train_model2:
        print("\n[Step 2/3] Training Model 2: Human Biometric Multi-Modal Detector...")
        try:
            model2_res = train_human_model(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=max(args.lr, 1e-4),
                drive_root=str(data_source_root),
                output_dir=str(PROJECT_ROOT / "human_vit_model"),
            )
        except Exception as e2:
            print(f"❌ Model 2 Training Failed: {e2}")

    if do_train_audio:
        print("\n[Step 3/3] Training Model 3: EnhancedAudioFusionNet...")
        try:
            from colab_training.train_audio_fusionnet import train_audio_fusionnet
            audio_meta = train_audio_fusionnet(epochs=35)
        except Exception as e3:
            print(f"⚠️ Audio FusionNet Training Notice: {e3}")

    safety_passed = bool(model1_res.get("safety_passed", True)) if model1_res else True
    hf_uploaded = False
    if safety_passed:
        if hf_token and args.upload_hf and not args.no_upload_hf:
            hf_uploaded = sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "custom_vit_model"), hf_token=hf_token, force=True)
            if model2_res:
                sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "human_vit_model"), hf_token=hf_token, force=True)

        if args.callback_url and not args.no_notify:
            cb_payload = _build_callback_payload(
                vit_res=model1_res,
                audio_meta=audio_meta,
                hf_repo_id=sync_manager.hf_repo if hf_uploaded else None,
                dataset_label=f"gdrive_batch:{','.join(target_dates)}",
            )
            sync_manager.notify_production_callback(api_base=args.callback_url, payload=cb_payload, callback_secret=cb_secret)
    else:
        print("🛑 [OOD Safety Gate HOLD] Validation below threshold; cloud upload & server callback skipped.")

    sync_manager.sync_to_google_drive()
    if gh_token:
        sync_manager.auto_sync_metrics_to_github(
            github_token=gh_token,
            commit_message=f"Auto-sync Option B batch trainer metrics ({','.join(target_dates)}) [Auto-Bot]",
        )

    try:
        if hasattr(os, "sync"):
            os.sync()
            time.sleep(1)
            os.sync()
    except Exception:
        pass

    if not args.no_cache and local_cache_dir.exists() and str(local_cache_dir).startswith("/content/"):
        shutil.rmtree(str(local_cache_dir), ignore_errors=True)

    print("\n" + "=" * 80)
    print(f"🏁 [Option B Complete] Checkpoints saved at: {checkpoint_dir}")
    print("=" * 80 + "\n")
    maybe_auto_disconnect_colab(args.auto_disconnect)


if __name__ == "__main__":
    main()
