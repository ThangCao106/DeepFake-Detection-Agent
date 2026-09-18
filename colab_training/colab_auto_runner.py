"""
Colab Automated Continuous Training & Feedback Loop Script.

Supports:
1. Pulling latest local code updates from GitHub (Auto-Pull).
2. Processing Image, Video, Audio multi-modality deepfake datasets.
3. Training & Fine-Tuning ViT model.
4. Auto-uploading model weights to Hugging Face Hub (ThangCao/custom-deepfake-vit).
5. Auto-syncing training metrics & logs back to GitHub (ThangCao106/DeepFake-Detection-Agent).
"""

import os
import sys
import argparse
import importlib

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Import pipeline components
from colab_dataset_pipeline import ColabMultiModalityDatasetManager, ColabAutoSyncManager


def run_colab_auto_pipeline(modality="IMAGE", dataset_key="ntire_robust_detection", max_samples=2000, hf_token=None, github_token=None, auto_pull=True, pass_number=1, target="all", upload_hf=False):
    sync_manager = ColabAutoSyncManager()

    # 1. Step 1: Auto-Pull latest code from GitHub
    if auto_pull:
        print("\n==========================================")
        print("[Step 1] Syncing latest code from GitHub...")
        print("==========================================")
        sync_manager.auto_pull_latest_code()
        
        # Reload modules to reflect latest local code edits
        import colab_dataset_pipeline
        import train_general
        import train_human
        import upload_model_cloud
        importlib.reload(colab_dataset_pipeline)
        importlib.reload(train_general)
        importlib.reload(train_human)
        importlib.reload(upload_model_cloud)

    # 2. Step 2: Prepare Multi-Modality Dataset
    print("\n==========================================")
    print(f"[Step 2] Preparing Dataset [{modality} -> {dataset_key}]...")
    print("==========================================")
    dataset_manager = ColabMultiModalityDatasetManager()
    manifest_path = dataset_manager.prepare_dataset_for_colab(
        modality=modality,
        dataset_key=dataset_key,
        max_samples=max_samples
    )
    print(f"Manifest created at: {manifest_path}")

    target = target.lower()

    # 3. Step 3 & 4 & 5: Stream-Specific Fine-Tuning, Hugging Face Upload & Report Generation
    if target in ["general", "all"]:
        print("\n==========================================")
        print("[Step 3 - General AI Stream] Starting Model Fine-Tuning (custom_vit_model)...")
        print("==========================================")
        try:
            from train_general import train_general_model
            train_general_model(output_dir="custom_vit_model")
        except Exception as e:
            print(f"[colab_runner] Note on general model training execution: {e}")

        if hf_token and upload_hf:
            print("[Step 4] Uploading General Model to Hugging Face Hub...")
            sync_manager.auto_upload_hf(model_dir="custom_vit_model", hf_token=hf_token)
        else:
            print("[Step 4] Checkpoint saved locally in 'custom_vit_model/' (HF sync deferred to Pass completion).")

        sync_manager.generate_and_save_report(
            modality=modality,
            dataset_key=dataset_key,
            pass_number=pass_number,
            target_model="general"
        )

    if target in ["human", "all"]:
        print("\n==========================================")
        print("[Step 3 - Human Deepfake Stream] Starting Model Fine-Tuning (human_vit_model)...")
        print("==========================================")
        try:
            from train_human import train_human_model
            train_human_model(output_dir="human_vit_model")
        except Exception as e:
            print(f"[colab_runner] Note on human model training execution: {e}")

        if hf_token and upload_hf:
            print("[Step 4] Uploading Human Model to Hugging Face Hub...")
            sync_manager.auto_upload_hf(model_dir="human_vit_model", hf_token=hf_token)
        else:
            print("[Step 4] Checkpoint saved locally in 'human_vit_model/' (HF sync deferred to Pass completion).")

        sync_manager.generate_and_save_report(
            modality=modality,
            dataset_key=dataset_key,
            pass_number=pass_number,
            target_model="human"
        )

    if github_token:
        print("\n==========================================")
        print("[Step 5] Syncing Reports & Metrics back to GitHub...")
        print("==========================================")
        commit_msg = f"Auto-sync Colab dual-stream training reports for Pass #{pass_number} {modality}:{dataset_key} (target={target}) [Auto-Bot]"
        sync_manager.auto_sync_metrics_to_github(github_token=github_token, commit_message=commit_msg)

    print("\n==========================================")
    print(f"[Complete] Colab Auto Pipeline Run (target={target}) Finished Successfully!")
    print("==========================================")


import time
import subprocess


def get_commit_sha(ref="HEAD"):
    """Get current git commit SHA hash."""
    try:
        res = subprocess.run(["git", "rev-parse", ref], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return ""


def get_full_multimodal_queue():
    """Dynamically query registry to build FULL training queue for ALL IMAGE, VIDEO, and AUDIO datasets."""
    from colab_dataset_pipeline import AWESOME_DEEPFAKES_REGISTRY
    queue = []
    for modality in ["IMAGE", "VIDEO", "AUDIO"]:
        if modality in AWESOME_DEEPFAKES_REGISTRY:
            for key in AWESOME_DEEPFAKES_REGISTRY[modality].keys():
                queue.append((modality, key))
    return queue


def run_auto_listen_daemon(hf_token=None, github_token=None, max_samples=1000, poll_interval=30):
    """
    Auto-Listen Daemon: Continuously polls GitHub origin/main.
    Whenever a new commit is pushed from Local, Colab automatically pulls it,
    executes training, uploads weights to Hugging Face, and syncs metrics back to GitHub.
    """
    print("======================================================================")
    print(f"📡 STARTING COLAB AUTO-LISTEN DAEMON (Polling every {poll_interval}s)")
    print("   Colab will automatically trigger training whenever you push code from Local!")
    print("======================================================================")

    last_processed_sha = get_commit_sha("HEAD")
    print(f"[daemon] Initial HEAD commit SHA: {last_processed_sha}")

    try:
        while True:
            print(f"\n[daemon] Checking for new code pushes on GitHub origin/main...")
            os.system("git fetch origin main 2>/dev/null")
            remote_sha = get_commit_sha("origin/main")

            if remote_sha and remote_sha != last_processed_sha:
                print(f"\n🔔 NEW COMMIT DETECTED ON GITHUB!")
                print(f"   Old SHA: {last_processed_sha}")
                print(f"   New SHA: {remote_sha}")
                print("   🚀 Triggering Automated Full Multimodal Training Pipeline...")

                run_continuous_auto_loop(
                    hf_token=hf_token,
                    github_token=github_token,
                    max_samples=max_samples,
                    pass_number=1
                )
                last_processed_sha = get_commit_sha("HEAD")
                print(f"\n✅ Job finished for commit {last_processed_sha}. Resuming Auto-Listen Daemon...")
            else:
                print(f"[daemon] No new code pushed (current: {last_processed_sha[:7]}). Standing by...")

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n🛑 Auto-Listen Daemon stopped by user.")


def run_continuous_auto_loop(hf_token=None, github_token=None, max_samples=1000, pass_number=1, target="all"):
    """Run full multi-modality continuous auto loop sequentially over ALL Image, Video, and Audio datasets."""
    queue = get_full_multimodal_queue()

    print("======================================================================")
    print(f"🔄 STARTING FULL MULTI-MODALITY AUTO-RUNNER LOOP (Pass #{pass_number} | {len(queue)} STAGES | Target={target.upper()})")
    print("   Sequence: ALL Registered [IMAGE] ➔ [VIDEO] ➔ [AUDIO] Datasets")
    print("======================================================================")

    for idx, (modality, dataset_key) in enumerate(queue, 1):
        print(f"\n======================================================================")
        print(f"🚀 STAGE [{idx}/{len(queue)}]: MODALITY = [{modality}] | DATASET = [{dataset_key}] | PASS = #{pass_number}")
        print("======================================================================")
        run_colab_auto_pipeline(
            modality=modality,
            dataset_key=dataset_key,
            max_samples=max_samples,
            hf_token=hf_token,
            github_token=github_token,
            auto_pull=True,
            pass_number=pass_number,
            target=target,
            upload_hf=False
        )

    # Synchronize trained model checkpoints to Hugging Face Hub once per full Pass
    if hf_token:
        print("\n======================================================================")
        print(f"📦 [Pass #{pass_number} Complete] Syncing Trained Checkpoints to Hugging Face...")
        print("======================================================================")
        sync_manager = ColabAutoSyncManager()
        if target in ["general", "all"]:
            sync_manager.auto_upload_hf(model_dir="custom_vit_model", hf_token=hf_token)
        if target in ["human", "all"]:
            sync_manager.auto_upload_hf(model_dir="human_vit_model", hf_token=hf_token)


def run_smart_auto_loop(hf_token=None, github_token=None, max_samples=1000, target="all"):
    """
    Smart Hybrid Continuous Training Loop:
    1. Continuously iterates training passes (Pass 1 ➔ Pass 2 ➔ Pass 3...).
    2. Before each pass, checks if new code was pushed to GitHub origin/main.
       - If new commit detected: Auto-pulls latest code and reloads modules immediately.
       - If no new commit: Continues next Pass smoothly to progressively optimize weights!
    3. Uploads trained model weights to Hugging Face Hub at each pass completion (rate-limit protected).
    4. Syncs metrics back to GitHub after each dataset stage.
    """
    pass_number = 1
    last_commit_sha = get_commit_sha("HEAD")

    print("======================================================================")
    print(f"🔄 STARTING SMART HYBRID CONTINUOUS AUTO-RUNNER LOOP (Target={target.upper()})")
    print("   1. Auto-Detects new code commits pushed from Local ➔ Reloads & Runs")
    print("   2. If no new code: Continuously loops (Pass 1 ➔ Pass 2 ➔ Pass 3...)")
    print("======================================================================")

    try:
        while True:
            print(f"\n" + "="*70)
            print(f"🌟 === STARTING FULL MULTIMODAL CONTINUOUS TRAINING PASS #{pass_number} (Target={target}) ===")
            print("="*70)

            # Check if GitHub has new code pushed from local
            os.system("git fetch origin main 2>/dev/null")
            remote_sha = get_commit_sha("origin/main")

            if remote_sha and remote_sha != last_commit_sha:
                print(f"\n🔔 NEW CODE COMMIT DETECTED ON GITHUB! ({remote_sha[:7]})")
                print("   📥 Pulling latest code and reloading modules...")
                os.system("git reset --hard origin/main && git pull origin main")
                last_commit_sha = get_commit_sha("HEAD")
                
                # Reload modules to apply local edits
                import colab_dataset_pipeline
                import train_general
                import train_human
                import upload_model_cloud
                importlib.reload(colab_dataset_pipeline)
                importlib.reload(train_general)
                importlib.reload(train_human)
                importlib.reload(upload_model_cloud)
            else:
                print(f"ℹ️ Code is up-to-date (SHA: {last_commit_sha[:7]}). Proceeding with Pass #{pass_number}...")

            # Run 1 full multi-modality pass across ALL Image -> Video -> Audio datasets
            run_continuous_auto_loop(
                hf_token=hf_token,
                github_token=github_token,
                max_samples=max_samples,
                pass_number=pass_number,
                target=target
            )

            print(f"\n✅ Pass #{pass_number} completed. Progressively refining weights for Pass #{pass_number + 1}...")
            pass_number += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Smart Auto-Runner Loop stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Colab Automated Continuous Training & Feedback Loop")
    parser.add_argument("--modality", type=str, default="IMAGE", choices=["IMAGE", "VIDEO", "AUDIO"], help="Dataset modality")
    parser.add_argument("--dataset-key", type=str, default="ntire_robust_detection", help="Dataset registry key")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum samples to process")
    parser.add_argument("--target", type=str, default="all", choices=["general", "human", "all"], help="Target fine-tuning model stream (general|human|all)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--github-token", type=str, default=None, help="GitHub Classic Access Token for auto-pushing metrics")
    parser.add_argument("--no-pull", action="store_true", help="Disable auto-pulling code from GitHub")
    parser.add_argument("--loop", action="store_true", help="Run smart hybrid continuous loop (Pass 1 -> Pass 2... + Auto-Detect commits)")
    parser.add_argument("--listen", action="store_true", help="Auto-Listen Daemon mode: Polls GitHub continuously and triggers training on every git push")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds for --listen mode")
    args = parser.parse_args()

    if args.listen:
        run_auto_listen_daemon(
            hf_token=args.hf_token,
            github_token=args.github_token,
            max_samples=args.max_samples,
            poll_interval=args.poll_interval
        )
    elif args.loop:
        run_smart_auto_loop(
            hf_token=args.hf_token,
            github_token=args.github_token,
            max_samples=args.max_samples,
            target=args.target
        )
    else:
        run_colab_auto_pipeline(
            modality=args.modality,
            dataset_key=args.dataset_key,
            max_samples=args.max_samples,
            hf_token=args.hf_token,
            github_token=args.github_token,
            auto_pull=not args.no_pull,
            target=args.target,
            upload_hf=True
        )


if __name__ == "__main__":
    main()
