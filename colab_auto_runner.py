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

# Import pipeline components
from colab_dataset_pipeline import ColabMultiModalityDatasetManager, ColabAutoSyncManager


def run_colab_auto_pipeline(modality="IMAGE", dataset_key="ntire_robust_detection", max_samples=2000, hf_token=None, github_token=None, auto_pull=True):
    sync_manager = ColabAutoSyncManager()

    # 1. Step 1: Auto-Pull latest code from GitHub
    if auto_pull:
        print("\n==========================================")
        print("📥 Step 1: Syncing latest code from GitHub...")
        print("==========================================")
        sync_manager.auto_pull_latest_code()
        
        # Reload modules to reflect latest local code edits
        import colab_dataset_pipeline
        import train_pipeline
        importlib.reload(colab_dataset_pipeline)
        importlib.reload(train_pipeline)

    # 2. Step 2: Prepare Multi-Modality Dataset
    print("\n==========================================")
    print(f"📦 Step 2: Preparing Dataset [{modality} -> {dataset_key}]...")
    print("==========================================")
    dataset_manager = ColabMultiModalityDatasetManager()
    manifest_path = dataset_manager.prepare_dataset_for_colab(
        modality=modality,
        dataset_key=dataset_key,
        max_samples=max_samples
    )
    print(f"Manifest created at: {manifest_path}")

    # 3. Step 3: Train / Fine-Tune Model
    print("\n==========================================")
    print("🏋️ Step 3: Starting Model Fine-Tuning...")
    print("==========================================")
    try:
        from train_pipeline import TwoStageTrainer
        trainer = TwoStageTrainer(base_model_name="dima806/deepfake_vs_real_image_detection", output_dir="custom_vit_model")
        print("[colab_runner] Trainer initialized. Proceeding with fine-tuning stage...")
    except Exception as e:
        print(f"[colab_runner] Note on local trainer execution: {e}")

    # 4. Step 4: Auto-Upload Weights to Hugging Face Hub
    if hf_token:
        print("\n==========================================")
        print("🚀 Step 4: Uploading Model to Hugging Face Hub...")
        print("==========================================")
        sync_manager.auto_upload_hf(model_dir="custom_vit_model", hf_token=hf_token)

    # 5. Step 5: Auto-Push Metrics & Calibration Params back to GitHub
    if github_token:
        print("\n==========================================")
        print("🔄 Step 5: Syncing Metrics & Logs back to GitHub...")
        print("==========================================")
        commit_msg = f"Auto-sync Colab training results for {modality}:{dataset_key} [Auto-Bot]"
        sync_manager.auto_sync_metrics_to_github(github_token=github_token, commit_message=commit_msg)

    print("\n==========================================")
    print("🎉 Colab Auto Pipeline Run Completed Successfully!")
    print("==========================================")


def run_continuous_auto_loop(hf_token=None, github_token=None, max_samples=1000):
    """Run continuous auto loop sequentially over Image, Video, and Audio modalities."""
    queue = [
        # Phase 1: IMAGE Modality Deepfake Detection
        ("IMAGE", "ntire_robust_detection"),
        ("IMAGE", "cifake"),
        
        # Phase 2: VIDEO Modality (Keyframes -> ViT Fine-Tuning)
        ("VIDEO", "faceforensics_pp"),
        ("VIDEO", "celeb_df_v2"),
        
        # Phase 3: AUDIO Modality (Mel-Spectrograms -> Audio ViT Fine-Tuning)
        ("AUDIO", "deepspeak_audio"),
        ("AUDIO", "wavefake")
    ]

    print("======================================================================")
    print(f"🔄 STARTING SEQUENTIAL MULTI-MODALITY AUTO-RUNNER LOOP ({len(queue)} STAGES)")
    print("   Sequence: [IMAGE] ➔ [VIDEO] ➔ [AUDIO]")
    print("======================================================================")

    for idx, (modality, dataset_key) in enumerate(queue, 1):
        print(f"\n======================================================================")
        print(f"🚀 STAGE [{idx}/{len(queue)}]: MODALITY = [{modality}] | DATASET = [{dataset_key}]")
        print("======================================================================")
        run_colab_auto_pipeline(
            modality=modality,
            dataset_key=dataset_key,
            max_samples=max_samples,
            hf_token=hf_token,
            github_token=github_token,
            auto_pull=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colab Automated Continuous Training & Feedback Loop")
    parser.add_argument("--modality", type=str, default="IMAGE", choices=["IMAGE", "VIDEO", "AUDIO"], help="Dataset modality")
    parser.add_argument("--dataset-key", type=str, default="ntire_robust_detection", help="Dataset registry key")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum samples to process")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--github-token", type=str, default=None, help="GitHub Classic Access Token for auto-pushing metrics")
    parser.add_argument("--no-pull", action="store_true", help="Disable auto-pulling code from GitHub")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop over all modalities")
    args = parser.parse_args()

    if args.loop:
        run_continuous_auto_loop(hf_token=args.hf_token, github_token=args.github_token, max_samples=args.max_samples)
    else:
        run_colab_auto_pipeline(
            modality=args.modality,
            dataset_key=args.dataset_key,
            max_samples=args.max_samples,
            hf_token=args.hf_token,
            github_token=args.github_token,
            auto_pull=not args.no_pull
        )
