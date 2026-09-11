"""
Model Upload & Cloud Synchronization Module.
Uploads the local fine-tuned ViT model (weights, configs, calibration parameters, ONNX)
to Hugging Face Hub or packages it for Google Drive / Cloud Storage.
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo


def upload_model_to_hf(model_dir="custom_vit_model", repo_id=None, hf_token=None, private=False):
    if not repo_id:
        print("[upload_cloud] Error: --repo-id is required (e.g., 'username/custom-deepfake-vit')")
        return False

    if not os.path.exists(model_dir):
        print(f"[upload_cloud] Error: Model directory '{model_dir}' does not exist.")
        return False

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[upload_cloud] Note: HF_TOKEN not set in environment or argument.")
        print("[upload_cloud] Dry run mode / checking model files integrity...")

    api = HfApi(token=token)

    print(f"[upload_cloud] Checking files in {model_dir}...")
    files_to_upload = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    for f in files_to_upload:
        f_size_mb = os.path.getsize(os.path.join(model_dir, f)) / (1024 * 1024)
        print(f"  - {f} ({f_size_mb:.2f} MB)")

    if not token:
        print("[upload_cloud] To upload to Hugging Face Hub, set your HF_TOKEN:")
        print(f"  $env:HF_TOKEN=\"your_huggingface_write_token\"")
        print(f"  py -3 upload_model_cloud.py --repo-id {repo_id}")
        return True

    try:
        print(f"[upload_cloud] Creating/Verifying repository '{repo_id}' on Hugging Face Hub...")
        create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")

        print(f"[upload_cloud] Uploading folder '{model_dir}' to '{repo_id}'...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"[upload_cloud] Upload Successful! Model is now live at: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"[upload_cloud] Upload failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned ViT model to Hugging Face Hub / Cloud")
    parser.add_argument("--model-dir", type=str, default="custom_vit_model", help="Path to local custom_vit_model directory")
    parser.add_argument("--repo-id", type=str, default="ThangCao106/custom-deepfake-vit", help="Hugging Face repo ID (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    upload_model_to_hf(model_dir=args.model_dir, repo_id=args.repo_id, hf_token=args.token, private=args.private)
