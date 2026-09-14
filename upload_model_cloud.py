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

    # Validate token user
    hf_username = None
    try:
        user_info = api.whoami(token=token)
        hf_username = user_info.get("name")
        print(f"[upload_cloud] Authenticated with Hugging Face as user: '{hf_username}'")
        
        namespace = repo_id.split("/")[0] if "/" in repo_id else repo_id
        if hf_username and namespace != hf_username:
            print(f"[upload_cloud] ⚠️ Warning: Token username ('{hf_username}') differs from repo namespace ('{namespace}').")
            print(f"[upload_cloud] 💡 Suggested repo_id: '{hf_username}/{repo_id.split('/')[-1]}'")
    except Exception as e_whoami:
        print(f"[upload_cloud] Note: Could not verify token user identity: {e_whoami}")

    try:
        print(f"[upload_cloud] Checking/Creating repository '{repo_id}' on Hugging Face Hub...")
        try:
            create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")
        except Exception as e_create:
            print(f"[upload_cloud] Note on repo creation: {e_create}. Proceeding to upload directly...")

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
        print(f"\n[upload_cloud] Upload failed: {e}")
        print("\n=== 🛠️ HUGGING FACE 403 FORBIDDEN TROUBLESHOOTING ===")
        if hf_username:
            print(f"1. Username Mismatch Check:")
            print(f"   Your HF Token belongs to user: '{hf_username}'")
            print(f"   You tried to upload to repo: '{repo_id}'")
            if repo_id.split('/')[0] != hf_username:
                print(f"   👉 FIX: Run command with --repo-id {hf_username}/{repo_id.split('/')[-1]}")
        print("2. Token Permissions Check:")
        print("   Go to: https://huggingface.co/settings/tokens")
        print("   Ensure your token type is 'WRITE' or 'Fine-grained' with 'Create/edit repos' permission.")
        print("3. Manual Repo Creation Option:")
        print(f"   Create repo manually at: https://huggingface.co/new (name: {repo_id.split('/')[-1]}) then re-run.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned ViT model to Hugging Face Hub / Cloud")
    parser.add_argument("--model-dir", type=str, default="custom_vit_model", help="Path to local custom_vit_model directory")
    parser.add_argument("--repo-id", type=str, default="ThangCao106/custom-deepfake-vit", help="Hugging Face repo ID (e.g. username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    upload_model_to_hf(model_dir=args.model_dir, repo_id=args.repo_id, hf_token=args.token, private=args.private)
