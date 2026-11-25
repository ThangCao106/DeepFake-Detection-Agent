"""
Model Upload & Cloud Synchronization Module.
Uploads the local fine-tuned ViT model (model.safetensors, audio_fusionnet.pt, calibration JSONs)
to Hugging Face Hub with automatic token-owner namespace resolution and rate-limit protection.
"""
import os
import sys
import time
import json
import argparse
from huggingface_hub import HfApi, create_repo

DEFAULT_MIN_INTERVAL_SECONDS = 900  # 15 minutes cooldown between continuous loop uploads
DEFAULT_HF_REPO_ID = "ThangCao106/custom-deepfake-vit"


def _get_cache_path():
    return os.path.join(os.path.dirname(__file__), ".last_hf_upload.json")


def _get_last_upload_time(model_dir):
    cache_file = _get_cache_path()
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get(model_dir, 0.0))
    except Exception:
        pass
    return 0.0


def _record_upload_time(model_dir):
    cache_file = _get_cache_path()
    try:
        data = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[model_dir] = time.time()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def resolve_hf_repo_id(hf_token=None, default_repo=DEFAULT_HF_REPO_ID) -> str:
    """Resolve target Hugging Face model repo ID dynamically from env or token owner."""
    explicit = (os.environ.get("CUSTOM_VIT_HF_REPO") or "").strip()
    if explicit:
        return explicit

    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    repo_name = default_repo.split("/")[-1] if "/" in default_repo else "custom-deepfake-vit"
    if token:
        try:
            api = HfApi(token=token)
            info = api.whoami(token=token)
            username = (info.get("name") or "").strip()
            if username:
                return f"{username}/{repo_name}"
        except Exception:
            pass
    return default_repo


def upload_model_to_hf(
    model_dir="custom_vit_model",
    repo_id=None,
    hf_token=None,
    private=False,
    force=False,
    min_interval=DEFAULT_MIN_INTERVAL_SECONDS,
):
    token = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    if not token:
        print("[upload_cloud] Note: HF_TOKEN not set in environment or argument. Skipping upload.")
        return False

    repo_id = repo_id or resolve_hf_repo_id(hf_token=token)
    if not os.path.exists(model_dir):
        print(f"[upload_cloud] Error: Model directory '{model_dir}' does not exist.")
        return False

    now = time.time()
    last_time = _get_last_upload_time(model_dir)
    elapsed = now - last_time
    if not force and elapsed < min_interval:
        rem_min = (min_interval - elapsed) / 60
        print(f"[upload_cloud] ⏳ Throttling active: '{model_dir}' was uploaded {elapsed/60:.1f}m ago ({rem_min:.1f}m remaining).")
        return True

    api = HfApi(token=token)

    print(f"[upload_cloud] Checking files in {model_dir}...")
    files_to_upload = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    for f in files_to_upload:
        f_size_mb = os.path.getsize(os.path.join(model_dir, f)) / (1024 * 1024)
        print(f"  - {f} ({f_size_mb:.2f} MB)")

    hf_username = None
    try:
        user_info = api.whoami(token=token)
        hf_username = (user_info.get("name") or "").strip()
        print(f"[upload_cloud] Authenticated with Hugging Face as user: '{hf_username}'")

        namespace = repo_id.split("/")[0] if "/" in repo_id else repo_id
        explicit_env = (os.environ.get("CUSTOM_VIT_HF_REPO") or "").strip()
        if hf_username and namespace != hf_username and not explicit_env:
            adjusted_repo = f"{hf_username}/{repo_id.split('/')[-1]}"
            print(f"[upload_cloud] Auto-aligned repo namespace '{repo_id}' -> '{adjusted_repo}'")
            repo_id = adjusted_repo
    except Exception as e_whoami:
        print(f"[upload_cloud] Note: Could not verify token user identity: {e_whoami}")

    try:
        print(f"[upload_cloud] Ensuring repository '{repo_id}' exists on Hugging Face Hub...")
        try:
            create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="model")
        except Exception as e_create:
            if "429" in str(e_create):
                print(f"[upload_cloud] ⚠️ HF Rate limit reached during repo check: {e_create}.")
                _record_upload_time(model_dir)
                return False
            print(f"[upload_cloud] Note on repo creation: {e_create}. Proceeding to upload...")

        print(f"[upload_cloud] Uploading folder '{model_dir}' to '{repo_id}'...")
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"[upload_cloud] Upload Successful! Model is live at: https://huggingface.co/{repo_id}")
        _record_upload_time(model_dir)
        return True
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "rate limit" in err_str.lower():
            print("[upload_cloud] ⚠️ HUGGING FACE COMMIT RATE LIMIT REACHED (429). Local weights are preserved.")
            _record_upload_time(model_dir)
            return False

        print(f"\n[upload_cloud] Upload failed: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned ViT model to Hugging Face Hub")
    parser.add_argument("--model-dir", type=str, default="custom_vit_model", help="Path to local model directory")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_HF_REPO_ID, help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--force", action="store_true", help="Bypass cooldown timer and force upload")
    args = parser.parse_args()

    upload_model_to_hf(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        hf_token=args.token,
        private=args.private,
        force=args.force,
    )
