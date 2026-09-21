"""
Hugging Face Model Uploader for custom-deepfake-vit.
Uploads model weights, ONNX export, configuration files, benchmark results,
and Model Card README to Hugging Face Hub (ThangCao/custom-deepfake-vit).
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "custom_vit_model"
REPO_ID = "ThangCao/custom-deepfake-vit"

def upload_to_huggingface():
    print("=" * 70)
    print(f"🚀 Starting Hugging Face Model Uploader for {REPO_ID}...")
    print("=" * 70)

    if not MODEL_DIR.exists():
        print(f"❌ Error: Model directory does not exist at {MODEL_DIR}")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ Error: `huggingface_hub` package not found. Please install via: pip install huggingface_hub")
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # 1. Ensure repository exists on Hugging Face Hub
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, token=token)
        print(f"  ✅ Repository verified/created: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"  ⚠️ Repository check/create warning: {e}")

    # 2. Upload model directory contents
    print(f"  📦 Uploading model files from: {MODEL_DIR}")
    for file_path in MODEL_DIR.glob("*"):
        if file_path.is_file():
            print(f"    - Uploading {file_path.name} ({file_path.stat().st_size / (1024*1024):.2f} MB)...")

    try:
        future = api.upload_folder(
            folder_path=str(MODEL_DIR),
            repo_id=REPO_ID,
            repo_type="model",
            token=token
        )
        print("=" * 70)
        print(f"🎉 [SUCCESS] Model files successfully uploaded to Hugging Face Hub!")
        print(f"🔗 Model URL: https://huggingface.co/{REPO_ID}")
        print("=" * 70)
    except Exception as e:
        print(f"❌ Error during folder upload: {e}")
        print("💡 Hint: Ensure you are logged in via `huggingface-cli login` or set HF_TOKEN environment variable.")

if __name__ == "__main__":
    upload_to_huggingface()
