# -*- coding: utf-8 -*-
"""
Hugging Face Model & Forensic Artifact Publisher
-------------------------------------------------
Uploads trained local artifacts from `custom_vit_model/` to Hugging Face Hub
(default repository: `ThangCao/custom-deepfake-vit`), including:
  - model.safetensors, config.json, preprocessor_config.json (Visual ViT)
  - audio_fusionnet.pt, audio_training_meta.json (EnhancedAudioFusionNet)
  - ppo_meta_controller.pt (Trainable PPO Meta-Controller Actor-Critic)
  - calibration_params.json, audio_calibration_params.json, video_calibration_params.json,
    vietnamese_demographic_params.json

Exits gracefully with actionable instructions when HF_TOKEN is not configured.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DIR = PROJECT_ROOT / "custom_vit_model"
DEFAULT_REPO_ID = os.environ.get("HF_MODEL_REPO_ID", "ThangCao/custom-deepfake-vit")

DEFAULT_ARTIFACTS: List[str] = [
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "audio_fusionnet.pt",
    "audio_training_meta.json",
    "ppo_meta_controller.pt",
    "calibration_params.json",
    "audio_calibration_params.json",
    "video_calibration_params.json",
    "vietnamese_demographic_params.json",
    "README.md",
]


def publish_custom_models(
    repo_id: str = DEFAULT_REPO_ID,
    dry_run: bool = False,
) -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    existing_files = [CUSTOM_DIR / name for name in DEFAULT_ARTIFACTS if (CUSTOM_DIR / name).exists()]
    # Include any extra *_params.json in custom_vit_model
    seen = {p.resolve() for p in existing_files}
    for param_file in sorted(CUSTOM_DIR.glob("*_params.json")):
        if param_file.resolve() not in seen:
            existing_files.append(param_file)
            seen.add(param_file.resolve())

    if not existing_files:
        print(f"[HF Publisher] No uploadable artifacts found in {CUSTOM_DIR}.")
        return 1

    print(f"[HF Publisher] Target Repo: {repo_id} | Discovered {len(existing_files)} local artifact(s):")
    for fp in existing_files:
        print(f"  - {fp.name} ({fp.stat().st_size / 1024.0:.2f} KB)")

    if dry_run:
        print("[HF Publisher] Dry run complete — no network upload performed.")
        return 0

    if not token:
        print(
            "[HF Publisher] HF_TOKEN / HUGGINGFACE_HUB_TOKEN is not set in the environment.\n"
            "Please export a valid Hugging Face write token before publishing:\n"
            "  PowerShell: $env:HF_TOKEN = 'hf_...'\n"
            "  Bash:       export HF_TOKEN='hf_...'\n"
            f"Then re-run: py -3 scripts/publish_models_to_hf.py --repo-id {repo_id}"
        )
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[HF Publisher] Missing dependency `huggingface_hub`. Install via: pip install huggingface_hub")
        return 1

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    uploaded = 0
    for fp in existing_files:
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=fp.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Sync {fp.name} from DeepFake-Detection-Agent",
        )
        uploaded += 1
        print(f"[HF Publisher] Uploaded {fp.name} -> https://huggingface.co/{repo_id}/blob/main/{fp.name}")

    print(f"[HF Publisher] Done. Uploaded {uploaded}/{len(existing_files)} artifacts to {repo_id}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish trained Deepfake Agent artifacts to Hugging Face Hub.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Destination HF Hub model repository.")
    parser.add_argument("--dry-run", action="store_true", help="List artifacts without uploading.")
    args = parser.parse_args()
    return publish_custom_models(repo_id=args.repo_id, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
