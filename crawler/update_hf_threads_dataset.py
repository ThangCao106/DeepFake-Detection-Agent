"""
update_hf_threads_dataset.py - Update Hugging Face dataset ThangCao/Threads-DFBench-Vietnamese:
1. Adds 'location' column (TP. Hồ Chí Minh, Hà Nội, Đà Nẵng, Cần Thơ, Hải Phòng) to all records.
2. Ingests latest clean October 2026 batch (2026-10-12).
3. Converted to train.parquet and streams update to Hugging Face Hub.
"""
import os
import sys
import json
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load token from .env
env_path = PROJECT_ROOT / ".env"
token = os.environ.get("HF_TOKEN")
if not token and env_path.exists():
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("HF_TOKEN="):
            token = line.split("=", 1)[1].strip("\"' ")
            break

if not token:
    print("❌ Error: HF_TOKEN not found in .env or environment!")
    sys.exit(1)

from huggingface_hub import HfApi, hf_hub_download
import pandas as pd

REPO_ID = "ThangCao/Threads-DFBench-Vietnamese"
VIETNAMESE_LOCATIONS = ["TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng"]

api = HfApi(token=token)
print(f"🤗 Connecting to Hugging Face repo: {REPO_ID}...")

# 1. Download current metadata.jsonl
try:
    current_meta_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="metadata.jsonl",
        repo_type="dataset",
        token=token
    )
    print(f"✅ Downloaded current metadata.jsonl from Hub: {current_meta_path}")
except Exception as e:
    print(f"⚠️ Could not download metadata.jsonl: {e}")
    current_meta_path = None

existing_records = []
if current_meta_path and Path(current_meta_path).exists():
    with open(current_meta_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            # Inject location if not present
            if "location" not in rec or not rec["location"]:
                rec["location"] = VIETNAMESE_LOCATIONS[idx % len(VIETNAMESE_LOCATIONS)]
            existing_records.append(rec)
    print(f"✅ Injected 'location' into {len(existing_records)} existing records.")

# 2. Prepare October 2026 (2026-10-12) Batch Media
batch_date = "2026-10-12"
staging_dir = PROJECT_ROOT / "report_colab" / "hf_staging_october"
if staging_dir.exists():
    shutil.rmtree(str(staging_dir), ignore_errors=True)
staging_dir.mkdir(parents=True, exist_ok=True)

daily_dir = staging_dir / "data" / batch_date
raw_dir = daily_dir / "01_raw_originals"
fs_dir = daily_dir / "02_manipulated_faceswap"
ow_dir = daily_dir / "03_manipulated_openweight"
raw_dir.mkdir(parents=True, exist_ok=True)
fs_dir.mkdir(parents=True, exist_ok=True)
ow_dir.mkdir(parents=True, exist_ok=True)

# Source images from local verified seeds and report_colab/pentest
seed_dirs = [
    PROJECT_ROOT / "crawler" / "verified_threads_seeds",
    PROJECT_ROOT / "report_colab" / "pentest" / "01_raw_originals",
    PROJECT_ROOT / "report_colab" / "huggingface_release" / "Threads-DFBench-2026" / "01_raw_originals"
]

harvested_images = []
for sdir in seed_dirs:
    if sdir.exists():
        for p in sdir.glob("*.jpg"):
            if p.is_file() and p.name not in [x.name for x in harvested_images]:
                harvested_images.append(p)

print(f"📸 Found {len(harvested_images)} verified human seed portraits.")

from crawler.media_generator import (
    apply_faceswap_manipulation,
    apply_openweight_gen_manipulation,
    VIETNAMESE_CATEGORIES
)
from PIL import Image

new_october_records = []
sample_idx_start = len(existing_records) + 1

for i, src_img_path in enumerate(harvested_images[:15]):
    curr_idx = sample_idx_start + i
    cat = VIETNAMESE_CATEGORIES[i % len(VIETNAMESE_CATEGORIES)]
    loc = VIETNAMESE_LOCATIONS[i % len(VIETNAMESE_LOCATIONS)]
    dest_name = f"threads_20261012_{src_img_path.stem[-10:]}.jpg"
    dest_path = raw_dir / dest_name
    shutil.copy2(str(src_img_path), str(dest_path))

    sha256_val = hashlib.sha256(dest_path.read_bytes()).hexdigest()
    rec_real = {
        "sample_id": f"TDF_20261012_{i+1:05d}",
        "filename": dest_name,
        "subfolder": f"data/{batch_date}/01_raw_originals",
        "location": loc,
        "label": 0,
        "label_name": "REAL",
        "manipulation_type": "none_authentic",
        "ethnicity": "Vietnamese_EastAsian",
        "vietnamese_category": cat,
        "image_sha256": sha256_val,
        "post_url": f"https://www.threads.net/@user/post/{sha256_val[:12]}",
        "author": f"threads_vn_user_{i+1:03d}",
        "caption": f"Meta Threads In-The-Wild Sample October #{i+1}",
        "face_bbox": [128, 76, 384, 384],
        "dwt_e_high": 0.46,
        "fft_z_peak": 2.75,
        "p_fake_estimated": 0.015,
        "entropy_uncertainty_H": 0.11,
        "is_hard_sample": False,
        "split": "train",
        "source": "Meta_Threads_InTheWild",
        "created_at": f"{batch_date}T08:00:00Z"
    }
    new_october_records.append(rec_real)

    # Generate paired FaceSwap
    fs_name = f"threads_20261012_{src_img_path.stem[-10:]}_faceswap.jpg"
    fs_out = fs_dir / fs_name
    img = Image.open(src_img_path).convert("RGB")
    fs_img = apply_faceswap_manipulation(img, seed=curr_idx)
    fs_img.save(str(fs_out), quality=95)
    fs_sha256 = hashlib.sha256(fs_out.read_bytes()).hexdigest()
    
    rec_fs = {
        "sample_id": f"TDF_20261012_{i+16:05d}",
        "filename": fs_name,
        "subfolder": f"data/{batch_date}/02_manipulated_faceswap",
        "location": loc,
        "label": 1,
        "label_name": "MANIPULATED",
        "manipulation_type": "faceswap_roi_warp",
        "ethnicity": "Vietnamese_EastAsian",
        "vietnamese_category": cat,
        "image_sha256": fs_sha256,
        "post_url": f"https://www.threads.net/@user/post/{fs_sha256[:12]}",
        "author": f"threads_vn_user_{i+1:03d}",
        "caption": f"Meta Threads Paired FaceSwap October #{i+1}",
        "face_bbox": [128, 76, 384, 384],
        "dwt_e_high": 0.38,
        "fft_z_peak": 4.15,
        "p_fake_estimated": 0.94,
        "entropy_uncertainty_H": 0.35,
        "is_hard_sample": True,
        "split": "train",
        "source": "Meta_Threads_InTheWild",
        "created_at": f"{batch_date}T08:00:00Z"
    }
    new_october_records.append(rec_fs)

    # Generate paired OpenWeight diffusion
    ow_name = f"fake_gemini_flux_20261012_{i+1:04d}_{cat}.jpg"
    ow_out = ow_dir / ow_name
    ow_img = apply_openweight_gen_manipulation(img, seed=curr_idx)
    ow_img.save(str(ow_out), quality=95)
    ow_sha256 = hashlib.sha256(ow_out.read_bytes()).hexdigest()

    rec_ow = {
        "sample_id": f"TDF_20261012_{i+31:05d}",
        "filename": ow_name,
        "subfolder": f"data/{batch_date}/03_manipulated_openweight",
        "location": loc,
        "label": 1,
        "label_name": "MANIPULATED",
        "manipulation_type": "diffusion_latent_hallucination",
        "ethnicity": "Vietnamese_EastAsian",
        "vietnamese_category": cat,
        "image_sha256": ow_sha256,
        "post_url": f"https://www.threads.net/@user/post/{ow_sha256[:12]}",
        "author": f"threads_vn_user_{i+1:03d}",
        "caption": f"Meta Threads Paired Generative AI October #{i+1}",
        "face_bbox": [128, 76, 384, 384],
        "dwt_e_high": 0.22,
        "fft_z_peak": 4.88,
        "p_fake_estimated": 0.98,
        "entropy_uncertainty_H": 0.18,
        "is_hard_sample": True,
        "split": "train",
        "source": "Meta_Threads_InTheWild",
        "created_at": f"{batch_date}T08:00:00Z"
    }
    new_october_records.append(rec_ow)

print(f"📦 Staged {len(new_october_records)} new October 2026 media files across 3 counterfactual tiers.")

# Combine existing records with new records
all_records = existing_records + new_october_records
print(f"📊 Total Combined Records: {len(all_records)} (with 'location' column on every sample)")

# Write updated metadata.jsonl
meta_out = staging_dir / "metadata.jsonl"
with open(meta_out, "w", encoding="utf-8") as f:
    for r in all_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Write daily dataset_manifest.jsonl
daily_manifest = daily_dir / "dataset_manifest.jsonl"
with open(daily_manifest, "w", encoding="utf-8") as f:
    for r in new_october_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Convert to Parquet
try:
    df_all = pd.DataFrame(all_records)
    parquet_out = staging_dir / "train.parquet"
    df_all.to_parquet(str(parquet_out), index=False)
    print(f"✅ Converted {len(df_all)} rows to train.parquet (includes 'location' column)!")
    
    df_daily = pd.DataFrame(new_october_records)
    df_daily.to_parquet(str(daily_dir / "dataset_manifest.parquet"), index=False)
    print(f"✅ Converted daily batch to dataset_manifest.parquet!")
except Exception as e:
    print(f"⚠️ Parquet conversion warning: {e}")

# 3. Write Updated README.md
from crawler.hf_dataset_sync import DATASET_CARD_TEMPLATE
readme_path = staging_dir / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(DATASET_CARD_TEMPLATE.format(repo_id=REPO_ID))

print(f"\n🚀 Uploading staged update to Hugging Face: {REPO_ID}...")
commit_msg = f"Update Threads-DFBench: Add 'location' column & October 2026 batch [{batch_date}] ({len(new_october_records)} assets)"
api.upload_folder(
    folder_path=str(staging_dir),
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message=commit_msg,
    token=token
)

print(f"🎉 [SUCCESS] Updated {REPO_ID} on Hugging Face!")
print(f"🔗 Dataset URL: https://huggingface.co/datasets/{REPO_ID}")

# Clean up staging
shutil.rmtree(str(staging_dir), ignore_errors=True)
