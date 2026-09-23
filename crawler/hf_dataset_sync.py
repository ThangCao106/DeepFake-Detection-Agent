"""
Hugging Face Dataset Hub Synchronizer (Option A Hub Separation + Option C Cloud-to-Cloud Stream).
Synchronizes In-The-Wild Meta Threads human benchmark datasets directly to Hugging Face Datasets Hub:
  - Primary Novel Hub: ThangCao106/Threads-DFBench-Vietnamese (In-The-Wild 5-Taxonomy Vietnamese Face Dataset)
  - Foundational Manifest Hub: ThangCao106/DualStream-Foundational-Manifests (Baseline dataset connectors & metadata)

Formats for native Hugging Face Viewer:
  - 01_raw_originals / 02_manipulated_faceswap / 03_manipulated_openweight
  - dataset_manifest.jsonl & dataset_manifest.parquet
  - Standardized Dataset Card (README.md) with metadata & taxonomy tags
"""
import os
import sys
import time
import json
import shutil
from pathlib import Path

# Safe encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure .env is loaded
try:
    from crawler.crawler_config import load_dotenv_safely
    load_dotenv_safely()
except ImportError:
    pass

DEFAULT_HF_DATASET_REPO = "ThangCao/Threads-DFBench-Vietnamese"
DEFAULT_BASELINES_REPO = "ThangCao/DualStream-Foundational-Manifests"

DATASET_CARD_TEMPLATE = """---
language:
- vi
- en
license: apache-2.0
task_categories:
- image-classification
- feature-extraction
tags:
- deepfake
- vietnamese-faces
- in-the-wild
- active-learning
- dual-stream
- ncs-thesis
pretty_name: "Threads-DFBench: In-The-Wild Vietnamese Facial DeepFake Benchmark"
size_categories:
- 10K<n<100K
---

# Threads-DFBench: In-The-Wild Vietnamese Facial DeepFake Benchmark 🇻🇳

**Threads-DFBench** is an active-learning, in-the-wild benchmark dataset designed to evaluate the generalization and robustness of Deepfake detection models on East Asian / Vietnamese facial morphologies.

## 📌 Dataset Overview
- **Source**: Public in-the-wild Meta Threads posts from Vietnam (verified via OpenCV Haar cascade face verification).
- **Taxonomy**: 5-Category Vietnamese Facial Biometric Taxonomy:
  1. `vn_id_official`: Studio/ID card framing (CCCD/Passport), neutral lighting, unobstructed frontal faces.
  2. `vn_selfie_lifestyle`: Everyday mobile front-camera selfies, natural indoor/cafe lighting, diverse expressions.
  3. `vn_street_outdoor`: Outdoor street contexts, harsh shadows, complex vehicle/urban backgrounds.
  4. `vn_occlusion_glasses_mask`: Real occlusions (prescription glasses, medical/cloth masks, helmets).
  5. `vn_mature_aging`: Middle-aged and elderly individuals with natural biological skin textures and wrinkles.

## 📂 Tri-Modal Counterfactual Structure
For each authentic original portrait (`01_raw_originals`), the benchmark constructs paired counterfactuals:
- `01_raw_originals/`: 100% authentic human photographic portraits from mobile camera sensors (`Label: 0 (REAL)`).
- `02_manipulated_faceswap/`: Region-of-Interest (RoI) facial boundary swaps (`Label: 1 (MANIPULATED)`).
- `03_manipulated_openweight/`: Latent diffusion synthesis preserving context while hallucinating AI skin texture (`Label: 1 (MANIPULATED)`).

## 🔬 Mathematical Biometric Features in Manifest
The accompanying `dataset_manifest.jsonl` and `dataset_manifest.parquet` contain:
- **Haar DWT 2D Wavelet High-Frequency Energy Ratio ($E_{{high}}$)**: Captures natural sub-band dermis details ($E_{{high}} \\ge 0.30$ for authentic skin).
- **Azimuthal Fourier Spectral Peak Energy ($Z_{{peak}}$)**: Identifies upsampling grid artifacts.
- **Predictive Shannon Entropy ($H(P)$)**: Identifies hard-to-classify boundary samples ($H(P) \\ge 0.70$) for Active Learning.

## 🚀 How to Load in Python

```python
from datasets import load_dataset

# Load full benchmark
dataset = load_dataset("{repo_id}")

# Access entries
sample = dataset["train"][0]
print(sample["sample_id"], sample["label"], sample["vietnamese_category"])
```

## 📜 Citation & Research Attribution
```bibtex
@dataset{{threads_dfbench_2026,
  author = {{Thang Cao}},
  title = {{Threads-DFBench: In-The-Wild Vietnamese Facial DeepFake Benchmark}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```
"""


FOUNDATIONAL_BASELINES_CARD = """---
language:
- en
- vi
license: openrail
task_categories:
- image-classification
tags:
- deepfake-detection
- dual-stream
- benchmark-suite
pretty_name: "Dual-Stream DeepFake Foundational Baseline Connectors"
---

# Dual-Stream DeepFake Foundational Baseline Connectors

This repository provides standardized data connectors, download manifests, and partition splits for the 8 foundational baseline datasets used in the Dual-Stream Deepfake Detection Framework.

## 📊 Dual-Stream Model Allocation

### 🖼️ Model 1: General Vision & Signal Model (>578,000 samples)
1. **NTIRE-RobustAIGenDetection** (~120,000 samples): Multi-generator synthetic artifacts (MSU 2024).
2. **CIFAKE** (120,000 samples): 60,000 real CIFAR-10 vs 60,000 Stable Diffusion.
3. **DF40** (40,000 samples): 40 advanced deepfake generation algorithms.
4. **DFFD** (298,000 samples): Diverse Fake Face Dataset (58k real / 240k GAN fakes).
*Augmented with Threads Hard Negatives ($H(P) \\ge 0.70$).*

### 🧬 Model 2: Human Biometric Model (>1,105,000 samples)
1. **MultiFakeVerse** (>845,000 samples): Person-centric VLM manipulation (arXiv:2403.07689).
2. **HiDF** (~70,000 samples): Human-Indistinguishable Deepfakes (62k img, 8k vid).
3. **140k Real & Fake Faces** (140,000 samples): 70k Flickr real vs 70k StyleGAN.
4. **DeepDetect-2025/2026** (50,000 samples): Adversarial Flux/Midjourney v6 vs Unsplash.
*Augmented with Threads Vietnamese 5-Taxonomy In-The-Wild Media.*
"""


def convert_manifest_to_parquet(jsonl_path: Path, parquet_path: Path) -> bool:
    """Converts JSONL manifest to Parquet format for native Hugging Face Viewer."""
    if not jsonl_path.exists():
        return False
        
    try:
        import pandas as pd
        df = pd.read_json(str(jsonl_path), lines=True)
        df.to_parquet(str(parquet_path), index=False)
        print(f"  [Parquet Converter] Successfully converted manifest -> {parquet_path}")
        return True
    except Exception as e:
        print(f"  [Parquet Converter] Notice (using jsonl directly): {e}")
        return False


def sync_threads_dataset_to_hf(
    date_str: str = None,
    source_dir: Path = None,
    repo_id: str = None,
    hf_token: str = None,
    private: bool = False
) -> bool:
    """
    Option C Cloud-to-Cloud Upload:
    Streams Threads-DFBench media and parquet metadata directly to Hugging Face Datasets Hub.
    """
    if not date_str:
        date_str = time.strftime("%Y-%m-%d")
        
    if not repo_id:
        repo_id = os.environ.get("HF_DATASET_REPO", DEFAULT_HF_DATASET_REPO)
        
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[HF Dataset Sync] Warning: HF_TOKEN not found in environment or arguments. Skipping Hugging Face upload.")
        return False

    if source_dir is None:
        # Check standard locations: GDrive mount on Colab vs local pentest dir
        colab_gdrive = Path(f"/content/drive/MyDrive/Thread_Data/{date_str}")
        local_pentest = PROJECT_ROOT / "report_colab" / "pentest"
        if colab_gdrive.exists():
            source_dir = colab_gdrive
        elif local_pentest.exists():
            source_dir = local_pentest
        else:
            print(f"[HF Dataset Sync] Error: Neither {colab_gdrive} nor {local_pentest} found.")
            return False

    print("\n" + "=" * 75)
    print(f"🤗 Hugging Face Datasets Hub Synchronization: {repo_id}")
    print(f"📁 Source: {source_dir} | Batch Date: {date_str}")
    print("=" * 75)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[HF Dataset Sync] Installing huggingface_hub...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "pyarrow", "pandas", "-q"], check=False)
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Option C: Smart Username Auto-Resolution
    try:
        who = api.whoami()
        authed_user = who.get("name") or who.get("username")
        if authed_user and "/" in repo_id:
            curr_namespace, repo_name = repo_id.split("/", 1)
            if curr_namespace.lower() != authed_user.lower():
                print(f"🔄 [Option C: Smart Namespace Resolution] Remapping '{curr_namespace}' -> '{authed_user}'")
                repo_id = f"{authed_user}/{repo_name}"
    except Exception as e:
        print(f"⚠️ [Option C: Auth Check Notice] {e}")

    # 1. Ensure Dataset Repository exists
    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, private=private, exist_ok=True)
        print(f"✅ [HF Repo Verified] Target dataset hub ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️ [HF Repo Warning] {e}")

    # 2. Prepare staging folder for Cloud-to-Cloud upload
    staging_dir = PROJECT_ROOT / "report_colab" / "hf_dataset_staging"
    if staging_dir.exists():
        shutil.rmtree(str(staging_dir), ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 2.1 Write Dataset Card README.md
    readme_path = staging_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(DATASET_CARD_TEMPLATE.format(repo_id=repo_id))

    # 2.2 Copy Media Assets under daily folder
    daily_staging = staging_dir / "data" / date_str
    subfolders = ["01_raw_originals", "02_manipulated_faceswap", "03_manipulated_openweight"]
    copied_media = 0

    for sub in subfolders:
        src_sub = source_dir / sub
        dest_sub = daily_staging / sub
        dest_sub.mkdir(parents=True, exist_ok=True)
        if src_sub.exists():
            for f in src_sub.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".mp4", ".wav", ".webm"]:
                    shutil.copy2(str(f), str(dest_sub / f.name))
                    copied_media += 1

    print(f"📦 [Staging Package] Staged {copied_media} images across 3 counterfactual categories.")

    # 2.3 Convert and Stage Manifests
    manifest_src = source_dir / "dataset_manifest.jsonl"
    if not manifest_src.exists():
        manifest_src = PROJECT_ROOT / "report_colab" / "pentest" / "dataset_manifest.jsonl"

    if manifest_src.exists():
        shutil.copy2(str(manifest_src), str(daily_staging / "dataset_manifest.jsonl"))
        shutil.copy2(str(manifest_src), str(staging_dir / "metadata.jsonl"))
        convert_manifest_to_parquet(manifest_src, daily_staging / "dataset_manifest.parquet")
        convert_manifest_to_parquet(manifest_src, staging_dir / "train.parquet")

    summary_src = source_dir / "dataset_summary.md"
    if not summary_src.exists():
        summary_src = PROJECT_ROOT / "report_colab" / "pentest" / "dataset_summary.md"
    if summary_src.exists():
        shutil.copy2(str(summary_src), str(daily_staging / "dataset_summary.md"))

    # 3. Stream Upload Folder to Hugging Face
    print(f"🚀 [Cloud-to-Cloud Upload] Streaming staged dataset directly to Hugging Face...")
    try:
        commit_msg = f"Auto-sync Threads-DFBench Vietnamese in-the-wild dataset [{date_str}] ({copied_media} media assets)"
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_msg,
            token=token
        )
        print(f"🎉 [SUCCESS] Threads-DFBench Dataset synchronized to Hugging Face!")
        print(f"🔗 Dataset URL: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as err:
        print(f"❌ [Upload Error] Failed to stream to Hugging Face: {err}")
        return False
    finally:
        # Clean up temporary staging
        shutil.rmtree(str(staging_dir), ignore_errors=True)


def sync_baselines_manifest_to_hf(
    repo_id: str = None,
    hf_token: str = None,
    private: bool = False
) -> bool:
    """
    Synchronizes the Foundational Baseline Connectors & Manifests Hub to Hugging Face:
    Links 8 standardized baseline datasets (Model 1 General & Model 2 Biometrics).
    """
    if not repo_id:
        repo_id = os.environ.get("HF_BASELINES_REPO", DEFAULT_BASELINES_REPO)

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[HF Baselines Sync] Warning: HF_TOKEN not found.")
        return False

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"], check=False)
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Option C: Smart Username Auto-Resolution
    try:
        who = api.whoami()
        authed_user = who.get("name") or who.get("username")
        if authed_user and "/" in repo_id:
            curr_namespace, repo_name = repo_id.split("/", 1)
            if curr_namespace.lower() != authed_user.lower():
                repo_id = f"{authed_user}/{repo_name}"
    except Exception as e:
        print(f"⚠️ [Option C: Auth Check Notice] {e}")

    try:
        create_repo(repo_id=repo_id, repo_type="dataset", token=token, private=private, exist_ok=True)
        print(f"✅ [HF Baselines Repo Verified] Target hub: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️ [HF Baselines Repo Warning] {e}")

    staging_dir = PROJECT_ROOT / "report_colab" / "hf_baselines_staging"
    if staging_dir.exists():
        shutil.rmtree(str(staging_dir), ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Write Dataset Card
    readme_path = staging_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(FOUNDATIONAL_BASELINES_CARD)

    # Write Connectors metadata JSON
    try:
        from colab_training.data_pipeline import AWESOME_DEEPFAKES_REGISTRY
        connectors_path = staging_dir / "dataset_connectors.json"
        with open(connectors_path, "w", encoding="utf-8") as f:
            json.dump(AWESOME_DEEPFAKES_REGISTRY, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Could not export registry: {e}")

    try:
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update Dual-Stream Foundational Baseline Connectors & Documentation",
            token=token
        )
        print(f"🎉 [SUCCESS] Baselines Hub synchronized: https://huggingface.co/datasets/{repo_id}")
        return True
    except Exception as err:
        print(f"❌ [Baselines Upload Error] {err}")
        return False
    finally:
        shutil.rmtree(str(staging_dir), ignore_errors=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload Threads-DFBench Vietnamese In-The-Wild Dataset to Hugging Face")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_HF_DATASET_REPO, help="Hugging Face Dataset repo ID")
    parser.add_argument("--date", type=str, default=None, help="Date batch (YYYY-MM-DD)")
    parser.add_argument("--private", action="store_true", help="Set dataset visibility to private")
    args = parser.parse_args()

    sync_threads_dataset_to_hf(
        date_str=args.date,
        repo_id=args.repo_id,
        private=args.private
    )
