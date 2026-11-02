#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crawler_data_feeder.py - Connects Threads Crawler & Curated Seeds to Daily Benchmark.
Harvests in-the-wild Vietnamese media samples across 5 Counterfactual Cohorts:
  1. Threads_vn_id_official: Citizen Identity Cards (CCCD), Driver Licenses.
  2. Threads_vn_selfie_lifestyle: Daily life selfies, beauty/smoothing filters.
  3. Threads_vn_street_outdoor: Outdoor street scenes, night markets, low-light motion blur.
  4. Threads_vn_occlusion_glasses_mask: Medical masks, sunglasses, motorbike helmets.
  5. Threads_vn_mature_aging: Elderly subjects with natural biological skin creases.
"""

import os
import sys
import json
import glob
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COHORT_MAPPING = {
    "vn_id_official": "Threads_vn_id_official",
    "vn_selfie_lifestyle": "Threads_vn_selfie_lifestyle",
    "vn_street_outdoor": "Threads_vn_street_outdoor",
    "vn_occlusion_glasses_mask": "Threads_vn_occlusion_glasses_mask",
    "vn_mature_aging": "Threads_vn_mature_aging"
}

COHORT_DESCRIPTIONS = {
    "Threads_vn_id_official": "CCCD / Ảnh thẻ công dân (Official ID)",
    "Threads_vn_selfie_lifestyle": "Selfie đời sống & Filter làm đẹp (Lifestyle Selfie)",
    "Threads_vn_street_outdoor": "Đường phố ngoại cảnh & Chợ đêm (Street & Outdoor)",
    "Threads_vn_occlusion_glasses_mask": "Che khuất khẩu trang & Kính râm (Occluded Face)",
    "Threads_vn_mature_aging": "Người cao tuổi & Lão hóa tự nhiên (Mature & Aging)"
}


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def get_daily_benchmark_samples(
    target_count: int = 30,
    output_manifest: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Harvests authentic and counterfactual media samples for daily benchmarking:
    Priority 1: Crawled images in report_colab/pentest/01_raw_originals or uploads.
    Priority 2: Curated in-the-wild seeds in crawler/verified_threads_seeds.
    Priority 3: Pairs with adversarial synthetic/deepfake counterfactuals.
    """
    samples = []
    seen_hashes = set()

    # Search Priority 1 & 2: Harvest REAL in-the-wild samples (Target: half of target_count)
    real_target = target_count // 2
    real_samples = []

    # 1. Check crawled originals
    crawled_dirs = [
        PROJECT_ROOT / "report_colab" / "pentest" / "01_raw_originals",
        PROJECT_ROOT / "crawler" / "data" / "01_raw_originals",
        PROJECT_ROOT / "uploads"
    ]
    for c_dir in crawled_dirs:
        if c_dir.exists():
            for p in c_dir.glob("*.*"):
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"] and "fake" not in p.name.lower():
                    fhash = compute_file_sha256(p)
                    if fhash in seen_hashes:
                        continue
                    seen_hashes.add(fhash)
                    
                    cohort = "Threads_vn_selfie_lifestyle"
                    for key, c_id in COHORT_MAPPING.items():
                        if key in p.name.lower():
                            cohort = c_id
                            break
                    
                    real_samples.append({
                        "sample_id": f"crawl_real_{fhash}",
                        "filename": p.name,
                        "path": str(p),
                        "dataset_source": cohort,
                        "cohort_name": COHORT_DESCRIPTIONS.get(cohort, cohort),
                        "ground_truth": "REAL",
                        "ground_truth_label": 0,
                        "modality": "image",
                        "source_origin": "THREADS_LIVE_CRAWLER"
                    })
                    if len(real_samples) >= real_target:
                        break
        if len(real_samples) >= real_target:
            break

    # 2. Check curated seeds to fulfill 5 cohorts
    seeds_dir = PROJECT_ROOT / "crawler" / "verified_threads_seeds"
    if seeds_dir.exists() and len(real_samples) < real_target:
        cohort_seeds: Dict[str, List[Path]] = {c: [] for c in COHORT_MAPPING.values()}
        for sp in seeds_dir.glob("*.jpg"):
            for key, c_id in COHORT_MAPPING.items():
                if key in sp.name.lower():
                    cohort_seeds[c_id].append(sp)
                    break
        
        per_cohort = max(1, (real_target - len(real_samples)) // len(COHORT_MAPPING) + 1)
        for c_id, seed_list in cohort_seeds.items():
            for sp in seed_list[:per_cohort]:
                fhash = compute_file_sha256(sp)
                if fhash in seen_hashes:
                    continue
                seen_hashes.add(fhash)
                real_samples.append({
                    "sample_id": f"seed_real_{fhash}",
                    "filename": sp.name,
                    "path": str(sp),
                    "dataset_source": c_id,
                    "cohort_name": COHORT_DESCRIPTIONS.get(c_id, c_id),
                    "ground_truth": "REAL",
                    "ground_truth_label": 0,
                    "modality": "image",
                    "source_origin": "VERIFIED_VIETNAMESE_SEED_POOL"
                })
                if len(real_samples) >= real_target:
                    break
            if len(real_samples) >= real_target:
                break

    # Search Priority 3: Harvest or Generate FAKE / Counterfactual samples (Target: half of target_count)
    fake_target = target_count - len(real_samples)
    fake_samples = []

    # Check existing evidence dirs first
    evidence_dirs = [
        PROJECT_ROOT / "report_colab" / "pentest" / "02_pentest_evidence",
        PROJECT_ROOT / "crawler" / "dropzone_consumer_ai"
    ]
    for e_dir in evidence_dirs:
        if e_dir.exists():
            for ep in e_dir.glob("*.*"):
                if ep.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    fhash = compute_file_sha256(ep)
                    if fhash in seen_hashes:
                        continue
                    seen_hashes.add(fhash)
                    fake_samples.append({
                        "sample_id": f"fake_{fhash}",
                        "filename": ep.name,
                        "path": str(ep),
                        "dataset_source": "Threads_vn_selfie_lifestyle",
                        "cohort_name": "Counterfactual Generative Deepfake",
                        "ground_truth": "FAKE",
                        "ground_truth_label": 1,
                        "modality": "image",
                        "source_origin": "ADVERSARIAL_DEEPFAKE_TESTBED"
                    })
                    if len(fake_samples) >= fake_target:
                        break
        if len(fake_samples) >= fake_target:
            break

    # If more fake samples needed, generate authentic paired counterfactuals on disk from real samples
    if len(fake_samples) < fake_target and real_samples:
        try:
            from PIL import Image
            from crawler.media_generator import (
                generate_paired_faceswap_manipulation,
                generate_paired_generative_diffusion_artifact
            )
            fake_cache_dir = PROJECT_ROOT / "report_colab" / "daily_benchmark" / "synthetic_counterfactuals"
            fake_cache_dir.mkdir(parents=True, exist_ok=True)

            for i, r_samp in enumerate(real_samples):
                if len(fake_samples) >= fake_target:
                    break
                r_path = Path(r_samp["path"])
                if not r_path.exists():
                    continue

                try:
                    with Image.open(r_path) as im:
                        im_rgb = im.convert("RGB")
                    
                    cat_key = r_samp.get("dataset_source", "").replace("Threads_", "")
                    # Alternate between FaceSwap and Generative Diffusion
                    if i % 2 == 0:
                        manip_im = generate_paired_faceswap_manipulation(im_rgb, source_cat=cat_key, sample_idx=i)
                        manip_type = "faceswap"
                    else:
                        manip_im = generate_paired_generative_diffusion_artifact(im_rgb, sample_idx=i)
                        manip_type = "diffusion"

                    out_name = f"fake_{manip_type}_{r_path.stem}.jpg"
                    out_path = fake_cache_dir / out_name
                    manip_im.save(out_path, format="JPEG", quality=92)

                    fhash = compute_file_sha256(out_path)
                    fake_samples.append({
                        "sample_id": f"fake_gen_{fhash}",
                        "filename": out_name,
                        "path": str(out_path),
                        "dataset_source": r_samp.get("dataset_source", "Threads_vn_selfie_lifestyle"),
                        "cohort_name": f"{r_samp.get('cohort_name', '')} (Adversarial {manip_type.upper()})",
                        "ground_truth": "FAKE",
                        "ground_truth_label": 1,
                        "modality": "image",
                        "source_origin": f"PAIRED_ADVERSARIAL_{manip_type.upper()}"
                    })
                except Exception as e_gen:
                    print(f"Warning generating counterfactual fake: {e_gen}")
        except Exception as e_imp:
            print(f"Warning importing generator: {e_imp}")

    samples = real_samples + fake_samples

    # Save manifest if requested
    if output_manifest is None:
        output_manifest = PROJECT_ROOT / "benchmark" / "daily_crawler_manifest.json"
    
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"📦 [crawler_data_feeder] Harvested {len(samples)} daily benchmark samples.")
    print(f"   Manifest saved to: {output_manifest}")
    return samples


if __name__ == "__main__":
    s = get_daily_benchmark_samples()
    print(f"Loaded {len(s)} samples successfully.")
