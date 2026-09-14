"""
Hugging Face Deepfake Dataset Scout & Audit Pipeline.
Automates 6-stage candidate dataset search, metadata inspection, label verification,
license compliance check, quantity evaluation, and training queue management.
"""

import os
import json
import argparse
from datetime import datetime, timezone
from huggingface_hub import HfApi, dataset_info
try:
    from datasets import load_dataset_builder
    DATASETS_LIB_AVAILABLE = True
except Exception:
    DATASETS_LIB_AVAILABLE = False


PERMISSIVE_LICENSES = ["mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "bsd-3-clause", "unlicense", "publicdomain"]
RESEARCH_LICENSES = ["cc-by-nc-4.0", "openrail", "other", "unknown"]


def search_hf_datasets(query="deepfake", limit=30):
    print(f"[hf_scout] Stage 1: HF Search - Searching datasets with query='{query}'...")
    api = HfApi()
    try:
        datasets = list(api.list_datasets(search=query, limit=limit, sort="downloads"))
        print(f"[hf_scout] Found {len(datasets)} candidate datasets from Hugging Face.")
        return datasets
    except Exception as e:
        print(f"[hf_scout] Error searching HF API: {e}")
        return []


def audit_dataset(ds_item):
    ds_id = ds_item.id
    timestamp = datetime.now(timezone.utc).isoformat()
    
    audit_record = {
        "dataset_id": ds_id,
        "timestamp": timestamp,
        "author": getattr(ds_item, "author", ds_id.split("/")[0] if "/" in ds_id else "unknown"),
        "downloads": getattr(ds_item, "downloads", 0),
        "likes": getattr(ds_item, "likes", 0),
        "tags": getattr(ds_item, "tags", []),
        "status": "REJECTED",
        "stage_passed": 1,
        "rejection_reason": None,
        "metadata": {},
        "license_info": "unknown",
        "license_type": "RESTRICTED",
        "sample_count": 0,
        "labels_found": []
    }

    tags = [str(t).lower() for t in audit_record["tags"]]
    description = str(getattr(ds_item, "description", "")).lower()

    # Stage 2: Filter Image + Deepfake
    is_image = any("image" in t for t in tags) or "image" in description or "cv" in tags or "computer-vision" in tags
    has_deepfake_keywords = any(kw in ds_id.lower() or kw in description for kw in ["deepfake", "fake", "real", "synthetic", "face"])
    
    if not is_image:
        audit_record["rejection_reason"] = "Stage 2 Failed: Non-image dataset modality"
        return audit_record

    if not has_deepfake_keywords:
        audit_record["rejection_reason"] = "Stage 2 Failed: No deepfake/real image keywords found"
        return audit_record

    audit_record["stage_passed"] = 2

    # Stage 3: Read Metadata
    try:
        info = dataset_info(ds_id)
        audit_record["metadata"]["gated"] = getattr(info, "gated", False)
        audit_record["metadata"]["private"] = getattr(info, "private", False)
        card_data = getattr(info, "cardData", {}) or {}
        license_str = str(card_data.get("license") or "unknown").lower()
        audit_record["license_info"] = license_str

        # Extract features and sample counts directly from info metadata if available
        features = card_data.get("dataset_info", {})
        if isinstance(features, list) and len(features) > 0:
            features = features[0]
        if isinstance(features, dict):
            feat_dict = features.get("features", {})
            if isinstance(feat_dict, dict):
                audit_record["labels_found"] = [k for k in feat_dict.keys() if any(lk in k.lower() for lk in ["label", "class", "target", "real", "fake", "image"])]
            splits_dict = features.get("splits", [])
            if isinstance(splits_dict, list):
                audit_record["sample_count"] = sum(s.get("num_examples", 0) for s in splits_dict if isinstance(s, dict))
    except Exception as e:
        license_str = "unknown"
        audit_record["metadata"]["fetch_error"] = str(e)

    audit_record["stage_passed"] = 3

    # Stage 4: Label Verification & Stage 5: License Check
    if any(lic in license_str for lic in PERMISSIVE_LICENSES):
        audit_record["license_type"] = "PERMISSIVE"
    elif any(lic in license_str for lic in RESEARCH_LICENSES):
        audit_record["license_type"] = "RESEARCH_ONLY"
    else:
        audit_record["license_type"] = "RESTRICTED"

    # Fallback label check from tags or naming convention if not explicitly extracted
    if not audit_record["labels_found"]:
        if any(lbl in tags or lbl in ds_id.lower() or lbl in description for lbl in ["binary", "classification", "deepfake", "real-vs-fake", "fake", "real"]):
            audit_record["labels_found"] = ["implicit_binary_label"]

    audit_record["stage_passed"] = 4

    # License check rejection rule
    if audit_record["license_type"] == "RESTRICTED" and "custom" not in license_str:
        audit_record["rejection_reason"] = f"Stage 5 Failed: License '{license_str}' is restricted or missing"
        audit_record["stage_passed"] = 5
        return audit_record

    audit_record["stage_passed"] = 5

    # Stage 6: Quantity Evaluation & Qualification
    sample_cnt = audit_record["sample_count"]
    if sample_cnt > 0 and sample_cnt < 10:
        audit_record["rejection_reason"] = f"Stage 6 Failed: Insufficient sample size ({sample_cnt} samples)"
        audit_record["stage_passed"] = 6
        return audit_record

    audit_record["status"] = "QUALIFIED"
    audit_record["stage_passed"] = 6
    return audit_record


def run_hf_dataset_scout(query="deepfake", max_results=25):
    candidates = search_hf_datasets(query=query, limit=max_results)
    
    pipeline_dir = os.path.join("dataset_pipeline")
    os.makedirs(pipeline_dir, exist_ok=True)
    
    history_file = os.path.join(pipeline_dir, "history_log.json")
    queue_file = os.path.join(pipeline_dir, "training_queue.json")

    # Load existing history if present
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    training_queue = []
    if os.path.exists(queue_file):
        try:
            with open(queue_file, "r", encoding="utf-8") as f:
                training_queue = json.load(f)
        except Exception:
            training_queue = []

    existing_ids = {h["dataset_id"] for h in history}

    qualified_count = 0
    rejected_count = 0

    print(f"\n[hf_scout] Running 6-Stage Audit on {len(candidates)} candidates...")
    for idx, ds_item in enumerate(candidates, start=1):
        ds_id = ds_item.id
        print(f"[{idx}/{len(candidates)}] Auditing dataset: {ds_id} ...")
        audit_record = audit_dataset(ds_item)
        
        # Avoid duplicate history entries
        history = [h for h in history if h["dataset_id"] != ds_id]
        history.append(audit_record)

        if audit_record["status"] == "QUALIFIED":
            qualified_count += 1
            if not any(q["dataset_id"] == ds_id for q in training_queue):
                training_queue.append({
                    "dataset_id": ds_id,
                    "added_at": audit_record["timestamp"],
                    "license": audit_record["license_info"],
                    "license_type": audit_record["license_type"],
                    "sample_count": audit_record["sample_count"],
                    "labels_found": audit_record["labels_found"],
                    "downloads": audit_record["downloads"]
                })
            print(f"  -> QUALIFIED! (License: {audit_record['license_info']}, Samples: {audit_record['sample_count']})")
        else:
            rejected_count += 1
            print(f"  -> REJECTED ({audit_record['rejection_reason']})")

    # Save updated history and queue
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(training_queue, f, indent=2)

    print("\n" + "="*60)
    print(f"[hf_scout] HF Scout Pipeline Complete!")
    print(f"Total Candidates Evaluated: {len(candidates)}")
    print(f"Qualified Datasets Added to Queue: {qualified_count}")
    print(f"Rejected Datasets Logged: {rejected_count}")
    print(f"History Log: {history_file}")
    print(f"Training Queue: {queue_file}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Face Deepfake Dataset Scout & Audit Pipeline")
    parser.add_argument("--search", type=str, default="deepfake", help="Search query on Hugging Face")
    parser.add_argument("--max-results", type=int, default=25, help="Max candidates to search and audit")
    args = parser.parse_args()

    run_hf_dataset_scout(query=args.search, max_results=args.max_results)
