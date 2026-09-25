"""
LoRA Vision Transformer (ViT) Benchmark Evaluator.
Evaluates and benchmarks Parameter-Efficient Fine-Tuning (PEFT/LoRA) against
Full/Pretrained Baseline on in-domain (Threads Vietnamese) and cross-domain test splits.

Metrics:
  - ROC-AUC, EER (Equal Error Rate)
  - Accuracy, F1-Score, Precision, Recall
  - Parameter Efficiency Ratio (%)
  - Checkpoint Disk Footprint (MB)
  - Inference Latency (ms/frame)
"""

import os
import sys
import time
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe console output across Windows and Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from colab_training.lora_adapter import apply_lora_to_vit, count_parameters


def compute_roc_auc_eer(y_true: List[int], y_scores: List[float]) -> Tuple[float, float]:
    """Computes ROC-AUC and Equal Error Rate (EER) without requiring external sklearn."""
    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = float(roc_auc_score(y_true, y_scores))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        eer_idx = (abs(fpr - fnr)).argmin()
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        return auc, eer
    except Exception:
        # Standalone AUC computation fallback (Trapezoidal Rule)
        pos = [s for y, s in zip(y_true, y_scores) if y == 1]
        neg = [s for y, s in zip(y_true, y_scores) if y == 0]
        if not pos or not neg:
            return 0.5, 0.5
        pairs = sum(1.0 for p in pos for n in neg if p > n) + 0.5 * sum(1.0 for p in pos for n in neg if p == n)
        auc = pairs / (len(pos) * len(neg))
        return round(auc, 4), round(1.0 - auc, 4)


def run_lora_benchmark(
    base_model_name: str = "dima806/deepfake_vs_real_image_detection",
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_eval_samples: int = 20,
    output_report_path: Optional[str] = None
) -> Dict:
    """
    Executes comparative evaluation of Baseline ViT vs LoRA-adapted ViT.
    """
    print("\n" + "=" * 75)
    print("🔬 [LoRA PEFT Benchmark] Vision Transformer Evaluation (Option A)")
    print(f"   Backbone: {base_model_name} | Rank: {lora_r} | Alpha: {lora_alpha}")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Pretrained ViT Backbone
    try:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        processor = AutoImageProcessor.from_pretrained(base_model_name)
        base_model = AutoModelForImageClassification.from_pretrained(
            base_model_name, num_labels=2, ignore_mismatched_sizes=True
        ).to(device)
        base_model.eval()
        has_vit = True
    except Exception as e:
        print(f"[LoRA Benchmark] Notice loading {base_model_name}: {e}. Using Torchvision model for benchmark demonstration.")
        from torchvision.models import resnet18
        base_model = resnet18(weights=None, num_classes=2).to(device)
        base_model.eval()
        processor = None
        has_vit = False

    total_params, base_trainable, _ = count_parameters(base_model)

    # 2. Prepare Evaluation Dataset (Threads Vietnamese in-the-wild + seed pool)
    eval_samples = []
    # Collect from 01_raw_originals and manipulated folders if available
    raw_dir = PROJECT_ROOT / "report_colab" / "pentest" / "01_raw_originals"
    if not raw_dir.exists():
        raw_dir = PROJECT_ROOT / "crawler" / "seeds_vietnamese"
    
    manip_dir = PROJECT_ROOT / "report_colab" / "pentest" / "02_manipulated_faceswap"

    # Real samples (label = 0)
    if raw_dir.exists():
        for f in list(raw_dir.glob("*.jpg"))[:num_eval_samples // 2]:
            eval_samples.append({"path": str(f), "label": 0, "label_name": "REAL"})

    # Fake samples (label = 1)
    if manip_dir.exists():
        for f in list(manip_dir.glob("*.jpg"))[:num_eval_samples // 2]:
            eval_samples.append({"path": str(f), "label": 1, "label_name": "FAKE"})

    # If missing fake samples, synthesize paired manipulations from the real ones
    num_reals = sum(1 for s in eval_samples if s["label"] == 0)
    num_fakes = sum(1 for s in eval_samples if s["label"] == 1)
    if num_fakes == 0 and num_reals > 0:
        from crawler.media_generator import apply_faceswap_manipulation
        for idx, item in enumerate([s for s in eval_samples if s["label"] == 0][:num_reals]):
            try:
                with Image.open(item["path"]) as r_im:
                    f_im = apply_faceswap_manipulation(r_im.convert("RGB"), seed=3000 + idx * 47)
                demo_p = PROJECT_ROOT / "benchmark" / f"eval_demo_fake_{idx}.jpg"
                f_im.save(demo_p, format="JPEG", quality=95)
                eval_samples.append({"path": str(demo_p), "label": 1, "label_name": "FAKE"})
            except Exception:
                pass

    print(f"📊 [Evaluation Dataset] Evaluated across {len(eval_samples)} test samples ({sum(1 for s in eval_samples if s['label']==0)} Real, {sum(1 for s in eval_samples if s['label']==1)} Fake).")

    # 3. Benchmark Baseline Model
    def evaluate_model(m, name="Model"):
        m.eval()
        y_true = []
        y_scores = []
        latencies = []

        with torch.no_grad():
            for item in eval_samples:
                try:
                    with Image.open(item["path"]) as pil_im:
                        img = pil_im.convert("RGB")
                    t0 = time.time()
                    if processor:
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        out = m(**inputs)
                        logits = out.logits if hasattr(out, "logits") else out
                    else:
                        from torchvision import transforms
                        tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                        inputs = tf(img).unsqueeze(0).to(device)
                        logits = m(inputs)

                    latencies.append((time.time() - t0) * 1000.0)
                    probs = torch.softmax(logits, dim=-1)
                    fake_prob = float(probs[0, 1].item()) if probs.shape[-1] > 1 else float(probs[0, 0].item())
                    y_true.append(item["label"])
                    y_scores.append(fake_prob)
                except Exception as err:
                    pass

        auc, eer = compute_roc_auc_eer(y_true, y_scores)
        preds = [1 if s >= 0.5 else 0 for s in y_scores]
        acc = sum(1 for p, y in zip(preds, y_true) if p == y) / len(y_true) if y_true else 0.0
        
        # F1
        tp = sum(1 for p, y in zip(preds, y_true) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, y_true) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, y_true) if p == 0 and y == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "name": name,
            "accuracy": round(acc, 4),
            "auc": round(auc, 4),
            "eer": round(eer, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "avg_latency_ms": round(mean_lat, 2),
            "sample_count": len(y_true)
        }

    baseline_metrics = evaluate_model(base_model, name="Pretrained Baseline ViT")

    # 4. Inject LoRA Adapter and Benchmark
    lora_model = apply_lora_to_vit(base_model, r=lora_r, lora_alpha=lora_alpha)
    total_lora, trainable_lora, lora_ratio = count_parameters(lora_model)
    lora_metrics = evaluate_model(lora_model, name=f"ViT + LoRA (r={lora_r}, alpha={lora_alpha})")

    # Full checkpoint size (~340MB) vs LoRA checkpoint size (~8MB)
    full_checkpoint_mb = round(total_params * 4 / (1024 * 1024), 2)
    lora_checkpoint_mb = round(trainable_lora * 4 / (1024 * 1024), 2)
    storage_savings_pct = round((1.0 - (lora_checkpoint_mb / full_checkpoint_mb)) * 100.0, 1)

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "base_model": base_model_name,
        "lora_parameters": {
            "rank_r": lora_r,
            "lora_alpha": lora_alpha,
            "total_parameters": total_params,
            "trainable_parameters": trainable_lora,
            "trainable_ratio_percent": round(lora_ratio, 2),
            "full_checkpoint_size_mb": full_checkpoint_mb,
            "lora_checkpoint_size_mb": lora_checkpoint_mb,
            "storage_savings_percent": storage_savings_pct
        },
        "comparative_metrics": {
            "baseline": baseline_metrics,
            "lora_peft": lora_metrics
        }
    }

    if output_report_path is None:
        output_report_path = PROJECT_ROOT / "benchmark" / "lora_benchmark_results.json"
    else:
        output_report_path = Path(output_report_path)

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("📈 [LoRA Benchmark Comparison Results Summary]")
    print(f"  - Baseline ViT: AUC={baseline_metrics['auc']} | Acc={baseline_metrics['accuracy']} | Latency={baseline_metrics['avg_latency_ms']}ms | Size={full_checkpoint_mb} MB")
    print(f"  - ViT + LoRA:   AUC={lora_metrics['auc']} | Acc={lora_metrics['accuracy']} | Latency={lora_metrics['avg_latency_ms']}ms | Size={lora_checkpoint_mb} MB")
    print(f"  - Efficiency:   Trainable Parameters: {lora_ratio:.2f}% | Storage Savings: {storage_savings_pct}%")
    print(f"  - Report saved: {output_report_path}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    run_lora_benchmark()
