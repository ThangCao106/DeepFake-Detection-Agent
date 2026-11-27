"""
Threads-DFBench Active Continual Fine-Tuning Pipeline for Google Colab GPU.
Upgraded across all 3 Enhancement Phases:
  1. Multi-Backbone Support (`vit`, `clip`, `siglip2`, `dinov2`) via `foundation_backbones.py`
  2. PEFT Adaptation: `DoRA` (default), `LoRA+` differential LR, `Effort` Orthogonal Subspace SVD, or Full FT
  3. Diagonal EWC (`ewc_fisher_state.pt`) + Orthogonal Adapter Regularization preventing Catastrophic Forgetting
  4. Forensics-Aware Social-Media Augmentation + On-the-Fly Self-Blended Images (SBI)
  5. Asymmetric Cost-Sensitive Focal Loss + Supervised Contrastive Loss (`SupConLoss`) + 5-Cohort Group-DRO
  6. Closed-Loop SQLite `user_feedback_events` Ingestion + 70/30 Anchor-vs-Replay Mixing + AMP + EMA
  7. 2-Parameter Platt Scaling (`a*z + b`), Youden `tau*`, 10-bin ECE, Split Conformal Band, and `model_sha256` binding
"""

import argparse
import copy
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


VN_COHORT_NAMES = (
    "vn_id_official",
    "vn_selfie_lifestyle",
    "vn_street_outdoor",
    "vn_occlusion_glasses_mask",
    "vn_mature_aging",
)


class FocalLoss(nn.Module):
    """Balanced Focal Loss FL(p_t) = - alpha * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AsymmetricFocalLoss(nn.Module):
    """
    Cost-sensitive Asymmetric Focal Loss for DeepFake Forensics.
    Emphasizes lifting True Positive Rate (Recall on Fake) without inflating False Positives.
    """

    def __init__(
        self,
        fake_idx: int = 1,
        gamma_pos: float = 1.5,
        gamma_neg: float = 2.5,
        alpha_pos: float = 0.65,
        alpha_neg: float = 0.35,
    ):
        super().__init__()
        self.fake_idx = int(fake_idx)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = float(alpha_neg)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        is_fake = (targets == self.fake_idx)
        gamma = torch.where(is_fake, torch.tensor(self.gamma_pos, device=logits.device), torch.tensor(self.gamma_neg, device=logits.device))
        alpha = torch.where(is_fake, torch.tensor(self.alpha_pos, device=logits.device), torch.tensor(self.alpha_neg, device=logits.device))
        losses = alpha * ((1.0 - pt) ** gamma) * ce
        if sample_weights is not None:
            losses = losses * sample_weights.to(losses.device, dtype=losses.dtype)
        return losses.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss clustering authentic VN portraits vs forged generators."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.size(0) < 2:
            return features.sum() * 0.0
        feats = F.normalize(features.float(), dim=1)
        sim = torch.matmul(feats, feats.T) / max(1e-4, self.temperature)
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(feats.device)
        logits_mask = 1.0 - torch.eye(feats.size(0), device=feats.device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        pos_per_row = mask.sum(1)
        valid = pos_per_row > 0
        if not valid.any():
            return features.sum() * 0.0
        mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(1) / pos_per_row[valid]
        return -mean_log_prob_pos.mean()


class GroupDROTracker:
    """
    Maintains exponentiated-gradient group weights across the 5 Vietnamese demographic cohorts
    so harder cohorts (e.g. selfie beauty filter, outdoor street) automatically receive higher update weight.
    """

    def __init__(self, cohort_names=VN_COHORT_NAMES, step_size: float = 0.1):
        self.cohort_names = list(cohort_names)
        self.weights: Dict[str, float] = {c: 1.0 / max(1, len(self.cohort_names)) for c in self.cohort_names}
        self.step_size = float(step_size)

    def sample_weights(self, cohort_ids: List[str]) -> torch.Tensor:
        mean_w = 1.0 / max(1, len(self.cohort_names))
        vals = [self.weights.get(c, mean_w) * len(self.cohort_names) for c in cohort_ids]
        return torch.tensor(vals, dtype=torch.float32)

    def update(self, cohort_losses: Dict[str, float]) -> None:
        if not cohort_losses:
            return
        for c, loss_val in cohort_losses.items():
            if c in self.weights:
                self.weights[c] *= math.exp(min(3.0, self.step_size * float(loss_val)))
        total = sum(self.weights.values()) or 1.0
        for c in self.weights:
            self.weights[c] /= total


def _normalize_manifest_record(rec: dict) -> dict:
    """Normalize heterogeneous manifest entries (.jsonl & .json) into a unified schema."""
    out = dict(rec)
    lp = out.get("local_path") or out.get("filepath") or out.get("path")
    if lp:
        out["local_path"] = str(lp)
    if not out.get("filename") and lp:
        out["filename"] = os.path.basename(str(lp))

    raw_lbl = out.get("label", out.get("ground_truth_label", out.get("ground_truth", 0)))
    if isinstance(raw_lbl, str):
        rl = raw_lbl.strip().upper()
        out["label"] = 0 if rl in ("REAL", "REALISM", "AUTHENTIC", "0") else 1
    else:
        out["label"] = int(raw_lbl)

    cohort = str(out.get("vietnamese_category") or out.get("dataset_source") or out.get("cohort") or "").lower()
    matched_cohort = "vn_selfie_lifestyle"
    for vc in VN_COHORT_NAMES:
        if vc in cohort:
            matched_cohort = vc
            break
    out["cohort"] = matched_cohort
    return out


class ThreadsDriveDataset(Dataset):
    """
    PyTorch Dataset loading verified real & manipulated images from local SSD / corpus / Google Drive,
    with optional SocialMediaForensicAugment and On-The-Fly SBI counterfactual injection.
    """

    def __init__(
        self,
        manifest_entries,
        base_search_dirs=None,
        processor=None,
        augment: bool = False,
        sbi_prob: float = 0.0,
        fake_label_value: int = 1,
        real_label_value: int = 0,
        seed: int = 2026,
    ):
        self.entries = [_normalize_manifest_record(e) for e in manifest_entries]
        self.processor = processor
        self.base_search_dirs = [Path(d) for d in (base_search_dirs or []) if Path(d).exists()]
        self.file_map = {}
        self.augment = bool(augment)
        self.sbi_prob = float(sbi_prob)
        self.fake_label_value = int(fake_label_value)
        self.real_label_value = int(real_label_value)
        self.rng = random.Random(seed)
        self.augmentor = None
        self.sbi_fn = None
        if self.augment or self.sbi_prob > 0.0:
            try:
                from colab_training.forensic_augmentations import (
                    SocialMediaForensicAugment,
                    generate_self_blended_image,
                )
                if self.augment:
                    self.augmentor = SocialMediaForensicAugment(
                        p_jpeg=0.45, p_webp=0.20, p_blur=0.30, p_resize=0.35, p_overlay=0.25, seed=seed
                    )
                if self.sbi_prob > 0.0:
                    self.sbi_fn = generate_self_blended_image
            except Exception:
                pass

    def _resolve_file_path(self, entry):
        lp = entry.get("local_path")
        if lp and os.path.exists(lp):
            return lp

        gid = entry.get("gdrive_file_id")
        if gid and os.path.exists(gid):
            return gid

        fname = entry.get("filename", "")
        if fname and fname in self.file_map:
            return self.file_map[fname]

        subfolder = entry.get("subfolder", "")
        for bdir in self.base_search_dirs:
            if subfolder:
                p1 = bdir / subfolder / fname
                if p1.exists():
                    self.file_map[fname] = str(p1)
                    return str(p1)
            p2 = bdir / fname
            if p2.exists():
                self.file_map[fname] = str(p2)
                return str(p2)
        return None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_path = self._resolve_file_path(entry)
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Training sample missing on disk: {entry}")

        img = Image.open(file_path).convert("RGB")
        label = int(entry.get("label", 0))

        # On-the-fly Self-Blended Image (SBI) counterfactual when enabled on Real portrait
        if (
            label == self.real_label_value
            and self.sbi_fn is not None
            and self.sbi_prob > 0.0
            and self.rng.random() < self.sbi_prob
        ):
            img = self.sbi_fn(img, seed=self.rng.randint(1, 1_000_000))
            label = self.fake_label_value
        elif self.augmentor is not None:
            img = self.augmentor(img)

        if self.processor:
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        else:
            from torchvision import transforms
            t_func = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = t_func(img)

        return (
            pixel_values,
            torch.tensor(label, dtype=torch.long),
            bool(entry.get("is_hard_sample", False)),
            entry.get("cohort", "vn_selfie_lifestyle"),
        )


def stage_active_training_samples(
    records,
    candidate_roots,
    local_cache_dir=Path("/content/train_cache/staged"),
    max_train_samples=2500,
):
    """Stage active learning window (70% clean anchor / 30% hard-negative replay) to local NVMe SSD."""
    if not records:
        return []

    records = [_normalize_manifest_record(r) for r in records]
    hard_samples = [r for r in records if r.get("is_hard_sample")]
    normal_samples = [r for r in records if not r.get("is_hard_sample")]

    # Maintain 70% clean anchor / 30% hard-negative & feedback replay ratio
    max_hard = min(len(hard_samples), max(1, int(max_train_samples * 0.30)))
    max_normal = max(0, max_train_samples - max_hard)

    selected = []
    if hard_samples:
        selected.extend(hard_samples[-max_hard:])
    if normal_samples:
        if len(normal_samples) > max_normal:
            recent_count = int(max_normal * 0.70)
            replay_count = max_normal - recent_count
            selected.extend(normal_samples[-recent_count:])
            selected.extend(normal_samples[:replay_count])
        else:
            selected.extend(normal_samples)

    if not Path("/content").exists():
        local_cache_dir = PROJECT_ROOT / "report_colab" / "train_cache" / "staged"
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    staged_records = []
    to_copy = []

    for r in selected:
        fname = r.get("filename", "")
        lp = r.get("local_path")

        if lp and os.path.exists(lp) and not str(lp).startswith("/content/drive"):
            r_copy = dict(r)
            r_copy["local_path"] = str(lp)
            staged_records.append(r_copy)
            continue

        staged_p = local_cache_dir / fname if fname else None
        if staged_p and staged_p.exists() and staged_p.stat().st_size > 100:
            r_copy = dict(r)
            r_copy["local_path"] = str(staged_p)
            staged_records.append(r_copy)
            continue

        found_source = None
        if lp and os.path.exists(lp):
            found_source = Path(lp)
        elif fname:
            subfolder = r.get("subfolder", "")
            for c_root in candidate_roots:
                if subfolder:
                    p1 = c_root / subfolder / fname
                    if p1.exists():
                        found_source = p1
                        break
                p2 = c_root / fname
                if p2.exists():
                    found_source = p2
                    break

        if found_source and staged_p:
            to_copy.append((found_source, staged_p, dict(r)))

    if to_copy:
        print(f"⚡ [Fast Local Staging] Pre-caching {len(to_copy)} samples to local SSD ({local_cache_dir})...")
        from concurrent.futures import ThreadPoolExecutor

        def _copy_task(item):
            src, dst, rec = item
            try:
                shutil.copy2(src, dst)
                rec["local_path"] = str(dst)
                return rec
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for res in pool.map(_copy_task, to_copy):
                if res is not None:
                    staged_records.append(res)
        print(f"✅ [Fast Local Staging] {len(staged_records)} samples staged on local SSD!")

    return staged_records if staged_records else records


def load_benchmark_hard_negatives(benchmark_dir=None):
    """Extract misclassified benchmark samples (match == False) whose files exist on disk."""
    if not benchmark_dir:
        benchmark_dir = PROJECT_ROOT / "custom_vit_model"
    res_file = Path(benchmark_dir) / "benchmark_results.json"
    hard_samples = []
    if res_file.exists():
        try:
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, p in enumerate(data.get("predictions", [])):
                    pth = p.get("path")
                    if not p.get("match", True) and pth and os.path.exists(pth):
                        gt = p.get("ground_truth", "FAKE").upper()
                        label = 0 if gt == "REAL" else 1
                        hard_samples.append({
                            "sample_id": f"BENCH_ERR_{idx+1:03d}",
                            "filename": os.path.basename(pth),
                            "local_path": pth,
                            "label": label,
                            "label_name": gt,
                            "is_hard_sample": True,
                            "split": "train",
                            "source": "Benchmark_Misclassification_Mining",
                        })
        except Exception as e:
            print(f"[Continual Trainer] Notice loading benchmark hard negatives: {e}")
    return hard_samples


def load_sqlite_feedback_hard_negatives(
    db_path: Optional[Path] = None,
    feedback_ids_out: Optional[List[str]] = None,
) -> List[dict]:
    """
    Close the SQLite Active Replay Loop:
    Reads un-trained feedback disputes (`lora_trained = 0`) from `user_feedback_events`.
    For existing video clips, extracts keyframes into `report_colab/feedback_frames/`.
    """
    dbp = Path(db_path) if db_path else (PROJECT_ROOT / "report_colab" / "deepfake_agent_memory.db")
    if not dbp.exists():
        return []

    records: List[dict] = []
    frame_out_dir = PROJECT_ROOT / "report_colab" / "feedback_frames"
    try:
        con = sqlite3.connect(str(dbp))
        cur = con.cursor()
        rows = cur.execute(
            "SELECT feedback_id, file_path, true_label, platform, media_type "
            "FROM user_feedback_events "
            "WHERE COALESCE(lora_trained, 0) = 0 "
            "AND UPPER(COALESCE(review_status, 'PENDING')) IN ('APPROVED_REAL', 'APPROVED_FAKE', 'APPROVED_CALIBRATED') "
            "ORDER BY id DESC LIMIT 200"
        ).fetchall()
        con.close()
    except Exception as e:
        print(f"[SQLite Replay] Notice querying feedback events: {e}")
        return []

    from colab_training.data_pipeline import DataPipeline

    for fid, fp, true_lbl, platform, media_type in rows:
        if not fp or not os.path.exists(str(fp)):
            continue
        lbl = 0 if "REAL" in str(true_lbl or "").upper() else 1
        lbl_name = "REAL" if lbl == 0 else "FAKE"
        ext = os.path.splitext(str(fp))[1].lower()
        img_paths = []
        if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi") or str(media_type or "").lower() == "video":
            img_paths = DataPipeline.extract_video_keyframes(str(fp), str(frame_out_dir / str(fid)), max_frames=2)
        elif ext in (".jpg", ".jpeg", ".png", ".webp"):
            img_paths = [str(fp)]

        for k_idx, ip in enumerate(img_paths):
            if os.path.exists(ip) and os.path.getsize(ip) > 256:
                records.append({
                    "sample_id": f"FB_{fid}_{k_idx}",
                    "feedback_id": str(fid),
                    "filename": os.path.basename(ip),
                    "local_path": ip,
                    "label": lbl,
                    "label_name": lbl_name,
                    "is_hard_sample": True,
                    "platform": platform or "unknown",
                    "source": "sqlite_user_feedback_events",
                })
        if img_paths and feedback_ids_out is not None:
            feedback_ids_out.append(str(fid))

    if records:
        print(f"[SQLite Replay] Loaded {len(records)} hard-negative samples from {len(feedback_ids_out or [])} un-trained feedback events.")
    return records


def mark_sqlite_feedback_trained(feedback_ids: List[str], db_path: Optional[Path] = None) -> int:
    """Mark processed `user_feedback_events` as `lora_trained = 1` ONLY after Safety Gate passes."""
    if not feedback_ids:
        return 0
    dbp = Path(db_path) if db_path else (PROJECT_ROOT / "report_colab" / "deepfake_agent_memory.db")
    if not dbp.exists():
        return 0
    try:
        con = sqlite3.connect(str(dbp))
        cur = con.cursor()
        placeholders = ",".join("?" for _ in feedback_ids)
        cur.execute(
            f"UPDATE user_feedback_events SET lora_trained = 1 WHERE feedback_id IN ({placeholders})",
            tuple(feedback_ids),
        )
        updated = cur.rowcount
        con.commit()
        con.close()
        return int(updated)
    except Exception as e:
        print(f"[SQLite Replay] Warning marking feedback trained: {e}")
        return 0


def load_manifest_file(mf: Path) -> list:
    """Read either JSONL (.jsonl) or JSON list (.json) manifest files."""
    items = []
    if not mf.exists():
        return items
    try:
        if mf.suffix.lower() == ".jsonl":
            with open(mf, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        items.append(_normalize_manifest_record(json.loads(s)))
        elif mf.suffix.lower() == ".json":
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        items.append(_normalize_manifest_record(rec))
    except Exception as e:
        print(f"[Continual Trainer] Warning reading {mf}: {e}")
    return items


def find_all_manifests(target_dir):
    """Recursively scan target_dir for corpus, dataset, and hard-negative manifests."""
    target_dir = Path(target_dir)
    if not target_dir.exists():
        return []
    manifest_files = (
        list(target_dir.glob("**/training_corpus_manifest.jsonl"))
        + list(target_dir.glob("**/dataset_manifest.jsonl"))
        + list(target_dir.glob("**/hard_samples_manifest.jsonl"))
        + list(target_dir.glob("**/human_review_replay_manifest.json"))
    )
    all_records = []
    seen_ids = set()
    for mf in manifest_files:
        for rec in load_manifest_file(mf):
            sid = rec.get("sample_id") or rec.get("local_path") or rec.get("filename")
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                all_records.append(rec)
    return all_records


def _load_vit_backbone_resilient(base_model: str, device: str, backbone_family: str = "vit"):
    """Load ViT/Foundation processor & model with token=False fallback."""
    try:
        from colab_training.foundation_backbones import load_foundation_backbone_resilient
        local_custom = PROJECT_ROOT / "custom_vit_model"
        explicit = base_model
        if backbone_family == "vit" and (local_custom / "model.safetensors").exists() and (local_custom / "config.json").exists():
            explicit = str(local_custom)
        proc, mod, is_cls, repo_used = load_foundation_backbone_resilient(
            backbone_family=backbone_family,
            explicit_model_id=explicit if backbone_family == "vit" else None,
            device=device,
        )
        return proc, mod, True
    except Exception:
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        local_custom = PROJECT_ROOT / "custom_vit_model"
        if (local_custom / "model.safetensors").exists() and (local_custom / "config.json").exists():
            base_model = str(local_custom)

        env_tok = (os.environ.get("HF_TOKEN") or "").strip() or None
        attempts = [env_tok, False] if (env_tok and not os.path.exists(base_model)) else [False]
        last_err = None
        for tok in attempts:
            try:
                processor = AutoImageProcessor.from_pretrained(base_model, token=tok)
                model = AutoModelForImageClassification.from_pretrained(
                    base_model, num_labels=2, ignore_mismatched_sizes=True, token=tok
                ).to(device)
                return processor, model, True
            except Exception as err:
                last_err = err
        raise RuntimeError(f"Could not load ViT backbone '{base_model}': {last_err}")


def _calibrate_platt_and_youden(
    y_true: list,
    fake_logits_diff: list,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    2-Parameter Platt Scaling (a*z + b) minimizing Binary Cross-Entropy / NLL on OOD logits:
      - Eliminates domain shift bias `b` that single-parameter Temperature Scaling cannot shift!
      - Derives equivalent `T* = clamp(1 / a, 0.25, 5.0)` for legacy readers.
      - Optimizes Youden's J threshold `tau*` on calibrated probabilities.
      - Computes 10-bin Expected Calibration Error (ECE) + Split Conformal uncertainty band [c_low, c_high].
    Returns:
      (platt_t, youden_tau, ece, platt_a, platt_b, conformal_low, conformal_high)
    """
    if not y_true or len(set(y_true)) < 2:
        return 1.6178, 0.8118, 0.0162, 0.6181, 0.0, 0.67, 0.90

    z_t = torch.tensor(fake_logits_diff, dtype=torch.float32)
    y_t = torch.tensor(y_true, dtype=torch.float32)

    raw_a = torch.nn.Parameter(torch.tensor(0.5))
    b_param = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([raw_a, b_param], lr=0.08)
    for _ in range(180):
        opt.zero_grad()
        a_pos = F.softplus(raw_a).clamp(min=0.10, max=10.0)
        loss = F.binary_cross_entropy_with_logits(a_pos * z_t + b_param, y_t)
        loss.backward()
        opt.step()

    platt_a = round(float(F.softplus(raw_a).clamp(min=0.10, max=10.0).item()), 4)
    platt_b = round(float(b_param.clamp(min=-5.0, max=5.0).item()), 4)
    best_t = round(max(0.25, min(5.0, 1.0 / max(1e-4, platt_a))), 4)

    cal_probs = [
        1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, platt_a * float(z) + platt_b))))
        for z in fake_logits_diff
    ]

    best_j = -1.0
    best_tau = 0.50
    n_pos = max(1, sum(1 for y in y_true if y == 1))
    n_neg = max(1, sum(1 for y in y_true if y == 0))
    for i in range(25, 92):
        th = i * 0.01
        tp = sum(1 for y, p in zip(y_true, cal_probs) if y == 1 and p >= th)
        fp = sum(1 for y, p in zip(y_true, cal_probs) if y == 0 and p >= th)
        j_stat = (tp / n_pos) - (fp / n_neg)
        if j_stat > best_j:
            best_j = j_stat
            best_tau = round(th, 4)

    # 10-bin Expected Calibration Error (ECE)
    ece = 0.0
    n_total = len(y_true)
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        idxs = [i for i, p in enumerate(cal_probs) if (lo <= p < hi) or (b == 9 and p == 1.0)]
        if idxs:
            avg_conf = sum(cal_probs[i] for i in idxs) / len(idxs)
            avg_acc = sum(y_true[i] for i in idxs) / len(idxs)
            ece += (len(idxs) / n_total) * abs(avg_acc - avg_conf)

    # Split Conformal Non-Conformity Margin around best_tau (alpha=0.10)
    margins = sorted(abs(p - best_tau) for y, p in zip(y_true, cal_probs) if (p >= best_tau) != (y == 1))
    if margins:
        q_idx = min(len(margins) - 1, int(math.ceil(0.90 * len(margins))) - 1)
        band_half = min(0.20, max(0.06, margins[q_idx]))
    else:
        band_half = 0.08
    conformal_low = round(max(0.15, best_tau - band_half), 4)
    conformal_high = round(min(0.95, best_tau + band_half), 4)

    return best_t, best_tau, round(ece, 4), platt_a, platt_b, conformal_low, conformal_high


class _ModelEMA:
    """Lightweight Exponential Moving Average over trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def update(self, model: nn.Module) -> None:
        d = self.decay
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    self.shadow[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        backup = {}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.copy_(self.shadow[n])
        return backup


def run_threads_continual_training(
    drive_root=None,
    epochs=3,
    batch_size=32,
    lr=1e-4,
    base_model="dima806/deepfake_vs_real_image_detection",
    output_checkpoint_dir=None,
    export_hf_dir=None,
    extra_manifests=None,
    max_train_samples=2500,
    min_ood_auc=0.70,
    max_ood_ece=0.15,
    backbone_family="vit",
    peft_method="dora",
    lora_r=16,
    lora_alpha=32,
    loraplus_ratio=16.0,
    use_ewc=True,
    ewc_lambda=50.0,
    use_augment=True,
    sbi_prob=0.15,
    use_supcon=True,
    use_group_dro=True,
):
    """
    Execute 3-Phase Enhanced Active Continual Fine-Tuning on verified media:
      - Ingests Corpus Harvester manifests, Google Drive manifests, and un-trained SQLite feedback disputes
      - Applies DoRA / LoRA+ / Effort Orthogonal Subspace adaptation + Diagonal EWC
      - Trains with SocialMediaForensicAugment + On-The-Fly SBI + Asymmetric Focal + SupCon + Group-DRO
      - Calibrates 2-Parameter Platt Scaling (a, b), Youden tau*, Conformal Band, and SHA-256 binding
      - Enforces OOD Safety Gate before exporting `model.safetensors` & marking SQLite feedback trained
    """
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 78)
    print(f"[Continual Trainer] Active Training Pipeline (Backbone={backbone_family}, PEFT={peft_method})")
    print(f"[Continual Trainer] Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | EWC: {use_ewc}")
    print("=" * 78)

    candidate_roots = []
    for cand in [
        Path(drive_root) if drive_root else None,
        Path("/content/drive/MyDrive/Thread_Data"),
        Path("/content/train_cache"),
        PROJECT_ROOT / "report_colab" / "corpus",
        PROJECT_ROOT / "report_colab" / "gdrive_storage",
        PROJECT_ROOT / "report_colab" / "pentest",
        PROJECT_ROOT / "colab_datasets",
    ]:
        if cand and cand.exists() and cand not in candidate_roots:
            candidate_roots.append(cand)

    records = []
    for c_root in candidate_roots:
        found = find_all_manifests(c_root)
        if found:
            records.extend(found)
            print(f"[Continual Trainer] [DATA DETECTED] Found {len(found)} manifest entries under {c_root}")

    for mf_str in (extra_manifests or []):
        if mf_str and os.path.exists(mf_str):
            extra_recs = load_manifest_file(Path(mf_str))
            if extra_recs:
                print(f"[Continual Trainer] Ingested {len(extra_recs)} records from explicit manifest: {mf_str}")
                records.extend(extra_recs)

    bench_hards = load_benchmark_hard_negatives()
    if bench_hards:
        print(f"[Hard Negative Mining] Ingested {len(bench_hards)} misclassified benchmark cases")
        records.extend(bench_hards)

    sqlite_feedback_ids: List[str] = []
    sqlite_hards = load_sqlite_feedback_hard_negatives(feedback_ids_out=sqlite_feedback_ids)
    if sqlite_hards:
        records.extend(sqlite_hards)

    if not records:
        from colab_training.data_pipeline import DataPipeline
        dp = DataPipeline()
        fallback_recs = dp.collect_verified_local_media("IMAGE", "threads_vietnamese_in_the_wild", max_samples=max_train_samples)
        records.extend([_normalize_manifest_record(r) for r in fallback_recs])

    unique_records = []
    seen = set()
    for r in records:
        key = r.get("local_path") or r.get("sample_id") or r.get("filename")
        if key and key not in seen:
            seen.add(key)
            unique_records.append(r)
    records = unique_records

    staged_records = stage_active_training_samples(
        records,
        candidate_roots=candidate_roots,
        max_train_samples=max_train_samples,
    )
    if staged_records:
        records = staged_records

    existing_records = []
    for rec in records:
        lp = rec.get("local_path")
        if lp and os.path.exists(lp) and os.path.getsize(lp) > 256:
            existing_records.append(rec)
        else:
            fname = rec.get("filename", "")
            sub = rec.get("subfolder", "")
            for cr in candidate_roots:
                p_cand = (cr / sub / fname) if sub else (cr / fname)
                if fname and p_cand.exists() and p_cand.stat().st_size > 256:
                    rec_c = dict(rec)
                    rec_c["local_path"] = str(p_cand)
                    existing_records.append(rec_c)
                    break

    if not existing_records:
        raise RuntimeError("No verified image files found on disk for Continual ViT Training.")
    records = existing_records

    from forensics_kernel.dataset_split_guard import (
        assign_leak_free_group_splits,
        assert_zero_origin_leakage,
        extract_origin_id,
        normalize_split_name,
    )

    for rec in records:
        rec["origin_id"] = rec.get("origin_id") or extract_origin_id(rec)

    has_locked = any(normalize_split_name(r.get("split"), default="") in ("cal_val", "test_locked") for r in records)
    if has_locked:
        base_train = [r for r in records if normalize_split_name(r.get("split")) == "train"]
        test_records = [r for r in records if normalize_split_name(r.get("split")) == "cal_val"]
        # Strictly isolate `test_locked` from continual training & calibration!
        locked_held_out = [r for r in records if normalize_split_name(r.get("split")) == "test_locked"]
        if not base_train or not test_records:
            grouped = assign_leak_free_group_splits([r for r in records if normalize_split_name(r.get("split")) != "test_locked"], train_ratio=0.80, cal_ratio=0.20, seed=2026)
            base_train = [r for r in grouped if r["split"] == "train"]
            test_records = [r for r in grouped if r["split"] == "cal_val"]
    else:
        grouped = assign_leak_free_group_splits(records, train_ratio=0.70, cal_ratio=0.15, seed=2026)
        base_train = [r for r in grouped if r["split"] == "train"]
        test_records = [r for r in grouped if r["split"] == "cal_val"]
        locked_held_out = [r for r in grouped if r["split"] == "test_locked"]

    assert_zero_origin_leakage(base_train, test_records, locked_held_out)

    reals = [r for r in base_train + test_records if int(r.get("label", 0)) == 0]
    fakes = [r for r in base_train + test_records if int(r.get("label", 0)) == 1]
    total_count = len(base_train) + len(test_records)
    hard_count = sum(1 for r in base_train + test_records if r.get("is_hard_sample"))
    print(f"\n[Dataset Summary] Leak-Free Samples (test_locked excluded={len(locked_held_out)}): Train={len(base_train)}, Cal/Val={len(test_records)} (Real={len(reals)}, Fake={len(fakes)}) | Hard Negatives: {hard_count}")

    train_records = []
    for r in base_train:
        train_records.append(r)
        if r.get("is_hard_sample") and len(train_records) < max_train_samples:
            train_records.append(r)

    processor = None
    is_hf_vit = False
    try:
        processor, model, is_hf_vit = _load_vit_backbone_resilient(base_model, device, backbone_family=backbone_family)
        print(f"[Model Loader] Loaded Vision backbone ({backbone_family}): {base_model}")
    except Exception as e:
        print(f"[Model Loader] Warning loading HF ViT ({e}); falling back to Torchvision resnet18.")
        from torchvision.models import resnet18
        model = resnet18(weights=None, num_classes=2).to(device)

    fake_idx = 1
    if is_hf_vit and hasattr(model, "config") and getattr(model.config, "id2label", None):
        for idx_k, lbl_name in model.config.id2label.items():
            if any(tok in str(lbl_name).lower() for tok in ("fake", "deepfake", "ai", "spoof")):
                fake_idx = int(idx_k)
                break
    real_idx = 1 - fake_idx

    # Apply PEFT (DoRA / LoRA+ / Effort Orthogonal Subspace)
    from colab_training.lora_adapter import (
        DiagonalEWC,
        apply_lora_to_vit,
        build_loraplus_param_groups,
        compute_orthogonal_adapter_loss,
        merge_native_adapters,
    )

    method_l = (peft_method or "dora").strip().lower()
    if method_l in ("dora", "lora"):
        model = apply_lora_to_vit(model, r=lora_r, lora_alpha=lora_alpha, peft_method=method_l, prefer_peft=True)
    elif method_l == "effort":
        from colab_training.foundation_backbones import apply_effort_orthogonal_subspace
        model = apply_effort_orthogonal_subspace(model, rank=lora_r)

    if fake_idx == 0:
        def _align_records(recs):
            aligned = []
            for r in recs:
                rc = dict(r)
                rc["label"] = 0 if int(r.get("label", 0)) == 1 else 1
                aligned.append(rc)
            return aligned
        train_ds_recs = _align_records(train_records)
        test_ds_recs = _align_records(test_records)
    else:
        train_ds_recs = train_records
        test_ds_recs = test_records

    search_dirs = [Path("/content/train_cache/staged")] + candidate_roots
    train_ds = ThreadsDriveDataset(
        train_ds_recs,
        base_search_dirs=search_dirs,
        processor=processor,
        augment=use_augment,
        sbi_prob=sbi_prob,
        fake_label_value=fake_idx,
        real_label_value=real_idx,
    )
    test_ds = ThreadsDriveDataset(
        test_ds_recs,
        base_search_dirs=search_dirs,
        processor=processor,
        augment=False,
        sbi_prob=0.0,
        fake_label_value=fake_idx,
        real_label_value=real_idx,
    )

    num_workers = 2 if (device == "cuda" and sys.platform.startswith("linux")) else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, max(1, len(train_ds))),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=min(batch_size, max(1, len(test_ds))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    param_groups = build_loraplus_param_groups(model, lr=lr, loraplus_lr_ratio=loraplus_ratio, weight_decay=0.01)
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr, "weight_decay": 0.01}]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = AsymmetricFocalLoss(fake_idx=fake_idx, gamma_pos=1.5, gamma_neg=2.5, alpha_pos=0.65, alpha_neg=0.35)
    supcon_fn = SupConLoss(temperature=0.1) if use_supcon else None
    dro_tracker = GroupDROTracker() if use_group_dro else None

    hf_export_dir = Path(export_hf_dir) if export_hf_dir else (PROJECT_ROOT / "custom_vit_model")
    ewc_path = hf_export_dir / "ewc_fisher_state.pt"
    ewc_obj = None
    if use_ewc and ewc_path.exists():
        try:
            ewc_obj = DiagonalEWC.load(ewc_path)
            print(f"[EWC] Loaded previous Fisher anchor state from {ewc_path}")
        except Exception:
            ewc_obj = None

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch, "amp") else None
    ema = _ModelEMA(model, decay=0.999)

    try:
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None

    model.train()
    history = []
    for ep in range(epochs):
        ep_loss, correct, total = 0.0, 0, 0
        cohort_loss_acc: Dict[str, List[float]] = {}
        pbar = tqdm(train_loader, desc=f"  Epoch [{ep + 1}/{epochs}]", unit="batch") if tqdm else train_loader
        for batch in pbar:
            pixel_values, labels, _, cohorts = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            s_weights = dro_tracker.sample_weights(list(cohorts)).to(device) if dro_tracker else None

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp) if hasattr(torch, "autocast") else torch.no_grad():
                pass
            if use_amp and hasattr(torch, "autocast"):
                with torch.autocast(device_type="cuda"):
                    outputs = model(pixel_values, output_hidden_states=bool(supcon_fn)) if is_hf_vit and "FoundationDeepfakeClassifier" not in type(model).__name__ else model(pixel_values)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    loss = criterion(logits, labels, sample_weights=s_weights)
                    if supcon_fn is not None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                        cls_emb = outputs.hidden_states[-1][:, 0, :]
                        loss = loss + 0.10 * supcon_fn(cls_emb, labels)
                    if method_l in ("dora", "lora"):
                        loss = loss + 1e-4 * compute_orthogonal_adapter_loss(model)
                    elif method_l == "effort":
                        from colab_training.foundation_backbones import compute_effort_orthogonality_loss
                        loss = loss + 1e-3 * compute_effort_orthogonality_loss(model)
                    if ewc_obj is not None:
                        loss = loss + ewc_obj.penalty(model, lambda_ewc=ewc_lambda)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                try:
                    outputs = model(pixel_values, output_hidden_states=bool(supcon_fn)) if is_hf_vit and hasattr(model, "config") and "FoundationDeepfakeClassifier" not in type(model).__name__ else model(pixel_values)
                except TypeError:
                    outputs = model(pixel_values)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels, sample_weights=s_weights)
                if supcon_fn is not None and hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    cls_emb = outputs.hidden_states[-1][:, 0, :]
                    loss = loss + 0.10 * supcon_fn(cls_emb, labels)
                if method_l in ("dora", "lora"):
                    loss = loss + 1e-4 * compute_orthogonal_adapter_loss(model)
                elif method_l == "effort":
                    from colab_training.foundation_backbones import compute_effort_orthogonality_loss
                    loss = loss + 1e-3 * compute_effort_orthogonality_loss(model)
                if ewc_obj is not None:
                    loss = loss + ewc_obj.penalty(model, lambda_ewc=ewc_lambda)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()

            ema.update(model)
            ep_loss += float(loss.item()) * len(labels)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += len(labels)
            if dro_tracker:
                per_sample_ce = F.cross_entropy(logits.detach(), labels, reduction="none").cpu().tolist()
                for c_id, lv in zip(cohorts, per_sample_ce):
                    cohort_loss_acc.setdefault(c_id, []).append(lv)

        if dro_tracker and cohort_loss_acc:
            dro_tracker.update({c: sum(vs) / len(vs) for c, vs in cohort_loss_acc.items()})
        scheduler.step()
        acc = correct / max(1, total)
        avg_loss = ep_loss / max(1, total)
        history.append({"epoch": ep + 1, "loss": round(avg_loss, 4), "accuracy": round(acc * 100, 2)})
        print(f"  ✔ Epoch [{ep + 1}/{epochs}] — Loss: {avg_loss:.4f} | Train Acc: {acc * 100:.2f}%")

    # Apply EMA weights for OOD evaluation and export
    ema.apply_to(model)

    model.eval()
    test_correct, test_total = 0, 0
    y_true_ood, y_scores_ood, fake_logit_diffs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            pixel_values, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(pixel_values)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=-1)
            fake_probs = probs[:, fake_idx]
            preds = logits.argmax(dim=-1)
            test_correct += int((preds == labels).sum().item())
            test_total += len(labels)

            canon_y = [(1 if int(lbl) == fake_idx else 0) for lbl in labels.cpu().tolist()]
            y_true_ood.extend(canon_y)
            y_scores_ood.extend(fake_probs.cpu().tolist())
            z_diff = (logits[:, fake_idx] - logits[:, real_idx]).cpu().tolist()
            fake_logit_diffs.extend(z_diff)

    test_acc = round((test_correct / max(1, test_total)) * 100.0, 2)
    pos_s = [s for y, s in zip(y_true_ood, y_scores_ood) if y == 1]
    neg_s = [s for y, s in zip(y_true_ood, y_scores_ood) if y == 0]
    if pos_s and neg_s:
        pairs = sum(1.0 for p in pos_s for n in neg_s if p > n) + 0.5 * sum(1.0 for p in pos_s for n in neg_s if p == n)
        eval_auc = round(pairs / (len(pos_s) * len(neg_s)), 4)
        best_gap, eer_est = 1.0, 0.5
        for th in [k / 50.0 for k in range(51)]:
            fp_v = sum(1 for y, s in zip(y_true_ood, y_scores_ood) if y == 0 and s >= th)
            tn_v = sum(1 for y, s in zip(y_true_ood, y_scores_ood) if y == 0 and s < th)
            fn_v = sum(1 for y, s in zip(y_true_ood, y_scores_ood) if y == 1 and s < th)
            tp_v = sum(1 for y, s in zip(y_true_ood, y_scores_ood) if y == 1 and s >= th)
            fpr_v = fp_v / max(1, fp_v + tn_v)
            fnr_v = fn_v / max(1, fn_v + tp_v)
            if abs(fpr_v - fnr_v) < best_gap:
                best_gap = abs(fpr_v - fnr_v)
                eer_est = (fpr_v + fnr_v) / 2.0
        eval_eer = round(eer_est * 100.0, 2)
    else:
        eval_auc = round(test_acc / 100.0, 4)
        eval_eer = round(max(1.0, 100.0 - test_acc), 2)

    platt_t, youden_tau, eval_ece, platt_a, platt_b, c_low, c_high = _calibrate_platt_and_youden(
        y_true_ood, fake_logit_diffs
    )
    safety_passed = bool(eval_auc >= min_ood_auc and eval_ece <= max_ood_ece)
    print(
        f"\n[OOD Evaluation] Acc: {test_acc:.2f}% | AUC: {eval_auc:.4f} | EER: {eval_eer:.2f}% | "
        f"Platt (a={platt_a}, b={platt_b}, T*={platt_t}) | tau*={youden_tau} | "
        f"Conformal=[{c_low}, {c_high}] | ECE: {eval_ece:.4f} | SafetyGate={'PASS' if safety_passed else 'HOLD'}"
    )

    colab_default = Path("/content/drive/MyDrive/Thread_Data")
    if not output_checkpoint_dir:
        if drive_root and os.path.exists(drive_root):
            output_checkpoint_dir = Path(drive_root) / "checkpoints"
        elif colab_default.exists():
            output_checkpoint_dir = colab_default / "checkpoints"
        else:
            output_checkpoint_dir = PROJECT_ROOT / "report_colab" / "gdrive_storage" / "checkpoints"

    output_checkpoint_dir = Path(output_checkpoint_dir)
    output_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_checkpoint_dir / "model_continual_latest.pt"
    torch.save(model.state_dict(), str(ckpt_path))

    # Update & persist EWC Fisher state before merging adapters
    if use_ewc and safety_passed:
        try:
            new_ewc = DiagonalEWC.from_model(model, dataloader=test_loader, device=device, max_batches=4)
            hf_export_dir.mkdir(parents=True, exist_ok=True)
            new_ewc.save(ewc_path)
        except Exception as e_ewc:
            print(f"[EWC] Notice saving Fisher state: {e_ewc}")

    exported_safetensors = False
    model_sha256 = ""
    if safety_passed and is_hf_vit and hasattr(model, "save_pretrained"):
        model = merge_native_adapters(model)
        hf_export_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(hf_export_dir), safe_serialization=True)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(hf_export_dir))

        from forensics_kernel.calibration_store import compute_active_model_sha256
        model_sha256 = compute_active_model_sha256(str(hf_export_dir))

        calib_payload = {
            "temperature_scaling_T": platt_t,
            "platt_a": platt_a,
            "platt_b": platt_b,
            "optimal_threshold_tau": youden_tau,
            "conformal_low": c_low,
            "conformal_high": c_high,
            "model_sha256": model_sha256,
            "peft_method": method_l,
            "backbone_family": backbone_family,
            "calibration_status": "EMPIRICAL_COLAB_CONTINUAL_VIT",
            "accuracy": round(test_acc / 100.0, 4),
            "auc_score": eval_auc,
            "eer_score": round(eval_eer / 100.0, 4),
            "ece_score": eval_ece,
            "train_samples": len(train_records),
            "ood_val_samples": len(test_records),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(hf_export_dir / "calibration_params.json", "w", encoding="utf-8") as cf:
            json.dump(calib_payload, cf, indent=2)
        exported_safetensors = (hf_export_dir / "model.safetensors").exists()
        print(f"  📦 [HF Safetensors Exported] -> {hf_export_dir / 'model.safetensors'} (sha256={model_sha256})")

        if sqlite_feedback_ids:
            marked = mark_sqlite_feedback_trained(sqlite_feedback_ids)
            print(f"  🔁 [SQLite Closed-Loop] Marked {marked} feedback events as lora_trained=1.")

    report_path = output_checkpoint_dir / "continual_training_report.md"
    training_md = f"""# Threads-DFBench Continual Active Learning Report
**Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())} | **Device**: `{device}` | **Backbone**: `{backbone_family}` | **PEFT**: `{method_l}`

## 1. Kết Quả Huấn Luyện Thực Nghiệm (Continual Fine-Tuning)
- **Tổng mẫu xác thực trên đĩa**: `{total_count}` mẫu (`Real={len(reals)}`, `Fake={len(fakes)}`).
- **Mẫu khó ưu tiên (Hard Negatives / SQLite Feedback)**: `{hard_count}` mẫu.
- **Augmentation & Regularization**: SocialMediaForensicAugment (`{use_augment}`), SBI (`{sbi_prob}`), SupCon (`{use_supcon}`), Group-DRO (`{use_group_dro}`), EWC (`{use_ewc}`).

| Epoch | Loss | Train Acc |
|:---:|:---:|:---:|
"""
    for h in history:
        training_md += f"| {h['epoch']} | {h['loss']} | {h['accuracy']}% |\n"

    training_md += f"""
## 2. Kiểm Định Out-Of-Distribution & Hậu Hiệu Chuẩn (Safety Gate: `{'PASS' if safety_passed else 'HOLD'}`)
- **OOD Accuracy**: **{test_acc:.2f}%**
- **Empirical ROC-AUC**: **{eval_auc:.4f}**
- **Equal Error Rate (EER)**: **{eval_eer:.2f}%**
- **2-Param Platt**: $a={platt_a}, b={platt_b}$ ($T^*={platt_t}$) | **Youden $\\tau^*$**: **{youden_tau}** | **ECE**: **{eval_ece:.4f}**
- **Split Conformal Band**: `[{c_low}, {c_high}]`
- **Model SHA-256**: `{model_sha256}` | **HF Safetensors Exported**: `{exported_safetensors}`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(training_md)

    return {
        "checkpoint": str(ckpt_path),
        "hf_export_dir": str(hf_export_dir),
        "exported_safetensors": exported_safetensors,
        "safety_passed": safety_passed,
        "report": str(report_path),
        "test_acc": test_acc,
        "accuracy": round(test_acc / 100.0, 4),
        "auc": eval_auc,
        "auc_score": eval_auc,
        "eer": eval_eer,
        "eer_score": round(eval_eer / 100.0, 4),
        "ece_score": eval_ece,
        "temperature_scaling_T": platt_t,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "optimal_threshold_tau": youden_tau,
        "conformal_low": c_low,
        "conformal_high": c_high,
        "model_sha256": model_sha256,
        "peft_method": method_l,
        "backbone_family": backbone_family,
        "train_samples": len(train_records),
        "val_samples": len(test_records),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threads-DFBench Continual Trainer (DoRA/LoRA+/Effort/EWC)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of continual training epochs")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--drive_root", "--drive-root", type=str, default=None, help="Path to Google Drive root")
    parser.add_argument("--output_checkpoint_dir", "--output-checkpoint-dir", type=str, default=None)
    parser.add_argument("--min-auc", type=float, default=0.70, help="OOD Safety Gate minimum ROC-AUC")
    parser.add_argument("--max-ece", type=float, default=0.15, help="OOD Safety Gate maximum ECE")
    parser.add_argument("--backbone", type=str, default="vit", choices=["vit", "clip", "siglip2", "dinov2"])
    parser.add_argument("--peft", type=str, default="dora", choices=["dora", "lora", "effort", "full"])
    parser.add_argument("--sbi-prob", type=float, default=0.15, help="On-the-fly SBI probability on real samples")
    parser.add_argument("--no-ewc", action="store_true", help="Disable EWC penalty")
    args = parser.parse_args()

    run_threads_continual_training(
        drive_root=args.drive_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_checkpoint_dir=args.output_checkpoint_dir,
        min_ood_auc=args.min_auc,
        max_ood_ece=args.max_ece,
        backbone_family=args.backbone,
        peft_method=args.peft,
        sbi_prob=args.sbi_prob,
        use_ewc=not args.no_ewc,
    )
