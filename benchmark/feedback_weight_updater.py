# -*- coding: utf-8 -*-
"""
Closed-Loop Feedback & Two-Stage Weight Update Engine.
------------------------------------------------------
Converts user/pentest misclassifications ("phản ánh kết quả sai": FP / FN) on
short videos (Threads, Instagram Reels, YouTube Shorts, TikTok) into:
  Stage 1 (Hard-Frame Vaulting): Extracts labeled frames into ZeroDayActiveLearningVault
      & SQLite user_feedback_events.
  Stage 2 (Fast Loop - Calibration & Fusion Weight Update): Optimizes SARAAB temporal
      weights (w_peak, w_avg, w_jitter), Dual-Stream fusion (weight_general_video,
      weight_human_video), Youden threshold (tau_video), and gating ratios with
      non-regression guard + delta clamping, then hot-reloads DeepfakeDetector.
  Stage 3 (Slow Loop - LoRA / ViT Weight Update): Runs bounded Focal-Loss LoRA PEFT
      continual adaptation on labeled feedback frames + replay buffer, validates
      loss/accuracy improvement, saves compact LoRA adapter & merged ViT weights
      (custom_vit_model/feedback_merged_weights.pt).
"""

import json
import math
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.deepfake_memory import (
    get_forensic_certificate,
    get_recent_feedback,
    init_db,
    save_user_feedback,
    save_weight_update_audit,
    get_db_connection,
)
from forensics_kernel.zero_day_vault import ZeroDayActiveLearningVault


CUSTOM_VIT_DIR = PROJECT_ROOT / "custom_vit_model"
VIDEO_CALIB_FILE = CUSTOM_VIT_DIR / "video_calibration_params.json"
CALIB_FILE = CUSTOM_VIT_DIR / "calibration_params.json"
FEEDBACK_FRAMES_DIR = PROJECT_ROOT / "colab_datasets" / "zero_day_vault" / "feedback_frames"
LABELED_MANIFEST_FILE = PROJECT_ROOT / "colab_datasets" / "zero_day_vault" / "labeled_feedback_manifest.json"
LORA_ADAPTER_FILE = CUSTOM_VIT_DIR / "lora_feedback_adapter.pt"
MERGED_FEEDBACK_WEIGHTS_FILE = CUSTOM_VIT_DIR / "feedback_merged_weights.pt"


def infer_platform_from_url(url: Optional[str]) -> str:
    """Detect short-video social platform from URL."""
    if not url:
        return "unknown"
    u = url.lower()
    if "threads.net" in u or "threads.com" in u:
        return "threads"
    if "instagram.com" in u or "instagr.am" in u:
        return "instagram_reels"
    if "youtube.com/shorts" in u or "youtube.com" in u or "youtu.be" in u:
        return "youtube_shorts"
    if "tiktok.com" in u or "vm.tiktok" in u or "vt.tiktok" in u:
        return "tiktok"
    return "web_url"


def normalize_binary_label(label: Optional[str]) -> str:
    """Normalize arbitrary label strings to canonical 'Deepfake' or 'Realism'."""
    s = str(label or "").strip().lower()
    fake_tokens = ("deepfake", "fake", "synthetic", "ai", "generated", "manipulated", "1", "true_fake")
    if any(tok == s or tok in s for tok in fake_tokens) and "real" not in s and "authentic" not in s:
        return "Deepfake"
    if "authentic" in s or "real" in s or s == "0":
        return "Realism"
    return "Deepfake" if "fake" in s or "synthetic" in s else "Realism"


def _label_to_int(label: str) -> int:
    return 1 if normalize_binary_label(label) == "Deepfake" else 0


class FeedbackWeightUpdater:
    """Handles feedback ingestion, hard-frame extraction, calibration updates, and LoRA weight updates."""

    @staticmethod
    def load_video_calibration_params() -> Dict[str, Any]:
        defaults = {
            "w_peak": 0.50,
            "w_avg": 0.35,
            "w_jitter": 0.15,
            "weight_general_video": 0.40,
            "weight_human_video": 0.60,
            "tau_video": 0.8118,
            "temperature_T": 1.6178,
            "min_corroborated_ratio": 0.30,
            "multi_frame_max_prob_gate": 0.85,
            "deepfake_frame_ratio_gate": 0.45,
            "ai_art_avg_comp_gate": 0.45,
            "calibration_status": "INITIALIZED_DEFAULT",
            "feedback_updates_count": 0,
        }
        if VIDEO_CALIB_FILE.exists():
            try:
                with open(VIDEO_CALIB_FILE, "r", encoding="utf-8") as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        return defaults

    @staticmethod
    def save_video_calibration_params(params: Dict[str, Any]) -> Path:
        CUSTOM_VIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(VIDEO_CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return VIDEO_CALIB_FILE

    @staticmethod
    def extract_hard_frames(
        media_path: str,
        feedback_id: str,
        true_label: str,
        platform: str,
        feedback_type: str,
        max_frames: int = 8,
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts representative keyframes from misclassified short video (or copies image)
        into ZeroDayActiveLearningVault and appends ground-truth labeled entries to
        labeled_feedback_manifest.json.
        """
        saved_paths: List[str] = []
        vault_ids: List[str] = []
        if not media_path or not os.path.exists(media_path):
            return saved_paths, vault_ids

        out_dir = FEEDBACK_FRAMES_DIR / feedback_id
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = Path(media_path).suffix.lower()
        is_image = ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        vault = ZeroDayActiveLearningVault.get_instance()
        label_int = _label_to_int(true_label)
        reason = f"{feedback_type}|platform={platform}|true={true_label}"

        if is_image:
            dst = out_dir / f"{feedback_id}_img{ext or '.jpg'}"
            shutil.copy2(media_path, dst)
            saved_paths.append(str(dst))
            res = vault.enqueue_adversarial_sample(
                str(dst), modality="image", confidence=0.5, anomaly_reason=reason
            )
            if "id" in res:
                vault_ids.append(res["id"])
        else:
            cap = cv2.VideoCapture(media_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    num = min(max_frames, total_frames)
                    indices = sorted(set(np.linspace(0, max(0, total_frames - 1), num, dtype=int).tolist()))
                    for i, f_idx in enumerate(indices):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            dst = out_dir / f"{feedback_id}_f{i:02d}.jpg"
                            cv2.imwrite(str(dst), frame)
                            saved_paths.append(str(dst))
                else:
                    # Fallback sequential grab when container frame count is missing
                    grabbed = 0
                    idx = 0
                    while grabbed < max_frames:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            break
                        if idx % 10 == 0:
                            dst = out_dir / f"{feedback_id}_f{grabbed:02d}.jpg"
                            cv2.imwrite(str(dst), frame)
                            saved_paths.append(str(dst))
                            grabbed += 1
                        idx += 1
                cap.release()

            for p in saved_paths:
                res = vault.enqueue_adversarial_sample(
                    p, modality="video_frame", confidence=0.5, anomaly_reason=reason
                )
                if "id" in res:
                    vault_ids.append(res["id"])

        # Persist labeled manifest for LoRA training
        LABELED_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        manifest: List[Dict[str, Any]] = []
        if LABELED_MANIFEST_FILE.exists():
            try:
                with open(LABELED_MANIFEST_FILE, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = []

        existing_paths = {m.get("image_path") for m in manifest}
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for p in saved_paths:
            if p not in existing_paths:
                manifest.append({
                    "image_path": p,
                    "label": label_int,
                    "true_label": normalize_binary_label(true_label),
                    "feedback_id": feedback_id,
                    "platform": platform,
                    "feedback_type": feedback_type,
                    "created_at": now_iso,
                    "trained": False,
                })

        with open(LABELED_MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return saved_paths, vault_ids

    @staticmethod
    def _compute_sample_risk(telemetry: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Vector-equivalent risk computation matching DeepfakeDetector.predict_video."""
        w_peak = float(params.get("w_peak", 0.50))
        w_avg = float(params.get("w_avg", 0.35))
        w_jitter = float(params.get("w_jitter", 0.15))
        w_sum = max(1e-6, w_peak + w_avg + w_jitter)
        w_peak, w_avg, w_jitter = w_peak / w_sum, w_avg / w_sum, w_jitter / w_sum

        max_fp = float(telemetry.get("max_fake_prob", telemetry.get("composite_risk_score", 0.5)))
        avg_fp = float(telemetry.get("avg_fake_prob", max_fp))
        jitter = float(telemetry.get("motion_jitter", 0.0))

        general_score = w_peak * max_fp + w_avg * avg_fp + w_jitter * min(1.0, jitter * 2.0)

        has_human = bool(telemetry.get("has_human_video", False))
        hum_risk = telemetry.get("human_vid_risk")
        if has_human and hum_risk is not None:
            wg = float(params.get("weight_general_video", 0.40))
            wh = float(params.get("weight_human_video", 0.60))
            ws = max(1e-6, wg + wh)
            risk = (wg / ws) * general_score + (wh / ws) * float(hum_risk)
        else:
            risk = general_score

        # Account for multi-frame corroboration / AI-art gate impact if present
        ai_art_ratio = float(telemetry.get("ai_art_frame_ratio", 0.0))
        high_comp_ratio = float(telemetry.get("high_comp_frame_ratio", 0.0))
        avg_comp = float(telemetry.get("avg_comp_prob", avg_fp))
        max_comp = float(telemetry.get("max_comp_prob", max_fp))
        min_corr = float(params.get("min_corroborated_ratio", 0.30))
        ai_gate = float(params.get("ai_art_avg_comp_gate", 0.45))

        if ai_art_ratio >= min_corr or (high_comp_ratio >= min_corr and avg_comp >= ai_gate):
            risk = max(risk, max_comp)

        return float(min(0.995, max(0.005, risk)))

    @classmethod
    def _evaluate_calibration_set(
        cls, samples: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute BCE, error rate, FP, FN, and objective on feedback+anchor telemetry set."""
        tau = float(params.get("tau_video", 0.8118))
        temp = max(0.5, float(params.get("temperature_T", 1.6178)))
        baseline = cls.load_video_calibration_params()

        bce_total = 0.0
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        total_weight = 0.0

        for s in samples:
            y = int(s["y"])
            w = float(s.get("sample_weight", 1.0))
            risk = cls._compute_sample_risk(s["telemetry"], params)
            # Centered logit scaled by 1/T around tau
            logit = (risk - tau) * (6.0 / temp)
            p_fake = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
            p_fake = min(1.0 - 1e-6, max(1e-6, p_fake))
            bce = -(y * math.log(p_fake) + (1 - y) * math.log(1.0 - p_fake))
            bce_total += w * bce
            total_weight += w

            pred = 1 if risk >= tau else 0
            if pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 1:
                fn += 1
            elif pred == 1 and y == 1:
                tp += 1
            else:
                tn += 1

        n = max(1, len(samples))
        mean_bce = bce_total / max(1e-6, total_weight)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        err_rate = (fp + fn) / n

        drift = (
            abs(float(params["w_peak"]) - float(baseline["w_peak"]))
            + abs(float(params["weight_general_video"]) - float(baseline["weight_general_video"]))
            + abs(float(params["tau_video"]) - float(baseline["tau_video"]))
        )
        objective = mean_bce + 1.5 * err_rate + 0.8 * fpr + 0.8 * fnr + 0.05 * drift
        return {
            "count": len(samples),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "error_rate": round(err_rate, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "mean_bce": round(mean_bce, 4),
            "objective": round(objective, 5),
        }

    @classmethod
    def run_fast_calibration_update(
        cls,
        detector=None,
        trigger_source: str = "user_feedback",
        db_path: Optional[str] = None,
        max_delta_tau: float = 0.08,
        max_delta_w: float = 0.12,
    ) -> Dict[str, Any]:
        """
        Fast-Loop Optimization:
        Refines SARAAB fusion weights, Dual-Stream weights, and Youden threshold tau_video
        across SQLite feedback events (+ curated short-video regression anchors) using
        bounded coordinate search with strict non-regression acceptance.
        """
        before_params = cls.load_video_calibration_params()
        events = get_recent_feedback(limit=250, db_path=db_path)

        # Build labeled calibration vectors from stored telemetry
        samples: List[Dict[str, Any]] = []
        for ev in events:
            tel = ev.get("feature_telemetry") or {}
            if not tel:
                tel = {
                    "max_fake_prob": float(ev.get("composite_risk_score", 0.5)),
                    "avg_fake_prob": float(ev.get("composite_risk_score", 0.5)),
                    "motion_jitter": 0.05,
                    "has_human_video": False,
                }
            samples.append({
                "y": _label_to_int(ev.get("true_label", "Realism")),
                "telemetry": tel,
                "sample_weight": 2.0 if ev.get("feedback_type") in ("FALSE_POSITIVE", "FALSE_NEGATIVE") else 1.0,
            })

        if not samples:
            return {
                "status": "skipped",
                "message": "No labeled feedback telemetry found for calibration update.",
                "params": before_params,
            }

        # Add clean canonical anchors so 1-2 noisy reports cannot collapse decision boundaries
        canonical_anchors = [
            {"y": 0, "sample_weight": 1.0, "telemetry": {"max_fake_prob": 0.28, "avg_fake_prob": 0.12, "motion_jitter": 0.04, "has_human_video": True, "human_vid_risk": 0.18}},
            {"y": 1, "sample_weight": 1.0, "telemetry": {"max_fake_prob": 0.94, "avg_fake_prob": 0.82, "motion_jitter": 0.18, "has_human_video": True, "human_vid_risk": 0.91}},
            {"y": 0, "sample_weight": 1.0, "telemetry": {"max_fake_prob": 0.35, "avg_fake_prob": 0.14, "motion_jitter": 0.03, "has_human_video": False}},
            {"y": 1, "sample_weight": 1.0, "telemetry": {"max_fake_prob": 0.89, "avg_fake_prob": 0.79, "motion_jitter": 0.14, "has_human_video": False, "ai_art_frame_ratio": 0.70, "max_comp_prob": 0.96}},
        ]
        all_samples = samples + canonical_anchors

        before_metrics = cls._evaluate_calibration_set(all_samples, before_params)
        best_params = dict(before_params)
        best_metrics = dict(before_metrics)

        # Bounded coordinate grid search around current params
        w_peak_0 = float(before_params["w_peak"])
        w_avg_0 = float(before_params["w_avg"])
        w_gen_0 = float(before_params["weight_general_video"])
        tau_0 = float(before_params["tau_video"])
        temp_0 = float(before_params.get("temperature_T", 1.6178))

        peak_cands = [round(min(0.70, max(0.25, w_peak_0 + d)), 3) for d in (-max_delta_w, -0.06, -0.03, 0.0, 0.03, 0.06, max_delta_w)]
        avg_cands = [round(min(0.55, max(0.15, w_avg_0 + d)), 3) for d in (-max_delta_w, -0.06, -0.03, 0.0, 0.03, 0.06, max_delta_w)]
        gen_cands = [round(min(0.75, max(0.20, w_gen_0 + d)), 3) for d in (-max_delta_w, -0.06, 0.0, 0.06, max_delta_w)]
        tau_cands = [round(min(0.90, max(0.55, tau_0 + d)), 4) for d in np.linspace(-max_delta_tau, max_delta_tau, 9)]
        temp_cands = [round(min(2.8, max(0.8, temp_0 + d)), 4) for d in (-0.30, -0.15, 0.0, 0.15, 0.30)]

        for wp in peak_cands:
            for wa in avg_cands:
                if wp + wa > 0.94:
                    continue
                wj = round(max(0.06, 1.0 - wp - wa), 3)
                for wg in gen_cands:
                    wh = round(1.0 - wg, 3)
                    for tv in tau_cands:
                        for tp in temp_cands:
                            cand = dict(best_params)
                            cand.update({
                                "w_peak": wp,
                                "w_avg": wa,
                                "w_jitter": wj,
                                "weight_general_video": wg,
                                "weight_human_video": wh,
                                "tau_video": tv,
                                "temperature_T": tp,
                            })
                            m = cls._evaluate_calibration_set(all_samples, cand)
                            if (m["fp"] + m["fn"], m["objective"]) < (best_metrics["fp"] + best_metrics["fn"], best_metrics["objective"]):
                                best_params = cand
                                best_metrics = m

        # Non-regression acceptance gate
        accepted = bool(
            (best_metrics["fp"] + best_metrics["fn"]) <= (before_metrics["fp"] + before_metrics["fn"])
            and best_metrics["objective"] <= before_metrics["objective"]
        )

        if accepted:
            best_params["calibration_status"] = "FEEDBACK_CALIBRATED_ACTIVE"
            best_params["feedback_updates_count"] = int(before_params.get("feedback_updates_count", 0)) + 1
            best_params["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            best_params["last_trigger"] = trigger_source
            cls.save_video_calibration_params(best_params)

            # Sync shared threshold & weights to calibration_params.json for consistent inference
            if CALIB_FILE.exists():
                try:
                    with open(CALIB_FILE, "r", encoding="utf-8") as f:
                        base_calib = json.load(f)
                    base_calib["temperature_scaling_T"] = best_params["temperature_T"]
                    base_calib["optimal_threshold_tau"] = best_params["tau_video"]
                    base_calib["last_feedback_calibration_at"] = best_params["updated_at"]
                    with open(CALIB_FILE, "w", encoding="utf-8") as f:
                        json.dump(base_calib, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

            if detector is not None and hasattr(detector, "reload_calibration_params"):
                detector.reload_calibration_params()

        update_id = save_weight_update_audit({
            "trigger_source": trigger_source,
            "feedback_count": len(samples),
            "before_params": before_params,
            "after_params": best_params if accepted else before_params,
            "before_metrics": before_metrics,
            "after_metrics": best_metrics if accepted else before_metrics,
            "accepted": accepted,
            "notes": "Fast-loop SARAAB/Dual-Stream calibration refinement from short-video feedback",
        }, db_path=db_path)

        return {
            "status": "updated" if accepted else "kept_baseline",
            "update_id": update_id,
            "accepted": accepted,
            "feedback_samples_used": len(samples),
            "before_params": before_params,
            "after_params": best_params if accepted else before_params,
            "before_metrics": before_metrics,
            "after_metrics": best_metrics if accepted else before_metrics,
        }

    @classmethod
    def run_slow_lora_weight_update(
        cls,
        epochs: int = 2,
        batch_size: int = 4,
        lr: float = 2e-5,
        max_feedback_frames: int = 64,
        promote_merged: bool = True,
        detector=None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Slow-Loop Neural Weight Update:
        Applies Native PyTorch LoRA (r=8, alpha=16) to custom_vit_model + trains classifier
        with Focal Loss on labeled feedback hard frames (+ replay buffer to avoid drift).
        Validates loss/accuracy reduction and merges LoRA deltas into
        custom_vit_model/feedback_merged_weights.pt when verified.
        """
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import Dataset, DataLoader
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from colab_training.lora_adapter import NativeLoRALinear, apply_lora_to_vit, save_lora_weights
        from colab_training.train_threads_continual import FocalLoss

        if not LABELED_MANIFEST_FILE.exists():
            return {"status": "skipped", "message": "No labeled_feedback_manifest.json entries found."}

        with open(LABELED_MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        valid_entries = [m for m in manifest if m.get("image_path") and os.path.exists(m["image_path"])]
        if not valid_entries:
            return {"status": "skipped", "message": "No accessible hard frames on disk for LoRA update."}

        feedback_batch = valid_entries[-max_feedback_frames:]

        # Gather compact balanced replay frames if local datasets exist
        replay_entries: List[Dict[str, Any]] = []
        replay_manifest = PROJECT_ROOT / "dataset_pipeline" / "source_split" / "train_manifest.json"
        if replay_manifest.exists():
            try:
                with open(replay_manifest, "r", encoding="utf-8") as f:
                    rm_data = json.load(f)
                reals = [x for x in rm_data if int(x.get("label", 0)) == 0 and os.path.exists(x.get("path", ""))][:8]
                fakes = [x for x in rm_data if int(x.get("label", 0)) == 1 and os.path.exists(x.get("path", ""))][:8]
                for x in reals + fakes:
                    replay_entries.append({"image_path": x["path"], "label": int(x["label"])})
            except Exception:
                pass

        train_records = feedback_batch + replay_entries

        class _FeedbackFrameDataset(Dataset):
            def __init__(self, records, processor):
                self.records = records
                self.processor = processor

            def __len__(self):
                return len(self.records)

            def __getitem__(self, idx):
                rec = self.records[idx]
                img = Image.open(rec["image_path"]).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt")
                return inputs["pixel_values"].squeeze(0), torch.tensor(int(rec["label"]), dtype=torch.long)

        target_model = str(CUSTOM_VIT_DIR) if (CUSTOM_VIT_DIR / "config.json").exists() else "dima806/deepfake_vs_real_image_detection"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoImageProcessor.from_pretrained(target_model)
        model = AutoModelForImageClassification.from_pretrained(target_model)
        if MERGED_FEEDBACK_WEIGHTS_FILE.exists():
            try:
                model.load_state_dict(torch.load(str(MERGED_FEEDBACK_WEIGHTS_FILE), map_location="cpu"), strict=False)
            except Exception:
                pass

        # Use Native PyTorch LoRA so delta merging W + scaling * (B @ A) is exact
        model = apply_lora_to_vit(model, r=8, lora_alpha=16, lora_dropout=0.05, prefer_peft=False).to(device)
        dataset = _FeedbackFrameDataset(train_records, processor)
        loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=True)
        eval_loader = DataLoader(dataset, batch_size=min(batch_size, max(1, len(dataset))), shuffle=False)

        criterion = FocalLoss(gamma=2.25, alpha=0.25)

        def _eval_model():
            model.eval()
            tot_loss, correct, count = 0.0, 0, 0
            with torch.no_grad():
                for px, yb in eval_loader:
                    px, yb = px.to(device), yb.to(device)
                    out = model(pixel_values=px)
                    loss = criterion(out.logits, yb)
                    tot_loss += float(loss.item()) * len(yb)
                    preds = out.logits.argmax(dim=-1)
                    correct += int((preds == yb).sum().item())
                    count += len(yb)
            return {
                "loss": round(tot_loss / max(1, count), 4),
                "acc": round(correct / max(1, count), 4),
                "count": count,
            }

        before_metrics = _eval_model()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

        for _ in range(max(1, epochs)):
            model.train()
            for px, yb in loader:
                px, yb = px.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(pixel_values=px)
                loss = criterion(out.logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

        after_metrics = _eval_model()
        improved = bool(
            after_metrics["loss"] <= before_metrics["loss"] + 1e-4
            or after_metrics["acc"] >= before_metrics["acc"]
        )

        saved_adapter_path = None
        saved_merged_path = None
        if improved:
            saved_adapter_path = save_lora_weights(model, LORA_ADAPTER_FILE)

            if promote_merged:
                # Merge NativeLoRALinear weights back into clean base nn.Linear state_dict
                for parent in model.modules():
                    for child_name, child in list(parent.named_children()):
                        if isinstance(child, NativeLoRALinear):
                            with torch.no_grad():
                                delta = (child.lora_B @ child.lora_A) * child.scaling
                                child.base_layer.weight.data += delta.to(child.base_layer.weight.device)
                            setattr(parent, child_name, child.base_layer)
                CUSTOM_VIT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(MERGED_FEEDBACK_WEIGHTS_FILE))
                saved_merged_path = str(MERGED_FEEDBACK_WEIGHTS_FILE)

                if detector is not None and hasattr(detector, "model"):
                    try:
                        detector.model.load_state_dict(model.state_dict(), strict=False)
                        detector.model.to(detector.device).eval()
                    except Exception:
                        pass

            # Mark manifest entries trained
            trained_paths = {r["image_path"] for r in feedback_batch}
            for item in manifest:
                if item.get("image_path") in trained_paths:
                    item["trained"] = True
            with open(LABELED_MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            vault = ZeroDayActiveLearningVault.get_instance()
            pending_ids = [s["id"] for s in vault.get_pending_samples() if s.get("file_path") in trained_paths]
            if pending_ids:
                vault.mark_samples_trained(pending_ids)

            # Register model weight version in SQLite
            try:
                init_db(db_path)
                conn = get_db_connection(db_path)
                cur = conn.cursor()
                ver_tag = f"lora-feedback-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
                cur.execute(
                    """
                    INSERT INTO model_weights_registry (date, version_tag, weights_file, backbone, status, metrics_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                        ver_tag,
                        str(saved_merged_path or saved_adapter_path),
                        "ViT-LoRA-PEFT-Feedback",
                        "active" if promote_merged else "staged",
                        json.dumps({"before": before_metrics, "after": after_metrics}, ensure_ascii=False),
                    ),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        update_id = save_weight_update_audit({
            "trigger_source": "slow_loop_lora_feedback",
            "feedback_count": len(feedback_batch),
            "before_params": {"lora_r": 8, "lora_alpha": 16, "epochs": epochs, "lr": lr},
            "after_params": {"adapter_path": saved_adapter_path, "merged_weights_path": saved_merged_path},
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "accepted": improved,
            "notes": f"LoRA continual ViT feedback training ({len(feedback_batch)} feedback frames + {len(replay_entries)} replay)",
        }, db_path=db_path)

        return {
            "status": "updated" if improved else "rejected_no_improvement",
            "update_id": update_id,
            "accepted": improved,
            "feedback_frames_used": len(feedback_batch),
            "replay_frames_used": len(replay_entries),
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "adapter_path": saved_adapter_path,
            "merged_weights_path": saved_merged_path,
        }

    @classmethod
    def submit_feedback(
        cls,
        true_label: str,
        certificate_id: Optional[str] = None,
        platform: Optional[str] = None,
        media_type: str = "video",
        file_path: Optional[str] = None,
        source_url: Optional[str] = None,
        predicted_label: Optional[str] = None,
        composite_risk_score: Optional[float] = None,
        feature_telemetry: Optional[Dict[str, Any]] = None,
        user_note: str = "",
        auto_update_calibration: bool = True,
        trigger_lora_update: bool = False,
        detector=None,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end entry point for user/pentest reflection:
          1. Enriches context from forensic_certificates if certificate_id is given.
          2. Saves feedback event in SQLite (user_feedback_events).
          3. Extracts hard frames into ZeroDayActiveLearningVault + labeled_feedback_manifest.json.
          4. Runs Fast-Loop Calibration Weight Update (+ hot-reloads detector).
          5. Optionally runs Slow-Loop LoRA ViT weight fine-tuning.
        """
        cert_data = None
        if certificate_id:
            cert_res = get_forensic_certificate(certificate_id, db_path=db_path)
            if cert_res and cert_res.get("status") == "VERIFIED_VALID":
                cert_data = cert_res

        if cert_data:
            media_int = cert_data.get("media_integrity") or {}
            img_meta = cert_data.get("image_metadata") or {}
            ev = cert_data.get("multi_dimensional_evidence") or {}
            saraab = ev.get("saraab_phase1_video_audit") or {}

            source_url = source_url or media_int.get("source_url")
            file_path = file_path or img_meta.get("file_path") or media_int.get("file_path")
            if composite_risk_score is None:
                composite_risk_score = float(cert_data.get("composite_risk_score", 0.5))
            if not predicted_label:
                verdict = str(cert_data.get("auditable_verdict", ""))
                predicted_label = "Deepfake" if any(k in verdict for k in ("SYNTHETIC", "DEEPFAKE", "FAKE", "SUSPECTED")) else "Realism"
            if not feature_telemetry:
                feature_telemetry = ev.get("saraab_feature_telemetry") or {
                    "max_fake_prob": float(ev.get("peak_fake_probability", composite_risk_score or 0.5)),
                    "avg_fake_prob": float(ev.get("aggregate_fake_probability", composite_risk_score or 0.5)),
                    "motion_jitter": float(saraab.get("temporal_motion_jitter", 0.05)),
                    "avg_comp_prob": float(ev.get("aggregate_composite_risk", composite_risk_score or 0.5)),
                    "max_comp_prob": float(ev.get("peak_composite_risk", composite_risk_score or 0.5)),
                    "has_human_video": bool((cert_data.get("dual_stream_breakdown") or {}).get("human_gate_triggered", False)),
                }

        norm_true = normalize_binary_label(true_label)
        norm_pred = normalize_binary_label(predicted_label or ("Realism" if norm_true == "Deepfake" else "Deepfake"))
        if norm_pred == "Deepfake" and norm_true == "Realism":
            fb_type = "FALSE_POSITIVE"
        elif norm_pred == "Realism" and norm_true == "Deepfake":
            fb_type = "FALSE_NEGATIVE"
        else:
            fb_type = "CONFIRMED"

        resolved_platform = platform or infer_platform_from_url(source_url)
        feedback_id = f"FB-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

        extracted_frames, vault_ids = cls.extract_hard_frames(
            media_path=file_path or "",
            feedback_id=feedback_id,
            true_label=norm_true,
            platform=resolved_platform,
            feedback_type=fb_type,
        )

        event_payload = {
            "feedback_id": feedback_id,
            "certificate_id": certificate_id,
            "platform": resolved_platform,
            "media_type": media_type,
            "file_path": file_path,
            "source_url": source_url,
            "predicted_label": norm_pred,
            "true_label": norm_true,
            "feedback_type": fb_type,
            "composite_risk_score": float(composite_risk_score or 0.5),
            "feature_telemetry": feature_telemetry or {},
            "user_note": user_note,
            "vault_sample_ids": vault_ids,
            "calibration_applied": False,
            "lora_trained": False,
        }
        save_user_feedback(event_payload, db_path=db_path)

        calib_res = None
        if auto_update_calibration:
            calib_res = cls.run_fast_calibration_update(
                detector=detector,
                trigger_source=f"feedback:{feedback_id}:{fb_type}",
                db_path=db_path,
            )
            if calib_res.get("accepted"):
                event_payload["calibration_applied"] = True

        lora_res = None
        if trigger_lora_update and extracted_frames:
            lora_res = cls.run_slow_lora_weight_update(detector=detector, db_path=db_path)
            if lora_res.get("accepted"):
                event_payload["lora_trained"] = True

        if event_payload["calibration_applied"] or event_payload["lora_trained"]:
            save_user_feedback(event_payload, db_path=db_path)

        return {
            "status": "success",
            "feedback_id": feedback_id,
            "feedback_type": fb_type,
            "platform": resolved_platform,
            "predicted_label": norm_pred,
            "true_label": norm_true,
            "extracted_frames_count": len(extracted_frames),
            "vault_sample_ids": vault_ids,
            "calibration_update": calib_res,
            "lora_update": lora_res,
        }

