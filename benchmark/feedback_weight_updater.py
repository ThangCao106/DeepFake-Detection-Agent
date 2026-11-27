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
    APPROVED_REVIEW_STATUSES,
    get_approved_feedback_for_weights,
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
AUDIO_CALIB_FILE = CUSTOM_VIT_DIR / "audio_calibration_params.json"
CALIB_FILE = CUSTOM_VIT_DIR / "calibration_params.json"
FEEDBACK_FRAMES_DIR = PROJECT_ROOT / "colab_datasets" / "zero_day_vault" / "feedback_frames"
LABELED_MANIFEST_FILE = PROJECT_ROOT / "colab_datasets" / "zero_day_vault" / "labeled_feedback_manifest.json"
LORA_ADAPTER_FILE = CUSTOM_VIT_DIR / "lora_feedback_adapter.pt"
MERGED_FEEDBACK_WEIGHTS_FILE = CUSTOM_VIT_DIR / "feedback_merged_weights.pt"
WEIGHT_UPDATES_REPORT_DIR = PROJECT_ROOT / "report_colab" / "weight_updates"


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

    @classmethod
    def _persist_weight_update_report_artifact(
        cls,
        *,
        update_id: str,
        modality: str,
        trigger_source: str,
        feedback_count: int,
        before_params: Dict[str, Any],
        after_params: Dict[str, Any],
        before_metrics: Dict[str, Any],
        after_metrics: Dict[str, Any],
        accepted: bool,
        notes: str,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write persistent JSON & Markdown weight-update audit reports under report_colab/weight_updates/."""
        from observability.app.calibration_audit import (
            build_weight_update_detailed_report,
            compute_metric_diffs,
            compute_param_diffs,
            render_weight_update_markdown_report,
        )

        p_diffs = compute_param_diffs(before_params, after_params)
        m_diffs = compute_metric_diffs(before_metrics, after_metrics)
        out_dir = Path(db_path).parent / "weight_updates" if db_path else WEIGHT_UPDATES_REPORT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        payload = {
            "update_id": update_id,
            "modality": modality,
            "trigger_source": trigger_source,
            "feedback_count": int(feedback_count),
            "accepted": bool(accepted),
            "status_label": "ACCEPTED" if accepted else "KEPT_BASELINE",
            "notes": notes,
            "created_at": ts_utc,
            "timestamp_utc": ts_utc,
            "before_params": before_params,
            "after_params": after_params,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "parameter_diffs": p_diffs,
            "changed_params_count": sum(1 for d in p_diffs if d.get("changed")),
            "metric_diffs": m_diffs,
        }

        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(update_id or "update"))
        art_json = out_dir / f"update_{safe_id}.json"
        art_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        consolidated = build_weight_update_detailed_report(
            repo_root=PROJECT_ROOT,
            custom_vit_dir=CUSTOM_VIT_DIR,
            db_path=Path(db_path) if db_path else None,
        )
        latest_json = out_dir / "latest_weight_update_report.json"
        latest_md = out_dir / "latest_weight_update_report.md"
        latest_json.write_text(json.dumps(consolidated, indent=2, ensure_ascii=False), encoding="utf-8")
        latest_md.write_text(render_weight_update_markdown_report(consolidated), encoding="utf-8")

        return {
            "parameter_diffs": p_diffs,
            "metric_diffs": m_diffs,
            "report_artifact_path": str(art_json),
            "latest_report_md_path": str(latest_md),
        }

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
    def load_audio_calibration_params() -> Dict[str, Any]:
        defaults = {
            "w_vocoder": 0.45,
            "w_pitch": 0.35,
            "w_phase": 0.20,
            "tau_audio": 0.8118,
            "temperature_T": 1.6178,
            "vocoder_gate": 0.72,
            "pitch_gate": 0.55,
            "calibration_status": "AUDIO_SEPARATED_ACTIVE",
            "feedback_updates_count": 0,
        }
        if AUDIO_CALIB_FILE.exists():
            try:
                with open(AUDIO_CALIB_FILE, "r", encoding="utf-8") as f:
                    defaults.update(json.load(f))
            except Exception:
                pass
        return defaults

    @staticmethod
    def save_audio_calibration_params(params: Dict[str, Any]) -> Path:
        CUSTOM_VIT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIO_CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return AUDIO_CALIB_FILE

    @staticmethod
    def extract_hard_frames(
        media_path: str,
        feedback_id: str,
        true_label: str,
        platform: str,
        feedback_type: str,
        max_frames: int = 8,
        approved: bool = False,
        review_status: str = "PENDING",
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
        is_audio = ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

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
        elif is_audio:
            dst = out_dir / f"{feedback_id}_aud{ext or '.wav'}"
            shutil.copy2(media_path, dst)
            saved_paths.append(str(dst))
            res = vault.enqueue_adversarial_sample(
                str(dst), modality="audio", confidence=0.5, anomaly_reason=reason
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
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for p in saved_paths:
            if Path(p).suffix.lower() in image_exts and p not in existing_paths:
                manifest.append({
                    "image_path": p,
                    "label": label_int,
                    "true_label": normalize_binary_label(true_label),
                    "feedback_id": feedback_id,
                    "platform": platform,
                    "feedback_type": feedback_type,
                    "created_at": now_iso,
                    "approved": bool(approved),
                    "review_status": str(review_status or "PENDING").upper(),
                    "trained": False,
                })

        with open(LABELED_MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return saved_paths, vault_ids

    @staticmethod
    def sync_manifest_approval_status(
        feedback_id: str,
        review_status: str,
        true_label: Optional[str] = None,
    ) -> int:
        """Synchronize /dfagent human-review decision into labeled_feedback_manifest.json."""
        if not LABELED_MANIFEST_FILE.exists():
            return 0
        try:
            manifest = json.loads(LABELED_MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            return 0
        if not isinstance(manifest, list):
            return 0

        r_up = str(review_status or "PENDING").upper()
        is_approved = r_up in APPROVED_REVIEW_STATUSES
        updated = 0
        for item in manifest:
            if str(item.get("feedback_id") or "") == str(feedback_id):
                item["review_status"] = r_up
                item["approved"] = is_approved
                if true_label:
                    norm_lbl = normalize_binary_label(true_label)
                    item["true_label"] = norm_lbl
                    item["label"] = _label_to_int(norm_lbl)
                updated += 1
        if updated > 0:
            LABELED_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return updated

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
        events = get_approved_feedback_for_weights(limit=250, media_type="video", db_path=db_path)

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

            if detector is not None and hasattr(detector, "reload_calibration_params"):
                detector.reload_calibration_params()

        final_after_params = best_params if accepted else before_params
        final_after_metrics = best_metrics if accepted else before_metrics
        update_id = save_weight_update_audit({
            "trigger_source": trigger_source,
            "feedback_count": len(samples),
            "before_params": before_params,
            "after_params": final_after_params,
            "before_metrics": before_metrics,
            "after_metrics": final_after_metrics,
            "accepted": accepted,
            "notes": "Fast-loop SARAAB/Dual-Stream calibration refinement from short-video feedback",
        }, db_path=db_path)

        art_info = cls._persist_weight_update_report_artifact(
            update_id=update_id,
            modality="video",
            trigger_source=trigger_source,
            feedback_count=len(samples),
            before_params=before_params,
            after_params=final_after_params,
            before_metrics=before_metrics,
            after_metrics=final_after_metrics,
            accepted=accepted,
            notes="Fast-loop SARAAB/Dual-Stream calibration refinement from short-video feedback",
            db_path=db_path,
        )

        return {
            "status": "updated" if accepted else "kept_baseline",
            "update_id": update_id,
            "accepted": accepted,
            "feedback_samples_used": len(samples),
            "before_params": before_params,
            "after_params": final_after_params,
            "before_metrics": before_metrics,
            "after_metrics": final_after_metrics,
            "parameter_diffs": art_info["parameter_diffs"],
            "metric_diffs": art_info["metric_diffs"],
            "report_artifact_path": art_info["report_artifact_path"],
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

        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        valid_entries = [
            m for m in manifest
            if m.get("image_path") and os.path.exists(m["image_path"]) and Path(m["image_path"]).suffix.lower() in image_exts
        ]
        # Strict /dfagent approval filter: ONLY approved feedback IDs can enter Slow-Loop LoRA updates
        approved_fids = {
            str(e.get("feedback_id") or "")
            for e in get_approved_feedback_for_weights(limit=500, db_path=db_path)
            if e.get("feedback_id")
        }
        valid_entries = [
            m for m in valid_entries
            if (str(m.get("feedback_id") or "") in approved_fids)
            or (bool(m.get("approved")) and str(m.get("review_status") or "").upper() in APPROVED_REVIEW_STATUSES)
        ]
        if not valid_entries:
            return {
                "status": "skipped_no_approved_samples",
                "accepted": False,
                "message": "Chưa có mẫu nào được Quản trị viên phê duyệt trên /dfagent để cập nhật LoRA.",
            }

        feedback_batch = valid_entries[-max_feedback_frames:]

        # Gather balanced replay frames to prevent catastrophic single-class forgetting
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

        real_cnt = sum(1 for r in feedback_batch + replay_entries if int(r["label"]) == 0)
        fake_cnt = sum(1 for r in feedback_batch + replay_entries if int(r["label"]) == 1)
        if real_cnt < 4:
            for d_real in (
                PROJECT_ROOT / "crawler" / "verified_threads_seeds",
                PROJECT_ROOT / "report_colab" / "pentest" / "01_raw_originals",
            ):
                if d_real.exists():
                    for p in sorted(d_real.glob("*.jpg"))[: max(0, 4 - real_cnt)]:
                        replay_entries.append({"image_path": str(p), "label": 0})
                        real_cnt += 1
                if real_cnt >= 4:
                    break
        if fake_cnt < 4:
            for d_fake in (
                PROJECT_ROOT / "report_colab" / "pentest" / "02_manipulated_faceswap",
                PROJECT_ROOT / "report_colab" / "pentest" / "03_manipulated_openweight",
                PROJECT_ROOT / "scratch" / "test_full_abc_out" / "03_manipulated_openweight",
            ):
                if d_fake.exists():
                    for p in sorted(d_fake.glob("*.jpg"))[: max(0, 4 - fake_cnt)]:
                        replay_entries.append({"image_path": str(p), "label": 1})
                        fake_cnt += 1
                if fake_cnt >= 4:
                    break

        all_records = feedback_batch + replay_entries
        reals_all = [r for r in all_records if int(r["label"]) == 0]
        fakes_all = [r for r in all_records if int(r["label"]) == 1]
        val_subset: List[Dict[str, Any]] = []
        train_subset: List[Dict[str, Any]] = []
        for grp in (reals_all, fakes_all):
            if len(grp) >= 2:
                n_val = max(1, int(round(0.2 * len(grp))))
                val_subset.extend(grp[:n_val])
                train_subset.extend(grp[n_val:])
            else:
                train_subset.extend(grp)
        if not val_subset:
            val_subset = list(all_records)
        if not train_subset:
            train_subset = list(all_records)

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

        has_local_vit = (
            (CUSTOM_VIT_DIR / "config.json").exists()
            and any((CUSTOM_VIT_DIR / w).exists() for w in ("model.safetensors", "pytorch_model.bin", "model.bin"))
        )
        target_model = str(CUSTOM_VIT_DIR) if has_local_vit else "dima806/deepfake_vs_real_image_detection"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_kw = {} if has_local_vit else {"token": False}
        processor = AutoImageProcessor.from_pretrained(target_model, **hf_kw)
        model = AutoModelForImageClassification.from_pretrained(target_model, **hf_kw)
        if MERGED_FEEDBACK_WEIGHTS_FILE.exists():
            try:
                model.load_state_dict(torch.load(str(MERGED_FEEDBACK_WEIGHTS_FILE), map_location="cpu"), strict=False)
            except Exception:
                pass

        # Use Native PyTorch LoRA so delta merging W + scaling * (B @ A) is exact
        model = apply_lora_to_vit(model, r=8, lora_alpha=16, lora_dropout=0.05, prefer_peft=False).to(device)
        train_dataset = _FeedbackFrameDataset(train_subset, processor)
        val_dataset = _FeedbackFrameDataset(val_subset, processor)
        loader = DataLoader(train_dataset, batch_size=min(batch_size, max(1, len(train_dataset))), shuffle=True)
        eval_loader = DataLoader(val_dataset, batch_size=min(batch_size, max(1, len(val_dataset))), shuffle=False)

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
            after_metrics["loss"] < before_metrics["loss"]
            and after_metrics["acc"] >= before_metrics["acc"]
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

        lora_before_params = {"lora_r": 8, "lora_alpha": 16, "epochs": epochs, "lr": lr}
        lora_after_params = {"lora_r": 8, "lora_alpha": 16, "epochs": epochs, "lr": lr, "adapter_path": saved_adapter_path, "merged_weights_path": saved_merged_path}
        lora_notes = f"LoRA continual ViT feedback training ({len(feedback_batch)} feedback frames + {len(replay_entries)} replay)"
        update_id = save_weight_update_audit({
            "trigger_source": "slow_loop_lora_feedback",
            "feedback_count": len(feedback_batch),
            "before_params": lora_before_params,
            "after_params": lora_after_params,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "accepted": improved,
            "notes": lora_notes,
        }, db_path=db_path)

        art_info = cls._persist_weight_update_report_artifact(
            update_id=update_id,
            modality="lora",
            trigger_source="slow_loop_lora_feedback",
            feedback_count=len(feedback_batch),
            before_params=lora_before_params,
            after_params=lora_after_params,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            accepted=improved,
            notes=lora_notes,
            db_path=db_path,
        )

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
            "parameter_diffs": art_info["parameter_diffs"],
            "metric_diffs": art_info["metric_diffs"],
            "report_artifact_path": art_info["report_artifact_path"],
        }

    @staticmethod
    def _compute_audio_sample_risk(telemetry: Dict[str, Any], params: Dict[str, Any]) -> float:
        """Vector-equivalent acoustic risk matching AudioForensicAnalyzer."""
        wv = float(params.get("w_vocoder", 0.45))
        wp = float(params.get("w_pitch", 0.35))
        wph = float(params.get("w_phase", 0.20))
        ws = max(1e-6, wv + wp + wph)
        wv, wp, wph = wv / ws, wp / ws, wph / ws

        voc = max(
            float(telemetry.get("vocoder_anomaly_score", 0.5)),
            float(telemetry.get("hf_dither_score", 0.0)),
        )
        pit = float(telemetry.get("pitch_score", 0.5))
        pha = float(telemetry.get("phase_anomaly_score", 0.2))
        if telemetry.get("is_non_speech", False):
            return 0.15
        if telemetry.get("is_telephony_bandlimited", False) and pit < 0.72:
            voc = 0.25

        risk = wv * voc + wp * pit + wph * pha
        voc_gate = float(params.get("vocoder_gate", 0.72))
        pit_gate = float(params.get("pitch_gate", 0.55))
        if (voc >= voc_gate and pit >= pit_gate) or telemetry.get("partial_splice_detected", False) or float(telemetry.get("hf_dither_score", 0.0)) >= 0.76:
            risk = max(risk, 0.86)
        return float(min(0.985, max(0.02, risk)))

    @classmethod
    def _evaluate_audio_calibration_set(
        cls, samples: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        tau = float(params.get("tau_audio", 0.8118))
        temp = max(0.5, float(params.get("temperature_T", 1.6178)))
        baseline = cls.load_audio_calibration_params()
        bce_total, total_weight = 0.0, 0.0
        tp = tn = fp = fn = 0
        for s in samples:
            y = int(s["y"])
            w = float(s.get("sample_weight", 1.0))
            risk = cls._compute_audio_sample_risk(s["telemetry"], params)
            logit = (risk - tau) * (6.0 / temp)
            p_fake = min(1.0 - 1e-6, max(1e-6, 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))))
            bce_total += w * (-(y * math.log(p_fake) + (1 - y) * math.log(1.0 - p_fake)))
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
            abs(float(params["w_vocoder"]) - float(baseline["w_vocoder"]))
            + abs(float(params["w_pitch"]) - float(baseline["w_pitch"]))
            + abs(float(params["tau_audio"]) - float(baseline["tau_audio"]))
        )
        objective = mean_bce + 1.5 * err_rate + 0.8 * fpr + 0.8 * fnr + 0.05 * drift
        return {
            "count": len(samples),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "error_rate": round(err_rate, 4),
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "mean_bce": round(mean_bce, 4),
            "objective": round(objective, 5),
        }

    @classmethod
    def run_audio_calibration_update(
        cls,
        detector=None,
        trigger_source: str = "audio_feedback_event",
        db_path: Optional[str] = None,
        max_delta_tau: float = 0.08,
        max_delta_w: float = 0.12,
    ) -> Dict[str, Any]:
        """
        Fast-loop acoustic coordinate optimization updating audio_calibration_params.json
        over stored audio feedback telemetry + canonical real/fake anchors with non-regression gate.
        """
        before_params = cls.load_audio_calibration_params()
        events = get_approved_feedback_for_weights(limit=250, media_type="audio", db_path=db_path)
        if not events:
            return {
                "status": "no_audio_feedback_samples",
                "accepted": False,
                "before_params": before_params,
                "after_params": before_params,
            }

        samples: List[Dict[str, Any]] = []
        for ev in events:
            tel = ev.get("feature_telemetry") or {}
            if not tel:
                tel = {
                    "vocoder_anomaly_score": float(ev.get("composite_risk_score", 0.5)),
                    "pitch_score": float(ev.get("composite_risk_score", 0.5)),
                    "phase_anomaly_score": 0.25,
                }
            samples.append({
                "y": _label_to_int(ev.get("true_label", "Realism")),
                "telemetry": tel,
                "sample_weight": 2.0 if ev.get("feedback_type") in ("FALSE_POSITIVE", "FALSE_NEGATIVE") else 1.0,
            })

        canonical_anchors = [
            {"y": 0, "sample_weight": 1.0, "telemetry": {"vocoder_anomaly_score": 0.05, "pitch_score": 0.14, "phase_anomaly_score": 0.22}},
            {"y": 0, "sample_weight": 1.0, "telemetry": {"vocoder_anomaly_score": 0.25, "pitch_score": 0.15, "phase_anomaly_score": 0.24, "is_telephony_bandlimited": True}},
            {"y": 1, "sample_weight": 1.0, "telemetry": {"vocoder_anomaly_score": 1.0, "pitch_score": 0.72, "phase_anomaly_score": 0.23}},
            {"y": 1, "sample_weight": 1.0, "telemetry": {"vocoder_anomaly_score": 0.10, "hf_dither_score": 0.90, "pitch_score": 0.72, "phase_anomaly_score": 0.15}},
        ]
        all_samples = samples + canonical_anchors
        before_metrics = cls._evaluate_audio_calibration_set(all_samples, before_params)
        best_params = dict(before_params)
        best_metrics = dict(before_metrics)

        wv_0 = float(before_params.get("w_vocoder", 0.45))
        wp_0 = float(before_params.get("w_pitch", 0.35))
        tau_0 = float(before_params.get("tau_audio", 0.8118))

        wv_cands = [round(min(0.65, max(0.25, wv_0 + d)), 3) for d in (-max_delta_w, -0.05, 0.0, 0.05, max_delta_w)]
        wp_cands = [round(min(0.55, max(0.20, wp_0 + d)), 3) for d in (-max_delta_w, -0.05, 0.0, 0.05, max_delta_w)]
        tau_cands = [round(min(0.90, max(0.58, tau_0 + d)), 4) for d in np.linspace(-max_delta_tau, max_delta_tau, 7)]

        for wv in wv_cands:
            for wp in wp_cands:
                if wv + wp > 0.92:
                    continue
                wph = round(max(0.08, 1.0 - wv - wp), 3)
                for ta in tau_cands:
                    cand = dict(best_params)
                    cand.update({"w_vocoder": wv, "w_pitch": wp, "w_phase": wph, "tau_audio": ta})
                    m = cls._evaluate_audio_calibration_set(all_samples, cand)
                    if (m["fp"] + m["fn"], m["objective"]) < (best_metrics["fp"] + best_metrics["fn"], best_metrics["objective"]):
                        best_params = cand
                        best_metrics = m

        accepted = bool(
            (best_metrics["fp"] + best_metrics["fn"]) <= (before_metrics["fp"] + before_metrics["fn"])
            and best_metrics["objective"] <= before_metrics["objective"]
        )

        if accepted:
            best_params["calibration_status"] = "AUDIO_FEEDBACK_CALIBRATED"
            best_params["feedback_updates_count"] = int(before_params.get("feedback_updates_count", 0)) + 1
            best_params["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            best_params["last_trigger"] = trigger_source
            cls.save_audio_calibration_params(best_params)
            if detector is not None and hasattr(detector, "reload_calibration_params"):
                detector.reload_calibration_params()

        aud_after_params = best_params if accepted else before_params
        aud_after_metrics = best_metrics if accepted else before_metrics
        update_id = save_weight_update_audit({
            "trigger_source": trigger_source,
            "feedback_count": len(samples),
            "before_params": before_params,
            "after_params": aud_after_params,
            "before_metrics": before_metrics,
            "after_metrics": aud_after_metrics,
            "accepted": accepted,
            "notes": "Fast-loop acoustic coordinate calibration refinement from audio feedback",
        }, db_path=db_path)

        art_info = cls._persist_weight_update_report_artifact(
            update_id=update_id,
            modality="audio",
            trigger_source=trigger_source,
            feedback_count=len(samples),
            before_params=before_params,
            after_params=aud_after_params,
            before_metrics=before_metrics,
            after_metrics=aud_after_metrics,
            accepted=accepted,
            notes="Fast-loop acoustic coordinate calibration refinement from audio feedback",
            db_path=db_path,
        )

        return {
            "status": "updated" if accepted else "kept_baseline",
            "update_id": update_id,
            "accepted": accepted,
            "feedback_samples_used": len(samples),
            "before_params": before_params,
            "after_params": aud_after_params,
            "before_metrics": before_metrics,
            "after_metrics": aud_after_metrics,
            "parameter_diffs": art_info["parameter_diffs"],
            "metric_diffs": art_info["metric_diffs"],
            "report_artifact_path": art_info["report_artifact_path"],
        }

    @classmethod
    def run_image_calibration_update(
        cls,
        detector=None,
        trigger_source: str = "dfagent_image_review",
        db_path: Optional[str] = None,
        max_delta_tau: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Fast-Loop calibration optimization for Image threshold (calibration_params.json)
        strictly using /dfagent-approved image feedback events + clean regression anchors.
        """
        before_params: Dict[str, Any] = {
            "temperature_scaling_T": 1.6178,
            "optimal_threshold_tau": 0.8118,
            "calibration_status": "INITIALIZED_DEFAULT",
        }
        if CALIB_FILE.exists():
            try:
                before_params.update(json.loads(CALIB_FILE.read_text(encoding="utf-8")))
            except Exception:
                pass

        events = get_approved_feedback_for_weights(limit=250, media_type="image", db_path=db_path)
        if not events:
            return {
                "status": "no_approved_image_feedback_samples",
                "accepted": False,
                "before_params": before_params,
                "after_params": before_params,
            }

        samples = [
            {
                "y": _label_to_int(ev.get("true_label", "Realism")),
                "risk": float(ev.get("composite_risk_score", 0.5)),
                "w": 2.0,
            }
            for ev in events
        ]
        anchors = [
            {"y": 0, "risk": 0.22, "w": 1.0},
            {"y": 0, "risk": 0.36, "w": 1.0},
            {"y": 1, "risk": 0.89, "w": 1.0},
            {"y": 1, "risk": 0.94, "w": 1.0},
        ]
        all_pts = samples + anchors

        def _eval_tau(tau_val: float) -> Tuple[int, float]:
            errs, bce = 0, 0.0
            t_s = max(0.5, float(before_params.get("temperature_scaling_T", 1.6178)))
            for pt in all_pts:
                pred = 1 if pt["risk"] >= tau_val else 0
                if pred != pt["y"]:
                    errs += 1
                z = (pt["risk"] - tau_val) * (6.0 / t_s)
                p = min(1.0 - 1e-6, max(1e-6, 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))))
                bce += pt["w"] * (-(pt["y"] * math.log(p) + (1 - pt["y"]) * math.log(1.0 - p)))
            return errs, round(bce + 0.05 * abs(tau_val - float(before_params.get("optimal_threshold_tau", 0.8118))), 5)

        tau_0 = float(before_params.get("optimal_threshold_tau", 0.8118))
        best_tau = tau_0
        best_score = _eval_tau(tau_0)
        for d in np.linspace(-max_delta_tau, max_delta_tau, 11):
            cand_tau = round(min(0.92, max(0.45, tau_0 + float(d))), 4)
            sc = _eval_tau(cand_tau)
            if sc < best_score:
                best_score = sc
                best_tau = cand_tau

        before_score = _eval_tau(tau_0)
        accepted = bool(best_score <= before_score)
        after_params = dict(before_params)
        if accepted:
            after_params["optimal_threshold_tau"] = best_tau
            after_params["calibration_status"] = "IMAGE_DFAGENT_APPROVED_CALIBRATED"
            after_params["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            after_params["last_trigger"] = trigger_source
            CUSTOM_VIT_DIR.mkdir(parents=True, exist_ok=True)
            CALIB_FILE.write_text(json.dumps(after_params, indent=2, ensure_ascii=False), encoding="utf-8")
            if detector is not None and hasattr(detector, "reload_calibration_params"):
                detector.reload_calibration_params()

        img_after_params = after_params if accepted else before_params
        img_before_metrics = {"errors": before_score[0], "objective": before_score[1]}
        img_after_metrics = {"errors": best_score[0], "objective": best_score[1]} if accepted else img_before_metrics
        update_id = save_weight_update_audit({
            "trigger_source": trigger_source,
            "feedback_count": len(samples),
            "before_params": before_params,
            "after_params": img_after_params,
            "before_metrics": img_before_metrics,
            "after_metrics": img_after_metrics,
            "accepted": accepted,
            "notes": "Image threshold refinement strictly from /dfagent-approved feedback",
        }, db_path=db_path)

        art_info = cls._persist_weight_update_report_artifact(
            update_id=update_id,
            modality="image",
            trigger_source=trigger_source,
            feedback_count=len(samples),
            before_params=before_params,
            after_params=img_after_params,
            before_metrics=img_before_metrics,
            after_metrics=img_after_metrics,
            accepted=accepted,
            notes="Image threshold refinement strictly from /dfagent-approved feedback",
            db_path=db_path,
        )

        return {
            "status": "updated" if accepted else "kept_baseline",
            "update_id": update_id,
            "accepted": accepted,
            "feedback_samples_used": len(samples),
            "before_params": before_params,
            "after_params": img_after_params,
            "before_metrics": img_before_metrics,
            "after_metrics": img_after_metrics,
            "parameter_diffs": art_info["parameter_diffs"],
            "metric_diffs": art_info["metric_diffs"],
            "report_artifact_path": art_info["report_artifact_path"],
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
        auto_update_calibration: bool = False,
        trigger_lora_update: bool = False,
        detector=None,
        db_path: Optional[str] = None,
        source_type: str = "HUMAN_REVIEW",
        review_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end entry point for user/pentest reflection:
          1. Enriches context from forensic_certificates if certificate_id is given.
          2. Saves feedback event in SQLite (user_feedback_events) with PENDING status for HUMAN_REVIEW.
          3. Extracts hard frames into ZeroDayActiveLearningVault + labeled_feedback_manifest.json (approved=False).
          4. Runs calibration/LoRA ONLY if authorized (AUTO_PENTEST or already approved on /dfagent).
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

        norm_source = str(source_type or "HUMAN_REVIEW").upper()
        norm_review = str(review_status or ("AUTO_LOGGED" if norm_source == "AUTO_PENTEST" else "PENDING")).upper()
        is_pre_approved = norm_review in APPROVED_REVIEW_STATUSES

        # Strict governance gate: unapproved HUMAN_REVIEW items CANNOT trigger calibration or LoRA
        if norm_source != "AUTO_PENTEST" and not is_pre_approved:
            auto_update_calibration = False
            trigger_lora_update = False

        extracted_frames, vault_ids = cls.extract_hard_frames(
            media_path=file_path or "",
            feedback_id=feedback_id,
            true_label=norm_true,
            platform=resolved_platform,
            feedback_type=fb_type,
            approved=is_pre_approved,
            review_status=norm_review,
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
            "source_type": norm_source,
            "review_status": norm_review,
        }
        save_user_feedback(event_payload, db_path=db_path)

        calib_res = None
        if auto_update_calibration:
            mtype_norm = str(media_type or "video").strip().lower()
            if mtype_norm == "audio":
                calib_res = cls.run_audio_calibration_update(
                    detector=detector,
                    trigger_source=f"feedback:{feedback_id}:{fb_type}",
                    db_path=db_path,
                )
            elif mtype_norm == "video":
                calib_res = cls.run_fast_calibration_update(
                    detector=detector,
                    trigger_source=f"feedback:{feedback_id}:{fb_type}",
                    db_path=db_path,
                )
            else:
                calib_res = {"status": "skipped_image_uses_council_or_lora", "accepted": False}
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
            "review_status": norm_review,
            "requires_dfagent_approval": (norm_source != "AUTO_PENTEST" and not is_pre_approved),
            "extracted_frames_count": len(extracted_frames),
            "vault_sample_ids": vault_ids,
            "calibration_update": calib_res,
            "lora_update": lora_res,
        }

