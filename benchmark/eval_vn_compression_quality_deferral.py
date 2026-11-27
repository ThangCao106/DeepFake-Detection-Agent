# -*- coding: utf-8 -*-
"""
Vietnamese Portrait Compression Survival, Signal-Quality-Aware Fusion & DiCoME Deferral Benchmark
==================================================================================================
Implements the 2026 PhD Thesis Experimental Protocol grounded in:
  - GenD (WACV 2026): strict origin_id/identity isolation & paired counterfactual hygiene
  - NTIRE 2026 (CVPR Workshop): Dual-protocol separation of paired compression sweeps vs
    verified capture conditions
  - FreqFusion (Nov 2026): Frequency vs Vision/Biometric survival under JPEG/WebP/blur
  - Light2Lie (NDSS 2026): Corneal/periocular cue reliability across eye ROI size, luminance & occlusion
  - DiCoME (ICML 2026): Semantic-vs-forensic conflict + face/eye quality selective deferral

Strict anti-leakage guarantee:
  1. `train`, `cal_val`, `test_locked` loaded from locked manifest with 0 `origin_id` overlap.
  2. Branch linear probes trained strictly on `train` (with paired JPEG augmentations).
  3. Platt calibration, quality-aware fusion weights, `tau_fpr1pct`, and `tau_youden` fitted and
     locked strictly on `cal_val` (`locked_protocol_and_thresholds.json`) BEFORE opening `test_locked`.
  4. `test_locked` evaluated ONCE across Protocols A & B, Signal Survival, and Matched-Coverage Deferral.
"""

import argparse
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forensics_kernel.biometric_corneal import CornealReflectionAnalyzer
from forensics_kernel.calibration_store import (
    apply_platt_2param,
    compute_signal_quality_weights,
    evaluate_dicome_deferral,
    select_matched_coverage_decisions,
)
from forensics_kernel.dataset_split_guard import (
    _roc_auc_binary,
    assert_zero_origin_leakage,
    compute_bootstrap_ci,
    compute_wilson_ci,
    load_locked_manifest_splits,
)
from forensics_kernel.spectral_wavelet import WaveletSpectralAnalyzer

DEFAULT_MANIFEST = PROJECT_ROOT / "report_colab" / "corpus" / "training_corpus_manifest.jsonl"
DEFAULT_VIT_DIR = PROJECT_ROOT / "custom_vit_model"
DEFAULT_OUT_DIR = PROJECT_ROOT / "report_colab" / "vn_compression_deferral_eval"


# ---------------------------------------------------------------------------
# Controlled Image Transformations (NTIRE 2026 Paired & OOD Protocols)
# ---------------------------------------------------------------------------

def apply_compression_chain(im: Image.Image, preset: str) -> Image.Image:
    """Apply identical deterministic compression/resize chain to both Real and Fake images."""
    rgb = im.convert("RGB")
    w, h = rgb.size
    if preset == "clean_raw":
        return rgb
    if preset == "mild_jpeg_q80":
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if preset == "medium_social_q55_r75":
        small = rgb.resize((max(48, int(w * 0.75)), max(48, int(h * 0.75))), Image.Resampling.BILINEAR)
        restored = small.resize((w, h), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        restored.save(buf, format="JPEG", quality=55)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if preset == "heavy_social_webp_q32_r50_blur":
        small = rgb.resize((max(36, int(w * 0.50)), max(36, int(h * 0.50))), Image.Resampling.BILINEAR)
        restored = small.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=0.80))
        buf = io.BytesIO()
        fmt = "WEBP" if "WEBP" in Image.SAVE else "JPEG"
        restored.save(buf, format=fmt, quality=32)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(f"Unknown compression preset: {preset}")


def make_unseen_diffusion_blend_ood(im: Image.Image, seed: int) -> Image.Image:
    """
    Generate an unseen manipulation family NOT present in training (radial diffusion-like
    inner facial smoothing + subtle channel temperature shift + JPEG Q=65 re-save).
    """
    rng = np.random.default_rng(int(seed))
    rgb = im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32)
    blurred = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=2.1)), dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    cx = 128.0 + float(rng.uniform(-10, 10))
    cy = 120.0 + float(rng.uniform(-10, 10))
    rx = float(rng.uniform(46, 66))
    ry = float(rng.uniform(56, 78))
    mask = np.clip(1.0 - (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0.0, 1.0) ** 1.6
    mask3 = mask[:, :, None]
    gain = np.array([rng.uniform(0.92, 1.08), rng.uniform(0.95, 1.05), rng.uniform(0.90, 1.07)], dtype=np.float32)
    synth = np.clip(blurred * gain + rng.normal(0.0, 2.2, size=arr.shape), 0.0, 255.0)
    blended = np.clip(mask3 * synth + (1.0 - mask3) * arr, 0.0, 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended, mode="RGB").save(buf, format="JPEG", quality=65)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Feature Extraction & Regularized Branch Probes
# ---------------------------------------------------------------------------

def extract_records_features(
    records: Sequence[Dict[str, Any]],
    images: Sequence[Image.Image],
    vit_processor: AutoImageProcessor,
    vit_model: AutoModelForImageClassification,
    wavelet_analyzer: WaveletSpectralAnalyzer,
    corneal_analyzer: CornealReflectionAnalyzer,
    view_name: str,
    batch_size: int = 16,
) -> List[Dict[str, Any]]:
    """Run real pretrained ViT + deterministic Wavelet/FFT + Corneal/Eye extractors."""
    n = len(images)
    vit_feats_all: List[np.ndarray] = []
    vit_raw_logits_all: List[float] = []

    vit_model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_imgs = list(images[start : start + batch_size])
            inputs = vit_processor(images=batch_imgs, return_tensors="pt")
            vit_out = vit_model.vit(**inputs)
            cls_emb = vit_out.last_hidden_state[:, 0, :]  # [B, 768]
            logits = vit_model.classifier(cls_emb)  # [B, 2]
            raw_fake_margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            cls_np = cls_emb.cpu().numpy().astype(np.float32)
            # L2-normalize embedding for stable linear probing
            norms = np.linalg.norm(cls_np, axis=1, keepdims=True) + 1e-6
            cls_np = cls_np / norms
            vit_feats_all.append(cls_np)
            vit_raw_logits_all.extend([float(v) for v in raw_fake_margin])

    vit_emb_mat = np.concatenate(vit_feats_all, axis=0) if vit_feats_all else np.zeros((0, 768), dtype=np.float32)

    out: List[Dict[str, Any]] = []
    for idx, (rec, im) in enumerate(zip(records, images)):
        w_res = wavelet_analyzer.analyze(im)
        c_res = corneal_analyzer.analyze(im)
        q_freq = float(w_res["frequency_signal_quality"])
        q_eye = float(c_res["corneal_signal_quality"])
        jpeg_blk = float(w_res["jpeg_blockiness_score"])
        blur_sev = float(w_res["blur_severity_score"])
        q_vision = float(np.clip(1.0 - 0.30 * jpeg_blk - 0.35 * blur_sev, 0.25, 1.0))

        vit_vec = np.concatenate(
            [vit_emb_mat[idx], np.array([vit_raw_logits_all[idx] / 5.0], dtype=np.float32)],
            axis=0,
        )
        freq_vec = np.asarray(w_res["feature_vector"], dtype=np.float32)
        eye_vec = np.asarray(c_res["feature_vector"], dtype=np.float32)

        out.append(
            {
                "sample_id": rec.get("sample_id"),
                "origin_id": rec.get("origin_id"),
                "label": int(rec["label"]),
                "vietnamese_category": str(rec.get("vietnamese_category") or "unknown"),
                "view_name": view_name,
                "vit_vec": vit_vec,
                "freq_vec": freq_vec,
                "eye_vec": eye_vec,
                "base_vit_logit": float(vit_raw_logits_all[idx]),
                "q_vision": round(q_vision, 4),
                "q_freq": round(q_freq, 4),
                "q_eye": round(q_eye, 4),
                "jpeg_blockiness": round(jpeg_blk, 4),
                "blur_severity": round(blur_sev, 4),
                "compression_severity": round(float(np.clip(0.55 * jpeg_blk + 0.45 * blur_sev, 0.0, 1.0)), 4),
                "wavelet_hf_energy": float(w_res["wavelet_high_freq_energy"]),
                "fft_lattice_z_peak": float(w_res["fft_lattice_z_peak"]),
                "corneal_asymmetry": float(c_res["corneal_asymmetry_score"]),
                "fresnel_contrast": float(c_res["fresnel_specular_contrast"]),
                "eye_roi_pixels": int(c_res["eye_roi_pixels"]),
                "eye_luminance": float(c_res["eye_luminance"]),
                "occlusion_glare_score": float(c_res["occlusion_glare_score"]),
                "bilateral_eyes_detected": bool(c_res["bilateral_eyes_detected"]),
            }
        )
    return out


class RegularizedProbe:
    """Standardized L2-regularized logistic probe returning calibrated logits."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, weight: np.ndarray, bias: float):
        self.mean = mean
        self.std = std
        self.weight = weight
        self.bias = float(bias)

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        xn = (x - self.mean) / self.std
        return xn @ self.weight + self.bias


def fit_logistic_probe(
    x_train: np.ndarray,
    y_train: Sequence[int],
    l2_weight: float = 0.02,
    steps: int = 240,
    lr: float = 0.08,
    seed: int = 2026,
) -> RegularizedProbe:
    """Train a convex L2-regularized binary logistic head on `train` only."""
    torch.manual_seed(int(seed))
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True) + 1e-4
    xn = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    yt = torch.tensor([float(y) for y in y_train], dtype=torch.float32)

    linear = nn.Linear(xn.shape[1], 1)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=l2_weight)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = linear(xn).squeeze(-1)
        loss = bce(logits, yt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = linear.weight.squeeze(0).cpu().numpy().astype(np.float32)
        b = float(linear.bias.item())
    return RegularizedProbe(mean.squeeze(0).astype(np.float32), std.squeeze(0).astype(np.float32), w, b)


def fit_platt_2param_on_logits(logits: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    """Fit 2-parameter Platt scaling sigmoid(a * z + b) strictly on `cal_val`."""
    z_t = torch.tensor([float(z) for z in logits], dtype=torch.float32)
    y_t = torch.tensor([float(y) for y in labels], dtype=torch.float32)
    a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    opt = torch.optim.Adam([a, b], lr=0.06)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(180):
        opt.zero_grad()
        scaled = torch.clamp(a, 0.05, 8.0) * z_t + b
        loss = bce(scaled, y_t) + 0.002 * ((a - 1.0) ** 2 + b ** 2)
        loss.backward()
        opt.step()
    a_val = round(float(torch.clamp(a, 0.05, 8.0).item()), 6)
    b_val = round(float(b.item()), 6)
    return {"platt_a": a_val, "platt_b": b_val, "temperature_scaling_T": round(1.0 / max(1e-4, a_val), 6)}


def find_cal_thresholds(y_true: Sequence[int], probs: Sequence[float]) -> Dict[str, Any]:
    """
    Fit two operating thresholds strictly on `cal_val`:
      1) `tau_fpr1pct`: 99th percentile of authentic probabilities on `cal_val` (conservative higher quantile)
      2) `tau_youden`: Youden's J (TPR - FPR) maximizing cutoff on `cal_val`
    """
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(probs, dtype=float)
    neg_scores = np.sort(p_arr[y_arr == 0])

    if len(neg_scores) > 0:
        tau_fpr1 = float(np.quantile(neg_scores, 0.99, method="higher")) + 1e-4
        tau_fpr1 = float(min(0.995, max(0.05, tau_fpr1)))
    else:
        tau_fpr1 = 0.85

    candidates = np.unique(np.clip(np.concatenate([p_arr, np.linspace(0.05, 0.95, 91)]), 0.01, 0.99))
    best_j = -999.0
    best_tau = 0.5
    pos = y_arr == 1
    neg = y_arr == 0
    for c in candidates:
        tpr = float(np.mean(p_arr[pos] >= c)) if np.any(pos) else 0.0
        fpr = float(np.mean(p_arr[neg] >= c)) if np.any(neg) else 0.0
        j = tpr - fpr
        if j > best_j or (abs(j - best_j) < 1e-9 and abs(c - 0.5) < abs(best_tau - 0.5)):
            best_j = j
            best_tau = float(c)

    cal_real_fpr1 = float(np.mean(p_arr[neg] >= tau_fpr1)) if np.any(neg) else 0.0
    cal_real_tpr1 = float(np.mean(p_arr[pos] >= tau_fpr1)) if np.any(pos) else 0.0
    return {
        "tau_fpr1pct": round(tau_fpr1, 6),
        "tau_youden": round(best_tau, 6),
        "cal_val_youden_j": round(best_j, 4),
        "cal_val_realized_fpr_at_fpr1_tau": round(cal_real_fpr1, 4),
        "cal_val_tpr_at_fpr1_tau": round(cal_real_tpr1, 4),
        "cal_val_neg_count": int(np.sum(neg)),
        "finite_sample_fpr_step": round(1.0 / max(1, int(np.sum(neg))), 4),
    }


def _compute_ece(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    if len(y) == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return round(ece, 4)


def compute_full_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    tau_fpr1: float,
    tau_youden: float,
    n_boot: int = 500,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Compute ROC-AUC (with 95% Bootstrap CI), TPR@FPR1%, Realized FPR (with 95% Wilson CI), Balanced Acc, and ECE."""
    y_l = [int(y) for y in y_true]
    p_l = [float(p) for p in y_prob]
    n = len(y_l)
    if n == 0:
        return {"n": 0}

    n_pos = sum(1 for y in y_l if y == 1)
    n_neg = sum(1 for y in y_l if y == 0)
    auc = _roc_auc_binary(y_l, p_l) if (n_pos > 0 and n_neg > 0) else None

    rng = np.random.default_rng(int(seed))
    auc_ci = [None, None]
    if auc is not None and n >= 6:
        y_np = np.asarray(y_l)
        p_np = np.asarray(p_l)
        boot_aucs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y_np[idx]
            if np.any(yb == 1) and np.any(yb == 0):
                boot_aucs.append(_roc_auc_binary(yb.tolist(), p_np[idx].tolist()))
        if boot_aucs:
            auc_ci = [round(float(np.percentile(boot_aucs, 2.5)), 4), round(float(np.percentile(boot_aucs, 97.5)), 4)]

    def _op_stats(tau: float) -> Dict[str, Any]:
        tp = sum(1 for y, p in zip(y_l, p_l) if y == 1 and p >= tau)
        fn = sum(1 for y, p in zip(y_l, p_l) if y == 1 and p < tau)
        fp = sum(1 for y, p in zip(y_l, p_l) if y == 0 and p >= tau)
        tn = sum(1 for y, p in zip(y_l, p_l) if y == 0 and p < tau)
        tpr = round(tp / n_pos, 4) if n_pos > 0 else 0.0
        fpr = round(fp / n_neg, 4) if n_neg > 0 else 0.0
        fnr = round(fn / n_pos, 4) if n_pos > 0 else 0.0
        bacc = round(0.5 * (tpr + (1.0 - fpr)), 4)
        bci = compute_bootstrap_ci(y_l, p_l, tau=tau, n_boot=n_boot, seed=seed)
        boot_ci = bci.get("bootstrap_ci95", {})
        return {
            "tau": round(float(tau), 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "tpr": tpr,
            "tpr_wilson_ci95": list(compute_wilson_ci(tp, n_pos)),
            "tpr_boot_ci95": boot_ci.get("tpr"),
            "fpr": fpr,
            "fpr_wilson_ci95": list(compute_wilson_ci(fp, n_neg)),
            "fpr_boot_ci95": boot_ci.get("fpr"),
            "fnr": fnr,
            "balanced_accuracy": bacc,
        }

    return {
        "n": n,
        "pos_n": n_pos,
        "neg_n": n_neg,
        "roc_auc": round(float(auc), 4) if auc is not None else None,
        "roc_auc_ci95": auc_ci,
        "ece": _compute_ece(y_l, p_l),
        "at_fpr1pct_tau": _op_stats(tau_fpr1),
        "at_youden_tau": _op_stats(tau_youden),
    }


def _evaluate_matched_coverage(
    rows: Sequence[Dict[str, Any]],
    tau: float,
    coverages: Sequence[float] = (1.0, 0.90, 0.85, 0.80, 0.70),
) -> Dict[str, Any]:
    """Compare Single-ViT Margin Deferral vs DiCoME Conflict+Quality Deferral at matched coverages."""
    y_true = [int(r["label"]) for r in rows]
    p_vit = [float(r["p_single_vit"]) for r in rows]
    p_fused = [float(r["p_quality_aware"]) for r in rows]

    # Margin uncertainty score in [0, 1]: smaller distance to tau -> higher deferral score
    vit_margin_scores = [
        float(max(0.0, 1.0 - abs(pv - tau) / max(tau, 1.0 - tau, 0.5)))
        for pv in p_vit
    ]
    dicome_scores = [float(r["dicome_deferral_score"]) for r in rows]

    def _run_policy(policy_scores: Sequence[float], pred_probs: Sequence[float]) -> Dict[str, Any]:
        points: List[Dict[str, Any]] = []
        risks: List[float] = []
        covs: List[float] = []
        for cov in coverages:
            keep = select_matched_coverage_decisions(policy_scores, coverage=cov)
            kept_idx = [i for i, k in enumerate(keep) if k]
            defer_n = len(rows) - len(kept_idx)
            y_k = [y_true[i] for i in kept_idx]
            p_k = [pred_probs[i] for i in kept_idx]
            pos_k = sum(1 for y in y_k if y == 1)
            neg_k = sum(1 for y in y_k if y == 0)
            fp_k = sum(1 for y, p in zip(y_k, p_k) if y == 0 and p >= tau)
            fn_k = sum(1 for y, p in zip(y_k, p_k) if y == 1 and p < tau)
            err_k = fp_k + fn_k
            realized_cov = round(len(kept_idx) / max(1, len(rows)), 4)
            sel_err = round(err_k / max(1, len(kept_idx)), 4)
            fpr_k = round(fp_k / neg_k, 4) if neg_k > 0 else 0.0
            fnr_k = round(fn_k / pos_k, 4) if pos_k > 0 else 0.0
            points.append(
                {
                    "target_coverage": round(float(cov), 2),
                    "realized_coverage": realized_cov,
                    "auto_decided_n": len(kept_idx),
                    "deferred_n": defer_n,
                    "realized_fpr": fpr_k,
                    "realized_fpr_ci95": list(compute_wilson_ci(fp_k, neg_k)),
                    "realized_fnr": fnr_k,
                    "realized_fnr_ci95": list(compute_wilson_ci(fn_k, pos_k)),
                    "selective_error_rate": sel_err,
                    "selective_error_ci95": list(compute_wilson_ci(err_k, len(kept_idx))),
                }
            )
            covs.append(realized_cov)
            risks.append(sel_err)
        # Trapezoidal AURC over sorted coverage levels (lower is better)
        order = np.argsort(covs)
        c_s = np.asarray([covs[i] for i in order], dtype=float)
        r_s = np.asarray([risks[i] for i in order], dtype=float)
        span = max(1e-6, float(c_s[-1] - c_s[0]))
        aurc = round(float(np.trapezoid(r_s, c_s) / span) if hasattr(np, "trapezoid") else float(np.trapz(r_s, c_s) / span), 4)
        return {"points": points, "normalized_aurc": aurc}

    # Fair comparison: baseline uses ViT margin on ViT decisions AND also tested on Fused decisions
    return {
        "operating_tau": round(float(tau), 4),
        "n_total": len(rows),
        "vit_margin_baseline_on_vit": _run_policy(vit_margin_scores, p_vit),
        "vit_margin_on_quality_fusion": _run_policy(vit_margin_scores, p_fused),
        "dicome_conflict_quality_proposed": _run_policy(dicome_scores, p_fused),
    }


def _score_samples(
    feats: Sequence[Dict[str, Any]],
    vit_probe: RegularizedProbe,
    freq_probe: RegularizedProbe,
    eye_probe: RegularizedProbe,
    platt_vit: Dict[str, float],
    platt_freq: Dict[str, float],
    platt_eye: Dict[str, float],
    fixed_weights: Tuple[float, float, float],
    platt_fixed: Optional[Dict[str, float]],
    quality_base_weights: Tuple[float, float, float],
    quality_gamma: float,
    platt_quality: Optional[Dict[str, float]],
    tau_for_dicome: float = 0.5,
) -> List[Dict[str, Any]]:
    x_vit = np.stack([f["vit_vec"] for f in feats], axis=0)
    x_freq = np.stack([f["freq_vec"] for f in feats], axis=0)
    x_eye = np.stack([f["eye_vec"] for f in feats], axis=0)

    z_vit = vit_probe.predict_logits(x_vit)
    z_freq = freq_probe.predict_logits(x_freq)
    z_eye = eye_probe.predict_logits(x_eye)

    scored: List[Dict[str, Any]] = []
    for i, f in enumerate(feats):
        pv = apply_platt_2param(float(z_vit[i]), platt_vit, is_logit=True) if platt_vit else 1.0 / (1.0 + math.exp(-float(z_vit[i])))
        pf = apply_platt_2param(float(z_freq[i]), platt_freq, is_logit=True) if platt_freq else 1.0 / (1.0 + math.exp(-float(z_freq[i])))
        pe = apply_platt_2param(float(z_eye[i]), platt_eye, is_logit=True) if platt_eye else 1.0 / (1.0 + math.exp(-float(z_eye[i])))

        wf_v, wf_f, wf_e = fixed_weights
        z_fixed = wf_v * float(z_vit[i]) + wf_f * float(z_freq[i]) + wf_e * float(z_eye[i])
        p_fixed = apply_platt_2param(z_fixed, platt_fixed, is_logit=True) if platt_fixed else (wf_v * pv + wf_f * pf + wf_e * pe)

        qw = compute_signal_quality_weights(
            f["q_vision"],
            f["q_freq"],
            f["q_eye"],
            base_weights=quality_base_weights,
            reliability_gamma=quality_gamma,
        )
        z_qual = qw["w_vision"] * float(z_vit[i]) + qw["w_freq"] * float(z_freq[i]) + qw["w_eye"] * float(z_eye[i])
        p_qual = apply_platt_2param(z_qual, platt_quality, is_logit=True) if platt_quality else (qw["w_vision"] * pv + qw["w_freq"] * pf + qw["w_eye"] * pe)

        dicome = evaluate_dicome_deferral(
            p_vision=pv,
            p_freq=pf,
            p_eye=pe,
            p_fused=p_qual,
            q_face_eye=f["q_eye"],
            compression_severity=f["compression_severity"],
            tau=tau_for_dicome,
            deferral_threshold=0.40,
        )

        row = {
            k: v
            for k, v in f.items()
            if k not in ("vit_vec", "freq_vec", "eye_vec")
        }
        row.update(
            {
                "z_vit": round(float(z_vit[i]), 5),
                "z_freq": round(float(z_freq[i]), 5),
                "z_eye": round(float(z_eye[i]), 5),
                "z_fixed": round(float(z_fixed), 5),
                "z_quality": round(float(z_qual), 5),
                "p_single_vit": round(float(pv), 6),
                "p_freq_only": round(float(pf), 6),
                "p_corneal_only": round(float(pe), 6),
                "p_fixed_fusion": round(float(p_fixed), 6),
                "p_quality_aware": round(float(p_qual), 6),
                "w_vision_dyn": qw["w_vision"],
                "w_freq_dyn": qw["w_freq"],
                "w_eye_dyn": qw["w_eye"],
                "dicome_deferral_score": dicome["deferral_score"],
                "dicome_cross_conflict": dicome["cross_branch_conflict"],
                "dicome_degradation_risk": dicome["degradation_risk"],
            }
        )
        scored.append(row)
    return scored


def _load_pil_list(records: Sequence[Dict[str, Any]], transform_fn) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    for rec in records:
        with Image.open(rec["local_path"]) as raw_im:
            imgs.append(transform_fn(raw_im, rec))
    return imgs


def run_benchmark(
    manifest_path: Path = DEFAULT_MANIFEST,
    vit_dir: Path = DEFAULT_VIT_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Dict[str, Any]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leak-free manifest loading
    loaded = load_locked_manifest_splits(manifest_path, exclude_unverified=True, require_existing_files=True)
    splits = loaded["splits"]
    leak_audit = loaded["audit"]
    assert_zero_origin_leakage(splits["train"], splits["cal_val"], splits["test_locked"])

    # 2. Initialize real ViT + Wavelet + Corneal extractors
    vit_processor = AutoImageProcessor.from_pretrained(str(vit_dir), local_files_only=True)
    vit_model = AutoModelForImageClassification.from_pretrained(str(vit_dir), local_files_only=True)
    wavelet_analyzer = WaveletSpectralAnalyzer()
    corneal_analyzer = CornealReflectionAnalyzer()

    # 3. Extract `train` across paired compression chain (GenD + NTIRE training hygiene, strictly isolated to train)
    train_recs_aug: List[Dict[str, Any]] = []
    train_imgs_aug: List[Image.Image] = []
    for preset in ("clean_raw", "mild_jpeg_q80", "medium_social_q55_r75"):
        train_recs_aug.extend(splits["train"])
        train_imgs_aug.extend(_load_pil_list(splits["train"], lambda im, r, p=preset: apply_compression_chain(im, p)))

    print(f"[Train] Extracting real ViT + Wavelet + Corneal features for {len(train_imgs_aug)} paired train views...")
    train_feats = extract_records_features(
        train_recs_aug, train_imgs_aug, vit_processor, vit_model, wavelet_analyzer, corneal_analyzer, "train_multi"
    )

    y_train = [f["label"] for f in train_feats]
    vit_probe = fit_logistic_probe(np.stack([f["vit_vec"] for f in train_feats]), y_train, l2_weight=0.05, steps=260, lr=0.06, seed=2026)
    freq_probe = fit_logistic_probe(np.stack([f["freq_vec"] for f in train_feats]), y_train, l2_weight=0.02, steps=260, lr=0.08, seed=2027)
    eye_probe = fit_logistic_probe(np.stack([f["eye_vec"] for f in train_feats]), y_train, l2_weight=0.02, steps=260, lr=0.08, seed=2028)

    # 4. Extract `cal_val` across clean + medium compression + heavy compression (strictly unseen during probe fit)
    cal_recs_all: List[Dict[str, Any]] = []
    cal_imgs_all: List[Image.Image] = []
    for preset in ("clean_raw", "medium_social_q55_r75", "heavy_social_webp_q32_r50_blur"):
        cal_recs_all.extend(splits["cal_val"])
        cal_imgs_all.extend(_load_pil_list(splits["cal_val"], lambda im, r, p=preset: apply_compression_chain(im, p)))

    print(f"[CalVal] Extracting features & locking thresholds on {len(cal_imgs_all)} cal_val views...")
    cal_feats = extract_records_features(
        cal_recs_all, cal_imgs_all, vit_processor, vit_model, wavelet_analyzer, corneal_analyzer, "cal_val_multi"
    )
    y_cal = [f["label"] for f in cal_feats]
    z_cal_vit = vit_probe.predict_logits(np.stack([f["vit_vec"] for f in cal_feats]))
    z_cal_freq = freq_probe.predict_logits(np.stack([f["freq_vec"] for f in cal_feats]))
    z_cal_eye = eye_probe.predict_logits(np.stack([f["eye_vec"] for f in cal_feats]))

    platt_vit = fit_platt_2param_on_logits(z_cal_vit, y_cal)
    platt_freq = fit_platt_2param_on_logits(z_cal_freq, y_cal)
    platt_eye = fit_platt_2param_on_logits(z_cal_eye, y_cal)

    # Search optimal fixed weights and quality-aware hyperparameters strictly on `cal_val`
    best_fixed_bce = 1e9
    best_fixed_w = (0.55, 0.25, 0.20)
    for wv in (0.45, 0.55, 0.65, 0.75):
        for wf in (0.15, 0.25, 0.35):
            we = round(1.0 - wv - wf, 4)
            if we < 0.05:
                continue
            zf = wv * z_cal_vit + wf * z_cal_freq + we * z_cal_eye
            pf_arr = np.clip(1.0 / (1.0 + np.exp(-zf)), 1e-6, 1.0 - 1e-6)
            y_arr = np.asarray(y_cal, dtype=float)
            loss = -float(np.mean(y_arr * np.log(pf_arr) + (1.0 - y_arr) * np.log(1.0 - pf_arr)))
            if loss < best_fixed_bce:
                best_fixed_bce = loss
                best_fixed_w = (wv, wf, we)

    z_cal_fixed = best_fixed_w[0] * z_cal_vit + best_fixed_w[1] * z_cal_freq + best_fixed_w[2] * z_cal_eye
    platt_fixed = fit_platt_2param_on_logits(z_cal_fixed, y_cal)

    best_qual_score = -1e9
    best_qual_cfg = ((0.55, 0.25, 0.20), 1.5)
    for bv in (0.45, 0.55, 0.65):
        for bf in (0.15, 0.25, 0.35):
            be = round(1.0 - bv - bf, 4)
            if be < 0.05:
                continue
            for gamma in (1.0, 1.5, 2.2):
                zq_list = []
                for i, f in enumerate(cal_feats):
                    qw = compute_signal_quality_weights(f["q_vision"], f["q_freq"], f["q_eye"], (bv, bf, be), gamma)
                    zq_list.append(qw["w_vision"] * float(z_cal_vit[i]) + qw["w_freq"] * float(z_cal_freq[i]) + qw["w_eye"] * float(z_cal_eye[i]))
                auc_q = _roc_auc_binary(y_cal, zq_list)
                # Combine AUC and negative BCE to break ties toward well-calibrated probabilities
                pq_arr = np.clip(1.0 / (1.0 + np.exp(-np.asarray(zq_list))), 1e-6, 1.0 - 1e-6)
                y_arr = np.asarray(y_cal, dtype=float)
                bce_q = -float(np.mean(y_arr * np.log(pq_arr) + (1.0 - y_arr) * np.log(1.0 - pq_arr)))
                obj = auc_q - 0.15 * bce_q
                if obj > best_qual_score:
                    best_qual_score = obj
                    best_qual_cfg = ((bv, bf, be), gamma)

    q_base_w, q_gamma = best_qual_cfg
    z_cal_qual = []
    for i, f in enumerate(cal_feats):
        qw = compute_signal_quality_weights(f["q_vision"], f["q_freq"], f["q_eye"], q_base_w, q_gamma)
        z_cal_qual.append(qw["w_vision"] * float(z_cal_vit[i]) + qw["w_freq"] * float(z_cal_freq[i]) + qw["w_eye"] * float(z_cal_eye[i]))
    platt_quality = fit_platt_2param_on_logits(z_cal_qual, y_cal)

    cal_scored = _score_samples(
        cal_feats,
        vit_probe,
        freq_probe,
        eye_probe,
        platt_vit,
        platt_freq,
        platt_eye,
        best_fixed_w,
        platt_fixed,
        q_base_w,
        q_gamma,
        platt_quality,
    )

    locked_thresholds = {
        "single_vision_vit": find_cal_thresholds(y_cal, [r["p_single_vit"] for r in cal_scored]),
        "freq_wavelet_only": find_cal_thresholds(y_cal, [r["p_freq_only"] for r in cal_scored]),
        "corneal_eye_only": find_cal_thresholds(y_cal, [r["p_corneal_only"] for r in cal_scored]),
        "fixed_weight_fusion": find_cal_thresholds(y_cal, [r["p_fixed_fusion"] for r in cal_scored]),
        "quality_aware_fusion": find_cal_thresholds(y_cal, [r["p_quality_aware"] for r in cal_scored]),
    }

    locked_protocol = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path),
        "leakage_audit": leak_audit,
        "calibration_samples_total": len(cal_feats),
        "platt_params": {
            "single_vision_vit": platt_vit,
            "freq_wavelet_only": platt_freq,
            "corneal_eye_only": platt_eye,
            "fixed_weight_fusion": platt_fixed,
            "quality_aware_fusion": platt_quality,
        },
        "fusion_hyperparams": {
            "fixed_weights": {"w_vision": best_fixed_w[0], "w_freq": best_fixed_w[1], "w_eye": best_fixed_w[2]},
            "quality_aware_base_weights": {"w_vision": q_base_w[0], "w_freq": q_base_w[1], "w_eye": q_base_w[2]},
            "quality_aware_gamma": q_gamma,
        },
        "locked_thresholds": locked_thresholds,
    }
    (out_dir / "locked_protocol_and_thresholds.json").write_text(
        json.dumps(locked_protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5. Evaluate ONCE on `test_locked` across Protocols A & B
    tau_dicome_ref = locked_thresholds["quality_aware_fusion"]["tau_youden"]
    compression_presets = (
        "clean_raw",
        "mild_jpeg_q80",
        "medium_social_q55_r75",
        "heavy_social_webp_q32_r50_blur",
    )

    test_views_scored: Dict[str, List[Dict[str, Any]]] = {}
    all_test_rows: List[Dict[str, Any]] = []

    for preset in compression_presets:
        t_recs = splits["test_locked"]
        t_imgs = _load_pil_list(t_recs, lambda im, r, p=preset: apply_compression_chain(im, p))
        t_feats = extract_records_features(
            t_recs, t_imgs, vit_processor, vit_model, wavelet_analyzer, corneal_analyzer, preset
        )
        scored = _score_samples(
            t_feats,
            vit_probe,
            freq_probe,
            eye_probe,
            platt_vit,
            platt_freq,
            platt_eye,
            best_fixed_w,
            platt_fixed,
            q_base_w,
            q_gamma,
            platt_quality,
            tau_for_dicome=tau_dicome_ref,
        )
        test_views_scored[preset] = scored
        all_test_rows.extend(scored)

    # Unseen generator family on `test_locked`: Authentic real portraits vs unseen radial diffusion-blend fakes
    ood_recs: List[Dict[str, Any]] = []
    ood_imgs: List[Image.Image] = []
    for idx_r, rec in enumerate(splits["test_locked"]):
        if int(rec["label"]) == 0:
            r_real = dict(rec)
            r_real["label"] = 0
            ood_recs.append(r_real)
            with Image.open(rec["local_path"]) as im_r:
                ood_imgs.append(im_r.convert("RGB"))
            r_fake = dict(rec)
            r_fake["sample_id"] = f"ood_diff_{rec.get('sample_id')}"
            r_fake["label"] = 1
            ood_recs.append(r_fake)
            with Image.open(rec["local_path"]) as im_f:
                ood_imgs.append(make_unseen_diffusion_blend_ood(im_f, seed=9000 + idx_r))

    ood_feats = extract_records_features(
        ood_recs, ood_imgs, vit_processor, vit_model, wavelet_analyzer, corneal_analyzer, "unseen_diffusion_blend_ood"
    )
    ood_scored = _score_samples(
        ood_feats,
        vit_probe,
        freq_probe,
        eye_probe,
        platt_vit,
        platt_freq,
        platt_eye,
        best_fixed_w,
        platt_fixed,
        q_base_w,
        q_gamma,
        platt_quality,
        tau_for_dicome=tau_dicome_ref,
    )
    test_views_scored["unseen_diffusion_blend_ood"] = ood_scored
    all_test_rows.extend(ood_scored)

    # 6. Summarize Protocol A (Paired Compression Sweep + OOD Generator + Fusion Comparison)
    model_prob_key = {
        "single_vision_vit": "p_single_vit",
        "freq_wavelet_only": "p_freq_only",
        "corneal_eye_only": "p_corneal_only",
        "fixed_weight_fusion": "p_fixed_fusion",
        "quality_aware_fusion": "p_quality_aware",
    }

    protocol_a: Dict[str, Any] = {}
    for v_name, rows in test_views_scored.items():
        y_v = [r["label"] for r in rows]
        view_block: Dict[str, Any] = {
            "n": len(rows),
            "signal_quality_summary": {
                "mean_q_vision": round(float(np.mean([r["q_vision"] for r in rows])), 4),
                "mean_q_freq": round(float(np.mean([r["q_freq"] for r in rows])), 4),
                "mean_q_eye": round(float(np.mean([r["q_eye"] for r in rows])), 4),
                "mean_w_vision_dyn": round(float(np.mean([r["w_vision_dyn"] for r in rows])), 4),
                "mean_w_freq_dyn": round(float(np.mean([r["w_freq_dyn"] for r in rows])), 4),
                "mean_w_eye_dyn": round(float(np.mean([r["w_eye_dyn"] for r in rows])), 4),
            },
            "models": {},
        }
        for m_name, p_key in model_prob_key.items():
            th = locked_thresholds[m_name]
            view_block["models"][m_name] = compute_full_metrics(
                y_v,
                [r[p_key] for r in rows],
                tau_fpr1=th["tau_fpr1pct"],
                tau_youden=th["tau_youden"],
                n_boot=500,
                seed=2026,
            )
        protocol_a[v_name] = view_block

    # Compute Signal Survival Retention vs `clean_raw`
    signal_survival: Dict[str, Any] = {}
    for m_name in model_prob_key:
        clean_auc = protocol_a["clean_raw"]["models"][m_name]["roc_auc"] or 0.5
        clean_tpr1 = protocol_a["clean_raw"]["models"][m_name]["at_fpr1pct_tau"]["tpr"]
        tiers = {}
        for tier in compression_presets:
            auc_t = protocol_a[tier]["models"][m_name]["roc_auc"] or 0.5
            tpr1_t = protocol_a[tier]["models"][m_name]["at_fpr1pct_tau"]["tpr"]
            tiers[tier] = {
                "roc_auc": auc_t,
                "auc_retention_pct_vs_clean": round(100.0 * auc_t / max(1e-4, clean_auc), 2),
                "tpr_at_target_fpr1pct": tpr1_t,
                "realized_fpr_at_target_fpr1pct": protocol_a[tier]["models"][m_name]["at_fpr1pct_tau"]["fpr"],
                "tpr_youden": protocol_a[tier]["models"][m_name]["at_youden_tau"]["tpr"],
            }
        signal_survival[m_name] = tiers

    # Pooled test summary across all 5 evaluation views (N = 310)
    y_pool = [r["label"] for r in all_test_rows]
    pooled_fusion_comparison: Dict[str, Any] = {}
    for m_name in ("single_vision_vit", "fixed_weight_fusion", "quality_aware_fusion"):
        th = locked_thresholds[m_name]
        pooled_fusion_comparison[m_name] = compute_full_metrics(
            y_pool,
            [r[model_prob_key[m_name]] for r in all_test_rows],
            tau_fpr1=th["tau_fpr1pct"],
            tau_youden=th["tau_youden"],
            n_boot=800,
            seed=2026,
        )

    # 7. Protocol B: Verified Capture Conditions & Optical Covariate Stratification
    # Use clean + medium + heavy compression views of `test_locked` (N = 186)
    pb_rows = (
        test_views_scored["clean_raw"]
        + test_views_scored["medium_social_q55_r75"]
        + test_views_scored["heavy_social_webp_q32_r50_blur"]
    )

    by_vn_condition: Dict[str, Any] = {}
    for cat in sorted({r["vietnamese_category"] for r in pb_rows}):
        sub = [r for r in pb_rows if r["vietnamese_category"] == cat]
        y_s = [r["label"] for r in sub]
        by_vn_condition[cat] = {
            "n": len(sub),
            "mean_q_freq": round(float(np.mean([r["q_freq"] for r in sub])), 4),
            "mean_q_eye": round(float(np.mean([r["q_eye"] for r in sub])), 4),
            "mean_eye_roi_px": round(float(np.mean([r["eye_roi_pixels"] for r in sub])), 1),
            "mean_occlusion_glare": round(float(np.mean([r["occlusion_glare_score"] for r in sub])), 4),
            "single_vision_vit": compute_full_metrics(
                y_s,
                [r["p_single_vit"] for r in sub],
                locked_thresholds["single_vision_vit"]["tau_fpr1pct"],
                locked_thresholds["single_vision_vit"]["tau_youden"],
                n_boot=300,
            ),
            "corneal_eye_only": compute_full_metrics(
                y_s,
                [r["p_corneal_only"] for r in sub],
                locked_thresholds["corneal_eye_only"]["tau_fpr1pct"],
                locked_thresholds["corneal_eye_only"]["tau_youden"],
                n_boot=300,
            ),
            "quality_aware_fusion": compute_full_metrics(
                y_s,
                [r["p_quality_aware"] for r in sub],
                locked_thresholds["quality_aware_fusion"]["tau_fpr1pct"],
                locked_thresholds["quality_aware_fusion"]["tau_youden"],
                n_boot=300,
            ),
        }

    # Optical Covariate Stratification (Light2Lie analysis: eye ROI size, lighting, occlusion/glare)
    med_eye_px = float(np.median([r["eye_roi_pixels"] for r in pb_rows]))
    med_glare = float(np.median([r["occlusion_glare_score"] for r in pb_rows]))
    optical_strata_defs = {
        "eye_roi_small": [r for r in pb_rows if r["eye_roi_pixels"] < med_eye_px],
        "eye_roi_adequate": [r for r in pb_rows if r["eye_roi_pixels"] >= med_eye_px],
        "lighting_extreme": [r for r in pb_rows if r["eye_luminance"] < 0.36 or r["eye_luminance"] > 0.72],
        "lighting_balanced": [r for r in pb_rows if 0.36 <= r["eye_luminance"] <= 0.72],
        "occlusion_high_or_glare": [r for r in pb_rows if r["occlusion_glare_score"] > med_glare],
        "occlusion_clear_ocular": [r for r in pb_rows if r["occlusion_glare_score"] <= med_glare],
    }
    by_optical_covariate: Dict[str, Any] = {}
    for s_name, sub in optical_strata_defs.items():
        if len(sub) < 4:
            continue
        y_s = [r["label"] for r in sub]
        by_optical_covariate[s_name] = {
            "n": len(sub),
            "mean_q_eye": round(float(np.mean([r["q_eye"] for r in sub])), 4),
            "corneal_eye_only": compute_full_metrics(
                y_s,
                [r["p_corneal_only"] for r in sub],
                locked_thresholds["corneal_eye_only"]["tau_fpr1pct"],
                locked_thresholds["corneal_eye_only"]["tau_youden"],
                n_boot=300,
            ),
            "quality_aware_fusion": compute_full_metrics(
                y_s,
                [r["p_quality_aware"] for r in sub],
                locked_thresholds["quality_aware_fusion"]["tau_fpr1pct"],
                locked_thresholds["quality_aware_fusion"]["tau_youden"],
                n_boot=300,
            ),
        }

    # 8. Direction 2: Matched-Coverage Selective Deferral (DiCoME vs ViT Margin)
    deferral_eval = {
        "pooled_challenge_suite_youden_tau": _evaluate_matched_coverage(
            all_test_rows,
            tau=locked_thresholds["quality_aware_fusion"]["tau_youden"],
        ),
        "heavy_and_ood_subset_youden_tau": _evaluate_matched_coverage(
            test_views_scored["heavy_social_webp_q32_r50_blur"] + test_views_scored["unseen_diffusion_blend_ood"],
            tau=locked_thresholds["quality_aware_fusion"]["tau_youden"],
        ),
    }

    # Persist per-sample JSONL trace
    jsonl_path = out_dir / "per_sample_signal_and_deferral.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for row in all_test_rows:
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = {
        "evaluation_provenance": "EMPIRICAL_LOCKED_VN_COMPRESSION_QUALITY_DEFERRAL",
        "generated_at_utc": locked_protocol["generated_at_utc"],
        "runtime_seconds": round(time.time() - t0, 2),
        "leakage_audit": leak_audit,
        "locked_protocol_file": str(out_dir / "locked_protocol_and_thresholds.json"),
        "fusion_hyperparams": locked_protocol["fusion_hyperparams"],
        "locked_thresholds": locked_thresholds,
        "protocol_a_compression_and_ood": protocol_a,
        "signal_survival_matrix": signal_survival,
        "pooled_fusion_comparison": pooled_fusion_comparison,
        "protocol_b_vietnamese_conditions": {
            "by_capture_condition": by_vn_condition,
            "by_optical_covariate_light2lie": by_optical_covariate,
            "median_eye_roi_pixels": med_eye_px,
            "median_occlusion_glare": round(med_glare, 4),
        },
        "selective_deferral_matched_coverage": deferral_eval,
    }
    (out_dir / "vn_compression_deferral_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_markdown_and_latex_reports(results, out_dir)
    print(f"[Done] Wrote Vietnamese compression, quality-aware fusion & DiCoME deferral benchmark to {out_dir}")
    return results


def _fmt_ci(ci: Optional[Sequence[Any]]) -> str:
    if not ci or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return "--"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _write_markdown_and_latex_reports(res: Dict[str, Any], out_dir: Path) -> None:
    pa = res["protocol_a_compression_and_ood"]
    surv = res["signal_survival_matrix"]
    pooled = res["pooled_fusion_comparison"]
    pb_cond = res["protocol_b_vietnamese_conditions"]["by_capture_condition"]
    pb_opt = res["protocol_b_vietnamese_conditions"]["by_optical_covariate_light2lie"]
    def_pool = res["selective_deferral_matched_coverage"]["pooled_challenge_suite_youden_tau"]

    lines: List[str] = [
        "# Benchmark Anh Nguoi Viet Duoi Nen, Hop Nhat Theo Chat Luong Tin Hieu & Hoan Ket Luan DiCoME",
        "",
        f"- **Provenance**: `{res['evaluation_provenance']}` | Runtime: `{res['runtime_seconds']}s`",
        f"- **Zero-Leakage Audit**: train={res['leakage_audit']['train_samples']} / cal_val={res['leakage_audit']['cal_val_samples']} / test_locked={res['leakage_audit']['test_locked_samples']} (overlap = 0/0/0)",
        f"- **Quality-Aware Fusion Hyperparams (locked on `cal_val`)**: base={res['fusion_hyperparams']['quality_aware_base_weights']}, gamma={res['fusion_hyperparams']['quality_aware_gamma']}",
        "",
        "## 1. Giao thuc A: Song con Tin hieu sau Nen & So sanh 3 Chien luoc Hop nhat (`NTIRE 2026` / `FreqFusion`)",
        "",
        "| Muc nen / Bien the (`test_locked`) | `q_freq` | `q_eye` | ViT AUC | Wavelet/FFT AUC | Corneal AUC | Fixed Fusion AUC | **Quality-Aware AUC** | Quality TPR@FPR<=1% | Quality Realized FPR (95% CI) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for view in ("clean_raw", "mild_jpeg_q80", "medium_social_q55_r75", "heavy_social_webp_q32_r50_blur", "unseen_diffusion_blend_ood"):
        vb = pa[view]
        sq = vb["signal_quality_summary"]
        ms = vb["models"]
        qf_m = ms["quality_aware_fusion"]
        f1 = qf_m["at_fpr1pct_tau"]
        lines.append(
            f"| `{view}` | {sq['mean_q_freq']:.3f} | {sq['mean_q_eye']:.3f} | "
            f"{ms['single_vision_vit']['roc_auc']:.4f} | {ms['freq_wavelet_only']['roc_auc']:.4f} | "
            f"{ms['corneal_eye_only']['roc_auc']:.4f} | {ms['fixed_weight_fusion']['roc_auc']:.4f} | "
            f"**{qf_m['roc_auc']:.4f}** | {f1['tpr']:.4f} {_fmt_ci(f1['tpr_wilson_ci95'])} | {f1['fpr']:.4f} {_fmt_ci(f1['fpr_wilson_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "### Tong hop Tren Toan bo Tap Kiem thu Gop (`N=310`, `clean + compressed + unseen_ood`)",
            "",
            "| Mo hinh | ROC-AUC (95% Boot CI) | TPR @ Target FPR 1% (95% CI) | Realized FPR @ Target 1% (95% CI) | Youden TPR / FPR | ECE |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for m in ("single_vision_vit", "fixed_weight_fusion", "quality_aware_fusion"):
        mm = pooled[m]
        f1 = mm["at_fpr1pct_tau"]
        fy = mm["at_youden_tau"]
        lines.append(
            f"| `{m}` | {mm['roc_auc']:.4f} {_fmt_ci(mm['roc_auc_ci95'])} | "
            f"{f1['tpr']:.4f} {_fmt_ci(f1['tpr_wilson_ci95'])} | {f1['fpr']:.4f} {_fmt_ci(f1['fpr_wilson_ci95'])} | "
            f"{fy['tpr']:.4f} / {fy['fpr']:.4f} | {mm['ece']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 2. Giao thuc B: Dieu kien Chup Nguoi Viet & Phan tang Quang hoc Mat (`Light2Lie`)",
            "",
            "| Dieu kien chup / Phan tang | N | `q_freq` | `q_eye` | ViT AUC | Corneal AUC | Quality-Aware AUC | Quality BalAcc (`tau_youden`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for cat, cb in pb_cond.items():
        lines.append(
            f"| `{cat}` | {cb['n']} | {cb['mean_q_freq']:.3f} | {cb['mean_q_eye']:.3f} | "
            f"{cb['single_vision_vit']['roc_auc']} | {cb['corneal_eye_only']['roc_auc']} | "
            f"{cb['quality_aware_fusion']['roc_auc']} | {cb['quality_aware_fusion']['at_youden_tau']['balanced_accuracy']:.4f} |"
        )
    for sk, sb in pb_opt.items():
        lines.append(
            f"| `optical:{sk}` | {sb['n']} | -- | {sb['mean_q_eye']:.3f} | -- | "
            f"{sb['corneal_eye_only']['roc_auc']} | {sb['quality_aware_fusion']['roc_auc']} | "
            f"{sb['quality_aware_fusion']['at_youden_tau']['balanced_accuracy']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. Hoan Ket luan Co Chon loc (`DiCoME` ICML 2026 vs ViT Probability Margin) tren Cung Ty le Tu dong Ket luan",
            "",
            "| Coverage (Tu dong) | Baseline ViT-Margin Err (FPR / FNR) | **DiCoME Quality+Conflict Err (FPR / FNR)** | Giam Sai so Tuyet doi |",
            "| :---: | :---: | :---: | :---: |",
        ]
    )
    b_pts = def_pool["vit_margin_baseline_on_vit"]["points"]
    d_pts = def_pool["dicome_conflict_quality_proposed"]["points"]
    for bp, dp in zip(b_pts, d_pts):
        diff = round((bp["selective_error_rate"] - dp["selective_error_rate"]) * 100.0, 2)
        lines.append(
            f"| {int(round(dp['realized_coverage']*100))}% ({dp['auto_decided_n']}/{def_pool['n_total']}) | "
            f"{bp['selective_error_rate']:.4f} ({bp['realized_fpr']:.4f} / {bp['realized_fnr']:.4f}) | "
            f"**{dp['selective_error_rate']:.4f}** ({dp['realized_fpr']:.4f} / {dp['realized_fnr']:.4f}) | {diff:+.2f}% |"
        )
    lines.append(
        f"\n- **AURC (thap hon la tot hon)**: ViT-Margin Baseline = `{def_pool['vit_margin_baseline_on_vit']['normalized_aurc']:.4f}`, "
        f"DiCoME Proposed = `{def_pool['dicome_conflict_quality_proposed']['normalized_aurc']:.4f}`."
    )

    (out_dir / "vn_compression_deferral_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Write LaTeX tables ready for PhD thesis
    tex_fusion = [
        "% Auto-generated from real test_locked inference (DO NOT EDIT MANUALLY)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Hieu nang hop nhat tren tap kiem thu khoa nguoi Viet (N=310 luot suy luan gom anh goc, nen da tang va gia mao chua thay)}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Mo hinh & ROC-AUC (95\\% CI) & TPR@FPR1\\% & FPR Thuc te (95\\% CI) & Bal.Acc ($\\tau^*$) & ECE \\\\",
        "\\midrule",
    ]
    for m, label in (
        ("single_vision_vit", "Single Vision (ViT-Base)"),
        ("fixed_weight_fusion", "Fixed-Weight Fusion"),
        ("quality_aware_fusion", "\\textbf{Quality-Aware Fusion (Proposed)}"),
    ):
        mm = pooled[m]
        f1 = mm["at_fpr1pct_tau"]
        fy = mm["at_youden_tau"]
        tex_fusion.append(
            f"{label} & {mm['roc_auc']:.3f} & {f1['tpr']:.3f} & {f1['fpr']:.3f} & {fy['balanced_accuracy']:.3f} & {mm['ece']:.3f} \\\\"
        )
    tex_fusion.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (out_dir / "table_fusion_fpr1pct_comparison.tex").write_text("\n".join(tex_fusion), encoding="utf-8")

    tex_surv = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Phan tich song con tin hieu qua cac muc nen NTIRE-style tren anh nguoi Viet}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Nhanh tin hieu & Clean Raw & Mild JPEG Q80 & Social Q55+0.75x & Heavy WebP Q32+Blur \\\\",
        "\\midrule",
    ]
    for m, label in (
        ("single_vision_vit", "ViT Spatial"),
        ("freq_wavelet_only", "2D-FFT + Haar Wavelet"),
        ("corneal_eye_only", "Corneal / Eye Specular"),
        ("quality_aware_fusion", "\\textbf{Quality-Aware Fusion}"),
    ):
        vals = [f"{surv[m][t]['roc_auc']:.3f} ({surv[m][t]['auc_retention_pct_vs_clean']:.0f}\\%)" for t in ("clean_raw", "mild_jpeg_q80", "medium_social_q55_r75", "heavy_social_webp_q32_r50_blur")]
        tex_surv.append(f"{label} & " + " & ".join(vals) + " \\\\")
    tex_surv.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (out_dir / "table_vn_compression_signal_survival.tex").write_text("\n".join(tex_surv), encoding="utf-8")

    tex_def = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{So sanh hoan ket luan co chon loc DiCoME voi nguong xac suat ViT o cung ty le tu dong ket luan}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Coverage & ViT-Margin Err (FPR/FNR) & DiCoME Err (FPR/FNR) & Tu dong / Hoan \\\\",
        "\\midrule",
    ]
    for bp, dp in zip(b_pts, d_pts):
        tex_def.append(
            f"{int(round(dp['realized_coverage']*100))}\\% & "
            f"{bp['selective_error_rate']:.3f} ({bp['realized_fpr']:.3f}/{bp['realized_fnr']:.3f}) & "
            f"{dp['selective_error_rate']:.3f} ({dp['realized_fpr']:.3f}/{dp['realized_fnr']:.3f}) & "
            f"{dp['auto_decided_n']} / {dp['deferred_n']} \\\\"
        )
    tex_def.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (out_dir / "table_dicome_selective_deferral.tex").write_text("\n".join(tex_def), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Portrait Compression, Quality Fusion & DiCoME Deferral Benchmark")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--vit-dir", type=str, default=str(DEFAULT_VIT_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    run_benchmark(Path(args.manifest), Path(args.vit_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
